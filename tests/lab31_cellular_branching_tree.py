#!/usr/bin/env python3
"""
tests/lab31_cellular_branching_tree.py
═══════════════════════════════════════════════════════════════════════════════
LAB 31 — ÁRBOL DE BIFURCACIÓN CON SELECCIÓN DE NODO ÓPTIMO Y SOFTPLUS
SSOT: 
  • Proyección continua en R+ con Softplus analítico (cero floor discontinuo).
  • Selección del nodo con máximo contraste teleológico (sin dilución).
  • Capa 0 virgen, acoplamiento en Fact Band L*=19.
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, time, math, argparse
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

from mlx_vlm import load, stream_generate
import aether_native_c

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")
MAX_BRANCHES = 5
MAX_GENERATION_TOKENS = 2500
L_FACT_BAND = 19

def section(title):
    print("\n" + "═" * 84)
    print(f"  {title}")
    print("═" * 84)

def to_unit(vec_mx):
    norm = mx.sqrt(mx.sum(vec_mx * vec_mx)) + 1e-12
    return vec_mx / norm

def to_numpy_f32(mlx_arr):
    arr_f32 = mlx_arr.astype(mx.float32)
    mx.eval(arr_f32)
    return np.array(arr_f32, copy=True).reshape(-1)

CANONICAL_PUZZLE_PROMPT = (
    "Tres robots (Alfa, Beta y Gamma) deben cruzar un puente colgante estrecho en plena noche. "
    "Solo pueden cruzar un máximo de dos robots juntos a la vez. Tienen una sola linterna de mano portátil "
    "que funciona con batería eléctrica, y sin ella encendida es imposible ver los tablones rotos del suelo. "
    "Alfa tarda exactamente 1 minuto en cruzar el puente. "
    "Beta tarda exactamente 2 minutos en cruzar el puente. "
    "Gamma tarda exactamente 5 minutos en cruzar el puente. "
    "Reglas estrictas de traslado: "
    "1. Cuando dos robots cruzan juntos, deben caminar juntos llevando la linterna de mano y avanzan al ritmo del más lento. "
    "2. La linterna no se puede lanzar por el aire; para que vuelva al punto de partida, un robot debe caminar de regreso cruzando el puente a pie con la linterna en la mano, tardando su tiempo individual. "
    "Estrategia exacta paso a paso para que los tres robots crucen al otro lado en un tiempo total acumulado de exactamente 8 minutos (mostrando el tiempo de cada ida y cada vuelta): "
)

def run_deber_ser_branching():
    section("LAB 31 — SELECCIÓN DE NODO ÓPTIMO Y LOGITS CONTINUOS EN R+")
    print(f"  Modelo Base             : Qwen3.5-0.8B (Apple Silicon UMA)")
    print(f"  Capa 0                  : 100% VIRGEN en todas las ramas")
    print(f"  Frontera de Inyección   : Residual Stream en Capa L* = {L_FACT_BAND}")
    print(f"  Presupuesto de Salida   : Hasta {MAX_GENERATION_TOKENS} tokens")

    model, processor = load(MODEL_PATH)
    tok = getattr(processor, "tokenizer", processor)
    lm_model = model.language_model.model
    num_layers = len(lm_model.layers)
    orig_layers = list(lm_model.layers)

    embed = lm_model.embed_tokens if hasattr(lm_model, "embed_tokens") else model.language_model.embed_tokens
    bits = getattr(embed, "bits", 4)
    group_size = getattr(embed, "group_size", 64)
    deq_W = mx.dequantize(embed.weight, embed.scales, getattr(embed, "biases", None), group_size=group_size, bits=bits).astype(mx.float32)
    mx.eval(deq_W)
    D = deq_W.shape[-1]

    final_norm = lm_model.norm
    orig_lm_head = getattr(model.language_model, "lm_head", None)
    if orig_lm_head is None:
        class TiedHead:
            def __init__(self, emb): self.emb = emb
            def __call__(self, x): return self.emb.as_linear(x)
        orig_lm_head = TiedHead(embed)

    prompt_ids = mx.array(tok.encode(CANONICAL_PUZZLE_PROMPT))[None, :]
    T = prompt_ids.shape[1]
    X_embed = embed(prompt_ids)[0].astype(mx.float32)
    eos_id = tok.eos_token_id if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None else 151643
    w_eos = to_unit(deq_W[eos_id]).astype(mx.float32)
    mx.eval(X_embed, w_eos)

    poles = aether_native_c.extract_tetrapolar_poles_metal(X_embed, w_eos, T // 2)
    mx.eval(poles["u_onto"], poles["u_teleo"], poles["u_anti"], poles["u_eos"])

    aether_native_c.hilbert_memory_reset()

    baseline_h19 = None
    prev_u_advance = None
    early_stopped = False
    stopped_at_branch = None

    u_onto_np  = to_numpy_f32(poles["u_onto"])
    u_teleo_np = to_numpy_f32(poles["u_teleo"])
    u_anti_np  = to_numpy_f32(poles["u_anti"])
    u_eos_np   = to_numpy_f32(poles["u_eos"])

    # Estructura para registrar nodos y seleccionar el óptimo
    branch_nodes = []
    best_node_idx = 0
    max_observed_contrast = -1.0

    print(f"\n✓ Tetrapolo extraído en silicio.")

    for branch_idx in range(MAX_BRANCHES):
        branch_letter = chr(65 + branch_idx)
        section(f"RAMA {branch_letter} [Nodo {branch_idx + 1}/{MAX_BRANCHES}]")

        # Inyectar el avance del mejor nodo conocido hasta el momento
        if len(branch_nodes) > 0:
            u_markov = branch_nodes[best_node_idx]["vector"]
            print(f"• Célula 02 (Memoria): Inyectando vector óptimo (Nodo {chr(65+best_node_idx)}) en L*={L_FACT_BAND}")
        else:
            u_markov = None
            print(f"• Célula 02 (Memoria): Rama Raíz (Exploración virgen sin perturbación)")

        class FactBandHomologousHook:
            def __init__(self, layer, markov_dir, alpha=0.35):
                self.layer = layer
                self.markov_dir = mx.array(markov_dir)[None, None, :] if markov_dir is not None else None
                self.alpha = alpha
                self.captured_h = None

            def __getattr__(self, name): return getattr(self.layer, name)

            def __call__(self, x, **kwargs):
                out = self.layer(x, **kwargs)
                h_last = out[:, -1:, :].astype(mx.float32)
                norm_h = mx.sqrt(mx.sum(h_last * h_last, axis=-1, keepdims=True)) + 1e-12

                if self.markov_dir is not None:
                    h_steered = h_last + (self.alpha * norm_h) * self.markov_dir
                    norm_s = mx.sqrt(mx.sum(h_steered * h_steered, axis=-1, keepdims=True)) + 1e-12
                    h_steered = h_steered * (norm_h / norm_s)
                    out = mx.concatenate([out[:, :-1, :], h_steered.astype(out.dtype)], axis=1)
                    self.captured_h = h_steered[0, -1, :].astype(mx.float32)
                else:
                    self.captured_h = h_last[0, -1, :].astype(mx.float32)

                mx.eval(self.captured_h)
                return out

        class Layer18Hook:
            def __init__(self, layer): self.layer, self.h18 = layer, None
            def __getattr__(self, name): return getattr(self.layer, name)
            def __call__(self, x, **kwargs):
                out = self.layer(x, **kwargs)
                if self.layer is orig_layers[L_FACT_BAND - 1]:
                    self.h18 = out[0, -1, :].astype(mx.float32)
                    mx.eval(self.h18)
                return out

        hook_l18 = Layer18Hook(orig_layers[L_FACT_BAND - 1])
        hook_l19 = FactBandHomologousHook(orig_layers[L_FACT_BAND], u_markov, alpha=0.35)

        for l in range(num_layers):
            if l == L_FACT_BAND - 1: lm_model.layers[l] = hook_l18
            elif l == L_FACT_BAND:   lm_model.layers[l] = hook_l19
            else:                    lm_model.layers[l] = orig_layers[l]

        out_fwd = model.language_model(prompt_ids)
        logits_raw = out_fwd.logits[0, -1, :].astype(mx.float32)
        mx.eval(logits_raw)

        for l in range(num_layers): lm_model.layers[l] = orig_layers[l]

        h_current_19 = hook_l19.captured_h
        h_prev_18    = hook_l18.h18
        v_layer = (h_current_19 - h_prev_18).astype(mx.float32)
        mx.eval(h_current_19, v_layer)

        if branch_idx == 0:
            baseline_h19 = to_numpy_f32(h_current_19)

        pred_rest = aether_native_c.tetrapolar_predictor_step_metal(
            h_current_19, v_layer, poles["u_onto"], poles["u_teleo"], poles["u_anti"], poles["u_eos"], tau=0.0
        )
        pred_motion = aether_native_c.tetrapolar_predictor_step_metal(
            h_current_19, v_layer, poles["u_onto"], poles["u_teleo"], poles["u_anti"], poles["u_eos"], tau=0.5
        )

        h_star_motion = pred_motion["h_star"]
        tel = pred_motion["telemetry"]
        mx.eval(h_star_motion)

        norm_h19 = float(np.linalg.norm(to_numpy_f32(h_current_19)))
        omega    = tel["omega_angular_velocity"]
        kappa    = tel["curvature_kappa"]

        # Fuerzas dinámicas tangenciales en R+
        g_onto  = max(0.0, float(tel["grad_onto"]))
        g_teleo = max(0.0, float(tel["grad_teleo"]))
        g_anti  = max(0.0, float(tel["grad_anti"]))
        g_eos   = max(0.0, float(tel["grad_eos"]))

        # Proyecciones estáticas en R+
        h19_u_np = to_numpy_f32(h_current_19) / norm_h19
        pos_onto  = max(0.0, float(np.dot(h19_u_np, u_onto_np)))
        pos_teleo = max(0.0, float(np.dot(h19_u_np, u_teleo_np)))
        pos_anti  = max(0.0, float(np.dot(h19_u_np, u_anti_np)))

        contrast_dynamic = g_teleo / (g_anti + 1e-6)
        contrast_static  = pos_teleo / (pos_anti + 1e-6)

        # Voronoi
        h_norm_rest = final_norm(pred_rest["h_star"][None, None, :])
        h_norm_mot  = final_norm(h_star_motion[None, None, :])
        z_rest = orig_lm_head(h_norm_rest)[0, 0, :].astype(mx.float32)
        z_mot  = orig_lm_head(h_norm_mot)[0, 0, :].astype(mx.float32)
        mx.eval(z_rest, z_mot)

        token_rest_id = int(mx.argmax(z_rest))
        token_mot_id  = int(mx.argmax(z_mot))
        voronoi_consistent = (token_rest_id == token_mot_id)

        print("\n  [TELEMETRÍA CINEMÁTICA R+ EN L*=19]:")
        print(f"  • Norma Estado ||h_19||  : {norm_h19:.4f}")
        print(f"  • Velocidad Angular ω    : {omega:.6f} rad/paso")
        print(f"  • Curvatura κ            : {kappa:.6f}")
        print(f"  • ∇_teleo / ∇_anti (R+)  : {g_teleo:.4f} / {g_anti:.4f} (Contraste Dinámico = {contrast_dynamic:.4f})")
        print(f"  • <h,U_tel> / <h,U_anti> : {pos_teleo:.4f} / {pos_anti:.4f} (Contraste Estático  = {contrast_static:.4f})")
        print(f"  • Voronoi h*(0) vs h*(τ) : {repr(tok.decode([token_rest_id]))} vs {repr(tok.decode([token_mot_id]))} | {'Consistente ✅' if voronoi_consistent else 'En Giro 🔄'}")

        # Guardar nodo y evaluar contraste
        h_star_np = to_numpy_f32(h_star_motion)
        delta_advance = h_star_np - baseline_h19
        norm_adv = np.linalg.norm(delta_advance)

        shift_angle_deg = 90.0
        if norm_adv > 1e-5:
            u_adv_raw = delta_advance / norm_adv
            # Torsión dialéctica expulsiva ante colisión
            if g_anti >= 0.03:
                u_adv_steered = u_adv_raw - 0.40 * u_anti_np + 0.40 * u_teleo_np
                u_advance_current = u_adv_steered / np.linalg.norm(u_adv_steered)
                print(f"  • Célula 01 (Torsión)    : Activada elusión dialéctica (g_anti={g_anti:.4f})")
            else:
                u_advance_current = u_adv_raw

            slot = aether_native_c.hilbert_memory_ingest(
                mx.array(u_advance_current), timestamp=int(branch_idx * 100), energy=float(kappa)
            )["slot_idx"]

            if prev_u_advance is not None:
                dot_val = max(-1.0, min(1.0, float(np.dot(u_advance_current, prev_u_advance))))
                shift_angle_deg = math.degrees(math.acos(dot_val))
            prev_u_advance = u_advance_current

            node_data = {
                "branch_idx": branch_idx,
                "vector": u_advance_current,
                "contrast_dyn": contrast_dynamic,
                "contrast_stat": contrast_static,
                "pos_teleo": pos_teleo,
                "g_teleo": g_teleo
            }
            branch_nodes.append(node_data)

            # Rastrear el nodo con mejor alineamiento objetivo
            if contrast_dynamic > max_observed_contrast:
                max_observed_contrast = contrast_dynamic
                best_node_idx = len(branch_nodes) - 1

            print(f"  • Célula 02 (Memoria)    : Avance Δh_{branch_letter} en Slot {slot} | Deflexión inter-rama Δθ = {shift_angle_deg:.4f}°")
        else:
            print("  • Célula 02 (Memoria)    : Rama raíz registrada como ancla de referencia.")

        # Parada si la deflexión angular colapsa y el contraste ya superó el umbral
        is_in_positive_hemisphere = (pos_teleo > 0.02)
        has_real_contrast = (contrast_dynamic >= 2.0)
        is_kinematically_stable = (shift_angle_deg < 0.15)

        if branch_idx >= 1 and is_in_positive_hemisphere and has_real_contrast and is_kinematically_stable and voronoi_consistent:
            section("🏆 PARADA ANTICIPADA: CONVERGENCIA EN NODO ÓPTIMO")
            print(f"• El árbol colapsó en certidumbre en Rama {branch_letter} (Nodo {branch_idx + 1}).")
            early_stopped = True
            stopped_at_branch = branch_idx
            break

    # ── VERBALIZACIÓN TERMINAL CONTINUA EN R+ ──
    section(f"VERBALIZACIÓN CONTINUA EN SILICIO (LOGITS EN R+ CON SOFTPLUS)")
    best_letter = chr(65 + branch_nodes[best_node_idx]["branch_idx"])
    print(f"Inyectando vector del Nodo Óptimo [{best_letter}] (Contraste={branch_nodes[best_node_idx]['contrast_dyn']:.2f}):\n")

    u_final = branch_nodes[best_node_idx]["vector"]
    hook_final = FactBandHomologousHook(orig_layers[L_FACT_BAND], u_final, alpha=0.35)
    for l in range(num_layers):
        if l == L_FACT_BAND: lm_model.layers[l] = hook_final
        else:                lm_model.layers[l] = orig_layers[l]

    # Wrapper Softplus continuo (R+ analítico, conserva entropía y gradiente)
    class SoftplusLogitsHead:
        def __init__(self, base_head): self.base_head = base_head
        def __getattr__(self, name): return getattr(self.base_head, name)
        def __call__(self, h):
            z = self.base_head(h)
            # Softplus suave en R+: ln(1 + exp(z)), estrictamente no negativo y diferenciable
            return mx.log(1.0 + mx.exp(mx.clip(z, -20.0, 50.0)))

    model.language_model.lm_head = SoftplusLogitsHead(orig_lm_head)

    prompt_chat_final = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": CANONICAL_PUZZLE_PROMPT}]}
    ], add_generation_prompt=True)

    print("Emisión fluida en vivo:")
    print("─" * 84)
    toks_out = []
    t0_gen = time.perf_counter()
    for r in stream_generate(model, processor, prompt=prompt_chat_final, max_tokens=MAX_GENERATION_TOKENS):
        toks_out.append(r.text)
        print(r.text, end="", flush=True)
    t_gen = time.perf_counter() - t0_gen

    model.language_model.lm_head = orig_lm_head
    for l in range(num_layers): lm_model.layers[l] = orig_layers[l]

    print("\n" + "─" * 84)
    print(f"✓ Generación completada: {len(toks_out)} tokens en {t_gen:.2f}s ({len(toks_out)/max(t_gen, 1e-5):.1f} tok/s)")

    section("DICTAMEN FINAL LAB 31")
    print(f"• Nodos explorados          : {stopped_at_branch + 1 if early_stopped else MAX_BRANCHES}")
    print(f"• Nodo óptimo inyectado     : Rama {best_letter}")
    print(f"• Parada anticipada         : {'SÍ' if early_stopped else 'NO (LÍMITE ALCANZADO)'}")
    print(f"• Dominio de Logits         : Softplus continuo en R+")

if __name__ == "__main__":
    run_deber_ser_branching()
