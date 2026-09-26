#!/usr/bin/env python3
"""
tests/lab32_trajectory_conjugate_field.py
═══════════════════════════════════════════════════════════════════════════════
LAB 32 — CARTOGRAFÍA COMPLETA DE PLANOS CONJUGADOS DEL TETRAPOLO EN S^{D-1}
SSOT: 
  • Trayectoria completa a través de TODAS las capas (0..23) y tokens.
  • Cero parada prematura. Medición del bivector instantáneo h ∧ v.
  • Mapeo de los planos conjugados: [Onto ∧ Teleo], [Teleo ∧ Anti], [Onto ∧ Anti].
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, time, math
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

from mlx_vlm import load, stream_generate
import aether_native_c

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

def section(title):
    print("\n" + "═" * 86)
    print(f"  {title}")
    print("═" * 86)

def to_unit(vec_mx):
    return vec_mx / (mx.sqrt(mx.sum(vec_mx * vec_mx)) + 1e-12)

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

def run_conjugate_field_mapping():
    section("LAB 32 — ANÁLISIS DE PLANOS CONJUGADOS DEL TETRAPOLO [Qwen3.5-0.8B]")
    print("  Objetivo: Mapear la trayectoria completa y sus bivectores h ∧ v en silicio.")

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

    # 1. Extracción en silicio del Tetrapolo Nativo
    prompt_ids = mx.array(tok.encode(CANONICAL_PUZZLE_PROMPT))[None, :]
    T = prompt_ids.shape[1]
    X_embed = embed(prompt_ids)[0].astype(mx.float32)
    eos_id = tok.eos_token_id if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None else 151643
    w_eos = to_unit(deq_W[eos_id]).astype(mx.float32)
    mx.eval(X_embed, w_eos)

    poles = aether_native_c.extract_tetrapolar_poles_metal(X_embed, w_eos, T // 2)
    mx.eval(poles["u_onto"], poles["u_teleo"], poles["u_anti"], poles["u_eos"])

    u_O = to_numpy_f32(poles["u_onto"])
    u_T = to_numpy_f32(poles["u_teleo"])
    u_A = to_numpy_f32(poles["u_anti"])
    u_E = to_numpy_f32(poles["u_eos"])

    print(f"✓ Tetrapolo extraído (D={D}).")
    print(f"  • Acoplamientos mutuos entre polos:")
    print(f"    - <Onto, Teleo> : {np.dot(u_O, u_T):+.4f}")
    print(f"    - <Teleo, Anti> : {np.dot(u_T, u_A):+.4f} (Ortogonal estricto)")
    print(f"    - <Onto, Anti>  : {np.dot(u_O, u_A):+.4f}")
    print(f"    - <Teleo, EOS>  : {np.dot(u_T, u_E):+.4f}")

    # ── 2. MONITOREO DE TODAS LAS CAPAS EN PREFILL (CERO CORTES) ─────────────
    section("2. MAPEO EN PROFUNDIDAD: TRAYECTORIA Y BIVECTORES CONJUGADOS (CAPAS 0 A 23)")
    print(f"  {'Capa':<5} │ {'||h||':<7} │ {'κ_Lagrange':<11} │ {'Ω_OT (Giro)':<13} │ {'Ω_TA (Dialéc)':<13} │ {'Ω_OA (Frontera)':<15} │ Estado Dominante")
    print("  " + "─" * 86)

    captured_layers = {}
    class FullProfileHook:
        def __init__(self, layer, idx): self.layer, self.idx = layer, idx
        def __getattr__(self, name): return getattr(self.layer, name)
        def __call__(self, x, **kwargs):
            out = self.layer(x, **kwargs)
            h = out[0, -1, :].astype(mx.float32)
            mx.eval(h)
            captured_layers[self.idx] = to_numpy_f32(h)
            return out

    for l in range(num_layers):
        lm_model.layers[l] = FullProfileHook(orig_layers[l], l)

    _ = model.language_model(prompt_ids)

    for l in range(num_layers):
        lm_model.layers[l] = orig_layers[l]

    # Analizar la cinemática de Lie y los planos conjugados capa por capa
    for l in range(1, num_layers):
        h_curr = captured_layers[l]
        h_prev = captured_layers[l - 1]
        
        norm_h = np.linalg.norm(h_curr)
        h_u = h_curr / norm_h
        
        v = h_curr - h_prev
        v_perp = v - np.dot(v, h_u) * h_u
        norm_vp = np.linalg.norm(v_perp)
        v_u = v_perp / (norm_vp + 1e-12)

        # Curvatura
        if l >= 2:
            h_p2 = captured_layers[l - 2]
            v_p = h_prev - h_p2
            a = v - v_p
            biv_sq = max(0.0, np.dot(v, v)*np.dot(a, a) - np.dot(v, a)**2)
            kappa = math.sqrt(biv_sq) / (np.linalg.norm(v)**3 + 1e-12)
        else:
            kappa = 0.0

        # Proyecciones estáticas sobre los 4 polos
        p_O = np.dot(h_u, u_O)
        p_T = np.dot(h_u, u_T)
        p_A = np.dot(h_u, u_A)
        p_E = np.dot(h_u, u_E)

        # Proyecciones dinámicas (velocidad tangencial)
        v_O = np.dot(v_u, u_O)
        v_T = np.dot(v_u, u_T)
        v_A = np.dot(v_u, u_A)
        v_E = np.dot(v_u, u_E)

        # BIVECTORES CONJUGADOS (Momento angular proyectado sobre cada plano 2D)
        # Omega_XY = <h, X><v, Y> - <h, Y><v, X>
        omega_OT = (p_O * v_T - p_T * v_O) # Plano Giroscópico (Onto -> Teleo)
        omega_TA = (p_T * v_A - p_A * v_T) # Plano Dialéctico (Teleo <-> Anti)
        omega_OA = (p_O * v_A - p_A * v_O) # Plano de Tensión Factual (Onto <-> Anti)

        # Clasificación del régimen
        if abs(omega_TA) > abs(omega_OT):
            regime = "⚠️ CIJALLAMIENTO DIALÉCTICO"
        elif omega_OT > 0.01:
            regime = "✅ TRACCIÓN TELEOLÓGICA"
        else:
            regime = "🔄 DERIVA SINTÁCTICA"

        print(f"  L{l:02d}  │ {norm_h:6.2f}  │ {kappa:10.5f}  │ {omega_OT:+12.5f} │ {omega_TA:+12.5f} │ {omega_OA:+14.5f}  │ {regime}")

    # ── 3. GENERACIÓN COMPLETA OBSERVANDO LA EVOLUCIÓN DINÁMICA ──────────────
    section("3. VERBALIZACIÓN COMPLETA (PRESIONANDO EL RAZONAMIENTO HASTA 2500 TOKENS)")
    
    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": CANONICAL_PUZZLE_PROMPT}]}
    ], add_generation_prompt=True)

    print("Generando texto en vivo:")
    print("─" * 86)
    toks = []
    t0 = time.perf_counter()
    for r in stream_generate(model, processor, prompt=prompt_chat, max_tokens=1500):
        toks.append(r.text)
        print(r.text, end="", flush=True)
    t_gen = time.perf_counter() - t0
    print("\n" + "─" * 86)
    print(f"✓ Generación completada: {len(toks)} tokens en {t_gen:.2f}s ({len(toks)/max(t_gen, 1e-5):.1f} tok/s)")

    section("DICTAMEN LAB 32")
    print("  • Mapeo continuo de bivectores de Lie concluido.")

if __name__ == "__main__":
    run_conjugate_field_mapping()