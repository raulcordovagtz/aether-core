#!/usr/bin/env python3
"""
tests/lab11_fact_band_routing.py
═══════════════════════════════════════════════════════════════════════════════
LAB 11 — MATRIZ CAUSAL DE DIRECCIONAMIENTO EN LA CANDIDATE FACT BAND
Evaluación de Causalidad Pura con 8 Controles (C0 a C7) sobre Qwen3.5-0.8B
SSOT: docs/Harness/MarkovMemoryCell/Hito 2.2.md
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, time
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import aether_native_c
from mlx_vlm import load

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def run_lab11():
    section("LAB 11 — MATRIZ CAUSAL: DIRECCIONAMIENTO ASOCIATIVO EN LA CANDIDATE FACT BAND")
    print(f"  Modelo Base: {MODEL_PATH}")

    model, processor = load(MODEL_PATH)
    tok = getattr(processor, "tokenizer", processor)
    lm_model = model.language_model.model
    num_layers = len(lm_model.layers)
    D = lm_model.layers[0].self_attn.q_proj.weight.shape[-1] if hasattr(lm_model.layers[0], "self_attn") else 1024

    prompt_text = "The capital of France is"
    prompt_ids = mx.array(tok.encode(prompt_text))[None, :]
    tokens_target_str = " Paris"
    target_tok_id = tok.encode(tokens_target_str)[0] if len(tok.encode(tokens_target_str)) > 0 else tok.encode("Paris")[0]
    wrong_tok_id = tok.encode(" Rome")[0] if len(tok.encode(" Rome")) > 0 else tok.encode("Rome")[0]
    london_tok_id = tok.encode(" London")[0] if len(tok.encode(" London")) > 0 else tok.encode("London")[0]

    print(f"  Prompt evaluado      : \"{prompt_text}\"")
    print(f"  Token Objetivo       : \"{tokens_target_str}\" (ID={target_tok_id})")
    print(f"  Tokens de Contraste  : \" Rome\" (ID={wrong_tok_id}), \" London\" (ID={london_tok_id})")
    print(f"  Capas totales (N)    : {num_layers} | Dimensión residual D={D}")

    # ─── FASE 1: MAPEO CINEMÁTICO κ(l) EN PREFILL (MODO A: AGNÓSTICO PURO) ────
    section("1/4. MAPEO CINEMÁTICO κ(l) (PREFILL DE CONTROL VANILLA)")
    layer_states_vanilla = []

    class TapLayer:
        def __init__(self, layer, l_idx):
            self.layer = layer
            self.l_idx = l_idx
        def __getattr__(self, name):
            return getattr(self.layer, name)
        def __call__(self, *args, **kwargs):
            out = self.layer(*args, **kwargs)
            h_last = out[0, -1, :].astype(mx.float32)
            mx.eval(h_last)
            h_unit = h_last / mx.sqrt(mx.sum(h_last * h_last))
            mx.eval(h_unit)
            layer_states_vanilla.append(h_unit)
            return out

    orig_layers = list(lm_model.layers)
    for l in range(num_layers):
        lm_model.layers[l] = TapLayer(orig_layers[l], l)

    out_v = model.language_model(prompt_ids)
    mx.eval(out_v.logits)

    for l in range(num_layers):
        lm_model.layers[l] = orig_layers[l]

    peak_res = aether_native_c.fact_band_detect_peak(layer_states_vanilla)
    l_peak = peak_res["peak_layer"]
    kappas = np.array(peak_res["kappas"])
    print(f"  • Cresta Cinemática Detectada (argmax κ): Capa L* = {l_peak} / {num_layers} (Profundidad relativa = {peak_res['relative_depth']:.2f})")
    print(f"  • Curvatura en Cresta κ(L*) = {kappas[l_peak]:.4f}")
    print(f"  • Perfil de Curvatura:")
    for l in range(0, num_layers, 2):
        bar = "█" * int(min(25, kappas[l] * 5))
        marker = " ◄ [L* CRESTA]" if l == l_peak or (l+1 == l_peak and l+1 < num_layers) else ""
        print(f"     L{l:02d}: κ={kappas[l]:6.4f} |{bar:<25}|{marker}")

    # ─── FASE 2: SÍNTESIS DE DIRECCIONES Y BLINDAJE DE MEMORIA (FREEZE) ───────
    section("2/4. FASE A (WRITE) & FASE B (FREEZE) DE LA CÉLULA DE MEMORIA")

    # Extraer y des-cuantizar matriz de embeddings
    embed = lm_model.embed_tokens
    deq_W = mx.dequantize(embed.weight, embed.scales, getattr(embed, "biases", None), group_size=64, bits=4).astype(mx.float32)
    mx.eval(deq_W)

    def extract_unit_dir(token_id):
        v = deq_W[token_id]
        return v / mx.sqrt(mx.sum(v * v))

    u_target_fact = extract_unit_dir(target_tok_id) # Paris
    u_wrong_fact  = extract_unit_dir(wrong_tok_id)  # Rome
    u_london_fact = extract_unit_dir(london_tok_id) # London
    mx.eval(u_target_fact, u_wrong_fact, u_london_fact)

    # WRITE: Ingestar en ranuras de HilbertMemoryCell
    aether_native_c.hilbert_memory_reset()
    aether_native_c.hilbert_memory_ingest(u_target_fact, timestamp=0, energy=1.0) # Slot 0: Paris
    aether_native_c.hilbert_memory_ingest(u_wrong_fact,  timestamp=1, energy=1.0) # Slot 1: Rome
    aether_native_c.hilbert_memory_ingest(u_london_fact, timestamp=2, energy=1.0) # Slot 2: London

    # FREEZE: Tomar snapshot bit a bit de todos los slots para verificar inmutabilidad
    memory_snapshot_before = [
        np.array(aether_native_c.hilbert_memory_get_slot(k)["tensor"]) for k in range(3)
    ]
    print(f"  • Memoria Experimental Inicializada con 3 Slots:")
    print(f"     - Slot 0: \" Paris\" (Target)")
    print(f"     - Slot 1: \" Rome\"  (Wrong Fact)")
    print(f"     - Slot 2: \" London\" (Tercer Hecho)")
    print(f"  • Estado de Memoria: CONGELADO (READ-ONLY) para Inferencia Causal")

    # ─── FASE 3: MATRIZ EXPERIMENTAL DE 8 CONTROLES (C0 a C7) ─────────────────
    section("3/4. FASE C (READ): MATRIZ DE CAUSALIDAD (8 CONTROLES C0 A C7)")

    class SteeringHook:
        def __init__(self, layer, l_idx, cfg, l_peak_target):
            self.layer = layer
            self.l_idx = l_idx
            self.cfg = cfg
            self.l_peak = l_peak_target
            self.intervened = False
            self.applied_g = 0.0

        def __getattr__(self, name):
            return getattr(self.layer, name)

        def __call__(self, *args, **kwargs):
            out = self.layer(*args, **kwargs)
            mode = self.cfg["mode"]

            should_steer = False
            u_dir = None
            eff_tau = 0.15

            if mode == "passive" and self.l_idx == self.l_peak:
                # C1 Passive: g=0 forzado
                should_steer = True
                u_dir = u_target_fact
                force_g = 0.0
            elif mode == "random" and self.l_idx == self.l_peak:
                # C2 Random Band: u_random ⊥ h_l, ||u_random||=1, mismo presupuesto que C3
                should_steer = True
                force_g = -1.0
                h_now = out[0, -1, :].astype(mx.float32)
                mx.eval(h_now)
                rnd = mx.array(np.random.RandomState(42).randn(D).astype(np.float32))
                rnd_ortho = rnd - mx.sum(rnd * h_now) * h_now / (mx.sum(h_now * h_now) + 1e-12)
                u_dir = rnd_ortho / mx.sqrt(mx.sum(rnd_ortho * rnd_ortho))
                mx.eval(u_dir)
            elif mode == "fact" and self.l_idx == self.cfg.get("layer", self.l_peak):
                # C3 Fact Band / C5 Late Fact
                should_steer = True
                u_dir = u_target_fact
                force_g = -1.0
            elif mode == "wrong" and self.l_idx == self.l_peak:
                # C4 Wrong Fact Band
                should_steer = True
                u_dir = u_wrong_fact
                force_g = -1.0
            elif mode == "weighted" and self.l_idx in self.cfg.get("window", []):
                # C6 Curvature Weighted: tau ponderado por curvatura
                should_steer = True
                u_dir = u_target_fact
                force_g = -1.0
                weight = self.cfg["weights"].get(self.l_idx, 0.33)
                eff_tau = 0.15 * weight
            elif mode == "mismatch" and self.l_idx == self.l_peak:
                # C7 Mismatch / Competition: usar router asociativo
                should_steer = True
                force_g = -1.0
                h_now = out[0, -1, :].astype(mx.float32)
                mx.eval(h_now)
                norm_now = mx.sqrt(mx.sum(h_now * h_now))
                h_now_unit = h_now / (norm_now + 1e-12)
                dec = aether_native_c.fact_band_route_layer(h_now_unit, threshold=0.00, beta=16.0)
                slot_picked = dec["selected_slot"]
                slot_dict = aether_native_c.hilbert_memory_get_slot(slot_picked)
                u_dir = slot_dict["tensor"]
                self.cfg["decision"] = dec

            if should_steer:
                h_last = out[0, -1, :].astype(mx.float32)
                mx.eval(h_last)
                orig_norm = mx.sqrt(mx.sum(h_last * h_last))
                h_unit = h_last / (orig_norm + 1e-12)
                routed = aether_native_c.dispatch_conformal_coupling(
                    h_unit, u_dir, step=self.l_idx,
                    tau_eff=eff_tau, kappa_att=0.80, beta_gate=12.0, theta_gate=0.10,
                    mode=1, force_g=force_g if "force_g" in locals() else -1.0
                )
                self.applied_g = routed["permeability_g"]
                self.intervened = True
                h_steered = (routed["h_steered"] * orig_norm).astype(out.dtype)
                out = mx.concatenate([out[:, :-1, :], h_steered[None, None, :]], axis=1)

            return out

    # Preparar pesos de ventana para C6
    w_window = [max(0, l_peak - 1), l_peak, min(num_layers - 1, l_peak + 1)]
    sum_k = sum(kappas[w] for w in w_window) + 1e-12
    weights_dict = {w: float(kappas[w] / sum_k) for w in w_window}

    controls = [
        ("C0_Vanilla",       {"mode": "vanilla"}),
        ("C1_Passive",       {"mode": "passive"}),
        ("C2_Random_Band",   {"mode": "random", "layer": l_peak}),
        ("C3_Fact_Band",     {"mode": "fact",   "layer": l_peak}),
        ("C4_Wrong_Band",    {"mode": "wrong",  "layer": l_peak}),
        ("C5_Late_Fact",     {"mode": "fact",   "layer": num_layers - 1}),
        ("C6_Curv_Weight",   {"mode": "weighted", "window": w_window, "weights": weights_dict}),
        ("C7_Mismatch_Comp", {"mode": "mismatch", "layer": l_peak}),
    ]

    print(f"  {'Control':<18} │ {'Logit Target':<14} │ {'Prob Target':<13} │ {'Rank':<6} │ {'KL Divergence':<14} │ {'Detalle / g'}")
    print("  " + "─" * 88)

    baseline_logit = 0.0
    baseline_probs = None
    results = {}

    for name, cfg in controls:
        hooks = []
        for l in range(num_layers):
            h = SteeringHook(orig_layers[l], l, cfg, l_peak)
            hooks.append(h)
            lm_model.layers[l] = h

        out = model.language_model(prompt_ids)
        logits = out.logits[0, -1, :]
        mx.eval(logits)

        # Restaurar capas originales
        for l in range(num_layers):
            lm_model.layers[l] = orig_layers[l]

        target_logit = float(logits[target_tok_id])
        probs = mx.softmax(logits)
        mx.eval(probs)
        target_prob = float(probs[target_tok_id])
        sorted_indices = mx.argsort(-logits).tolist()
        rank = sorted_indices.index(target_tok_id) + 1

        if name == "C0_Vanilla":
            baseline_logit = target_logit
            baseline_probs = probs
            kl_val = 0.0
        else:
            kl_val = float(mx.sum(probs * mx.log((probs + 1e-12) / (baseline_probs + 1e-12))))

        delta_z = target_logit - baseline_logit
        g_report = f"g={max([h.applied_g for h in hooks]):.4f}" if any(h.intervened for h in hooks) else "g=0.0"

        if name == "C7_Mismatch_Comp" and "decision" in cfg:
            dec = cfg["decision"]
            g_report += f" | Slot {dec['selected_slot']} (Δr={dec['resonance_margin']:.4f})"

        results[name] = {
            "target_logit": target_logit,
            "delta_z": delta_z,
            "target_prob": target_prob,
            "rank": rank,
            "kl": kl_val
        }

        print(f"  {name:<18} │ {target_logit:8.4f} ({delta_z:+5.2f}) │ {target_prob:11.4f} │ {rank:<6d} │ {kl_val:<14.4f} │ {g_report}")

    # ─── FASE 4: VERIFICACIÓN BIT A BIT DE INMUTABILIDAD DE MEMORIA ───────────
    section("4/4. FASE D: VERIFICACIÓN BIT A BIT DE INMUTABILIDAD DE MEMORIA")
    memory_snapshot_after = [
        np.array(aether_native_c.hilbert_memory_get_slot(k)["tensor"]) for k in range(3)
    ]

    max_diff_mem = 0.0
    for k in range(3):
        diff_k = float(np.max(np.abs(memory_snapshot_after[k] - memory_snapshot_before[k])))
        max_diff_mem = max(max_diff_mem, diff_k)
        print(f"  • Ranura {k}: max |after - before| = {diff_k:.2e}")

    assert max_diff_mem == 0.0, f"Violación de inmutabilidad en memoria: {max_diff_mem}"
    print(f"  [✅ PASS] Inmutabilidad Estricta Certificada: max_diff = {max_diff_mem:.2e} (Cero contaminación)")

    # ─── DICTAMEN CIENTÍFICO FINAL ────────────────────────────────────────────
    section("DICTAMEN CIENTÍFICO FINAL DEL PROTOCOLO LAB 11")
    c0 = results["C0_Vanilla"]
    c1 = results["C1_Passive"]
    c2 = results["C2_Random_Band"]
    c3 = results["C3_Fact_Band"]
    c5 = results["C5_Late_Fact"]

    # Invariante 1: Identidad Pasiva
    diff_pass = abs(c1["target_logit"] - c0["target_logit"])
    print(f"  1. Identidad Pasiva (C0 == C1): |Δz| = {diff_pass:.2e} (Cero coste del hook)")
    assert diff_pass < 1e-4, "Passive alteró los logits de Vanilla"

    # Invariante 2: Selectividad Causal (C3 Fact vs C2 Random)
    delta_z_fact = c3["delta_z"]
    delta_z_rand = c2["delta_z"]
    print(f"  2. Selectividad Causal: Δz(Fact)={delta_z_fact:+.2f} vs Δz(Random)={delta_z_rand:+.2f}")
    assert delta_z_fact > delta_z_rand, "Perturbación aleatoria no fue inferior a la dirección fáctica"

    # Invariante 3: Efecto de Profundidad (C3 Cresta vs C5 Última Capa)
    print(f"  3. Localización de Profundidad: Δz(Cresta L*={l_peak})={delta_z_fact:+.2f} vs Δz(Última Capa)={c5['delta_z']:+.2f}")

    print("\n  🏆 HITO 2.2-R1 CERTIFICADO: SUSTRATO DE ENRUTAMIENTO Y CAUSALIDAD OPERATIVOS")

if __name__ == "__main__":
    run_lab11()
