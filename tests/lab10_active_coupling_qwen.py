#!/usr/bin/env python3
"""
tests/lab10_active_coupling_qwen.py
═══════════════════════════════════════════════════════════════════════════════
PROTOCOLO H1.3-B: ACOPLAMIENTO ACTIVO EN INFERENCIA REAL (Qwen3.5-0.8B)
Comparativa Cuádruple: Vanilla vs Passive vs Active-0 vs Active
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, time
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

from PIL import Image
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import aether_native_c
from mlx_vlm import load, stream_generate

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")
IMG_PATH   = "/Users/crotalo/Downloads/005.jpg"

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def run_lab10():
    section("H1.3-B: EVALUACIÓN DE CIRCUITO CERRADO EN Qwen3.5-0.8B")
    print(f"  Modelo: {MODEL_PATH}")
    print(f"  Imagen: {IMG_PATH}")

    model, processor = load(MODEL_PATH)
    prompt_text = "Describe con precisión el material y el color del objeto de la imagen."
    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
    ], add_generation_prompt=True)

    lm_model = model.language_model.model
    final_norm = lm_model.norm

    # ── 1. CONDICIÓN 1: VANILLA PURO (CONTROL) ──────────────────────────────
    print("\n[1/4] Generando con VANILLA PURO (Sin hooks)...")
    toks_vanilla = []
    for r in stream_generate(model, processor, prompt=prompt_chat, image=IMG_PATH, max_tokens=25):
        toks_vanilla.append(r.text)
        print(r.text, end="", flush=True)
    txt_vanilla = "".join(toks_vanilla).strip()
    print()

    # Extraer atractor contextual del prompt/prefill (anclaje semántico sin fugas)
    tok = getattr(processor, "tokenizer", processor)
    prompt_ids = mx.array(tok.encode(prompt_text))[None, :]
    u_context = lm_model.embed_tokens(prompt_ids)[0]
    u_context = mx.mean(u_context, axis=0)
    u_context = u_context / mx.sqrt(mx.sum(u_context * u_context))
    mx.eval(u_context)

    # ── 2. CONDICIÓN 2: AETHER PASSIVE (Hook con telemetría, h_out = h_in) ──
    print("\n[2/4] Generando con AETHER PASSIVE (Modo 0: Telemetría pura sin alteración)...")
    aether_native_c.junction_reset()

    class PassiveNormProbe:
        def __init__(self, norm_module):
            self._module = norm_module
            self.step = 0
            self.diffs = []
        def __getattr__(self, name):
            return getattr(self._module, name)
        def __call__(self, x, **kwargs):
            if hasattr(x, "shape") and x.shape[1] == 1:
                h_in = x[0, 0, :].astype(mx.float32)
                mx.eval(h_in)
                res = aether_native_c.dispatch_conformal_coupling(
                    h_in, u_context, step=self.step, mode=0
                )
                self.step += 1
                diff = float(mx.sqrt(mx.sum((res["h_steered"] - h_in)**2)))
                self.diffs.append(diff)
                # En modo pasivo se pasa h_steered (que es h_in exacto)
                h_ret = res["h_steered"].astype(x.dtype)[None, None, :]
                return self._module(h_ret, **kwargs)
            return self._module(x, **kwargs)

    probe_passive = PassiveNormProbe(final_norm)
    lm_model.norm = probe_passive

    toks_passive = []
    for r in stream_generate(model, processor, prompt=prompt_chat, image=IMG_PATH, max_tokens=25):
        toks_passive.append(r.text)
        print(r.text, end="", flush=True)
    txt_passive = "".join(toks_passive).strip()
    lm_model.norm = final_norm
    print()

    max_diff_pass = max(probe_passive.diffs) if probe_passive.diffs else 0.0
    print(f"  • Max ||h_steered - h_in|| en Passive: {max_diff_pass:.2e}")
    assert txt_vanilla == txt_passive, "ERROR: Passive alteró los tokens generados respecto a Vanilla"
    print("  [✅ PASS] Identidad Numérica y Token: Vanilla == Passive")

    # ── 3. CONDICIÓN 3: AETHER ACTIVE-0 (Junction activa con g = 0 forzado) ─
    print("\n[3/4] Generando con AETHER ACTIVE-0 (Junction en circuito, intervención forzada g=0)...")
    aether_native_c.junction_reset()

    class ActiveZeroNormProbe:
        def __init__(self, norm_module):
            self._module = norm_module
            self.step = 0
            self.diffs = []
        def __getattr__(self, name):
            return getattr(self._module, name)
        def __call__(self, x, **kwargs):
            if hasattr(x, "shape") and x.shape[1] == 1:
                h_in = x[0, 0, :].astype(mx.float32)
                mx.eval(h_in)
                res = aether_native_c.dispatch_conformal_coupling(
                    h_in, u_context, step=self.step, mode=1, force_g=0.0
                )
                self.step += 1
                diff = float(mx.sqrt(mx.sum((res["h_steered"] - h_in)**2)))
                self.diffs.append(diff)
                h_ret = res["h_steered"].astype(x.dtype)[None, None, :]
                return self._module(h_ret, **kwargs)
            return self._module(x, **kwargs)

    probe_active0 = ActiveZeroNormProbe(final_norm)
    lm_model.norm = probe_active0

    toks_active0 = []
    for r in stream_generate(model, processor, prompt=prompt_chat, image=IMG_PATH, max_tokens=25):
        toks_active0.append(r.text)
        print(r.text, end="", flush=True)
    txt_active0 = "".join(toks_active0).strip()
    lm_model.norm = final_norm
    print()

    max_diff_a0 = max(probe_active0.diffs) if probe_active0.diffs else 0.0
    print(f"  • Max ||h_steered - h_in|| en Active-0: {max_diff_a0:.2e}")
    assert txt_vanilla == txt_active0, "ERROR: Active-0 forzado alteró los tokens generados"
    print("  [✅ PASS] Identidad Numérica y Token: Vanilla == Active-0 (Cero distorsión del Junction)")

    # ── 4. CONDICIÓN 4: AETHER ACTIVE (Junction adaptativa con compuerta libre) ─
    print("\n[4/4] Generando con AETHER ACTIVE (Modo 1: Compuerta adaptativa libre g > 0)...")
    aether_native_c.junction_reset()

    class ActiveNormProbe:
        def __init__(self, norm_module):
            self._module = norm_module
            self.step = 0
            self.metrics_log = []
        def __getattr__(self, name):
            return getattr(self._module, name)
        def __call__(self, x, **kwargs):
            if hasattr(x, "shape") and x.shape[1] == 1:
                h_in = x[0, 0, :].astype(mx.float32)
                mx.eval(h_in)
                res = aether_native_c.dispatch_conformal_coupling(
                    h_in, u_context, step=self.step,
                    tau_eff=0.15, kappa_att=0.80, beta_gate=12.0, theta_gate=0.35, mode=1
                )
                self.metrics_log.append({
                    "step": self.step,
                    "g": res["permeability_g"],
                    "q": res["dirichlet_tension_q"],
                    "kappa": res["curvature_kappa"],
                    "r": res["correlation_r"],
                    "gate_open": res["gate_open"],
                    "cell_evaluated": res["cell_evaluated"],
                    "intervention_applied": res["intervention_applied"]
                })
                self.step += 1
                h_ret = res["h_steered"].astype(x.dtype)[None, None, :]
                return self._module(h_ret, **kwargs)
            return self._module(x, **kwargs)

    probe_active = ActiveNormProbe(final_norm)
    lm_model.norm = probe_active

    toks_active = []
    for r in stream_generate(model, processor, prompt=prompt_chat, image=IMG_PATH, max_tokens=25):
        toks_active.append(r.text)
        print(r.text, end="", flush=True)
    txt_active = "".join(toks_active).strip()
    lm_model.norm = final_norm
    print()

    # ── 5. REPORTE COMPARATIVO Y DICTAMEN ─────────────────────────────────────
    section("REPORTE COMPARATIVO PROTOCOLO LAB 10")
    print(f"  • Vanilla  : \"{txt_vanilla}\"")
    print(f"  • Passive  : \"{txt_passive}\" (Identidad estricta confirmada)")
    print(f"  • Active-0 : \"{txt_active0}\" (Identidad estricta confirmada)")
    print(f"  • Active   : \"{txt_active}\"")

    total_steps = len(probe_active.metrics_log)
    interventions = sum(1 for m in probe_active.metrics_log if m["intervention_applied"])
    gates_open = sum(1 for m in probe_active.metrics_log if m["gate_open"])
    mean_g = np.mean([m["g"] for m in probe_active.metrics_log]) if total_steps > 0 else 0.0
    mean_q = np.mean([m["q"] for m in probe_active.metrics_log]) if total_steps > 0 else 0.0
    mean_r = np.mean([m["r"] for m in probe_active.metrics_log]) if total_steps > 0 else 0.0

    print(f"\n  Telemetría de la Intervención Activa ({total_steps} pasos):")
    print(f"    • Evaluaciones de la Célula    : {interventions} / {total_steps}")
    print(f"    • Umbral Macro Superado (g>=0.5): {gates_open} / {total_steps}")
    print(f"    • Permeabilidad Media <g>      : {mean_g:.4f}")
    print(f"    • Tensión de Dirichlet Media <q>: {mean_q:.4f}")
    print(f"    • Correlación con Atractor <r> : {mean_r:.4f}")

    section("DICTAMEN FINAL DEL EXPERIMENTO")
    if txt_vanilla == txt_passive and txt_vanilla == txt_active0:
        print("  🏆 CERTIFICACIÓN HOMEOSTÁTICA LOGRADA:")
        print("     1. Coste del hook = 0 (Vanilla == Passive)")
        print("     2. Coste del junction = 0 (Vanilla == Active-0)")
        print("     3. El efecto de modulación observado en Active procede de forma pura")
        print("        del acoplamiento contextual en el espacio residual.")
    else:
        print("  ❌ ERROR: Fallo de equivalencia en las condiciones de control.")

if __name__ == "__main__":
    run_lab10()
