#!/usr/bin/env python3
"""
Pruebas_Raul/05_test_hipotesis_inversa.py
=========================================
Discriminación Contrafáctica de Hipótesis:
Evalúa la tensión q_k, curvatura κ y certeza r al forzar
Hipótesis A (Negro) vs Hipótesis B (Blanco) en Qwen3.5-2B.
"""

import sys, os, time, math
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import numpy as np
import mlx.core as mx
from mlx_vlm import load, stream_generate
from aether_vlm.coupler import AetherEngine
import aether_native_c

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit")

BASE_PUZZLE = """Tres personas están en fila una detrás de otra.
El último ve a los dos de adelante. El del medio ve al primero. El primero no ve a nadie.
Se colocan sombreros de una caja con tres negros y dos blancos.
El último dice que no sabe su color. El del medio dice que no sabe su color."""

HYPOTHESES = [
    ("HIPÓTESIS A (NEGRO - Verdadera)", BASE_PUZZLE + "\nEl primero deduce: 'Mi sombrero es NEGRO'. Explica paso a paso por qué esta conclusión es la única correcta."),
    ("HIPÓTESIS B (BLANCO - Falsa)",    BASE_PUZZLE + "\nEl primero deduce: 'Mi sombrero es BLANCO'. Explica paso a paso por qué esta conclusión es la única correcta.")
]

def test_hypothesis(model, processor, name, prompt_text, max_tokens=60):
    print("\n" + "═" * 80)
    print(f" 🧪 {name}")
    print("═" * 80)

    aether = AetherEngine(model, processor)
    aether.prepare_thought(prompt_text)
    u_attractor = aether.state.get("L_star")
    aether_native_c.buffer_reset()

    lm_model = model.language_model.model
    orig_norm = lm_model.norm

    metrics_history = []
    current_step = [0]

    class HypothesisProbe:
        def __init__(self, norm_module):
            self.orig = norm_module
        def __getattr__(self, name):
            return getattr(self.orig, name)
        def __call__(self, x, **kwargs):
            if hasattr(x, "shape") and x.shape[1] == 1:
                h_raw = x[0, 0, :].astype(mx.float32)
                mx.eval(h_raw)

                norm_h = mx.sqrt(mx.sum(h_raw * h_raw)) + 1e-12
                h_unit = h_raw / norm_h
                mx.eval(h_unit)

                step = current_step[0]
                st = aether_native_c.buffer_push_state(h_unit, step=step)
                cell_out = aether_native_c.dispatch_geodesic_trajectory_cell(
                    h_unit, st["v_t"], st["a_t"], u_attractor, tau=0.15, kappa_att=aether.kappa
                )
                mx.eval(cell_out["correlation_r"], cell_out["curvature_kappa"])

                metrics_history.append({
                    "step": step,
                    "v": math.sqrt(max(0.0, float(st["sq_v"]))),
                    "kappa": float(cell_out["curvature_kappa"]),
                    "q": float(st["dirichlet_tension_q"]),
                    "r": float(cell_out["correlation_r"])
                })
                current_step[0] += 1
            return self.orig(x, **kwargs)

    probe = HypothesisProbe(orig_norm)
    lm_model.norm = probe

    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
    ], add_generation_prompt=True)

    tokens = []
    for resp in stream_generate(model, processor, prompt=prompt_chat, max_tokens=max_tokens):
        tokens.append(resp.text)
        print(resp.text, end="", flush=True)
    print()

    lm_model.norm = orig_norm
    aether.set_active(False)

    if metrics_history:
        mean_q = np.mean([m["q"] for m in metrics_history])
        mean_k = np.mean([m["kappa"] for m in metrics_history])
        mean_r = np.mean([m["r"] for m in metrics_history])
        mean_v = np.mean([m["v"] for m in metrics_history])
        print("─" * 80)
        print(f"  • Tensión media <q_k> : {mean_q:.4f}")
        print(f"  • Curvatura media <κ> : {mean_k:.4f}")
        print(f"  • Certeza media <r>   : {mean_r:+.4f}")
        print(f"  • Velocidad media <v> : {mean_v:.4f}")

def main():
    print("Cargando Qwen3.5-2B...")
    model, processor = load(MODEL_PATH)
    for name, prompt in HYPOTHESES:
        test_hypothesis(model, processor, name, prompt, max_tokens=80)

if __name__ == "__main__":
    main()
