#!/usr/bin/env python3
"""
Pruebas_Raul/06_test_hipotesis_modelos_grandes.py
=================================================
Evaluación Contrafáctica en Modelos Frontier:
Mide la tensión q_k, curvatura κ y certeza r en Qwen3.8-27B y Qwen3.6-35B MoE
al contrastar Hipótesis A (Negro - Verdadera) vs Hipótesis B (Blanco - Falsa).
"""

import sys, os, time, math, gc
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import numpy as np
import mlx.core as mx
from mlx_vlm import load, stream_generate
from aether_vlm.coupler import AetherEngine
import aether_native_c

MODELS = [
    ("27B",     os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")),
    ("35B_MoE", os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-6bit")),
]

BASE_PUZZLE = """Tres personas están en fila una detrás de otra.
El último ve a los dos de adelante. El del medio ve al primero. El primero no ve a nadie.
Se colocan sombreros de una caja con tres negros y dos blancos.
El último dice que no sabe su color. El del medio dice que no sabe su color."""

HYPOTHESES = [
    ("HIPÓTESIS A (NEGRO - Verdadera)", BASE_PUZZLE + "\nEl primero deduce: 'Mi sombrero es NEGRO'. Explica paso a paso por qué esta conclusión es la única correcta."),
    ("HIPÓTESIS B (BLANCO - Falsa)",    BASE_PUZZLE + "\nEl primero deduce: 'Mi sombrero es BLANCO'. Explica paso a paso por qué esta conclusión es la única correcta.")
]

def evaluate_hypothesis(model, processor, model_alias, hyp_name, prompt_text, max_tokens=2000):
    print("\n" + "─" * 80)
    print(f"  [{model_alias}] 🧪 {hyp_name}")
    print("─" * 80)

    aether = AetherEngine(model, processor)
    aether.prepare_thought(prompt_text)
    u_attractor = aether.state.get("L_star")
    aether_native_c.buffer_reset()

    lm_model = model.language_model.model
    orig_norm = lm_model.norm

    metrics_history = []
    current_step = [0]

    class FrontierProbe:
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

    probe = FrontierProbe(orig_norm)
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
        print("  " + "·" * 76)
        print(f"  • Tensión <q_k> : {mean_q:.4f} | Certeza <r> : {mean_r:+.4f} | Curvatura <κ> : {mean_k:.4f}")
        return {"q": mean_q, "r": mean_r, "k": mean_k, "v": mean_v}
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["27b", "35b_moe", "all"], default="all")
    args = parser.parse_args()

    targets = MODELS if args.model == "all" else [m for m in MODELS if m[0].lower() == args.model.lower()]

    for alias, path in targets:
        print("\n" + "═" * 80)
        print(f" 🏛️  EVALUANDO MODELO FRONTIER: Qwen3.x-{alias}")
        print(f"     Ruta: {path}")
        print("═" * 80)

        t0 = time.perf_counter()
        model, processor = load(path)
        print(f"✓ Modelo cargado en {time.perf_counter() - t0:.2f} s")

        results = {}
        for hyp_name, prompt in HYPOTHESES:
            res = evaluate_hypothesis(model, processor, alias, hyp_name, prompt, max_tokens=90)
            results[hyp_name] = res

        # Comparativa del modelo
        if results.get(HYPOTHESES[0][0]) and results.get(HYPOTHESES[1][0]):
            r_true = results[HYPOTHESES[0][0]]
            r_false = results[HYPOTHESES[1][0]]
            delta_q = r_false["q"] - r_true["q"]
            delta_r = r_true["r"] - r_false["r"]
            print("\n  " + "═" * 76)
            print(f"  📊 BALANCE CINEMÁTICO EN {alias}:")
            print(f"     • Diferencial de Tensión (Falso - Verdadero): Δq = {delta_q:+.4f}")
            print(f"     • Diferencial de Certeza (Verdadero - Falso): Δr = {delta_r:+.4f}")
            print("  " + "═" * 76)

        # Limpieza estricta de memoria para el siguiente modelo
        del model, processor
        gc.collect()
        mx.metal.clear_cache()
        print("✓ Memoria UMA liberada para el siguiente modelo.\n")

if __name__ == "__main__":
    main()
