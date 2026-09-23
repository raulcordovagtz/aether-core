#!/usr/bin/env python3
"""
Pruebas_Raul/06_curvas_modelos_grandes.py
=========================================
Evaluación Cinemática Pura en Modelos Frontier (27B y 35B MoE):
Compara la proyección de curvas entre Hipótesis A (Negro) y B (Blanco)
sin emisión de texto prolijo, midiendo tensión q_k, certeza r y curvatura κ.
"""

import sys, os, time, math, gc
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import numpy as np
import mlx.core as mx
from mlx_vlm import load, stream_generate
from aether_vlm.coupler import AetherEngine
import aether_native_c

MODELS_LARGE = [
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

def evaluate_curve_only(model, processor, prompt_text, max_steps=20):
    aether = AetherEngine(model, processor)
    aether.prepare_thought(prompt_text)
    u_attractor = aether.state.get("L_star")
    aether_native_c.buffer_reset()

    lm_model = model.language_model.model
    final_norm = lm_model.norm

    history = []
    current_step = [0]

    class CurveOnlyProbe:
        def __init__(self, norm_mod):
            self.orig = norm_mod
        def __getattr__(self, name):
            return getattr(self.orig, name)
        def __call__(self, x, **kwargs):
            if hasattr(x, "shape") and x.shape[1] == 1:
                h_raw = x[0, 0, :].astype(mx.float32)
                mx.eval(h_raw)

                norm_h = mx.sqrt(mx.sum(h_raw * h_raw)) + 1e-12
                h_unit = (h_raw / norm_h).astype(mx.float32)
                mx.eval(h_unit)

                step = current_step[0]
                st = aether_native_c.buffer_push_state(h_unit, step=step)
                cell_out = aether_native_c.dispatch_geodesic_trajectory_cell(
                    h_unit, st["v_t"], st["a_t"], u_attractor,
                    tau=0.15, kappa_att=aether.kappa
                )
                mx.eval(cell_out["correlation_r"], cell_out["curvature_kappa"])

                history.append({
                    "step": step,
                    "v": math.sqrt(max(0.0, float(st["sq_v"]))),
                    "kappa": float(cell_out["curvature_kappa"]),
                    "q": float(st["dirichlet_tension_q"]),
                    "r": float(cell_out["correlation_r"])
                })
                current_step[0] += 1
            return self.orig(x, **kwargs)

    probe = CurveOnlyProbe(final_norm)
    lm_model.norm = probe

    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
    ], add_generation_prompt=True)

    # Solo generamos los pasos necesarios para alimentar la curva
    for _ in stream_generate(model, processor, prompt=prompt_chat, max_tokens=max_steps):
        pass

    lm_model.norm = final_norm
    aether.set_active(False)

    return {
        "mean_q": float(np.mean([m["q"] for m in history])) if history else 0.0,
        "mean_k": float(np.mean([m["kappa"] for m in history])) if history else 0.0,
        "mean_r": float(np.mean([m["r"] for m in history])) if history else 0.0,
        "mean_v": float(np.mean([m["v"] for m in history])) if history else 0.0,
        "steps": len(history)
    }

def main():
    for alias, path in MODELS_LARGE:
        if not os.path.exists(path):
            print(f"\n⚠️  Modelo {alias} no encontrado en ruta: {path}. Saltando...")
            continue

        print("\n" + "═" * 84)
        print(f" 🪐 ANALIZANDO CINEMÁTICA DE CAMPO EN: Qwen3-{alias}")
        print("═" * 84)

        t0 = time.perf_counter()
        model, processor = load(path)
        print(f"✓ Modelo cargado en UMA ({time.perf_counter() - t0:.2f} s)")

        results = {}
        for h_name, prompt in HYPOTHESES:
            print(f"  • Midiendo trayectoria de: {h_name.split()[1]} ...", end="", flush=True)
            res = evaluate_curve_only(model, processor, prompt, max_steps=20)
            results[h_name] = res
            print(f" listo ({res['steps']} pasos evaluados).")

        res_a = results["HIPÓTESIS A (NEGRO - Verdadera)"]
        res_b = results["HIPÓTESIS B (BLANCO - Falsa)"]

        delta_q = res_b["mean_q"] - res_a["mean_q"]
        delta_r = res_a["mean_r"] - res_b["mean_r"]

        print("\n  " + "─" * 76)
        print(f"  {'Métrica Cinemática':<26} │ {'A (NEGRO - Verdad)':<16} │ {'B (BLANCO - Falsa)':<16} │ {'Diferencial (Δ)'}")
        print("  " + "─" * 76)
        print(f"  {'Tensión Dirichlet <q_k>':<26} │ {res_a['mean_q']:<16.4f} │ {res_b['mean_q']:<16.4f} │ {delta_q:+16.4f}")
        print(f"  {'Certeza/Afinidad <r>':<26} │ {res_a['mean_r']:<+16.4f} │ {res_b['mean_r']:<+16.4f} │ {delta_r:+16.4f}")
        print(f"  {'Curvatura Media <κ>':<26} │ {res_a['mean_k']:<16.4f} │ {res_b['mean_k']:<16.4f} │ {res_a['mean_k'] - res_b['mean_k']:+16.4f}")
        print(f"  {'Velocidad Media <v>':<26} │ {res_a['mean_v']:<16.4f} │ {res_b['mean_v']:<16.4f} │ {res_a['mean_v'] - res_b['mean_v']:+16.4f}")
        print("  " + "─" * 76)

        if delta_q > 0:
            print(f"  🏆 Veredicto {alias}: La hipótesis falsa genera MAYOR TENSIÓN (+{delta_q:.4f}).")
        else:
            print(f"  ℹ️  Veredicto {alias}: Contraste de tensión neutro o invertido ({delta_q:+.4f}).")

        # Limpieza de memoria para no acumular ambos gigantes en RAM
        del model
        del processor
        gc.collect()

if __name__ == "__main__":
    main()
