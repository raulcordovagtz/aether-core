#!/usr/bin/env python3
"""
Pruebas_Raul/test_h3_escalon2_audit.py
═══════════════════════════════════════════════════════════════════════════════
HITO 3 — ESCALÓN 2.1: AUDITORÍA MULTI-CAPA Y GEODÉSICA ANALÍTICA EXACTA
Protocolo observacional riguroso (g = 0). CERO afirmaciones ontológicas.
Evalúa error angular e_pred frente a persistencia en:
  • Capas: 6, 12, 17, 23
  • 3 Prompts de naturaleza distinta (factual, lógico, narrativo)
  • Geodésica Riemanniana analítica exacta: h_hat = 2<h_{t-1}, h_t> h_t - h_{t-1}
═══════════════════════════════════════════════════════════════════════════════
"""

import sys, os, time, math

for conda_py in ["/opt/miniconda3/bin/python3", os.path.expanduser("~/miniconda3/bin/python3")]:
    if os.path.exists(conda_py) and sys.executable != conda_py:
        os.execv(conda_py, [conda_py] + sys.argv)

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import numpy as np
import mlx.core as mx
from mlx_vlm import load, stream_generate

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

PROMPTS = [
    ("Factual", "¿Cuál es la relación física entre la temperatura, la presión y el volumen de un gas ideal?"),
    ("Lógico", "Si todos los mamíferos respiran oxígeno y las ballenas son mamíferos, ¿qué se concluye y por qué?"),
    ("Narrativo", "Había una vez en un antiguo reino un alquimista que intentaba descifrar un manuscrito oculto.")
]

EVAL_LAYERS = [6, 12, 17, 23]
TOKENS_PER_PROMPT = 35

def angular_distance(u, v):
    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    return float(np.arccos(dot))

def exact_geodesic_next(h_tm1, h_t):
    """
    Continuación geodésica analítica exacta en S^{D-1} vía Log-map + Exp-map.
    h_hat_{t+1} = 2 * <h_{t-1}, h_t> * h_t - h_{t-1}
    Garantiza ||h_hat|| == 1.0 analítico exacto y arco angular idéntico.
    """
    cos_theta = np.dot(h_tm1, h_t)
    h_cand = 2.0 * cos_theta * h_t - h_tm1
    # Regularización de norma para precisión de máquina (IEEE 754)
    return h_cand / (np.linalg.norm(h_cand) + 1e-12)

def main():
    print("═" * 78)
    print("  H3-E2.1: AUDITORÍA DE PREDICCIÓN GEODÉSICA MULTI-CAPA (QWEN-0.8B)")
    print("═" * 78)

    model, processor = load(MODEL_PATH)
    lm_model = model.language_model.model
    total_layers = len(lm_model.layers)
    cfg = getattr(model, "config", None)
    D = getattr(getattr(cfg, "text_config", cfg), "hidden_size", 1024)

    print(f"• Topología: D={D}, Capas={total_layers} | Capas evaluadas: {EVAL_LAYERS}")
    print(f"• Batería: {len(PROMPTS)} prompts × {TOKENS_PER_PROMPT} tokens de decode")
    print("═" * 78)

    # Resultados globales agregados
    report_rows = []

    for prompt_label, prompt_text in PROMPTS:
        print(f"\n▶ EVALUANDO PROMPT [{prompt_label}]: \"{prompt_text[:50]}...\"")
        
        prompt_chat = processor.apply_chat_template([
            {"role": "user", "content": prompt_text}
        ], add_generation_prompt=True)

        for l_idx in EVAL_LAYERS:
            orig_layer = lm_model.layers[l_idx]
            captured = []

            class MultiLayerProbeHook:
                def __init__(self, layer):
                    self.layer = layer
                def __getattr__(self, name):
                    return getattr(self.layer, name)
                def __call__(self, x, **kwargs):
                    out = self.layer(x, **kwargs)
                    h_tensor = out[0] if isinstance(out, (tuple, list)) else out
                    if hasattr(h_tensor, "shape") and len(h_tensor.shape) == 3 and h_tensor.shape[1] == 1:
                        raw = h_tensor[0, 0, :].astype(mx.float32)
                        mx.eval(raw)
                        arr = np.array(raw, copy=True)
                        norm = np.linalg.norm(arr) + 1e-12
                        captured.append(arr / norm)
                    return out

            lm_model.layers[l_idx] = MultiLayerProbeHook(orig_layer)

            # Generar tokens observando pasivamente
            tokens = []
            for r in stream_generate(model, processor, prompt=prompt_chat, max_tokens=TOKENS_PER_PROMPT):
                tokens.append(r.text)

            lm_model.layers[l_idx] = orig_layer # Restaurar capa

            N = len(captured)
            if N < 5:
                continue

            # Evaluación fuera de muestra
            err_persist = []
            err_linear = []
            err_geodesic = []
            delta_thetas_real = []

            for t in range(2, N - 1):
                h_tm1 = captured[t - 1]
                h_t   = captured[t]
                h_real_next = captured[t + 1]

                # Desplazamiento real observado
                th_real = angular_distance(h_t, h_real_next)
                delta_thetas_real.append(th_real)

                # P0: Persistencia
                h_p0 = h_t
                err_persist.append(th_real)

                # P1: Velocidad Lineal Euclídea normalizada
                v_lin = h_t - h_tm1
                h_p1 = (h_t + v_lin) / (np.linalg.norm(h_t + v_lin) + 1e-12)
                err_linear.append(angular_distance(h_p1, h_real_next))

                # P2: Continuación Geodésica Exacta en S^{D-1}
                h_p2 = exact_geodesic_next(h_tm1, h_t)
                err_geodesic.append(angular_distance(h_p2, h_real_next))

            mean_th = np.mean(delta_thetas_real)
            mean_p0 = np.mean(err_persist)
            mean_p1 = np.mean(err_linear)
            mean_p2 = np.mean(err_geodesic)
            gain_geo = mean_p0 - mean_p2 # > 0 significa que la geodésica mejoró a la persistencia

            report_rows.append({
                "prompt": prompt_label,
                "layer": l_idx,
                "N": len(delta_thetas_real),
                "theta_actual": mean_th,
                "e_p0": mean_p0,
                "e_linear": mean_p1,
                "e_geodesic": mean_p2,
                "gain_geodesic": gain_geo
            })

            print(f"  • Capa {l_idx:02d} (N={len(delta_thetas_real):02d}): <θ_real>={mean_th:.4f} rad | "
                  f"P0={mean_p0:.4f} | P1={mean_p1:.4f} | P2(Geod)={mean_p2:.4f} | "
                  f"Δ(P0 - P2) = {gain_geo:+.5f} rad")

    # ─── TABLA DE SÍNTESIS CIENTÍFICA ─────────────────────────────────────────
    print("\n" + "═" * 78)
    print("  REPORTE CRÍTICO MULTI-CAPA (H3-E2.1)")
    print("═" * 78)
    print(f" {'Prompt':<10} │ {'Capa':<5} │ {'<θ_real>':<10} │ {'e(Persist)':<12} │ {'e(Geodésica)':<14} │ {'Ganancia Δe':<12}")
    print(" ───────────┼───────┼────────────┼──────────────┼────────────────┼─────────────")
    for r in report_rows:
        print(f" {r['prompt']:<10} │ L{r['layer']:02d}   │ {r['theta_actual']:10.4f} │ {r['e_p0']:12.4f} │ {r['e_geodesic']:14.4f} │ {r['gain_geodesic']:+12.5f}")

    print(" ───────────┴───────┴────────────┴──────────────┴────────────────┴─────────────")
    
    # Análisis agregado de ganancias
    gains = [r["gain_geodesic"] for r in report_rows]
    positives = sum(1 for g in gains if g > 0.0)
    total_evals = len(gains)

    print(f"\n• Casos donde la geodésica exacta mejoró a la persistencia: {positives} / {total_evals}")
    print(f"• Ganancia media de la geodésica sobre la persistencia: {np.mean(gains):+.5f} rad")
    print("═" * 78)

if __name__ == "__main__":
    main()
