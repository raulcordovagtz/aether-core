#!/usr/bin/env python3
"""
Pruebas_Raul/test_h3_escalon2_real_prediction.py
═══════════════════════════════════════════════════════════════════════════════
HITO 3 — ESCALÓN 2: CAPACIDAD PREDICTIVA FUERA DE MUESTRA EN QWEN-0.8B
Protocolo estrictamente observacional (g = 0). CERO intervenciones.
Evalúa el error angular e_pred = ∠(h_hat_{t+1}, h_{t+1}^{actual}) en decode real:
  • P0: Persistencia
  • P1: Velocidad Lineal Normalizada
  • P2: Exp-Map Riemanniano
  • P3: Extrapolación de Taylor (2º Orden)
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

PROMPT = "¿Cuál es la relación física entre la temperatura, la presión y el volumen de un gas ideal?"

def section(t):
    print("\n" + "═" * 78)
    print(f"  {t}")
    print("═" * 78)

def angular_distance(u, v):
    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    return float(np.arccos(dot))

def main():
    section("H3-E2: EVALUACIÓN PREDICTIVA FUERA DE MUESTRA (QWEN-0.8B REAL)")
    print(f"• Cargando modelo real desde {MODEL_PATH}...")
    model, processor = load(MODEL_PATH)
    lm_model = model.language_model.model

    # Extracción robusta de D desde config o embed_tokens
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    D = getattr(text_cfg, "hidden_size", 1024)
    print(f"✓ Topología detectada: D = {D} | Capas = {len(lm_model.layers)}")

    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": PROMPT}
    ], add_generation_prompt=True)

    # Captura observacional en la Capa 17 (Cresta kinemática κ(l))
    peak_layer_idx = 17
    orig_peak_layer = lm_model.layers[peak_layer_idx]

    captured_states: list[np.ndarray] = []

    class PassiveProbeHook:
        def __init__(self, layer):
            self.layer = layer
        def __getattr__(self, name):
            return getattr(self.layer, name)
        def __call__(self, x, **kwargs):
            out = self.layer(x, **kwargs)
            h_out = out[0] if isinstance(out, (tuple, list)) else out
            
            # Capturar exclusivamente estados unitarios durante el decode autorregresivo (shape [1, 1, D])
            if hasattr(h_out, "shape") and len(h_out.shape) == 3 and h_out.shape[1] == 1:
                h_raw = h_out[0, 0, :].astype(mx.float32)
                mx.eval(h_raw)
                h_np = np.array(h_raw, copy=True)
                h_unit = h_np / (np.linalg.norm(h_np) + 1e-12)
                captured_states.append(h_unit)
            return out

    hook = PassiveProbeHook(orig_peak_layer)
    lm_model.layers[peak_layer_idx] = hook

    print("• Ejecutando generación Vanilla libre (50 tokens)...")
    tokens = []
    t0 = time.perf_counter()
    for r in stream_generate(model, processor, prompt=prompt_chat, max_tokens=50):
        tokens.append(r.text)
        print(r.text, end="", flush=True)
    t_gen = time.perf_counter() - t0
    lm_model.layers[peak_layer_idx] = orig_peak_layer # Restaurar capa

    print(f"\n\n✓ Trayectoria real capturada: {len(captured_states)} estados residuales en Capa {peak_layer_idx}")
    print(f"• Velocidad de inferencia: {len(tokens)/t_gen:.1f} tok/s")

    N = len(captured_states)
    if N < 10:
        print("❌ Error: No se capturaron suficientes estados de decode.")
        return

    # ─── EVALUACIÓN FUERA DE MUESTRA (OUT-OF-SAMPLE) ───────────────────────────
    section("EVALUACIÓN DE ERROR ANGULAR e_pred = ∠(h_hat_{t+1}, h_{t+1}^{actual})")

    err_p0 = [] # Persistencia
    err_p1 = [] # Velocidad Lineal
    err_p2 = [] # Exp-Map Riemanniano
    err_p3 = [] # Taylor 2º Orden

    delta_thetas_real = [] # Giro real que ejecutó el modelo entre h_t y h_{t+1}

    for t in range(2, N - 1):
        h_tm2 = captured_states[t - 2]
        h_tm1 = captured_states[t - 1]
        h_t   = captured_states[t]
        h_next_real = captured_states[t + 1]

        # Cinemática discreta exacta observable en t
        v_t = h_t - h_tm1
        v_tm1 = h_tm1 - h_tm2
        a_t = v_t - v_tm1

        theta_real = angular_distance(h_t, h_next_real)
        delta_thetas_real.append(theta_real)

        # 1. P0: Persistencia
        h_hat_p0 = h_t

        # 2. P1: Velocidad Lineal Normalizada
        cand_p1 = h_t + v_t
        h_hat_p1 = cand_p1 / (np.linalg.norm(cand_p1) + 1e-12)

        # 3. P2: Exp-Map Riemanniano en S^{D-1}
        v_tangent = v_t - np.dot(v_t, h_t) * h_t
        norm_vt = np.linalg.norm(v_tangent)
        if norm_vt > 1e-12:
            v_hat = v_tangent / norm_vt
            h_hat_p2 = np.cos(norm_vt) * h_t + np.sin(norm_vt) * v_hat
        else:
            h_hat_p2 = h_t

        # 4. P3: Extrapolación de Taylor 2º orden en R^D normalizada
        cand_p3 = h_t + v_t + 0.5 * a_t
        h_hat_p3 = cand_p3 / (np.linalg.norm(cand_p3) + 1e-12)

        err_p0.append(angular_distance(h_hat_p0, h_next_real))
        err_p1.append(angular_distance(h_hat_p1, h_next_real))
        err_p2.append(angular_distance(h_hat_p2, h_next_real))
        err_p3.append(angular_distance(h_hat_p3, h_next_real))

    # ─── TABLA DE TELEMETRÍA CIENTÍFICA ───────────────────────────────────────
    print(f" {'Predictor':<24} │ {'Error Medio (rad)':<18} │ {'Error Mediano':<15} │ {'p90 (rad)':<12}")
    print(" ─────────────────────────┼────────────────────┼─────────────────┼──────────────")
    print(f" P0: Persistencia         │ {np.mean(err_p0):18.4f} │ {np.median(err_p0):15.4f} │ {np.percentile(err_p0, 90):12.4f}")
    print(f" P1: Velocidad Lineal     │ {np.mean(err_p1):18.4f} │ {np.median(err_p1):15.4f} │ {np.percentile(err_p1, 90):12.4f}")
    print(f" P2: Exp-Map Riemanniano  │ {np.mean(err_p2):18.4f} │ {np.median(err_p2):15.4f} │ {np.percentile(err_p2, 90):12.4f}")
    print(f" P3: Taylor 2º Orden      │ {np.mean(err_p3):18.4f} │ {np.median(err_p3):15.4f} │ {np.percentile(err_p3, 90):12.4f}")

    mean_theta_real = np.mean(delta_thetas_real)
    print(f"\n• Desplazamiento angular real del Transformer: <θ_actual> = {mean_theta_real:.4f} rad")

    # ─── VERIFICACIÓN DE LA REGLA DE ORO DE H3 ────────────────────────────────
    section("EVALUACIÓN DE LA REGLA DE ORO: PERMISO PARA CONTROLAR")
    
    delta_p1 = np.mean(err_p0) - np.mean(err_p1)
    delta_p2 = np.mean(err_p0) - np.mean(err_p2)
    delta_p3 = np.mean(err_p0) - np.mean(err_p3)

    print(f"• Ganancia frente a Persistencia (Δe > 0 indica que la cinemática ayuda):")
    print(f"   - P1 (Velocidad Lineal)    : {delta_p1:+.5f} rad")
    print(f"   - P2 (Exp-Map Riemanniano) : {delta_p2:+.5f} rad")
    print(f"   - P3 (Taylor 2º Orden)     : {delta_p3:+.5f} rad")

    tau_pred = mean_theta_real * 0.90
    best_err = min(np.mean(err_p1), np.mean(err_p2), np.mean(err_p3))
    has_predictive_authority = (best_err < np.mean(err_p0)) and (best_err < tau_pred)

    print(f"\n• Umbral de Autoridad Predictiva τ_pred : {tau_pred:.4f} rad")
    print(f"• Mejor Error Cinemático Fuera de Muestra: {best_err:.4f} rad")
    
    if has_predictive_authority:
        print("\n🏆 DICTAMEN: [AUTORIDAD CONCEDIDA]")
        print("   La cinemática neuronal es predictiva fuera de muestra.")
        print("   El controlador TIENE PERMISO para evaluar intervenciones en H3-E3.")
    else:
        print("\n⚠️ DICTAMEN: [FRENO DE SEGURIDAD ACTIVADO — g = 0 STRICT]")
        print("   La inercia cinemática NO supera consistentemente a la persistencia.")
        print("   El controlador NO TIENE PERMISO para intervenir la trayectoria aún.")
    print("═" * 78)

if __name__ == "__main__":
    main()
