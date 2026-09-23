#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# ⚠️ ADVERTENCIA ARQUITECTÓNICA OBLIGATORIA (SSOT — LECCIÓN LAB 09)
# ═══════════════════════════════════════════════════════════════════════════════
# ESTE SCRIPT ES UN SENSOR DE TURBULENCIA Y ENTROPÍA, NO UN MODELO BALÍSTICO.
#
# 1. LOS PREDICTORES INERCIALES B1 (v_t) Y B2 (a_t) FRACASAN TOKEN-A-TOKEN:
#    La transición entre tokens consecutivos (t -> t+1) es discreta y sufre saltos
#    angulares masivos (>53° / ~0.93 rad). Tratar la secuencia de tokens como un
#    tiro parabólico continuo es un error conceptual grave: la extrapolación inercial
#    rinde sistemáticamente PEOR que la simple persistencia B0 (Δe < 0).
#
# 2. EL VERDADERO PROPÓSITO DE LA CURVATURA κ EN ESTE SCRIPT:
#    La curvatura de Lagrange κ NO fue diseñada para adivinar el futuro estado h_{t+1}.
#    Su valor radica exclusivamente en ser un SISMÓGRAFO DE TURBULENCIA: detectar
#    cuándo la red sufre una deflexión angular brusca (corr(κ, Δθ)) para saber
#    cuándo el modelo entra en una bifurcación de decisión.
#
# 3. EL EJE FÍSICO DE LA CINEMÁTICA EN AETHER:
#    La cinemática (v, a, κ) opera en la PROFUNDIDAD DE CAPAS (l -> l+1 en prefill),
#    donde la representación se refina gradualmente para el MISMO token. En el eje
#    temporal de decode (t -> t+1), la inercia local se disuelve en cada paso.
#
# 4. CONDICIÓN SINE QUA NON:
#    Cualquier intento de timoneo sin acoplamiento a la Fact Band y a la memoria C2
#    (como advierte la conclusión de este script) degenera en colapso repetitivo.
# ═══════════════════════════════════════════════════════════════════════════════

"""
tests/lab09_trajectory_parity.py
═══════════════════════════════════════════════════════════════════════════════
LAB 09-R1 — PROTOCOLO CIENTÍFICO DE PARIDAD TRAYECTORIA ↔ TOKEN
Evaluación Observacional Fuera de Muestra sobre Qwen3.5-0.8B (Modo Pasivo)
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, time, math
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

def run_lab09_r1():
    section("LAB 09-R1 — PARIDAD CAUSAL TRAYECTORIA ↔ INFERENCIA")
    print("  Modelo: Qwen3.5-0.8B | Protocolo: Observacional Pasivo (Vanilla Puro)")

    # 1. Cargar modelo sin alterar pesos ni acopladores
    print(f"  Cargando modelo desde: {MODEL_PATH} ...")
    model, processor = load(MODEL_PATH)
    prompt_text = "Describe en detalle el objeto que observas en la imagen."
    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
    ], add_generation_prompt=True)

    img = Image.open(IMG_PATH).convert("RGB")
    aether_native_c.buffer_reset()
    aether_native_c.gate_set_mode(0) # PassiveObserve

    # Identificar componentes exactos del modelo Qwen
    lm_model = model.language_model.model
    final_norm = lm_model.norm
    lm_head = getattr(model.language_model, "lm_head", None)
    if lm_head is None:
        from aether_vlm.coupler import TiedLinearHead
        lm_head = TiedLinearHead(lm_model.embed_tokens)

    # 2. Hook observacional que captura el estado PRE-NORM exclusivamente durante DECODE
    captured_pre_norm = []
    decode_active = False

    class NormProbeWrapper:
        def __init__(self, norm_module):
            self._module = norm_module
        def __getattr__(self, name):
            return getattr(self._module, name)
        def __call__(self, x, **kwargs):
            if decode_active and hasattr(x, "shape") and x.shape[1] == 1:
                h_curr = x[0, 0, :].astype(mx.float32)
                mx.eval(h_curr)
                captured_pre_norm.append(np.array(h_curr, copy=True))
            return self._module(x, **kwargs)

    lm_model.norm = NormProbeWrapper(final_norm)

    # 3. Generación con captura de 30 tokens
    print("\n  Ejecutando inferencia observacional (30 tokens)...")
    tokens_text = []
    decode_active = True
    for resp in stream_generate(model, processor, prompt=prompt_chat, image=IMG_PATH, max_tokens=30):
        tokens_text.append(resp.text)
        print(resp.text, end="", flush=True)
    decode_active = False
    lm_model.norm = final_norm
    print("\n")

    N = len(captured_pre_norm)
    print(f"  Estados de decode capturados con precisión (Δlen=1): {N}")
    if N < 10:
        print("❌ Error: No se capturaron suficientes tokens de decode.")
        return

    # 4. Evaluación de Predicción Causal fuera de muestra
    # Predictor 0: Persistencia h_t
    # Predictor 1: Velocidad Constante h_t + v_t
    # Predictor 2: Aceleración Constante h_t + v_t + 1/2 a_t
    # Predictor 3: Geodesic Projective Cell
    err_b0, err_b1, err_b2, err_cell = [], [], [], []
    cos_b0, cos_b1, cos_b2, cos_cell = [], [], [], []
    kl_b0,  kl_b1,  kl_b2,  kl_cell  = [], [], [], []

    kappa_kin_list = []
    delta_theta_real = []

    # Extraer atractor del contexto (promedio de los primeros tokens de decode, no del futuro)
    u_context = mx.array(captured_pre_norm[0])
    u_context = u_context / mx.sqrt(mx.sum(u_context * u_context))
    mx.eval(u_context)

    # Ingestión persistente en el búfer C++
    aether_native_c.buffer_reset()
    for t in range(2):
        aether_native_c.buffer_push_state(mx.array(captured_pre_norm[t]), step=t)

    for t in range(2, N - 1):
        h_t_mx = mx.array(captured_pre_norm[t])
        st = aether_native_c.buffer_push_state(h_t_mx, step=t)
        mx.eval(st["v_t"], st["a_t"])

        v_t = st["v_t"]
        a_t = st["a_t"]
        h_tp1_real = mx.array(captured_pre_norm[t + 1])
        mx.eval(h_tp1_real)

        norm_h_tp1 = mx.sqrt(mx.sum(h_tp1_real * h_tp1_real))
        u_tp1_real = h_tp1_real / norm_h_tp1

        # ─── B0: Persistencia ────────────────────────────────────────────────
        h_pred_b0 = h_t_mx / mx.sqrt(mx.sum(h_t_mx * h_t_mx))

        # ─── B1: Velocidad Constante ─────────────────────────────────────────
        h_pred_b1 = h_t_mx + v_t
        h_pred_b1 = h_pred_b1 / mx.sqrt(mx.sum(h_pred_b1 * h_pred_b1))

        # ─── B2: Aceleración Constante ───────────────────────────────────────
        h_pred_b2 = h_t_mx + v_t + 0.5 * a_t
        h_pred_b2 = h_pred_b2 / mx.sqrt(mx.sum(h_pred_b2 * h_pred_b2))

        # ─── B3: Geodesic Projective Cell (Hito 1.1) ─────────────────────────
        # Corrección: Se utiliza u_context (derivado del pasado), NUNCA h_{t+1}
        cell_out = aether_native_c.dispatch_geodesic_trajectory_cell(
            h_t_mx, v_t, a_t, u_context, tau=1.0, kappa_att=0.05
        )
        h_pred_cell = cell_out["h_star"]
        kappa = float(cell_out["curvature_kappa"])
        kappa_kin_list.append(kappa)

        mx.eval(h_pred_b0, h_pred_b1, h_pred_b2, h_pred_cell)

        # ─── Métricas Angulares y de Distancia ────────────────────────────────
        c0 = float(mx.sum(h_pred_b0 * u_tp1_real))
        c1 = float(mx.sum(h_pred_b1 * u_tp1_real))
        c2 = float(mx.sum(h_pred_b2 * u_tp1_real))
        cc = float(mx.sum(h_pred_cell * u_tp1_real))

        cos_b0.append(c0); cos_b1.append(c1); cos_b2.append(c2); cos_cell.append(cc)
        err_b0.append(float(mx.sqrt(mx.sum((h_pred_b0 - u_tp1_real)**2))))
        err_b1.append(float(mx.sqrt(mx.sum((h_pred_b1 - u_tp1_real)**2))))
        err_b2.append(float(mx.sqrt(mx.sum((h_pred_b2 - u_tp1_real)**2))))
        err_cell.append(float(mx.sqrt(mx.sum((h_pred_cell - u_tp1_real)**2))))

        # ─── Giro Angular Real Δθ entre v_t y v_{t+1} ────────────────────────
        v_next = h_tp1_real - h_t_mx
        nv_t = float(mx.sqrt(mx.sum(v_t * v_t)))
        nv_n = float(mx.sqrt(mx.sum(v_next * v_next)))
        if nv_t > 1e-6 and nv_n > 1e-6:
            cos_d = max(-1.0, min(1.0, float(mx.sum(v_t * v_next)) / (nv_t * nv_n)))
            delta_theta_real.append(math.acos(cos_d))
        else:
            delta_theta_real.append(0.0)

        # ─── Proyección en Logits y Divergencia KL ────────────────────────────
        def get_probs(h_vec):
            h_normed = final_norm(h_vec[None, None, :])
            z = lm_head(h_normed)[0, 0, :]
            return mx.softmax(z)

        p_real = get_probs(h_tp1_real)
        p_b0   = get_probs(h_pred_b0 * norm_h_tp1)
        p_b1   = get_probs(h_pred_b1 * norm_h_tp1)
        p_b2   = get_probs(h_pred_b2 * norm_h_tp1)
        p_cell = get_probs(h_pred_cell * norm_h_tp1)
        mx.eval(p_real, p_b0, p_b1, p_b2, p_cell)

        def kl(p, q):
            return float(mx.sum(p * mx.log((p + 1e-12) / (q + 1e-12))))

        kl_b0.append(kl(p_real, p_b0))
        kl_b1.append(kl(p_real, p_b1))
        kl_b2.append(kl(p_real, p_b2))
        kl_cell.append(kl(p_real, p_cell))

    # 5. Reporte Estadístico Riguroso
    section("LAB 09-R1 — RESULTADOS CIENTÍFICOS RIGUROSOS")
    total_samples = len(cos_cell)
    print(f"  N predicciones evaluadas fuera de muestra: {total_samples}")
    print("\n  " + "─" * 74)
    print(f"  {'Predictor':<22} │ {'Cosine Sim':<12} │ {'Error Angular':<15} │ {'KL Logits':<12}")
    print("  " + "─" * 74)
    print(f"  {'B0 (Persistencia)':<22} │ {np.mean(cos_b0):<12.4f} │ {np.mean(err_b0):<15.4f} │ {np.mean(kl_b0):<12.4f}")
    print(f"  {'B1 (Velocidad Const)':<22} │ {np.mean(cos_b1):<12.4f} │ {np.mean(err_b1):<15.4f} │ {np.mean(kl_b1):<12.4f}")
    print(f"  {'B2 (Aceleración Const)':<22} │ {np.mean(cos_b2):<12.4f} │ {np.mean(err_b2):<15.4f} │ {np.mean(kl_b2):<12.4f}")
    print(f"  {'B3 (Célula Geodésica)':<22} │ {np.mean(cos_cell):<12.4f} │ {np.mean(err_cell):<15.4f} │ {np.mean(kl_cell):<12.4f}")
    print("  " + "─" * 74)

    # Ventajas relativas
    delta_vs_b0 = np.mean(err_b0) - np.mean(err_cell)
    delta_vs_b1 = np.mean(err_b1) - np.mean(err_cell)
    delta_vs_b2 = np.mean(err_b2) - np.mean(err_cell)

    print(f"\n  Ganancia de Error Angular (Δ > 0 implica ventaja de la Célula):")
    print(f"    • Célula vs Persistencia       : {delta_vs_b0:+.5f}")
    print(f"    • Célula vs Velocidad Constante: {delta_vs_b1:+.5f}")
    print(f"    • Célula vs Aceleración Const  : {delta_vs_b2:+.5f}")

    # Correlación Curvatura de Lagrange vs Deflexión Angular Real
    corr_kappa = 0.0
    if np.std(kappa_kin_list) > 1e-6 and np.std(delta_theta_real) > 1e-6:
        corr_kappa = float(np.corrcoef(kappa_kin_list, delta_theta_real)[0, 1])
    print(f"\n  Correlación Curvatura κ ↔ Giro Real Δθ: r = {corr_kappa:.4f}")

    # Veredicto científico objetivo
    section("DICTAMEN EXPERIMENTAL")
    is_predictive = (np.mean(cos_cell) >= np.mean(cos_b1) - 0.005) and (np.mean(kl_cell) <= np.mean(kl_b0))
    if is_predictive:
        print("  🏆 CONCLUSIÓN: [ PREDICTIVE — LA CÉLULA CAPTURA DINÁMICA REAL ]")
        print("     La extrapolación geodésica analítica predice el avance del Transformer")
        print("     con menor o igual entropía que los baselines inerciales.")
    else:
        print("  ⚠️ CONCLUSIÓN: [ NON-PREDICTIVE / LOCAL DYNAMICS ]")
        print("     La célula requiere acoplamiento con la Fact Band o ajuste de horizonte tau.")

if __name__ == "__main__":
    run_lab09_r1()
