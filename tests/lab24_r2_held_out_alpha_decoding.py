#!/usr/bin/env python3
"""
tests/lab24_r2_held_out_alpha_decoding.py
═══════════════════════════════════════════════════════════════════════════════
LAB 24-R2 — DECODIFICACIÓN INVERSA DE ALFA EN PUNTOS HELD-OUT
SSOT: Auditoría del Asesor Técnico:
  1. Calibración del decodificador inverso SOLO en rejilla gruesa:
     α_train ∈ {0.0, 0.25, 0.50, 0.75, 1.00}
  2. Evaluación EXCLUSIVA en valores HELD-OUT jamás vistos:
     α_held_out ∈ {0.10, 0.20, 0.35, 0.65, 0.85, 0.95}
  3. Cálculo de R² out-of-sample y error medio absoluto
  4. Texto sobrio y métricas puras sin redondeos inflados
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, time, math
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

from mlx_vlm import load
import aether_native_c

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")
L_SWAP = 19

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def to_numpy_f32(mlx_arr):
    arr_f32 = mlx_arr.astype(mx.float32)
    mx.eval(arr_f32)
    return np.array(arr_f32, copy=True).reshape(-1)

def run_lab24_r2():
    section("LAB 24-R2: DECODIFICACIÓN INVERSA EN PUNTOS HELD-OUT (CERO OVERFITTING)")
    print(f"  Modelo: Qwen3.5-0.8B | Capa de Intervención: L* = {L_SWAP}")
    print(f"  Calibración de Decodificador: α_train ∈ {{0.0, 0.25, 0.50, 0.75, 1.00}}")
    print(f"  Evaluación Out-of-Sample    : α_test  ∈ {{0.10, 0.20, 0.35, 0.65, 0.85, 0.95}}")

    model, processor = load(MODEL_PATH)
    tok = getattr(processor, "tokenizer", processor)
    lm_model = model.language_model.model
    num_layers = len(lm_model.layers)
    orig_layers = list(lm_model.layers)
    D_dim = lm_model.layers[0].self_attn.q_proj.weight.shape[-1] if hasattr(lm_model.layers[0], "self_attn") else 1024

    tmpl_base = (
        "Considera cinco enteros distintos A, B, C, D, E entre 1 y 15. "
        "Se cumple que A > B, B > C, D = C + 2, E = A + C, E < 13, D < B. "
        "A es par y C es impar. "
        "Con estas condiciones, el valor exacto de D es "
    )

    ids_base = mx.array(tok.encode(tmpl_base))[None, :]

    def extract_h19_and_logits(prompt_ids):
        storage = []
        class HookL19:
            def __init__(self, layer, idx): self.layer, self.idx = layer, idx
            def __getattr__(self, name): return getattr(self.layer, name)
            def __call__(self, *args, **kwargs):
                out = self.layer(*args, **kwargs)
                if self.idx == L_SWAP:
                    h = out[0, -1, :].astype(mx.float32)
                    mx.eval(h)
                    storage.append(to_numpy_f32(h))
                return out
        for l in range(num_layers): lm_model.layers[l] = HookL19(orig_layers[l], l)
        out = model.language_model(prompt_ids)
        logits = out.logits[0, -1, :].astype(mx.float32)
        mx.eval(logits)
        for l in range(num_layers): lm_model.layers[l] = orig_layers[l]
        return storage[0], to_numpy_f32(logits)

    h_base_k2, z_base_k2 = extract_h19_and_logits(ids_base)
    tmpl_k8 = tmpl_base.replace("C + 2", "C + 8")
    h_base_k8, _ = extract_h19_and_logits(mx.array(tok.encode(tmpl_k8))[None, :])

    # Construcción de W_ad
    np.random.seed(2026)
    P_frozen = np.random.randn(D_dim, 6).astype(np.float32) / np.sqrt(6.0)

    def make_q_basis(val_D):
        d = float(val_D)
        v = np.array([d, d**2, d**3 * 0.01, math.sqrt(d), 1.0 / d, 1.0], dtype=np.float32)
        return P_frozen @ v

    q3 = make_q_basis(3)
    q9 = make_q_basis(9)
    delta_q = q9 - q3
    delta_h = h_base_k8 - h_base_k2
    W_ad = np.outer(delta_h, delta_q) / float(np.dot(delta_q, delta_q))

    v_ray_raw = W_ad @ (q9 - q3)
    norm_v_max = float(np.linalg.norm(v_ray_raw))
    v_unit = v_ray_raw / norm_v_max

    def forward_steered(delta_vec):
        aether_native_c.junction_reset()
        delta_mx = mx.array(delta_vec)
        class Hook:
            def __init__(self, layer, idx): self.layer, self.idx = layer, idx
            def __getattr__(self, name): return getattr(self.layer, name)
            def __call__(self, x, **kwargs):
                out = self.layer(x, **kwargs)
                if self.idx == L_SWAP:
                    h_last = out[:, -1:, :].astype(mx.float32)
                    h_patched = h_last + delta_mx[None, None, :]
                    out = mx.concatenate([out[:, :-1, :], h_patched.astype(out.dtype)], axis=1)
                return out

        for l in range(num_layers): lm_model.layers[l] = Hook(orig_layers[l], l)
        out = model.language_model(ids_base)
        logits = out.logits[0, -1, :].astype(mx.float32)
        mx.eval(logits)
        for l in range(num_layers): lm_model.layers[l] = orig_layers[l]
        return to_numpy_f32(logits)

    # ── 1. FASE DE CALIBRACIÓN DEL DECODIFICADOR (SOLO 5 PUNTOS) ─────────────
    section("1. CALIBRACIÓN DEL DECODIFICADOR SOBRE REJILLA GRUESA α_train")
    alphas_train = [0.00, 0.25, 0.50, 0.75, 1.00]
    dz_train_list = []

    for a in alphas_train:
        dh = v_unit * (a * norm_v_max)
        z = forward_steered(dh)
        dz_train_list.append(z - z_base_k2)

    # Ajustar proyector inverso óptimo por mínimos cuadrados:
    # min_w sum_i (w^T dz_i - a_i)^2
    Z_train_mat = np.column_stack(dz_train_list) # [151936, 5]
    y_train_vec = np.array(alphas_train, dtype=np.float32) # [5]

    # w_inv = (Z Z^T + lambda I)^-1 Z y -> usando SVD de Z_train
    Uz, Sz, Vtz = np.linalg.svd(Z_train_mat, full_matrices=False)
    # Pseudo-inversa estable
    w_inv = (Uz / (Sz + 1e-4)) @ Vtz @ y_train_vec

    norm_w_inv = np.linalg.norm(w_inv)
    print(f"  • Vector proyector inverso ajustado sobre 5 puntos: ||w_inv|| = {norm_w_inv:.6f}")

    # Verificar ajuste in-sample
    train_recovered = [float(np.dot(dz, w_inv)) for dz in dz_train_list]
    train_errs = [abs(t - r) for t, r in zip(alphas_train, train_recovered)]
    print(f"  • Error máximo in-sample en calibración: {max(train_errs):.6f}")

    # ── 2. EVALUACIÓN OUT-OF-SAMPLE EN PUNTOS HELD-OUT (CERO FUGA) ───────────
    section("2. EVALUACIÓN OUT-OF-SAMPLE EN PUNTOS HELD-OUT")
    alphas_held_out = [0.10, 0.20, 0.35, 0.65, 0.85, 0.95]

    recovered_held_out = []
    dz_held_out_list = []

    print(f"  {'Alfa Real (Inyectado)':<24} │ {'Alfa Decodificado de Logits':<28} │ Error Absoluto")
    print("  " + "─" * 68)

    for a_test in alphas_held_out:
        dh_test = v_unit * (a_test * norm_v_max)
        z_test = forward_steered(dh_test)
        dz_test = z_test - z_base_k2
        dz_held_out_list.append(dz_test)

        a_hat = float(np.dot(dz_test, w_inv))
        recovered_held_out.append(a_hat)
        err = abs(a_test - a_hat)
        print(f"  α = {a_test:<18.2f} │ α̂ = {a_hat:<24.6f} │ {err:.6f}")

    # ── 3. MÉTRICAS RIGUROSAS OUT-OF-SAMPLE ───────────────────────────────────
    section("3. MÉTRICAS DE GENERALIZACIÓN OUT-OF-SAMPLE")
    y_true = np.array(alphas_held_out)
    y_pred = np.array(recovered_held_out)

    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2_held_out = 1.0 - (ss_res / (ss_tot + 1e-12))
    mae_held_out = float(np.mean(np.abs(y_true - y_pred)))
    max_err_held_out = float(np.max(np.abs(y_true - y_pred)))

    print(f"  • R² Out-of-Sample (Held-Out)           : {r2_held_out:.8f}")
    print(f"  • Error Medio Absoluto (MAE Held-Out)   : {mae_held_out:.6f}")
    print(f"  • Error Máximo Absoluto (Max Held-Out)  : {max_err_held_out:.6f}")

    assert r2_held_out > 0.99, f"Fallo en generalización out-of-sample: R²={r2_held_out}"
    print(f"\n  [✅ PASS] R² Out-of-Sample = {r2_held_out:.6f} > 0.99:")
    print("  Queda formalmente demostrado que la relación entre la coordenada continua α")
    print("  y la perturbación en los 151,936 logits generaliza fuera de muestra sin sobreajuste.")

    section("DICTAMEN FINAL LAB 24-R2 CONCLUIDO")

if __name__ == "__main__":
    run_lab24_r2()
