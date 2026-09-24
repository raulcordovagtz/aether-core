#!/usr/bin/env python3
"""
tests/lab23_symbolic_manifold_separation.py
═══════════════════════════════════════════════════════════════════════════════
LAB 23 — SEPARACIÓN CAUSAL FINA DE ESTADOS CERCANOS D ∈ {5, 6, 7, 8, 9}
SSOT: Enmiendas Metodológicas del Asesor Técnico:
  1. Contexto ciego e idéntico: k=2 (D=3)
  2. 5 estados continuos consecutivos: D ∈ {5, 6, 7, 8, 9}
  3. Geometría residual G_Δh y geometría de logits G_Δz (151,936 dims)
  4. Matriz de distancias relativas L2 y matriz de divergencias KL
  5. Test fino D=7 vs D=8: discriminación de logits de tokens específicos
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

def run_lab23():
    section("LAB 23: SEPARACIÓN CAUSAL FINA DE ESTADOS CERCANOS D ∈ {5, 6, 7, 8, 9}")
    print(f"  Modelo: Qwen3.5-0.8B | Capa de Intervención: L* = {L_SWAP}")
    print(f"  Contexto Base Ciego: k=2 (Formal) | Evaluando Estados: D ∈ [5, 6, 7, 8, 9]")

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

    # Token IDs de los dígitos evaluados
    digit_tokens = {d: tok.encode(f" {d}")[-1] for d in range(3, 10)}

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

    # 1. Capturar baseline natural de k=2
    h_base_k2, z_base_k2 = extract_h19_and_logits(ids_base)

    # Capturar baseline k=8 para calibración disjunta
    tmpl_k8 = tmpl_base.replace("C + 2", "C + 8")
    h_base_k8, _ = extract_h19_and_logits(mx.array(tok.encode(tmpl_k8))[None, :])

    # 2. Construcción del Adaptador No Léxico W_ad
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

    # 3. Generar las 5 perturbaciones consecutivas D ∈ {5, 6, 7, 8, 9}
    test_D_vals = [5, 6, 7, 8, 9]
    delta_h_dict = {
        d: W_ad @ (make_q_basis(d) - q3) for d in test_D_vals
    }

    # ── FASE 1: GEOMETRÍA EN EL ESPACIO RESIDUAL R^1024 ───────────────────────
    section("1. GEOMETRÍA EN EL ESPACIO RESIDUAL R^1024 (G_Δh Y DISTANCIAS L2)")
    print("  Matriz de Cosenos Residuales cos(Δh_i, Δh_j):")
    print(f"  {'':<6} " + "  ".join([f"D={d:<6}" for d in test_D_vals]))
    
    G_dh = np.zeros((5, 5))
    dist_dh = np.zeros((5, 5))

    for i, d1 in enumerate(test_D_vals):
        row = []
        for j, d2 in enumerate(test_D_vals):
            v1, v2 = delta_h_dict[d1], delta_h_dict[d2]
            c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            G_dh[i, j] = c
            dist_dh[i, j] = np.linalg.norm(v1 - v2)
            row.append(f"{c:.4f}")
        print(f"  D={d1:<4} [ " + "  ".join(row) + " ]")

    print("\n  Magnitudes de Perturbación Residual ||Δh(D)||:")
    for d in test_D_vals:
        print(f"    • D={d}: ||Δh|| = {np.linalg.norm(delta_h_dict[d]):.4f}")

    # ── FASE 2: FORWARD EN SILICIO Y CAPTURA DE LOGITS EN R^151936 ────────────
    section("2. FORWARD EN SILICIO UMA Y CAPTURA DE LOGITS (151,936 DIMS)")

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

    z_steered_dict = {d: forward_steered(delta_h_dict[d]) for d in test_D_vals}
    dz_dict = {d: z_steered_dict[d] - z_base_k2 for d in test_D_vals}

    # ── FASE 3: MATRIZ DE DISTANCIAS RELATIVAS L2 Y DIVERGENCIAS KL ──────────
    section("3. MATRIZ DE DISTANCIAS L2 Y DIVERGENCIAS KL SOBRE 151,936 LOGITS")

    def get_probs(z_vec):
        p = np.exp(z_vec - np.max(z_vec))
        return p / np.sum(p)

    probs_dict = {d: get_probs(z_steered_dict[d]) for d in test_D_vals}

    print("  Matriz de Cosenos de Logits cos(Δz_i, Δz_j):")
    print(f"  {'':<6} " + "  ".join([f"D={d:<6}" for d in test_D_vals]))
    G_dz = np.zeros((5, 5))
    for i, d1 in enumerate(test_D_vals):
        row = []
        for j, d2 in enumerate(test_D_vals):
            z1, z2 = dz_dict[d1], dz_dict[d2]
            c = float(np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2) + 1e-12))
            G_dz[i, j] = c
            row.append(f"{c:.4f}")
        print(f"  D={d1:<4} [ " + "  ".join(row) + " ]")

    print("\n  Matriz de Divergencias KL D_KL(P_i || P_j) [en milinats, mnat = 1e-3 nats]:")
    print(f"  {'':<6} " + "  ".join([f"D={d:<7}" for d in test_D_vals]))
    KL_mat = np.zeros((5, 5))
    for i, d1 in enumerate(test_D_vals):
        row = []
        for j, d2 in enumerate(test_D_vals):
            p1, p2 = probs_dict[d1], probs_dict[d2]
            kl_val = float(np.sum(p1 * np.log((p1 + 1e-12) / (p2 + 1e-12)))) * 1e3
            KL_mat[i, j] = kl_val
            row.append(f"{kl_val:6.2f}")
        print(f"  D={d1:<4} [ " + "  ".join(row) + " ]")

    # ── FASE 4: DISCRIMINACIÓN ESPECÍFICA DE TOKENS (D=7 vs D=8) ─────────────
    section("4. DISCRIMINACIÓN FINA: DESPLAZAMIENTOS DE LOGITS EN TOKENS ESPECÍFICOS")
    print("  Evaluando Δlogit = z_steered(token) - z_vanilla(token) para cada intervención:\n")
    print(f"  {'Intervención':<14} │ {'Token \"5\"':<12} │ {'Token \"6\"':<12} │ {'Token \"7\"':<12} │ {'Token \"8\"':<12} │ {'Token \"9\"':<12}")
    print("  " + "─" * 74)

    for d in test_D_vals:
        zs = z_steered_dict[d]
        row_str = []
        for target_d in test_D_vals:
            t_id = digit_tokens[target_d]
            delta_logit = float(zs[t_id] - z_base_k2[t_id])
            marker = f"[{delta_logit:+6.2f}]" if target_d == d else f" {delta_logit:+6.2f} "
            row_str.append(marker)
        print(f"  D = {d:<10} │ {row_str[0]:<12} │ {row_str[1]:<12} │ {row_str[2]:<12} │ {row_str[3]:<12} │ {row_str[4]:<12}")

    # Métricas específicas del par D=7 vs D=8
    diff_7_8 = np.linalg.norm(dz_dict[7] - dz_dict[8])
    norm_7 = np.linalg.norm(dz_dict[7])
    rel_err_7_8 = diff_7_8 / norm_7
    cos_7_8 = G_dz[2, 3]
    kl_7_8 = KL_mat[2, 3]

    print(f"\n  Comparativa Fina D=7 vs D=8:")
    print(f"    • Cos(Δz_7, Δz_8)                 : {cos_7_8:.6f}")
    print(f"    • Distancia Relativa ||Δz7 - Δz8|| : {rel_err_7_8:.4f} (Separación euclídea del {rel_err_7_8*100:.1f}%)")
    print(f"    • Divergencia KL(P_7 || P_8)       : {kl_7_8:.3f} mnat ({kl_7_8*1e-3:.5f} nats)")

    section("DICTAMEN EXPERIMENTAL LAB 23 CONCLUIDO")
    print("  ✓ Análisis de separación fina de la variedad continua completado.")

if __name__ == "__main__":
    run_lab23()
