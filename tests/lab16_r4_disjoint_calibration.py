#!/usr/bin/env python3
"""
tests/lab16_r4_disjoint_calibration.py
═══════════════════════════════════════════════════════════════════════════════
LAB 16-R4 — CALIBRACIÓN DISJUNTA TOTAL, TRIPLE MÉTRICA Y CONTROL DE FUGA
SSOT: Auditoría del Asesor Técnico:
  1. Calibración DISJUNTA: W_ad aprende SOLO de D ∈ {3, 9}.
  2. D=5 y D=7 son HELD-OUT ABSOLUTOS (cero fuga de h_nat en calibración).
  3. Triplete métrico: cos, ratio de norma ||Δz_s||/||Δz_n|| y error relativo.
  4. Controles negativos: Random q de igual norma y cero intervención.
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, time
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

def run_lab16_r4():
    section("LAB 16-R4: CALIBRACIÓN ESTRICTAMENTE DISJUNTA (CERO FUGA)")
    print(f"  Modelo: Qwen3.5-0.8B | Capa de Intervención: L* = {L_SWAP}")
    print(f"  Calibración de W_ad: EXCLUSIVA en D ∈ {{3, 9}} (k=2 y k=8)")
    print(f"  Evaluación HELD-OUT: D ∈ {{5, 7}} (k=4 y k=6 JAMÁS VISTOS EN W_ad)")

    model, processor = load(MODEL_PATH)
    tok = getattr(processor, "tokenizer", processor)
    lm_model = model.language_model.model
    num_layers = len(lm_model.layers)
    orig_layers = list(lm_model.layers)
    D_dim = lm_model.layers[0].self_attn.q_proj.weight.shape[-1] if hasattr(lm_model.layers[0], "self_attn") else 1024

    tmpl = (
        "Considera cinco enteros distintos A, B, C, D, E entre 1 y 15. "
        "Se cumple que A > B, B > C, D = C + {k}, E = A + C, E < 13, D < B. "
        "A es par y C es impar. "
        "Con estas condiciones, el valor exacto de D es "
    )

    k_list = [2, 4, 6, 8]
    D_targets = {2: 3, 4: 5, 6: 7, 8: 9}

    def extract_h19_and_logits(prompt_str):
        ids = mx.array(tok.encode(prompt_str))[None, :]
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
        out = model.language_model(ids)
        logits = out.logits[0, -1, :].astype(mx.float32)
        mx.eval(logits)
        for l in range(num_layers): lm_model.layers[l] = orig_layers[l]
        return storage[0], to_numpy_f32(logits)

    # ── 1. CAPTURA DE BASELINES NATURALES ─────────────────────────────────────
    section("1. CAPTURA DE BASELINES NATURALES")
    h_nat = {}
    z_nat = {}

    for k in k_list:
        h, z = extract_h19_and_logits(tmpl.format(k=k))
        h_nat[k] = h
        z_nat[k] = z
        print(f"  • Baseline k={k} (D={D_targets[k]}): ||h19|| = {np.linalg.norm(h):.2f}, ||z|| = {np.linalg.norm(z):.2f}")

    # ── 2. CONSTRUCCIÓN DISJUNTA DE W_ad (SOLO D=3 Y D=9) ────────────────────
    section("2. CALIBRACIÓN DISJUNTA DE W_ad (CERO FUGA DE D=5 NI D=7)")
    np.random.seed(2026)
    P_frozen = np.random.randn(D_dim, 3).astype(np.float32) / np.sqrt(3.0)

    def get_abstract_state(val_D):
        v = np.array([float(val_D), float(val_D)**2, 1.0], dtype=np.float32)
        return P_frozen @ v

    # Estados abstractos polinómicos continuos
    q3 = get_abstract_state(3)
    q5 = get_abstract_state(5)
    q7 = get_abstract_state(7)
    q9 = get_abstract_state(9)

    # CALIBRACIÓN: Usamos SOLO el delta entre los extremos (D=3 y D=9)
    # W_ad mapea (q9 - q3) hacia (h_nat[8] - h_nat[2])
    delta_q_train = q9 - q3
    delta_h_train = h_nat[8] - h_nat[2]

    # Operador lineal de rango 1 compartido W_ad = (delta_h @ delta_q^T) / ||delta_q||^2
    norm_dq_sq = float(np.dot(delta_q_train, delta_q_train))
    W_ad = np.outer(delta_h_train, delta_q_train) / norm_dq_sq

    print(f"  • Adaptador W_ad entrenado ÚNICAMENTE con el par extremo (D=3, D=9).")
    print(f"  • Matriz W_ad: R^{D_dim} -> R^{D_dim}, Rango=1 continuo, CERO conocimiento de D=5 y D=7.")

    # ── 3. EVALUACIÓN DE LAS INTERVENCIONES HELD-OUT D=5 Y D=7 ────────────────
    section("3. SÍNTESIS DE INTERVENCIONES HELD-OUT")

    # Sintetizar desplazamiento para D=5 y D=7 respecto a la base D=3
    delta_h_synth = {
        3: np.zeros(D_dim, dtype=np.float32),
        5: W_ad @ (q5 - q3), # HELD-OUT INTERPOLADO
        7: W_ad @ (q7 - q3), # HELD-OUT INTERPOLADO
        9: W_ad @ (q9 - q3)  # Reconstrucción de calibración
    }

    print(f"  • ||Δh_synth(D=5 held-out)|| = {np.linalg.norm(delta_h_synth[5]):.4f}")
    print(f"  • ||Δh_synth(D=7 held-out)|| = {np.linalg.norm(delta_h_synth[7]):.4f}")
    print(f"  • ||Δh_synth(D=9 extremo) || = {np.linalg.norm(delta_h_synth[9]):.4f}")

    # ── 4. FORWARD CON INTERVENCIÓN SINTÉTICA EN L19 ──────────────────────────
    section("4. EJECUCIÓN FORWARD CON INTERVENCIÓN SINTÉTICA EN SILICIO")

    def forward_with_delta(prompt_str, delta_vec):
        aether_native_c.junction_reset()
        ids = mx.array(tok.encode(prompt_str))[None, :]
        if delta_vec is None or np.linalg.norm(delta_vec) == 0.0:
            out = model.language_model(ids)
            logits = out.logits[0, -1, :].astype(mx.float32)
            mx.eval(logits)
            return to_numpy_f32(logits)

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
        out = model.language_model(ids)
        logits = out.logits[0, -1, :].astype(mx.float32)
        mx.eval(logits)
        for l in range(num_layers): lm_model.layers[l] = orig_layers[l]
        return to_numpy_f32(logits)

    # Inyectar en Contexto Base k=2 (D=3 natural) hacia D=5 y D=7
    z_base_k2 = z_nat[2]
    z_synth_k2_to_5 = forward_with_delta(tmpl.format(k=2), delta_h_synth[5])
    z_synth_k2_to_7 = forward_with_delta(tmpl.format(k=2), delta_h_synth[7])
    z_synth_k2_to_9 = forward_with_delta(tmpl.format(k=2), delta_h_synth[9])

    # Control Negativo: Random Q de igual norma que delta_q(7)
    np.random.seed(999)
    rnd_q = np.random.randn(D_dim).astype(np.float32)
    rnd_q = rnd_q * (np.linalg.norm(q7 - q3) / np.linalg.norm(rnd_q))
    delta_h_rand = W_ad @ rnd_q
    z_synth_random = forward_with_delta(tmpl.format(k=2), delta_h_rand)

    # ── 5. EL TRIPLETE MÉTRICO RIGUROSO (DIRECCIÓN, MAGNITUD, ERROR RELATIVO)
    section("5. EL TRIPLETE MÉTRICO RIGUROSO (CERO FUGA)")
    print("  Evaluado sobre los 151,936 logits en float32 continuo:\n")
    print(f"  {'Transición Evaluada':<26} │ {'Cos(Δz_s, Δz_n)':<16} │ {'Ratio ||s||/||n||':<18} │ {'Error Relativo':<15} │ Veredicto")
    print("  " + "─" * 90)

    eval_pairs = [
        ("k=2 -> D=5 [HELD-OUT]", z_synth_k2_to_5 - z_base_k2, z_nat[4] - z_nat[2]),
        ("k=2 -> D=7 [HELD-OUT]", z_synth_k2_to_7 - z_base_k2, z_nat[6] - z_nat[2]),
        ("k=2 -> D=9 [EXTREMO]",  z_synth_k2_to_9 - z_base_k2, z_nat[8] - z_nat[2]),
        ("Control Random Q",       z_synth_random - z_base_k2,   z_nat[6] - z_nat[2])
    ]

    for label, dz_s, dz_n in eval_pairs:
        norm_s = np.linalg.norm(dz_s)
        norm_n = np.linalg.norm(dz_n)
        dot_sn = np.dot(dz_s, dz_n)
        cos_val = float(dot_sn / (norm_s * norm_n + 1e-12))
        ratio_val = float(norm_s / (norm_n + 1e-12))
        err_rel = float(np.linalg.norm(dz_s - dz_n) / (norm_n + 1e-12))

        status = "✅ GENERALIZA" if (cos_val > 0.60 and "Random" not in label) else ("❌ INESPECÍFICO" if "Random" in label else "⚠️ DÉBIL")
        if "Random" in label and cos_val < 0.10: status = "✅ CONTROL VÁLIDO"

        print(f"  {label:<26} │ {cos_val:+16.6f} │ {ratio_val:18.4f} │ {err_rel:15.4f} │ {status}")

    section("DICTAMEN EXPERIMENTAL LAB 16-R4 CONCLUIDO")
    print("  ✓ Calibración disjunta ejecutada sin fuga supervisada.")

if __name__ == "__main__":
    run_lab16_r4()
