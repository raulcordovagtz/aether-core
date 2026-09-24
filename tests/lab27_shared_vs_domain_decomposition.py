#!/usr/bin/env python3
"""
tests/lab27_shared_vs_domain_decomposition.py
═══════════════════════════════════════════════════════════════════════════════
LAB 27 — DESCOMPOSICIÓN SVD: EJE COMPARTIDO vs DESVIACIÓN DE DOMINIO
SSOT: Formulación del Asesor Técnico:
  1. Extracción de los 3 ejes naturales: v_A, v_B, v_C en R^1024
  2. SVD del conjunto V = [v_A, v_B, v_C]: cálculo de la varianza compartida σ1² / Σσ²
  3. Extracción de v_shared = u_1 (el núcleo invariante de razonamiento numérico)
  4. Evaluación cruzada con v_shared en las 3 familias simultáneamente
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

def run_lab27():
    section("LAB 27: DESCOMPOSICIÓN SVD DEL EJE COMPARTIDO vs ESPECÍFICO")
    print(f"  Modelo: Qwen3.5-0.8B | Capa de Intervención: L* = {L_SWAP}")

    model, processor = load(MODEL_PATH)
    tok = getattr(processor, "tokenizer", processor)
    lm_model = model.language_model.model
    num_layers = len(lm_model.layers)
    orig_layers = list(lm_model.layers)

    families = {
        "Fam A (Puzle Lógico)": (
            "Considera cinco enteros distintos A, B, C, D, E entre 1 y 15. "
            "Se cumple que A > B, B > C, D = C + {k}, E = A + C, E < 13, D < B. "
            "A es par y C es impar. "
            "Con estas condiciones, el valor exacto de D es "
        ),
        "Fam B (Logística Almacén)": (
            "En un almacen central de logistica hay tres tipos de contenedores: cajas rojas, azules y verdes. "
            "Se sabe que la cantidad de cajas verdes es exactamente igual a la cantidad de cajas azules mas {k} unidades. "
            "Si las cajas azules son 1, el total de cajas verdes es "
        ),
        "Fam C (Distancias Tren)": (
            "Un tren de alta velocidad viaja entre tres estaciones Alfa, Beta y Gamma. "
            "La distancia de Alfa a Gamma supera a la distancia de Alfa a Beta en exactamente {k} kilometros. "
            "Sabiendo que la distancia a Beta es 1 kilometro, la distancia exacta a Gamma es "
        )
    }

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

    # ── 1. EXTRACCIÓN DE LOS 3 EJES NATURALES v_A, v_B, v_C ──────────────────
    section("1. EXTRACCIÓN DE LOS EJES NATURALES POR DOMINIO")
    v_domains = {}
    h_bases = {}
    z_bases = {}

    for name, tmpl in families.items():
        h2, z2 = extract_h19_and_logits(tmpl.format(k=2))
        h8, z8 = extract_h19_and_logits(tmpl.format(k=8))
        v_raw = h8 - h2
        v_domains[name] = v_raw
        h_bases[name] = h2
        z_bases[name] = z2
        print(f"  • {name:<26}: ||h8 - h2|| = {np.linalg.norm(v_raw):.4f}")

    # Matriz de cosenos directos entre los 3 ejes naturales en R^1024
    keys = list(families.keys())
    print("\n  Matriz de Cosenos entre Ejes Naturales en el Espacio Residual R^1024:")
    print(f"  {'':<26} " + "  ".join([f"Eje {chr(65+i):<6}" for i in range(3)]))
    for i, k1 in enumerate(keys):
        row = []
        for j, k2 in enumerate(keys):
            c = float(np.dot(v_domains[k1], v_domains[k2]) / (np.linalg.norm(v_domains[k1]) * np.linalg.norm(v_domains[k2])))
            row.append(f"{c:+.4f}")
        print(f"  {k1:<26} [ " + "  ".join(row) + " ]")

    # ── 2. DESCOMPOSICIÓN SVD: VARIANZA COMPARTIDA vs ESPECÍFICA ─────────────
    section("2. DESCOMPOSICIÓN SVD DEL ESPACIO DE LOS 3 EJES")
    # V = [v_A / ||v_A||, v_B / ||v_B||, v_C / ||v_C||]
    V_matrix = np.column_stack([v_domains[k] / np.linalg.norm(v_domains[k]) for k in keys]) # [1024, 3]

    U_svd, S_svd, Vt_svd = np.linalg.svd(V_matrix, full_matrices=False)
    var_explained = (S_svd ** 2) / np.sum(S_svd ** 2)

    v_shared = U_svd[:, 0] # Primer componente principal común
    v_shared = v_shared / np.linalg.norm(v_shared)

    print(f"  • Valores Singulares de los 3 Ejes (SVD) : {[round(float(s), 4) for s in S_svd]}")
    print(f"  • Varianza Explicada por Modo 1 (v_shared): {var_explained[0]*100:.2f}%")
    print(f"  • Varianza Modo 2 (Desviación Dominio 1)   : {var_explained[1]*100:.2f}%")
    print(f"  • Varianza Modo 3 (Desviación Dominio 2)   : {var_explained[2]*100:.2f}%")

    print("\n  Proyección de cada eje de dominio sobre v_shared:")
    for name in keys:
        c_proj = float(np.dot(v_domains[name] / np.linalg.norm(v_domains[name]), v_shared))
        print(f"    • cos({name.split()[1]}, v_shared) = {c_proj:+.6f} (Ángulo = {math.degrees(math.acos(c_proj)):.2f}°)")

    # ── 3. EVALUACIÓN CRUZADA USANDO v_shared SOBRE LAS 3 FAMILIAS ───────────
    section("3. EVALUACIÓN DE RECUPERACIÓN Y DECODIFICACIÓN CON v_shared")

    # Amplitud representativa promedio
    mean_amp = float(np.mean([np.linalg.norm(v_domains[k]) for k in keys]))
    alphas_test = [0.00, 0.25, 0.50, 0.75, 1.00]

    def forward_steered(ids_tensor, delta_vec):
        aether_native_c.junction_reset()
        if delta_vec is None or np.linalg.norm(delta_vec) == 0.0:
            out = model.language_model(ids_tensor)
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
        out = model.language_model(ids_tensor)
        logits = out.logits[0, -1, :].astype(mx.float32)
        mx.eval(logits)
        for l in range(num_layers): lm_model.layers[l] = orig_layers[l]
        return to_numpy_f32(logits)

    # Calibrar decodificador conjunto w_inv_shared usando el promedio de respuestas a v_shared
    dz_joint_cal = []
    ids_famA = mx.array(tok.encode(families["Fam A (Puzle Lógico)"].format(k=2)))[None, :]

    for a in alphas_test:
        dh = v_shared * (a * mean_amp)
        z = forward_steered(ids_famA, dh)
        dz_joint_cal.append(z - z_bases["Fam A (Puzle Lógico)"])

    Z_joint = np.column_stack(dz_joint_cal)
    Uz_j, Sz_j, Vtz_j = np.linalg.svd(Z_joint, full_matrices=False)
    w_inv_shared = (Uz_j / (Sz_j + 1e-4)) @ Vtz_j @ np.array(alphas_test, dtype=np.float32)

    print(f"  {'Familia Evaluada':<26} │ {'R² con v_shared':<16} │ {'R² previo (Lab 26)':<18} │ {'Δ R² Ganancia'}")
    print("  " + "─" * 78)

    for name in keys:
        ids_f = mx.array(tok.encode(families[name].format(k=6)))[None, :]
        z_base_f = forward_steered(ids_f, None)
        recovered_a = []

        for a in alphas_test:
            dh = v_shared * (a * mean_amp)
            z_s = forward_steered(ids_f, dh)
            dz = z_s - z_base_f
            a_hat = float(np.dot(dz, w_inv_shared))
            recovered_a.append(a_hat)

        y_true = np.array(alphas_test)
        y_pred = np.array(recovered_a)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        ss_res = np.sum((y_true - y_pred)**2)
        r2_new = 1.0 - (ss_res / (ss_tot + 1e-12))

        r2_prev = 0.9984 if "Puzle" in name else (0.6471 if "Logística" in name else 0.8758)
        delta_gain = r2_new - r2_prev
        print(f"  {name:<26} │ {r2_new:<16.6f} │ {r2_prev:<18.6f} │ {delta_gain:+12.4f}")

    section("DICTAMEN FINAL LAB 27 CONCLUIDO")
    print("  ✓ Descomposición SVD de eje compartido y análisis de varianza completados.")

if __name__ == "__main__":
    run_lab27()
