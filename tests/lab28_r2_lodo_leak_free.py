#!/usr/bin/env python3
"""
tests/lab28_r2_lodo_leak_free.py
═══════════════════════════════════════════════════════════════════════════════
LAB 28-R2 — LODO ESTRICTAMENTE LIBRE DE FUGAS (EJE Y DECODIFICADOR DISJUNTOS)
SSOT: Auditoría del Asesor Técnico:
  1. Eje v_LODO y decodificador w_inv entrenados SOLO en el par donante
  2. Dominio objetivo 100% HELD-OUT (cero fitting in-sample de w_dec)
  3. Rejilla de evaluación con alfas intermedias no vistas: {0.15, 0.35, 0.65, 0.85}
  4. Control nulo con vector aleatorio de norma estrictamente igualada
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

def run_lab28_r2():
    section("LAB 28-R2: LODO LIBRE DE FUGAS (EJE Y DECODIFICADOR DISJUNTOS)")
    print(f"  Modelo: Qwen3.5-0.8B | Capa de Intervención: L* = {L_SWAP}")
    print(f"  CERO fitting in-sample en el dominio objetivo. Alfas Held-Out: [0.15, 0.35, 0.65, 0.85]")

    model, processor = load(MODEL_PATH)
    tok = getattr(processor, "tokenizer", processor)
    lm_model = model.language_model.model
    num_layers = len(lm_model.layers)
    orig_layers = list(lm_model.layers)

    families = {
        "A": {
            "name": "Fam A (Puzle Lógico)",
            "tmpl": "Considera cinco enteros distintos A, B, C, D, E entre 1 y 15. Se cumple que A > B, B > C, D = C + {k}, E = A + C, E < 13, D < B. A es par y C es impar. Con estas condiciones, el valor exacto de D es "
        },
        "B": {
            "name": "Fam B (Logística Almacén)",
            "tmpl": "En un almacen central de logistica hay tres tipos de contenedores: cajas rojas, azules y verdes. Se sabe que la cantidad de cajas verdes es exactamente igual a la cantidad de cajas azules mas {k} unidades. Si las cajas azules son 1, el total de cajas verdes es "
        },
        "C": {
            "name": "Fam C (Distancias Tren)",
            "tmpl": "Un tren de alta velocidad viaja entre tres estaciones Alfa, Beta y Gamma. La distancia de Alfa a Gamma supera a la distancia de Alfa a Beta en exactamente {k} kilometros. Sabiendo que la distancia a Beta es 1 kilometro, la distancia exacta a Gamma es "
        }
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

    # 1. Extracción de ejes naturales por dominio
    v_natural = {}
    amp_eval = 0.50

    for dom_key in ["A", "B", "C"]:
        t = families[dom_key]["tmpl"]
        h2, _ = extract_h19_and_logits(t.format(k=2))
        h8, _ = extract_h19_and_logits(t.format(k=8))
        v_nat = h8 - h2
        v_natural[dom_key] = v_nat / np.linalg.norm(v_nat)

    def forward_steered_on_dom(dom_key, delta_vec):
        aether_native_c.junction_reset()
        ids = mx.array(tok.encode(families[dom_key]["tmpl"].format(k=6)))[None, :]
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

    # Rejillas: Calibración gruesa vs Evaluación fina held-out
    alphas_train = [0.00, 0.50, 1.00] # Rejilla gruesa
    alphas_test  = [0.15, 0.35, 0.65, 0.85] # Valores held-out jamás vistos

    # ── 2. FUNCIÓN DE CALIBRACIÓN PURA SOBRE PARES DONANTES ──────────────────
    def calibrate_lodo_pair(donor_keys):
        """Calibra el eje v_LODO y el decodificador w_inv usando SOLO donor_keys."""
        # 1. Eje v_LODO por SVD
        V_pair = np.column_stack([v_natural[donor_keys[0]], v_natural[donor_keys[1]]])
        U_p, S_p, _ = np.linalg.svd(V_pair, full_matrices=False)
        v_lodo = U_p[:, 0]
        v_lodo = v_lodo / np.linalg.norm(v_lodo)
        if np.dot(v_lodo, v_natural[donor_keys[0]]) < 0:
            v_lodo = -v_lodo

        # 2. Calibrar decodificador w_inv usando únicamente respuestas de los dos dominios donantes
        dz_train_all = []
        y_train_all = []

        for d_key in donor_keys:
            z_base = forward_steered_on_dom(d_key, None)
            for a in alphas_train:
                dh = v_lodo * (a * amp_eval)
                zs = forward_steered_on_dom(d_key, dh)
                dz_train_all.append(zs - z_base)
                y_train_all.append(a)

        Z_train = np.column_stack(dz_train_all) # [151936, 6]
        Uz, Sz, Vtz = np.linalg.svd(Z_train, full_matrices=False)
        w_inv = (Uz / (Sz + 1e-4)) @ Vtz @ np.array(y_train_all, dtype=np.float32)

        return v_lodo, w_inv

    # ── 3. EVALUACIÓN CRUZADA SIN FUGAS SOBRE CADA DOMINIO HELD-OUT ───────────
    section("EVALUACIÓN CRUZADA LODO LIBRE DE FUGAS")
    print(f"  {'Dominio Held-Out':<24} │ {'Donantes':<12} │ {'R² Held-Out':<14} │ {'MAE':<10} │ {'Max Err':<10} │ Control Random R²")
    print("  " + "─" * 88)

    lodo_experiments = [
        ("C (Distancias Tren)",   "C", ["A", "B"]),
        ("B (Logística Almacén)", "B", ["A", "C"]),
        ("A (Puzle Lógico)",      "A", ["B", "C"]),
    ]

    for label, target_key, donor_pair in lodo_experiments:
        # Calibrar eje y decodificador SIN el target_key
        v_lodo, w_inv_lodo = calibrate_lodo_pair(donor_pair)

        # Evaluar en el dominio target_key sobre ALFA HELD-OUT
        z_base_target = forward_steered_on_dom(target_key, None)
        recovered_test = []

        for a_test in alphas_test:
            dh_test = v_lodo * (a_test * amp_eval)
            zs_test = forward_steered_on_dom(target_key, dh_test)
            dz_test = zs_test - z_base_target
            # Decodificar con el w_inv que NUNCA vio target_key
            a_hat = float(np.dot(dz_test, w_inv_lodo))
            recovered_test.append(a_hat)

        # Métricas out-of-sample
        y_true = np.array(alphas_test)
        y_pred = np.array(recovered_test)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        ss_res = np.sum((y_true - y_pred)**2)
        r2_lodo = 1.0 - (ss_res / (ss_tot + 1e-12))
        mae_lodo = float(np.mean(np.abs(y_true - y_pred)))
        max_err = float(np.max(np.abs(y_true - y_pred)))

        # Control Nulo: Vector aleatorio de norma igualada
        np.random.seed(42)
        rnd_vec = np.random.randn(1024).astype(np.float32)
        rnd_vec = rnd_vec / np.linalg.norm(rnd_vec)
        recovered_rnd = []
        for a_test in alphas_test:
            dh_rnd = rnd_vec * (a_test * amp_eval)
            zs_rnd = forward_steered_on_dom(target_key, dh_rnd)
            dz_rnd = zs_rnd - z_base_target
            a_rnd_hat = float(np.dot(dz_rnd, w_inv_lodo))
            recovered_rnd.append(a_rnd_hat)

        ss_res_rnd = np.sum((y_true - np.array(recovered_rnd))**2)
        r2_rnd = 1.0 - (ss_res_rnd / (ss_tot + 1e-12))

        donors_str = f"{donor_pair[0]} + {donor_pair[1]}"
        print(f"  {label:<24} │ {donors_str:<12} │ {r2_lodo:<14.6f} │ {mae_lodo:<10.4f} │ {max_err:<10.4f} │ {r2_rnd:.4f}")

    section("DICTAMEN FINAL LAB 28-R2 CONCLUIDO")

if __name__ == "__main__":
    run_lab28_r2()
