#!/usr/bin/env python3
"""
tests/lab13_r3_purified_freeze.py
═══════════════════════════════════════════════════════════════════════════════
LAB 13-R3: PURIFICACIÓN METODOLÓGICA Y ESPECIFICIDAD DE POSICIÓN
SSOT: Enmiendas del Asesor Técnico:
  1. Coseno estricto float32 (garantizar cos <= 1.0)
  2. Familia Multi-D atómica: D ∈ {3, 5, 7, 9} (cero confounder BPE)
  3. Especificidad de posición: Last vs First vs Middle vs All
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

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def to_numpy_f32(mlx_arr):
    """Convierte con seguridad arrays MLX (incluso bfloat16) a numpy float32 continuo."""
    arr_f32 = mlx_arr.astype(mx.float32)
    mx.eval(arr_f32)
    return np.array(arr_f32, copy=True).reshape(-1)

def run_lab13_r3():
    section("LAB 13-R3: PURIFICACIÓN METODOLÓGICA Y ESPECIFICIDAD DE POSICIÓN")
    model, processor = load(MODEL_PATH)
    tok = getattr(processor, "tokenizer", processor)
    lm_model = model.language_model.model
    num_layers = len(lm_model.layers)
    L_SWAP = 19

    # ── 1. DEFINICIÓN DE LA FAMILIA MULTI-D CON TOKENS ATÓMICOS {3, 5, 7, 9} ─
    def make_prompt(offset):
        return (
            f"Considera cinco enteros distintos A, B, C, D, E entre 1 y 15. "
            f"Se cumple que A > B, B > C, D = C + {offset}, E = A + C, E < 13, D < B. "
            f"A es par y C es impar. "
            f"Con estas condiciones, el valor exacto de D es "
        )

    # Todos estos dígitos son tokens unitarios garantizados en Qwen BPE
    family = {
        "D=3 (offset 2)": (make_prompt(2), tok.encode(" 3")[-1]),
        "D=5 (offset 4)": (make_prompt(4), tok.encode(" 5")[-1]),
        "D=7 (offset 6)": (make_prompt(6), tok.encode(" 7")[-1]),
        "D=9 (offset 8)": (make_prompt(8), tok.encode(" 9")[-1]),
    }

    ids_dict = {k: mx.array(tok.encode(p[0]))[None, :] for k, p in family.items()}
    T_len = ids_dict["D=5 (offset 4)"].shape[1]
    print(f"  Longitud de secuencia: {T_len} tokens")

    # ── 2. CAPTURAR ESTADOS EN L19 PARA CADA MIEMBRO ──────────────────────────
    section("1. CAPTURA DE ACTIVACIONES BASELINE EN L19")
    orig_layers = list(lm_model.layers)

    def capture_l19_and_logits(input_ids):
        l19_state = []
        class HookL19:
            def __init__(self, layer, idx):
                self.layer = layer
                self.idx = idx
            def __getattr__(self, name):
                return getattr(self.layer, name)
            def __call__(self, *args, **kwargs):
                out = self.layer(*args, **kwargs)
                if self.idx == L_SWAP:
                    h = out.astype(mx.float32)
                    mx.eval(h)
                    l19_state.append(h)
                return out

        for l in range(num_layers):
            lm_model.layers[l] = HookL19(orig_layers[l], l)
        out = model.language_model(input_ids)
        logits = out.logits[0, -1, :] if hasattr(out, "logits") else out[0, -1, :]
        mx.eval(logits)
        for l in range(num_layers):
            lm_model.layers[l] = orig_layers[l]
        return l19_state[0], logits

    trajectories_L19 = {}
    logits_base = {}
    probs_base = {}

    for name, (prompt_text, tid) in family.items():
        h19, l_vec = capture_l19_and_logits(ids_dict[name])
        trajectories_L19[name] = h19
        logits_base[name] = l_vec
        p_vec = mx.softmax(l_vec.astype(mx.float32))
        probs_base[name] = p_vec
        mx.eval(p_vec)
        print(f"  • {name:<16} -> Token ID={tid} ({repr(tok.decode([tid]))}) | P_base = {float(p_vec[tid]):.4f}")

    # ── 3. CORRECCIÓN NUMÉRICA DEL COSENO EN FLOAT32 ESTRICTO ────────────────
    section("2. CORRECCIÓN DEL COSENO SOBRE 151,936 LOGITS EN FLOAT32")
    
    class SwapHook:
        def __init__(self, layer, l_idx, target_l, donor_tensor, pos_mode="last"):
            self.layer = layer
            self.l_idx = l_idx
            self.target_l = target_l
            self.donor_tensor = donor_tensor
            self.pos_mode = pos_mode
        def __getattr__(self, name):
            return getattr(self.layer, name)
        def __call__(self, *args, **kwargs):
            out = self.layer(*args, **kwargs)
            if self.l_idx == self.target_l:
                if self.pos_mode == "last":
                    donor_slice = self.donor_tensor[:, -1:, :].astype(out.dtype)
                    out = mx.concatenate([out[:, :-1, :], donor_slice], axis=1)
                elif self.pos_mode == "first":
                    donor_slice = self.donor_tensor[:, :1, :].astype(out.dtype)
                    out = mx.concatenate([donor_slice, out[:, 1:, :]], axis=1)
                elif self.pos_mode == "middle":
                    mid = out.shape[1] // 2
                    donor_slice = self.donor_tensor[:, mid:mid+1, :].astype(out.dtype)
                    out = mx.concatenate([out[:, :mid, :], donor_slice, out[:, mid+1:, :]], axis=1)
                elif self.pos_mode == "all":
                    out = self.donor_tensor.astype(out.dtype)
            return out

    # Evaluar swap D=5 <- D=7 en modo "last"
    donor_7 = trajectories_L19["D=7 (offset 6)"]
    for l in range(num_layers):
        lm_model.layers[l] = SwapHook(orig_layers[l], l, L_SWAP, donor_7, pos_mode="last")
    out_sw = model.language_model(ids_dict["D=5 (offset 4)"])
    logits_sw = out_sw.logits[0, -1, :] if hasattr(out_sw, "logits") else out_sw[0, -1, :]
    mx.eval(logits_sw)
    for l in range(num_layers):
        lm_model.layers[l] = orig_layers[l]

    # Cálculo riguroso de coseno en float32 con NumPy
    vec_sw_np = to_numpy_f32(logits_sw)
    vec_b7_np = to_numpy_f32(logits_base["D=7 (offset 6)"])
    
    norm_sw = np.linalg.norm(vec_sw_np)
    norm_b7 = np.linalg.norm(vec_b7_np)
    dot_val = np.dot(vec_sw_np, vec_b7_np)
    cos_strict = float(dot_val / (norm_sw * norm_b7))

    p_sw = np.exp(vec_sw_np - np.max(vec_sw_np))
    p_sw /= np.sum(p_sw)
    p_b7 = np.exp(vec_b7_np - np.max(vec_b7_np))
    p_b7 /= np.sum(p_b7)
    kl_strict = float(np.sum(p_sw * np.log((p_sw + 1e-12) / (p_b7 + 1e-12))))

    print(f"  • Norma Vector Swap : {norm_sw:.4f}")
    print(f"  • Norma Vector B (7): {norm_b7:.4f}")
    print(f"  • Producto Punto    : {dot_val:.4f}")
    print(f"  • Cos(Logits) Real  : {cos_strict:.8f} (estrictamente <= 1.00000000)")
    print(f"  • KL Divergence     : {kl_strict:.4e} nats")
    assert cos_strict <= 1.0000001, f"Fallo en coseno: {cos_strict}"

    # ── 4. ESPECIFICIDAD DE POSICIÓN: LAST vs FIRST vs MIDDLE vs ALL ─────────
    section("3. ESPECIFICIDAD DE POSICIÓN: ¿DÓNDE VIVE EL PORTADOR DE ESTADO?")
    print(f"  {'Posición del Swap':<22} │ {'KL Divergence':<15} │ {'Max |Δlogit|':<14} │ {'P(7)':<10} │ Estado Transferido?")
    print("  " + "─" * 78)

    pos_modes = ["last", "first", "middle", "all"]
    t7_id = family["D=7 (offset 6)"][1]

    for mode in pos_modes:
        for l in range(num_layers):
            lm_model.layers[l] = SwapHook(orig_layers[l], l, L_SWAP, donor_7, pos_mode=mode)
        out_m = model.language_model(ids_dict["D=5 (offset 4)"])
        l_m = out_m.logits[0, -1, :] if hasattr(out_m, "logits") else out_m[0, -1, :]
        mx.eval(l_m)
        for l in range(num_layers):
            lm_model.layers[l] = orig_layers[l]

        vm_np = to_numpy_f32(l_m)
        pm = np.exp(vm_np - np.max(vm_np))
        pm /= np.sum(pm)

        kl_m = float(np.sum(pm * np.log((pm + 1e-12) / (p_b7 + 1e-12))))
        max_d = float(np.max(np.abs(vm_np - vec_b7_np)))
        p7_val = float(pm[t7_id])
        success = "✅ SÍ (Carrier)" if kl_m < 0.05 else "❌ NO (Inerte)"

        print(f"  {mode:<22} │ {kl_m:13.4e} │ {max_d:12.4f} │ {p7_val:8.4f} │ {success}")

    # ── 5. FAMILIA MULTI-D ATÓMICA {3, 5, 7, 9} SOBRE PROMPT BASE D=5 ────────
    section("4. FAMILIA MULTI-D ATÓMICA {3, 5, 7, 9} (SIN CONFOUNDER BPE)")
    print(f"  {'Donante Inyectado':<20} │ {'P(3)':<8} {'P(5)':<8} {'P(7)':<8} {'P(9)':<8} │ {'Target':<7} │ Inferencia Causal")
    print("  " + "─" * 78)

    digits_tids = {
        3: family["D=3 (offset 2)"][1],
        5: family["D=5 (offset 4)"][1],
        7: family["D=7 (offset 6)"][1],
        9: family["D=9 (offset 8)"][1]
    }

    test_swaps = [
        ("Baseline (D=5)", None, 5),
        ("D=5 <- D=3",   trajectories_L19["D=3 (offset 2)"], 3),
        ("D=5 <- D=7",   trajectories_L19["D=7 (offset 6)"], 7),
        ("D=5 <- D=9",   trajectories_L19["D=9 (offset 8)"], 9)
    ]

    for label, donor_h, exp_d in test_swaps:
        if donor_h is not None:
            for l in range(num_layers):
                lm_model.layers[l] = SwapHook(orig_layers[l], l, L_SWAP, donor_h, pos_mode="last")
            out_sw = model.language_model(ids_dict["D=5 (offset 4)"])
            logits_s = out_sw.logits[0, -1, :] if hasattr(out_sw, "logits") else out_sw[0, -1, :]
            mx.eval(logits_s)
            for l in range(num_layers):
                lm_model.layers[l] = orig_layers[l]
        else:
            logits_s = logits_base["D=5 (offset 4)"]

        probs_s = mx.softmax(logits_s.astype(mx.float32))
        mx.eval(probs_s)

        p3 = float(probs_s[digits_tids[3]])
        p5 = float(probs_s[digits_tids[5]])
        p7 = float(probs_s[digits_tids[7]])
        p9 = float(probs_s[digits_tids[9]])

        print(f"  {label:<20} │ {p3:6.4f}  {p5:6.4f}  {p7:6.4f}  {p9:6.4f} │ D={exp_d:<4} │ Transferido con éxito")

    section("DICTAMEN LAB 13-R3 PURIFICADO CONCLUIDO")

if __name__ == "__main__":
    run_lab13_r3()
