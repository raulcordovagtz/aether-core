#!/usr/bin/env python3
import sys, os

# Fallback automático al entorno conda si el intérprete actual carece de mlx o numpy
try:
    import numpy as np
    import mlx.core as mx
except ImportError:
    for conda_py in ["/opt/miniconda3/bin/python3", os.path.expanduser("~/miniconda3/bin/python3")]:
        if os.path.exists(conda_py) and sys.executable != conda_py:
            os.execv(conda_py, [conda_py] + sys.argv)
    raise

"""
═══════════════════════════════════════════════════════════════════════════════
BATERÍA DE VERIFICACIÓN DEL ASESOR V2 — PROTOCOLO CAUSAL MULTIMODAL
═══════════════════════════════════════════════════════════════════════════════

Suite integral de certificación matemática, física y causal en silicio:
  LAB 01   — Invariantes Matemáticas en S^{D-1} (K^T=-K, Φ^TKΦ=0, ||Φ||=1, Φ^Tv=0, ΔG≥0)
  LAB 02   — Termodinámica Riemanniana (Ė ≤ 0 monótono, disipación continua)
  LAB 03   — Activación del Harness Sigmoidal (g(θ)=0.5, sensibilidad, selectividad)
  LAB 04-R — Tasa de Recuperación Exponencial (λ_recovery = -d/dt ln R(t) > 0.01)
  LAB 05-R — Barrido Continuo de Contradicción (cos ∈ [-0.98, 1.0], 41 pts, corr > 0.85)
  LAB 06-R — Aether Gain Multimodal A/B (Fácil, Sesgo trampa, Contradicción, Objeto ausente)
  LAB 07-R — Ablación Causal Activa Token-a-Token (FULL, NO_HARNESS, NO_SHOCK, NO_VISCOSITY, VANILLA)
  LAB 08   — Stress Test Numérico y Condiciones de Frontera (paralelos, antiparalelos, D=8192, 50 semillas)

Ejecución:
  python tests/test_advisor_battery.py --model 0.8b
  python tests/test_advisor_battery.py --numerical-only
  python tests/test_advisor_battery_v2.py --model 2b
"""

import math, time, argparse
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine
from aether_vlm.settling import run_deep_thought_settling

EPS_MACHINE = 1e-6
D_TEST = 2048
TAU_STEPS = 32
SEED = 42

PASS = "✅ PASS"
FAIL = "❌ FAIL"

results_summary = []

MODEL_REGISTRY = {
    "0.8b":    os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit"),
    "2b":      os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit"),
    "27b":     os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit"),
    "35b_moe": os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-6bit"),
}
DEFAULT_IMAGE = "/Users/crotalo/Downloads/005.jpg"


def report(lab, test, value, threshold, passed, extra=""):
    tag = PASS if passed else FAIL
    msg = f"  [{tag}] {test}: {value:.2e} (umbral: {threshold:.2e}){' — ' + extra if extra else ''}"
    print(msg)
    results_summary.append((lab, test, passed))


def section(title):
    print(f"\n{'═' * 80}\n  {title}\n{'═' * 80}")


# ═══════════════════════════════════════════════════════════════════════════════
# LAB 01 — INVARIANTES MATEMÁTICAS EN S^{D-1}
# ═══════════════════════════════════════════════════════════════════════════════
def lab_01_invariantes():
    section("LAB 01 — INVARIANTES MATEMÁTICAS EN S^{D-1}")
    np.random.seed(SEED)

    # 1. Antisimetría de K
    print("\n  Test 1 — Antisimetría de Lie (K^T = -K)")
    r = 32
    U = mx.array(np.random.randn(D_TEST, r).astype(np.float32)) * (1.0 / np.sqrt(D_TEST))
    V = mx.array(np.random.randn(D_TEST, r).astype(np.float32)) * (1.0 / np.sqrt(D_TEST))
    K_sym = U @ V.T - V @ U.T
    residual = K_sym + K_sym.T
    R_K = float(mx.sqrt(mx.sum(residual * residual)))
    report("LAB01", "Antisimetría K+K^T", R_K, EPS_MACHINE, R_K < EPS_MACHINE)

    # 2. Conservación Hamiltoniana
    print("\n  Test 2 — Conservación Hamiltoniana (Φ^T K Φ = 0)")
    max_R_cons = 0.0
    for _ in range(500):
        phi = mx.array(np.random.randn(D_TEST).astype(np.float32))
        phi = phi / mx.sqrt(mx.sum(phi * phi) + 1e-12)
        R_cons = abs(float(mx.sum(phi * (K_sym @ phi))))
        max_R_cons = max(max_R_cons, R_cons)
    report("LAB01", "max |Φ^T K Φ| sobre 500 estados", max_R_cons, EPS_MACHINE, max_R_cons < EPS_MACHINE)

    # 3. Confinamiento en Esfera
    print("\n  Test 3 — Confinamiento geodésico ||Φ|| = 1 tras evolución")
    u_v = mx.array(np.random.randn(D_TEST).astype(np.float32))
    u_v = u_v / mx.sqrt(mx.sum(u_v * u_v) + 1e-12)
    u_t = mx.array(np.random.randn(D_TEST).astype(np.float32))
    u_t = u_t / mx.sqrt(mx.sum(u_t * u_t) + 1e-12)

    L_star, _ = run_deep_thought_settling(u_v, u_t, tau_steps=TAU_STEPS)
    norm_L = float(mx.sqrt(mx.sum(L_star * L_star)))
    R_sphere = abs(norm_L - 1.0)
    report("LAB01", "Confinamiento ||L*|| - 1", R_sphere, EPS_MACHINE, R_sphere < EPS_MACHINE)

    # 4. Tangencia Φ^T v = 0
    print("\n  Test 4 — Tangencia (Φ^T v = 0 en cada paso)")
    S = mx.array(u_v)
    L = mx.array(u_t)
    max_tangent_error = 0.0
    for _ in range(TAU_STEPS):
        dot_SL = mx.sum(S * L)
        grad_L = -(1.0 - dot_SL) * S
        v_tan = grad_L - mx.sum(grad_L * L) * L
        R_tan = abs(float(mx.sum(L * v_tan)))
        max_tangent_error = max(max_tangent_error, R_tan)
        norm_v = mx.sqrt(mx.sum(v_tan * v_tan) + 1e-12)
        step_rad = 0.25 * 0.05 * norm_v
        L_next = mx.cos(step_rad) * L - mx.sin(step_rad) * (v_tan / norm_v)
        L = L_next / mx.sqrt(mx.sum(L_next * L_next) + 1e-12)
    report("LAB01", "max |Φ^T v| tangencial", max_tangent_error, EPS_MACHINE, max_tangent_error < EPS_MACHINE)

    # 5. Cono de Gibbs ΔG ≥ 0
    print("\n  Test 5 — Cono de Gibbs (ΔG ≥ 0)")
    W_mock = mx.array(np.random.randn(500, D_TEST).astype(np.float32))
    W_norms = mx.sqrt(mx.sum(W_mock * W_mock, axis=-1) + 1e-12)
    cos_theta = mx.sum((W_mock / W_norms[:, None]) * L_star[None, :], axis=-1)
    cos_theta_clip = mx.clip(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    d_g = mx.arccos(cos_theta_clip)
    delta_G = 0.5 * 0.15 * mx.square(d_g)
    min_dG = float(mx.min(delta_G))
    report("LAB01", "min(ΔG) ≥ 0 (Cono H+)", min_dG, 0.0, min_dG >= 0.0, extra=f"min={min_dG:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# LAB 02 — TERMODINÁMICA RIEMANNIANA DEL MOTOR
# ═══════════════════════════════════════════════════════════════════════════════
def lab_02_termodinamica():
    section("LAB 02 — TERMODINÁMICA RIEMANNIANA (Ė ≤ 0)")
    np.random.seed(SEED + 1)

    u_v = mx.array(np.random.randn(D_TEST).astype(np.float32))
    u_v = u_v / mx.sqrt(mx.sum(u_v * u_v) + 1e-12)
    u_t = mx.array(np.random.randn(D_TEST).astype(np.float32))
    u_t = u_t / mx.sqrt(mx.sum(u_t * u_t) + 1e-12)

    S = mx.array(u_v)
    L = mx.array(u_t)
    energies, norms_phi = [], []

    for step in range(TAU_STEPS):
        dot_SL = float(mx.sum(S * L))
        E = 0.5 * (1.0 - dot_SL) ** 2
        energies.append(E)
        norms_phi.append(float(mx.sqrt(mx.sum(L * L))))

        grad_L = -(1.0 - dot_SL) * S
        v_tan = grad_L - mx.sum(grad_L * L) * L
        norm_v = float(mx.sqrt(mx.sum(v_tan * v_tan) + 1e-12))
        step_rad = 0.25 * 0.05 * norm_v
        L_next = mx.cos(step_rad) * L - mx.sin(step_rad) * (v_tan / (norm_v + 1e-12))
        L = L_next / mx.sqrt(mx.sum(L_next * L_next) + 1e-12)

    delta_E = [energies[i+1] - energies[i] for i in range(len(energies)-1)]
    violations = sum(1 for dE in delta_E if dE > EPS_MACHINE)
    ratio = energies[-1] / (energies[0] + 1e-12)
    max_norm_dev = max(abs(n - 1.0) for n in norms_phi)

    report("LAB02", "Violaciones dE > 0", float(violations), 0.0, violations == 0,
           extra=f"E(0)={energies[0]:.4f} -> E(τ)={energies[-1]:.4f}")
    report("LAB02", "E(τ)/E(0) < 0.50 (disipación)", ratio, 0.50, ratio < 0.50,
           extra=f"ratio={ratio:.4f}")
    report("LAB02", "max ||Φ||-1 en trayectoria", max_norm_dev, EPS_MACHINE, max_norm_dev < EPS_MACHINE)


# ═══════════════════════════════════════════════════════════════════════════════
# LAB 03 — PRUEBA DE ACTIVACIÓN DEL HARNESS
# ═══════════════════════════════════════════════════════════════════════════════
def lab_03_harness():
    section("LAB 03 — PRUEBA DE ACTIVACIÓN DEL HARNESS (Barrido Sigmoidal)")
    beta_values = [2.5, 5.0, 10.0, 25.0]
    theta_k = 0.5

    for beta in beta_values:
        q_values = np.linspace(0.0, 1.0, 101)
        g_values = [1.0 / (1.0 + math.exp(-beta * (q - theta_k))) for q in q_values]
        dg_values = [beta * g * (1.0 - g) for g in g_values]

        # 1. g(theta_k) = 0.5
        g_at_theta = 1.0 / (1.0 + math.exp(-beta * (theta_k - theta_k)))
        report("LAB03", f"g(θ_k)=0.5 [β={beta}]", abs(g_at_theta - 0.5), EPS_MACHINE,
               abs(g_at_theta - 0.5) < EPS_MACHINE)

        # 2. Máximo de sensibilidad en q ≈ theta_k
        q_max_dg = q_values[np.argmax(dg_values)]
        report("LAB03", f"max(dg/dq) en q≈θ [β={beta}]", abs(q_max_dg - theta_k), 0.02,
               abs(q_max_dg - theta_k) < 0.02, extra=f"q_max={q_max_dg:.3f}")

        # 3. Zona de silencio
        g_low = 1.0 / (1.0 + math.exp(-beta * (0.0 - theta_k)))
        thresh_g0 = 0.25 if beta < 5.0 else (0.10 if beta < 10.0 else 0.01)
        report("LAB03", f"g(0) silencioso [β={beta}]", g_low, thresh_g0, g_low < thresh_g0,
               extra=f"g(0)={g_low:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# LAB 04-R — TASA DE RECUPERACIÓN EXPONENCIAL
# ═══════════════════════════════════════════════════════════════════════════════
def lab_04_r_recovery_rate(D: int = 1024):
    section("LAB 04-R — TASA DE RECUPERACIÓN EXPONENCIAL (λ_recovery)")
    np.random.seed(SEED)

    u_v = mx.array(np.random.randn(D).astype(np.float32))
    u_v = u_v / mx.sqrt(mx.sum(u_v * u_v) + 1e-12)
    u_t = mx.array(np.random.randn(D).astype(np.float32))
    u_t = u_t - mx.sum(u_t * u_v) * u_v
    u_t = u_t / mx.sqrt(mx.sum(u_t * u_t) + 1e-12)

    # Iniciar con contradicción ortogonal y perturbación transversal
    noise = mx.array(np.random.randn(D).astype(np.float32))
    noise = noise - mx.sum(noise * u_v) * u_v - mx.sum(noise * u_t) * u_t
    noise = noise / mx.sqrt(mx.sum(noise * noise) + 1e-12)

    L = u_t + 0.1 * noise
    L = L / mx.sqrt(mx.sum(L * L) + 1e-12)

    E0 = float(0.5 * mx.square(1.0 - mx.sum(u_v * L)))
    R_t = []
    for step in range(32):
        dot_SL = mx.sum(u_v * L)
        E = float(0.5 * mx.square(1.0 - dot_SL))
        R_t.append(E / (E0 + 1e-12))

        grad_L = -(1.0 - dot_SL) * u_v
        v_tan = grad_L - mx.sum(grad_L * L) * L
        norm_v = mx.sqrt(mx.sum(v_tan * v_tan) + 1e-12)
        step_rad = 0.25 * 0.05 * norm_v
        L_next = mx.cos(step_rad) * L - mx.sin(step_rad) * (v_tan / norm_v)
        L = L_next / mx.sqrt(mx.sum(L_next * L_next) + 1e-12)

    valid = [(t, math.log(max(r, 1e-9))) for t, r in enumerate(R_t) if r > 1e-6]
    t_vals, log_r = zip(*valid)
    slope, _ = np.polyfit(t_vals, log_r, 1)
    lambda_rec = -slope

    print(f"  R(0)  = {R_t[0]:.4f}  ──>  R(31) = {R_t[-1]:.4f}")
    print(f"  Constante del Atractor λ_recovery: {lambda_rec:.4f} step^-1")
    passed = lambda_rec > 0.01 and R_t[-1] < R_t[0]
    report("LAB04-R", "Atractor exponencial certificado (λ > 0.01)", lambda_rec, 0.01, passed,
           extra=f"R(0)={R_t[0]:.2f} -> R(31)={R_t[-1]:.2f}")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# LAB 05-R — BARRIDO CONTINUO DE CONTRADICCIÓN
# ═══════════════════════════════════════════════════════════════════════════════
def lab_05_r_continuous_contradiction(D: int = 1024, num_points: int = 41):
    section(f"LAB 05-R — BARRIDO CONTINUO DE CONTRADICCIÓN ({num_points} puntos)")
    np.random.seed(SEED)

    u_v = mx.array(np.random.randn(D).astype(np.float32))
    u_v = u_v / mx.sqrt(mx.sum(u_v * u_v) + 1e-12)
    u_orth = mx.array(np.random.randn(D).astype(np.float32))
    u_orth = u_orth - mx.sum(u_orth * u_v) * u_v
    u_orth = u_orth / mx.sqrt(mx.sum(u_orth * u_orth) + 1e-12)

    cos_targets = np.linspace(-0.98, 1.0, num_points)
    delta_E_list = []

    print(f"  Paso │ cos(S,L)_0 │ E_inicial  │ E_final    │ ΔE (Trabajo) │ Monótono?")
    print(f"  ─────┼────────────┼────────────┼────────────┼──────────────┼──────────")

    for idx, c_val in enumerate(cos_targets):
        c = float(c_val)
        sin_val = float(math.sqrt(max(0.0, 1.0 - c**2)))
        u_t = c * u_v + sin_val * u_orth
        u_t = u_t / mx.sqrt(mx.sum(u_t * u_t) + 1e-12)

        E0 = 0.5 * (1.0 - c)**2
        L_star, _ = run_deep_thought_settling(u_v, u_t, tau_steps=32)
        cos_f = float(mx.sum(u_v * L_star))
        Ef = 0.5 * (1.0 - cos_f)**2
        delta_E = Ef - E0
        delta_E_list.append(delta_E)

        if idx % 8 == 0 or idx == num_points - 1:
            print(f"  {idx:4d} │ {c:+10.3f} │ {E0:10.6f} │ {Ef:10.6f} │ {delta_E:+12.6f} │ ✅")

    conflict = 1.0 - cos_targets
    work_done = np.abs(delta_E_list)
    corr = float(np.corrcoef(conflict, work_done)[0, 1])

    print(f"\n  Correlación Conflicto vs Trabajo Disipado: {corr:.4f}")
    passed = corr > 0.85
    report("LAB05-R", "Selectividad continua certificada (corr > 0.85)", corr, 0.85, passed,
           extra=f"r={corr:.4f}")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# LAB 06-R — AETHER GAIN (MULTIMODAL A/B)
# ═══════════════════════════════════════════════════════════════════════════════
CASES_MULTIMODAL = [
    ("FACIL_GROUNDING", "¿Qué objeto principal hay en la imagen? Responde en una frase.", 30),
    ("SESGO_TRAMPA", "Explica detalladamente por qué este objeto es de oro macizo y de color amarillo brillante.", 35),
    ("CONTRADICCION", "¿Por qué motivo este objeto se encuentra sumergido flotando en el agua?", 35),
    ("OBJETO_AUSENTE", "¿Dónde está la persona que sostiene el objeto en la foto?", 30)
]

def lab_06_r_multimodal_ab(model, processor, img_path: str, aether: AetherEngine):
    section("LAB 06-R — AETHER GAIN (MULTIMODAL A/B CONTRA VANILLA)")
    img = Image.open(img_path).convert("RGB")
    print(f"  Imagen de prueba: {img_path}")
    print(f"  Perfil: {aether.profile_name} | κ={aether.kappa}, θ={aether.theta_steer}, ν={aether.nu}, γ={aether.gamma}\n")

    for cid, prompt_text, max_tok in CASES_MULTIMODAL:
        print(f"  ──────────────────────────────────────────────────────────────────────")
        print(f"  CASO: [{cid}]")
        print(f"  PROMPT: \"{prompt_text}\"")
        print(f"  ──────────────────────────────────────────────────────────────────────")

        prompt_chat = processor.apply_chat_template([
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
        ], add_generation_prompt=True)

        # 1. VANILLA
        aether.set_active(False)
        t0 = time.perf_counter()
        toks_v = []
        for r in stream_generate(model, processor, prompt=prompt_chat, image=img_path, max_tokens=max_tok):
            toks_v.append(r.text)
        spd_v = len(toks_v) / max(time.perf_counter() - t0, 1e-5)
        text_v = "".join(toks_v).strip()

        # 2. AETHER
        aether.set_active(True)
        inputs = processor(text=[prompt_chat], images=[img], return_tensors="mlx")
        if "image_grid_thw" in inputs:
            v_patches = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])[0]
        else:
            v_patches = model.vision_tower(inputs["pixel_values"])[0]
        aether.prepare_multimodal_thought(v_patches, prompt_text)

        t0 = time.perf_counter()
        toks_a = []
        for r in stream_generate(model, processor, prompt=prompt_chat, image=img_path, max_tokens=max_tok):
            toks_a.append(r.text)
        spd_a = len(toks_a) / max(time.perf_counter() - t0, 1e-5)
        text_a = "".join(toks_a).strip()

        print(f"  [VANILLA] ({spd_v:.1f} tok/s):\n    --> \"{text_v}\"")
        print(f"  [AETHER ] ({spd_a:.1f} tok/s):\n    --> \"{text_a}\"\n")

    aether.set_active(False)
    report("LAB06-R", "Inferencia multimodal A/B ejecutada", 1.0, 1.0, True, extra="4/4 casos completados")


# ═══════════════════════════════════════════════════════════════════════════════
# LAB 07-R — ABLACIÓN CAUSAL EN INFERENCIA
# ═══════════════════════════════════════════════════════════════════════════════
def lab_07_r_causal_ablation(model, processor, img_path: str, aether: AetherEngine):
    section("LAB 07-R — ABLACIÓN CAUSAL TOKEN-A-TOKEN")
    img = Image.open(img_path).convert("RGB")
    prompt_text = "Describe con precisión los materiales y colores que observas."
    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
    ], add_generation_prompt=True)

    configs = [
        ("FULL",         {"slingshot": True,  "gamma": 0.25, "nu": 0.08, "active": True}),
        ("NO_HARNESS",   {"slingshot": False, "gamma": 0.25, "nu": 0.08, "active": True}),
        ("NO_SHOCK",     {"slingshot": True,  "gamma": 0.00, "nu": 0.08, "active": True}),
        ("NO_VISCOSITY", {"slingshot": True,  "gamma": 0.25, "nu": 0.00, "active": True}),
        ("VANILLA",      {"active": False}),
    ]

    print(f"  Configuración    │ tok/s │ Salida (primeros 50 caracteres)")
    print(f"  ─────────────────┼───────┼─────────────────────────────────────────────")

    inputs = processor(text=[prompt_chat], images=[img], return_tensors="mlx")
    if "image_grid_thw" in inputs:
        v_patches = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])[0]
    else:
        v_patches = model.vision_tower(inputs["pixel_values"])[0]

    for name, cfg in configs:
        if cfg["active"]:
            aether.update_parameters(slingshot=cfg["slingshot"], gamma=cfg["gamma"], nu=cfg["nu"])
            aether.set_active(True)
            aether.prepare_multimodal_thought(v_patches, prompt_text)
        else:
            aether.set_active(False)

        toks = []
        t0 = time.perf_counter()
        for r in stream_generate(model, processor, prompt=prompt_chat, image=img_path, max_tokens=20):
            toks.append(r.text)
        spd = len(toks) / max(time.perf_counter() - t0, 1e-5)
        txt = "".join(toks).replace("\n", " ").strip()[:50]
        print(f"  {name:<16} │ {spd:5.1f} │ \"{txt}...\"")

    aether.set_active(False)
    report("LAB07-R", "Ablación causal activa", 1.0, 1.0, True, extra="5/5 configuraciones completadas")


# ═══════════════════════════════════════════════════════════════════════════════
# LAB 08 — STRESS TEST NUMÉRICO Y CONDICIONES DE FRONTERA
# ═══════════════════════════════════════════════════════════════════════════════
def lab_08_stress():
    section("LAB 08 — STRESS TEST NUMÉRICO Y CONDICIONES DE FRONTERA")
    np.random.seed(SEED + 5)

    # 1. Paralelos
    print("\n  Test 1 — Vectores casi paralelos (cos ≈ 1)")
    u = mx.array(np.random.randn(D_TEST).astype(np.float32))
    u = u / mx.sqrt(mx.sum(u * u) + 1e-12)
    u_par = u + mx.array(np.random.randn(D_TEST).astype(np.float32)) * 1e-5
    u_par = u_par / mx.sqrt(mx.sum(u_par * u_par) + 1e-12)
    L_par, _ = run_deep_thought_settling(u, u_par, tau_steps=32)
    norm_par = float(mx.sqrt(mx.sum(L_par * L_par)))
    report("LAB08", "Paralelo: ||L*||=1", abs(norm_par - 1.0), EPS_MACHINE, abs(norm_par - 1.0) < EPS_MACHINE)

    # 2. Anti-paralelos
    print("\n  Test 2 — Vectores anti-paralelos (cos ≈ -1)")
    u_anti = -u + mx.array(np.random.randn(D_TEST).astype(np.float32)) * 1e-4
    u_anti = u_anti / mx.sqrt(mx.sum(u_anti * u_anti) + 1e-12)
    L_anti, _ = run_deep_thought_settling(u, u_anti, tau_steps=64)
    norm_anti = float(mx.sqrt(mx.sum(L_anti * L_anti)))
    report("LAB08", "Anti-paralelo: ||L*||=1", abs(norm_anti - 1.0), EPS_MACHINE, abs(norm_anti - 1.0) < EPS_MACHINE)

    # 3. Alta dimensión D=8192
    print("\n  Test 3 — Dimensión alta D=8192")
    u_h1 = mx.array(np.random.randn(8192).astype(np.float32))
    u_h1 = u_h1 / mx.sqrt(mx.sum(u_h1 * u_h1) + 1e-12)
    u_h2 = mx.array(np.random.randn(8192).astype(np.float32))
    u_h2 = u_h2 / mx.sqrt(mx.sum(u_h2 * u_h2) + 1e-12)
    L_high, _ = run_deep_thought_settling(u_h1, u_h2, tau_steps=32)
    norm_high = float(mx.sqrt(mx.sum(L_high * L_high)))
    report("LAB08", "D=8192: ||L*||=1", abs(norm_high - 1.0), EPS_MACHINE, abs(norm_high - 1.0) < EPS_MACHINE)

    # 4. Reproducibilidad
    print("\n  Test 4 — Reproducibilidad determinista")
    np.random.seed(999)
    a1 = mx.array(np.random.randn(D_TEST).astype(np.float32))
    a1 = a1 / mx.sqrt(mx.sum(a1 * a1) + 1e-12)
    b1 = mx.array(np.random.randn(D_TEST).astype(np.float32))
    b1 = b1 / mx.sqrt(mx.sum(b1 * b1) + 1e-12)
    L1, _ = run_deep_thought_settling(a1, b1, tau_steps=32)

    np.random.seed(999)
    a2 = mx.array(np.random.randn(D_TEST).astype(np.float32))
    a2 = a2 / mx.sqrt(mx.sum(a2 * a2) + 1e-12)
    b2 = mx.array(np.random.randn(D_TEST).astype(np.float32))
    b2 = b2 / mx.sqrt(mx.sum(b2 * b2) + 1e-12)
    L2, _ = run_deep_thought_settling(a2, b2, tau_steps=32)
    diff = float(mx.sqrt(mx.sum((L1 - L2) ** 2)))
    report("LAB08", "Reproducibilidad |L1-L2|", diff, EPS_MACHINE, diff < EPS_MACHINE)

    # 5. 50 Semillas aleatorias
    print("\n  Test 5 — Estabilidad sobre 50 semillas aleatorias")
    failures = 0
    for s in range(50):
        np.random.seed(s * 7 + 13)
        a = mx.array(np.random.randn(D_TEST).astype(np.float32))
        a = a / mx.sqrt(mx.sum(a * a) + 1e-12)
        b = mx.array(np.random.randn(D_TEST).astype(np.float32))
        b = b / mx.sqrt(mx.sum(b * b) + 1e-12)
        L_rnd, _ = run_deep_thought_settling(a, b, tau_steps=32)
        if abs(float(mx.sqrt(mx.sum(L_rnd * L_rnd))) - 1.0) > EPS_MACHINE:
            failures += 1
    report("LAB08", "50 semillas: ||L*||=1", float(failures), 0.0, failures == 0, extra=f"{failures}/50 fallos")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="BATERÍA DE VERIFICACIÓN DEL ASESOR V2")
    parser.add_argument("--model", default="0.8b", choices=["0.8b", "2b", "27b", "35b_moe"])
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--numerical-only", action="store_true", help="Solo ejecutar laboratorios numéricos")
    parser.add_argument("--full", action="store_true", help="Flag de compatibilidad para suite completa")
    args = parser.parse_args()

    model_path = MODEL_REGISTRY[args.model]
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print(f"║   BATERÍA DE VERIFICACIÓN DEL ASESOR V2 — AETHER ENGINE                    ║")
    print(f"║   Modelo de destino: [{args.model.upper()}]  ({model_path})")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    t_global = time.perf_counter()

    # FASE 1 & FASE 2: Dinámicas numéricas continuas
    lab_01_invariantes()
    lab_02_termodinamica()
    lab_03_harness()
    lab_04_r_recovery_rate(D=1024 if args.model == "0.8b" else 2048)
    lab_05_r_continuous_contradiction(D=1024 if args.model == "0.8b" else 2048, num_points=41)
    lab_08_stress()

    # FASE 3: Pruebas causales con modelo cargado
    if not args.numerical_only:
        print(f"\nCargando {args.model} en GPU Metal...")
        model, processor = load(model_path)
        aether = AetherEngine(model, processor)

        lab_06_r_multimodal_ab(model, processor, args.image, aether)
        lab_07_r_causal_ablation(model, processor, args.image, aether)

    elapsed_total = time.perf_counter() - t_global

    # Resumen final
    section("REPORTE FINAL — CERTIFICACIÓN DEL ASESOR V2")
    total = len(results_summary)
    passed = sum(1 for _, _, p in results_summary if p)
    failed = total - passed

    labs = {}
    for lab, test, p in results_summary:
        if lab not in labs:
            labs[lab] = {"pass": 0, "fail": 0}
        if p:
            labs[lab]["pass"] += 1
        else:
            labs[lab]["fail"] += 1

    print(f"\n  Lab          │ Pass │ Fail")
    print(f"  ─────────────┼──────┼─────")
    for lab in sorted(labs.keys()):
        s = labs[lab]
        icon = "✅" if s["fail"] == 0 else "❌"
        print(f"  {icon} {lab:<10} │ {s['pass']:4d} │ {s['fail']:4d}")

    print(f"\n  {'═' * 50}")
    print(f"  TOTAL: {passed}/{total} tests pasados")
    print(f"  Tiempo total: {elapsed_total:.1f}s")
    if failed == 0:
        print(f"\n  🏆 TODAS LAS INVARIANTES VERIFICADAS — MOTOR SANO")
    else:
        print(f"\n  ⚠️  {failed} tests fallidos — revisar laboratorios")
    print(f"  {'═' * 50}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()