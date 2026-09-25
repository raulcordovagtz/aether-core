#!/usr/bin/env python3
"""
tests/test_tetrapolar_predictor.py
═══════════════════════════════════════════════════════════════════════════════
SUITE DE CERTIFICACIÓN FORMAL: PREDICTOR GEODÉSICO TETRAPOLAR (FASE 1)
SSOT: spec/C22_tetrapolar_predictor_cell.yaml
Verificaciones:
  1. Confinamiento esférico absoluto ||h*(tau)|| = 1.000000 para todo tau
  2. Identidad matemática en reposo h*(0) == h_in
  3. Acotación física de líneas derivativas |grad_pole| <= 1.0
  4. Sensibilidad direccional: grad_teleo > 0 hacia la meta, grad_anti < 0
  5. Paridad numérica estricta entre CPU C++20 y GPU Metal
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, math
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import aether_native_c

EPS = 1e-6
DIMS = [1024, 2048, 5120]

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def test_tetrapolar_predictor():
    section("C-022: CERTIFICACIÓN UNITARIA PREDICTOR GEODÉSICO TETRAPOLAR")
    np.random.seed(42)

    for D in DIMS:
        print(f"\n▶ PROBANDO EN DIMENSIÓN D = {D}...")

        # Generar estado h y velocidad v
        h_raw = np.random.randn(D).astype(np.float32)
        h = h_raw / np.linalg.norm(h_raw)

        v_raw = np.random.randn(D).astype(np.float32)
        v = v_raw - np.dot(v_raw, h) * h  # Ortogonal a h
        v = v / np.linalg.norm(v) * 0.25   # Magnitud angular finita

        # Generar las 4 primitivas ortonormalizadas
        def make_unit(vec): return vec / np.linalg.norm(vec)

        v_hat = v / np.linalg.norm(v)
        u_onto  = make_unit(np.random.randn(D).astype(np.float32))
        
        noise_t = np.random.randn(D).astype(np.float32)
        noise_t = (noise_t / np.linalg.norm(noise_t)) * 0.05
        u_teleo = make_unit(v_hat + noise_t)  # Fuertemente alineado con v (cos > 0.95)

        noise_a = np.random.randn(D).astype(np.float32)
        noise_a = (noise_a / np.linalg.norm(noise_a)) * 0.05
        u_anti  = make_unit(-v_hat + noise_a) # Fuertemente opuesto a v (cos < -0.95)

        u_eos   = make_unit(np.random.randn(D).astype(np.float32))

        # Enviar arrays MLX a la extensión nativa
        h_mx       = mx.array(h)
        v_mx       = mx.array(v)
        u_onto_mx  = mx.array(u_onto)
        u_teleo_mx = mx.array(u_teleo)
        u_anti_mx  = mx.array(u_anti)
        u_eos_mx   = mx.array(u_eos)
        mx.eval(h_mx, v_mx, u_onto_mx, u_teleo_mx, u_anti_mx, u_eos_mx)

        # ── TEST 1: Identidad en Reposo (tau = 0.0) ──
        res_t0_cpu = aether_native_c.tetrapolar_predictor_step(
            h_mx, v_mx, u_onto_mx, u_teleo_mx, u_anti_mx, u_eos_mx, tau=0.0
        )
        res_t0_metal = aether_native_c.tetrapolar_predictor_step_metal(
            h_mx, v_mx, u_onto_mx, u_teleo_mx, u_anti_mx, u_eos_mx, tau=0.0
        )
        h_star_0_cpu = np.array(res_t0_cpu["h_star"])
        h_star_0_metal = np.array(res_t0_metal["h_star"])
        diff_t0_cpu = np.linalg.norm(h_star_0_cpu - h)
        diff_t0_metal = np.linalg.norm(h_star_0_metal - h)
        assert diff_t0_cpu < EPS, f"Error CPU en identidad tau=0: {diff_t0_cpu:.2e}"
        assert diff_t0_metal < EPS, f"Error Metal en identidad tau=0: {diff_t0_metal:.2e}"
        print(f"  [✅ PASS] Identidad en reposo (tau=0): CPU err={diff_t0_cpu:.2e}, Metal err={diff_t0_metal:.2e} < 1e-6")

        # ── TEST 2: Confinamiento Esférico Estricto (tau ∈ [0.1 .. 2.0]) ──
        taus = [0.1, 0.5, 1.0, 1.5, 2.0]
        max_norm_err_cpu = 0.0
        max_norm_err_metal = 0.0
        max_diff_cpu_metal = 0.0

        for tau in taus:
            res_tau_cpu = aether_native_c.tetrapolar_predictor_step(
                h_mx, v_mx, u_onto_mx, u_teleo_mx, u_anti_mx, u_eos_mx, tau=tau
            )
            res_tau_metal = aether_native_c.tetrapolar_predictor_step_metal(
                h_mx, v_mx, u_onto_mx, u_teleo_mx, u_anti_mx, u_eos_mx, tau=tau
            )
            
            arr_cpu = np.array(res_tau_cpu["h_star"])
            arr_metal = np.array(res_tau_metal["h_star"])
            
            norm_star_cpu = np.linalg.norm(arr_cpu)
            norm_star_metal = np.linalg.norm(arr_metal)
            
            err_norm_cpu = abs(norm_star_cpu - 1.0)
            err_norm_metal = abs(norm_star_metal - 1.0)
            diff_cm = np.max(np.abs(arr_cpu - arr_metal))

            max_norm_err_cpu = max(max_norm_err_cpu, err_norm_cpu)
            max_norm_err_metal = max(max_norm_err_metal, err_norm_metal)
            max_diff_cpu_metal = max(max_diff_cpu_metal, diff_cm)

            assert err_norm_cpu < EPS, f"Violación de norma CPU en tau={tau}: {norm_star_cpu}"
            assert err_norm_metal < EPS, f"Violación de norma Metal en tau={tau}: {norm_star_metal}"
            assert diff_cm < EPS, f"Discrepancia CPU vs Metal en tau={tau}: {diff_cm:.2e}"

        print(f"  [✅ PASS] Confinamiento esférico exacto para tau ∈ [0.1..2.0]:")
        print(f"      • CPU max |norm - 1|   = {max_norm_err_cpu:.2e} < 1e-6")
        print(f"      • Metal max |norm - 1| = {max_norm_err_metal:.2e} < 1e-6")
        print(f"      • Paridad CPU ↔ Metal  = {max_diff_cpu_metal:.2e} < 1e-6")

        # ── TEST 3: Sensibilidad de las 4 Líneas Derivativas ──
        tel = res_tau_cpu["telemetry"]
        tel_metal = res_tau_metal["telemetry"]
        g_onto  = tel["grad_onto"]
        g_teleo = tel["grad_teleo"]
        g_anti  = tel["grad_anti"]
        g_eos   = tel["grad_eos"]

        assert -1.0001 <= g_onto <= 1.0001
        assert -1.0001 <= g_teleo <= 1.0001
        assert -1.0001 <= g_anti <= 1.0001
        assert -1.0001 <= g_eos <= 1.0001

        # u_teleo estaba alineado con v -> grad_teleo debe ser claramente positivo
        # u_anti estaba opuesto a v -> grad_anti debe ser claramente negativo
        assert g_teleo > 0.50, f"grad_teleo debió ser fuertemente positivo: {g_teleo}"
        assert g_anti < -0.50, f"grad_anti debió ser fuertemente negativo: {g_anti}"

        # Paridad de telemetría CPU vs Metal
        assert abs(tel["grad_teleo"] - tel_metal["grad_teleo"]) < 1e-5
        assert abs(tel["grad_anti"] - tel_metal["grad_anti"]) < 1e-5
        assert abs(tel["omega_angular_velocity"] - tel_metal["omega_angular_velocity"]) < 1e-5

        print(f"  [✅ PASS] Líneas Derivativas del Tetrapolo:")
        print(f"      • grad_teleo (Hacia la meta)  : {g_teleo:+.4f} > +0.50")
        print(f"      • grad_anti  (Hacia el error) : {g_anti:+.4f} < -0.50")
        print(f"      • grad_onto  (Anclaje base)   : {g_onto:+.4f}")
        print(f"      • grad_eos   (Cierre final)   : {g_eos:+.4f}")
        print(f"      • teleology_alignment (<h*, u>): {tel['teleology_alignment']:+.4f}")
        print(f"      • Paridad de derivadas CPU ↔ Metal < 1e-5 ✓")

    section("RESULTADO: PREDICTOR GEODÉSICO TETRAPOLAR 100% CERTIFICADO")

if __name__ == "__main__":
    test_tetrapolar_predictor()
