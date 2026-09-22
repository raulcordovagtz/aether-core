#!/usr/bin/env python3
"""
tests/test_hilbert_memory_cell.py
═══════════════════════════════════════════════════════════════════════════════
SUITE DE CERTIFICACIÓN FORMAL H2.1-A & H2.1-B: HILBERT MEMORY CELL
Validación Rigurosa: Superposición N-aria | Match-and-Peel | Ángulo Geodésico
Degeneraciones Duales (+A y -A) | Paridad Cuantitativa CPU ↔ Metal GPU
SSOT: spec/C13_boolean_attention_algebra.yaml
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, math
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))
import aether_native_c

EPS = 1e-5
DIMS_TO_TEST = [1024, 2048, 5120]

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def test_hilbert_memory_cell():
    section("H2.1-A & H2.1-B: CERTIFICACIÓN EXPERIMENTAL HILBERT MEMORY CELL")
    np.random.seed(42)

    for D in DIMS_TO_TEST:
        print(f"\n▶ VALIDANDO EN DIMENSIÓN D = {D} (Bytes por vector = {D*4} B)...")
        aether_native_c.hilbert_memory_reset()

        # ─── 1. Ingesta Persistente y Verificación Dimensional D_C1 == D_C2 ──
        h_subprod = mx.array(np.random.randn(D).astype(np.float32))
        h_subprod = h_subprod / mx.sqrt(mx.sum(h_subprod * h_subprod))
        mx.eval(h_subprod)

        res_ing = aether_native_c.hilbert_memory_ingest(h_subprod, timestamp=100, energy=0.45)
        assert res_ing["dimension"] == D
        assert res_ing["slot_idx"] == 0
        print(f"  [✅ PASS] Ingesta Persistente en UMA: Slot {res_ing['slot_idx']}, Count={res_ing['count']}")

        # ─── 2. Superposición de 2 Hechos Ortogonales (Resonancia 1/√2) ────────
        # Hipótesis: A y B unitarios, A ⟂ B
        v_A = np.random.randn(D).astype(np.float32)
        v_A /= np.linalg.norm(v_A)
        v_B_raw = np.random.randn(D).astype(np.float32)
        v_B = v_B_raw - np.dot(v_B_raw, v_A) * v_A
        v_B /= np.linalg.norm(v_B)

        mA = mx.array(v_A)
        mB = mx.array(v_B)
        mx.eval(mA, mB)

        pack_cpu = aether_native_c.hilbert_memory_pack_two(mA, mB)
        pack_gpu = aether_native_c.hilbert_memory_pack_two_metal(mA, mB)
        mx.eval(pack_cpu["result"], pack_gpu["result"])

        assert not pack_cpu["degenerate"]
        assert not pack_gpu["degenerate"]
        assert pack_cpu["degeneracy_reason"] == pack_gpu["degeneracy_reason"] == 0

        # Resonancia teórica: <m_pack, mA> = 1/sqrt(2) ≈ 0.707106
        m_pack_cpu = pack_cpu["result"]
        m_pack_gpu = pack_gpu["result"]
        res_A = float(mx.sum(m_pack_cpu * mA))
        res_B = float(mx.sum(m_pack_cpu * mB))
        exp_res = 1.0 / math.sqrt(2.0)
        assert abs(res_A - exp_res) < EPS, f"Error en resonancia A: {abs(res_A - exp_res)}"
        assert abs(res_B - exp_res) < EPS, f"Error en resonancia B: {abs(res_B - exp_res)}"

        # Paridad CPU ↔ Metal GPU
        diff_pack = float(mx.max(mx.abs(m_pack_cpu - m_pack_gpu)))
        assert diff_pack < EPS, f"Fallo de paridad en pack_two: {diff_pack}"
        print(f"  [✅ PASS] Superposición (AND-like): <pack, A>={res_A:.4f}, <pack, B>={res_B:.4f} (Teórico: {exp_res:.4f})")
        print(f"  [✅ PASS] Paridad CPU ↔ Metal (pack_two): max_diff = {diff_pack:.2e} < 1e-5")

        # ─── 3. Match-and-Peel: Desempaquetamiento por Deflación (NOT-like) ───
        peel_cpu = aether_native_c.hilbert_memory_deflate(m_pack_cpu, mA)
        peel_gpu = aether_native_c.hilbert_memory_deflate_metal(m_pack_gpu, mA)
        mx.eval(peel_cpu["result"], peel_gpu["result"])

        assert not peel_cpu["degenerate"]
        assert not peel_gpu["degenerate"]
        assert peel_cpu["degeneracy_reason"] == peel_gpu["degeneracy_reason"] == 0

        m_peeled_cpu = peel_cpu["result"]
        m_peeled_gpu = peel_gpu["result"]

        # A. Aniquilación de mA: |<m_peeled, mA>| < EPS
        residual_A = abs(float(mx.sum(m_peeled_cpu * mA)))
        assert residual_A < EPS, f"Fallo en aniquilación: {residual_A}"

        # B. Recuperación de mB bajo hipótesis A ⟂ B: <m_peeled, mB> ≈ 1.0
        dot_recovery_B = float(mx.sum(m_peeled_cpu * mB))
        assert abs(dot_recovery_B - 1.0) < EPS, f"Fallo en recuperación de B: {dot_recovery_B}"

        diff_peel = float(mx.max(mx.abs(m_peeled_cpu - m_peeled_gpu)))
        assert diff_peel < EPS, f"Fallo de paridad en deflate: {diff_peel}"
        print(f"  [✅ PASS] Match-and-Peel (Aniquilación mA): residual = {residual_A:.2e} < 1e-5")
        print(f"  [✅ PASS] Recuperación Pura mB: <peeled, mB> = {dot_recovery_B:.6f} ≈ 1.000000")
        print(f"  [✅ PASS] Paridad CPU ↔ Metal (deflate): max_diff = {diff_peel:.2e} < 1e-5")

        # ─── 4. Degeneración en Empaquetado: Paralelo (+A) vs Antiparalelo (-A)
        # Paralelo (+A + A): NO debe ser degenerado, debe normalizar a A
        pack_par = aether_native_c.hilbert_memory_pack_two(mA, mA)
        mx.eval(pack_par["result"])
        assert not pack_par["degenerate"], "Paralelo A + A no debió ser degenerado"
        dot_par = float(mx.sum(pack_par["result"] * mA))
        assert abs(dot_par - 1.0) < EPS
        print(f"  [✅ PASS] Paralelo (+A + A): No degenerado, preserva identidad <pack, A> = {dot_par:.6f}")

        # Antiparalelo (A + (-A) = 0): DEBE ser degenerado con motivo ZERO_SUPERPOSITION (1)
        pack_antipar_cpu = aether_native_c.hilbert_memory_pack_two(mA, -mA)
        pack_antipar_gpu = aether_native_c.hilbert_memory_pack_two_metal(mA, -mA)
        assert pack_antipar_cpu["degenerate"] and pack_antipar_gpu["degenerate"]
        assert pack_antipar_cpu["degeneracy_reason"] == pack_antipar_gpu["degeneracy_reason"] == 1
        print(f"  [✅ PASS] Antiparalelo (+A + (-A) = 0): Degenerado certificado en CPU y Metal (Motivo: ZERO_SUPERPOSITION)")

        # ─── 5. Degeneración Colineal en Deflación: (+A y -A) ─────────────────
        # Caso +A (m_pack = +A, target = A) -> Colineal (Motivo: COLLINEAR_TARGET = 2)
        defl_pos_cpu = aether_native_c.hilbert_memory_deflate(mA, mA)
        defl_pos_gpu = aether_native_c.hilbert_memory_deflate_metal(mA, mA)
        assert defl_pos_cpu["degenerate"] and defl_pos_gpu["degenerate"]
        assert defl_pos_cpu["degeneracy_reason"] == defl_pos_gpu["degeneracy_reason"] == 2

        # Caso -A (m_pack = -A, target = A) -> Colineal (Motivo: COLLINEAR_TARGET = 2)
        defl_neg_cpu = aether_native_c.hilbert_memory_deflate(-mA, mA)
        defl_neg_gpu = aether_native_c.hilbert_memory_deflate_metal(-mA, mA)
        assert defl_neg_cpu["degenerate"] and defl_neg_gpu["degenerate"]
        assert defl_neg_cpu["degeneracy_reason"] == defl_neg_gpu["degeneracy_reason"] == 2
        print(f"  [✅ PASS] Deflación Colineal Dual (+A y -A): Ambos detectados como COLLINEAR_TARGET en CPU y Metal")

        # ─── 6. Transporte Paralelo de Estilo: Ángulo Geodésico y Paridad ────
        h_truth = mA
        u_style_raw = np.random.randn(D).astype(np.float32)
        u_style = mx.array(u_style_raw / np.linalg.norm(u_style_raw))
        mx.eval(u_style)

        theta_s = 0.25 # radianes
        style_cpu = aether_native_c.hilbert_memory_style_transport(h_truth, u_style, theta_s)
        style_gpu = aether_native_c.hilbert_memory_style_transport_metal(h_truth, u_style, theta_s)
        mx.eval(style_cpu["result"], style_gpu["result"])

        assert not style_cpu["degenerate"] and not style_gpu["degenerate"]
        h_styled_cpu = style_cpu["result"]
        h_styled_gpu = style_gpu["result"]

        # A. Preservación esférica
        norm_styled = float(mx.sqrt(mx.sum(h_styled_cpu * h_styled_cpu)))
        assert abs(norm_styled - 1.0) < EPS

        # B. Proyección sobre la verdad == cos(theta_s)
        dot_truth = float(mx.sum(h_styled_cpu * h_truth))
        exp_truth = math.cos(theta_s)
        assert abs(dot_truth - exp_truth) < EPS

        # C. Distancia angular geodésica exacta: arccos(<h_styled, h_truth>) == theta_s
        d_geodesic = math.acos(max(-1.0, min(1.0, dot_truth)))
        err_geo = abs(d_geodesic - theta_s)
        assert err_geo < EPS, f"Fallo en distancia angular geodésica: {err_geo}"

        # D. Paridad CPU ↔ Metal
        diff_style = float(mx.max(mx.abs(h_styled_cpu - h_styled_gpu)))
        assert diff_style < EPS, f"Fallo de paridad en style transport: {diff_style}"
        print(f"  [✅ PASS] Transporte de Estilo: <h_styled, h_truth> = {dot_truth:.4f} (cos(θ)={exp_truth:.4f})")
        print(f"  [✅ PASS] Distancia Geodésica: d_S(h_truth, h_styled) = {d_geodesic:.6f} rad (esperado {theta_s:.6f}, err={err_geo:.2e})")
        print(f"  [✅ PASS] Paridad CPU ↔ Metal (style): max_diff = {diff_style:.2e} < 1e-5")

        # ─── 7. Degeneración Colineal en Transporte de Estilo (+Truth y -Truth)
        style_col_cpu = aether_native_c.hilbert_memory_style_transport(h_truth, h_truth, theta_s)
        style_col_gpu = aether_native_c.hilbert_memory_style_transport_metal(h_truth, h_truth, theta_s)
        assert style_col_cpu["degenerate"] and style_col_gpu["degenerate"]
        assert style_col_cpu["degeneracy_reason"] == style_col_gpu["degeneracy_reason"] == 3 # ZERO_TANGENT
        diff_col = float(mx.max(mx.abs(style_col_cpu["result"] - h_truth)))
        assert diff_col < 1e-6
        print(f"  [✅ PASS] Estilo Colineal (u_style || h_truth): Degeneración ZERO_TANGENT detectada, preserva h_truth en CPU y Metal")

    section("RESULTADO: H2.1-A & H2.1-B 100% CERTIFICADOS EN SILICIO (CPU & METAL)")

if __name__ == "__main__":
    test_hilbert_memory_cell()
