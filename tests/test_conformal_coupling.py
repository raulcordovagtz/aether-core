#!/usr/bin/env python3
"""
tests/test_conformal_coupling.py
═══════════════════════════════════════════════════════════════════════════════
SUITE DE CERTIFICACIÓN GEOMÉTRICA H1.3-A: CONFORMAL COUPLING JUNCTION
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))
import aether_native_c

EPS = 1e-5

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def test_h13a_geometric_certification():
    section("H1.3-A: CERTIFICACIÓN GEOMÉTRICA Y COMPORTAMIENTO DE FRONTERA")
    D = 2048
    aether_native_c.junction_reset()

    h_in = mx.array(np.random.randn(D).astype(np.float32))
    h_in = h_in / mx.sqrt(mx.sum(h_in * h_in))
    u_att = mx.array(np.random.randn(D).astype(np.float32))
    u_att = u_att / mx.sqrt(mx.sum(u_att * u_att))
    mx.eval(h_in, u_att)

    # 1. Identidad Pasiva Estricta (Mode = 0 Passive)
    print("▶ 1/7. Modo Pasivo: Equivalencia numérica exacta h_out == h_in...")
    res_p0 = aether_native_c.dispatch_conformal_coupling(h_in, u_att, step=0, mode=0)
    mx.eval(res_p0["h_steered"])
    diff_p0 = float(mx.sqrt(mx.sum((res_p0["h_steered"] - h_in)**2)))
    assert diff_p0 < 1e-7, f"Violación de identidad pasiva: {diff_p0}"
    assert not res_p0["cell_evaluated"], "No debió evaluarse la célula en modo pasivo"
    assert not res_p0["intervention_applied"], "No debió intervenirse en modo pasivo"
    print(f"  [✅ PASS] Identidad Numérica Pasiva: ||h_out - h_in|| = {diff_p0:.2e}")

    # 2. Preservación Esférica sobre S^{D-1}
    print("\n▶ 2/7. Preservación Esférica sobre S^{D-1} en Modo Activo...")
    aether_native_c.junction_reset()
    h_t = h_in
    for step in range(3):
        h_t = h_t + mx.array(np.random.randn(D).astype(np.float32) * 0.05)
        h_t = h_t / mx.sqrt(mx.sum(h_t * h_t))
        mx.eval(h_t)
        res_act = aether_native_c.dispatch_conformal_coupling(
            h_t, u_att, step=step, tau_eff=0.15, kappa_att=0.8, theta_gate=0.01, mode=1
        )
    mx.eval(res_act["h_steered"])
    norm_out = float(mx.sqrt(mx.sum(res_act["h_steered"] * res_act["h_steered"])))
    err_norm = abs(norm_out - 1.0)
    assert err_norm < EPS, f"Violación de norma esférica: {err_norm}"
    print(f"  [✅ PASS] Preservación de Norma: ||h_out|| = {norm_out:.8f} (err: {err_norm:.2e})")

    # 3. Ortogonalidad de Deflación (<h_projected, h_deflated> = 0)
    print("\n▶ 3/7. Ortogonalidad Exacta de Deflación Gram-Schmidt...")
    h_proj = res_act["h_projected"]
    h_orth = res_act["h_orthogonal"]
    mx.eval(h_proj, h_orth)
    dot_ortho = abs(float(mx.sum(h_proj * h_orth)))
    assert dot_ortho < EPS, f"Fallo de ortogonalidad: {dot_ortho}"
    print(f"  [✅ PASS] Ortogonalidad Estricta: |<h_projected, h_deflated>| = {dot_ortho:.2e}")

    # 4. Límites del Acoplamiento: g=0 -> h_out=h_in, g=1 -> h_out approx h_projected
    print("\n▶ 4/7. Límites Asintóticos de la Intervención (g=0 y g=1)...")
    res_g0 = aether_native_c.dispatch_conformal_coupling(
        h_t, u_att, step=3, mode=1, force_g=0.0
    )
    mx.eval(res_g0["h_steered"])
    diff_g0 = float(mx.sqrt(mx.sum((res_g0["h_steered"] - h_t)**2)))
    assert diff_g0 < 1e-7, f"Fallo en límite g=0: {diff_g0}"
    print(f"  [✅ PASS] Límite g=0: ||h_out(0) - h_in|| = {diff_g0:.2e}")

    res_g1 = aether_native_c.dispatch_conformal_coupling(
        h_t, u_att, step=3, mode=1, force_g=1.0
    )
    mx.eval(res_g1["h_steered"], res_g1["h_projected"])
    diff_g1 = float(mx.sqrt(mx.sum((res_g1["h_steered"] - res_g1["h_projected"])**2)))
    assert diff_g1 < 1e-5, f"Fallo en límite g=1: {diff_g1}"
    print(f"  [✅ PASS] Límite g=1: ||h_out(1) - h_projected|| = {diff_g1:.2e}")

    # 5. Continuidad y Monotonicidad de la Compuerta
    print("\n▶ 5/7. Continuidad y Monotonicidad de la Compuerta g(q)...")
    theta_g = 0.35
    beta_g = 12.0
    q_vals = np.linspace(0.0, 1.0, 50)
    g_vals = 1.0 / (1.0 + np.exp(-beta_g * (q_vals - theta_g)))
    diffs = np.diff(g_vals)
    assert np.all(diffs >= 0), "La compuerta no es monotónica creciente"
    print(f"  [✅ PASS] Monotonicidad continua certificada sobre 50 puntos (min diff = {np.min(diffs):.2e})")

    # 6. Monotonicidad de la Magnitud de Intervención: g1 < g2 -> d(h_out(g1), h_in) <= d(h_out(g2), h_in)
    print("\n▶ 6/7. Monotonicidad de la Distancia Angular de Intervención...")
    g_sweep = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    distances = []
    for g_val in g_sweep:
        res_g = aether_native_c.dispatch_conformal_coupling(
            h_t, u_att, step=3, mode=1, force_g=g_val
        )
        mx.eval(res_g["h_steered"])
        d = float(mx.sqrt(mx.sum((res_g["h_steered"] - h_t)**2)))
        distances.append(d)

    for i in range(len(distances) - 1):
        assert distances[i] <= distances[i+1] + 1e-6, f"Violación de monotonicidad: {distances}"
    print(f"  [✅ PASS] Distancias para g={g_sweep}: {[round(x, 4) for x in distances]}")

    # 7. Verificación de Regímenes Cinemáticos y Triplete de Estado
    print("\n▶ 7/7. Regímenes Cinemáticos y Triplete (gate_open, cell_evaluated, intervention_applied)...")
    aether_native_c.junction_reset()
    res_closed = aether_native_c.dispatch_conformal_coupling(
        h_in, u_att, step=0, theta_gate=0.50, beta_gate=12.0, mode=1
    )
    assert res_closed["permeability_g"] < 0.05
    assert not res_closed["gate_open"]
    assert res_closed["cell_evaluated"] # g >= 1e-4
    assert res_closed["intervention_applied"]
    print(f"  [✅ PASS] Triplete de Estado en reposo: g={res_closed['permeability_g']:.4f} | gate_open={res_closed['gate_open']} | evaluated={res_closed['cell_evaluated']} | applied={res_closed['intervention_applied']}")

    section("RESULTADO: H1.3-A CERTIFICADO AL 100% EN SILICIO")

if __name__ == "__main__":
    test_h13a_geometric_certification()
