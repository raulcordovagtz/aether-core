#!/usr/bin/env python3
"""
tests/test_fact_band_router.py
═══════════════════════════════════════════════════════════════════════════════
SUITE DE VERIFICACIÓN UNITARIA: FACT BAND ROUTER & CANDIDATE PEAK (HITO 2.2-R1)
Validación de:
  1. Cero fuga estricta en compuerta rectificada (r < theta => g == 0.00e+00)
  2. Activación selectiva y margen explícito delta (r >= theta + delta => g > 0.50)
  3. Competencia multi-slot con registro de r_second y margen (r_max - r_second)
  4. Detector de cresta cinemática kappa(l) con clamp numérico y manejo de ||v|| ≈ 0
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, math
import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))
import aether_native_c

EPS = 1e-5

def test_fact_band_router_unit():
    print("═" * 78)
    print("  VERIFICACIÓN UNITARIA: FACT BAND ROUTER & COMPUERTA RECTIFICADA")
    print("═" * 78)

    D = 2048
    np.random.seed(42)
    aether_native_c.hilbert_memory_reset()

    # 1. Cargar 3 slots de memoria ortogonales
    m0 = mx.array(np.random.randn(D).astype(np.float32))
    m0 = m0 / mx.sqrt(mx.sum(m0 * m0))
    m1 = mx.array(np.random.randn(D).astype(np.float32))
    m1 = m1 - mx.sum(m1 * m0) * m0
    m1 = m1 / mx.sqrt(mx.sum(m1 * m1))
    m2 = mx.array(np.random.randn(D).astype(np.float32))
    m2 = m2 - mx.sum(m2 * m0) * m0 - mx.sum(m2 * m1) * m1
    m2 = m2 / mx.sqrt(mx.sum(m2 * m2))
    mx.eval(m0, m1, m2)

    aether_native_c.hilbert_memory_ingest(m0, timestamp=0, energy=0.5)
    aether_native_c.hilbert_memory_ingest(m1, timestamp=1, energy=0.5)
    aether_native_c.hilbert_memory_ingest(m2, timestamp=2, energy=0.5)

    assert aether_native_c.hilbert_memory_slot_count() == 3

    # ─── TEST 1: CERO FUGA ESTRICTA (r < theta => g == 0.00e+00) ─────────────
    # Vector de consulta ortogonal a todos los slots (r ≈ 0 < theta=0.45)
    q_silent = mx.array(np.random.randn(D).astype(np.float32))
    q_silent = q_silent - mx.sum(q_silent * m0) * m0 - mx.sum(q_silent * m1) * m1 - mx.sum(q_silent * m2) * m2
    q_silent = q_silent / mx.sqrt(mx.sum(q_silent * q_silent))
    mx.eval(q_silent)

    dec_silent = aether_native_c.fact_band_route_layer(q_silent, threshold=0.45, beta=16.0)
    assert dec_silent["max_resonance_r"] < 0.45
    assert dec_silent["rectified_gate_g"] == 0.0, f"Fuga detectada: g={dec_silent['rectified_gate_g']}"
    assert not dec_silent["is_active"]
    print(f"  [✅ PASS] Cero Fuga Verificada: r_max={dec_silent['max_resonance_r']:.4f} < θ=0.45 => g = {dec_silent['rectified_gate_g']:.2e} (Silencio absoluto)")

    # ─── TEST 2: ACTIVACIÓN SELECTIVA Y MARGEN DELTA (r >= theta + delta) ────
    # Ajuste 1 del Analista: exigir r_max >= theta + delta (delta=0.05) para certificar g > 0.50
    theta_test = 0.45
    delta_test = 0.05
    noise = mx.array(np.random.randn(D).astype(np.float32))
    noise = noise / mx.sqrt(mx.sum(noise * noise)) * 0.20
    q_target = m1 + noise
    q_target = q_target / mx.sqrt(mx.sum(q_target * q_target))
    mx.eval(q_target)

    dec_target = aether_native_c.fact_band_route_layer(q_target, threshold=theta_test, beta=16.0)
    assert dec_target["selected_slot"] == 1, f"Slot incorrecto: {dec_target['selected_slot']}"
    assert dec_target["max_resonance_r"] >= theta_test + delta_test, \
        f"Resonancia insuficiente: {dec_target['max_resonance_r']} < {theta_test + delta_test}"
    assert dec_target["rectified_gate_g"] > 0.50, \
        f"Compuerta debió superar 0.50: {dec_target['rectified_gate_g']}"
    assert dec_target["is_active"]
    print(f"  [✅ PASS] Activación Selectiva: Slot {dec_target['selected_slot']} con r={dec_target['max_resonance_r']:.4f} >= θ+δ ({theta_test+delta_test:.2f}) => g = {dec_target['rectified_gate_g']:.4f} > 0.50")

    # ─── TEST 3: COMPETICIÓN MULTI-SLOT Y MARGEN (r_max - r_second) ──────────
    # Ajuste 5 del Analista: registrar y verificar r_max, r_second y margin
    # Crear consulta con afinidad controlada: 70% m1, 30% m2
    q_comp = 0.70 * m1 + 0.30 * m2
    q_comp = q_comp / mx.sqrt(mx.sum(q_comp * q_comp))
    mx.eval(q_comp)

    dec_comp = aether_native_c.fact_band_route_layer(q_comp, threshold=0.30, beta=16.0)
    assert dec_comp["selected_slot"] == 1
    r_max = dec_comp["max_resonance_r"]
    r_sec = dec_comp["second_resonance_r"]
    margin = dec_comp["resonance_margin"]
    expected_margin = r_max - r_sec
    assert abs(margin - expected_margin) < EPS, f"Error en calculo de margen: {margin} vs {expected_margin}"
    assert margin > 0.10, f"Margen de competición muy estrecho: {margin}"
    print(f"  [✅ PASS] Competición Multi-Slot: r_max={r_max:.4f} (Slot {dec_comp['selected_slot']}), r_second={r_sec:.4f}, Margin Δr={margin:.4f} > 0.10")

    # ─── TEST 4: DETECTOR DE CRESTA CINEMÁTICA κ(l) ───────────────────────────
    # Generar secuencia realista de 24 capas con velocidad no nula (||v|| ≈ 0.10)
    # y aceleración / giro pronunciado en capas 15..18
    h0 = mx.array(np.random.randn(D).astype(np.float32))
    h0 = h0 / mx.sqrt(mx.sum(h0 * h0))

    v_base = mx.array(np.random.randn(D).astype(np.float32))
    v_base = (v_base - mx.sum(v_base * h0) * h0)
    v_base = v_base / mx.sqrt(mx.sum(v_base * v_base)) * 0.10

    v_fact = mx.array(np.random.randn(D).astype(np.float32))
    v_fact = (v_fact - mx.sum(v_fact * h0) * h0 - mx.sum(v_fact * v_base) * v_base)
    v_fact = v_fact / mx.sqrt(mx.sum(v_fact * v_fact)) * 0.10

    layers = [h0]
    curr = h0
    v_curr = v_base
    for l in range(1, 24):
        if 15 <= l <= 18:
            v_curr = 0.3 * v_curr + 0.7 * v_fact
        elif l > 18:
            v_curr = v_fact
        curr = curr + v_curr
        curr = curr / mx.sqrt(mx.sum(curr * curr))
        layers.append(curr)

    peak_info = aether_native_c.fact_band_detect_peak(layers)
    p_l = peak_info["peak_layer"]
    print(f"  [✅ PASS] Detección de Cresta κ(l): Capa {p_l}/24 (Profundidad relativa = {peak_info['relative_depth']:.2f})")
    assert 14 <= p_l <= 19, f"Cresta fuera de ventana: {p_l}"

    # ─── TEST 5: RESILIENCIA CINEMÁTICA: VELOCIDAD CERO Y CLAMP DE RADICANDO ─
    # Ajuste 3 del Analista: probar secuencia con capas idénticas (v=0)
    layers_flat = []
    base_state = mx.array(np.random.randn(D).astype(np.float32))
    base_state = base_state / mx.sqrt(mx.sum(base_state * base_state))
    for l in range(10):
        # Capas 3, 4, 5 idénticas a capa 2
        if 2 <= l <= 5:
            layers_flat.append(base_state)
        else:
            base_state = base_state + mx.array(np.random.randn(D).astype(np.float32) * 0.02)
            base_state = base_state / mx.sqrt(mx.sum(base_state * base_state))
            layers_flat.append(base_state)

    flat_info = aether_native_c.fact_band_detect_peak(layers_flat)
    kappas_flat = np.array(flat_info["kappas"])
    # Las capas con v ≈ 0 no deben generar NaN ni crash
    assert not np.isnan(kappas_flat).any(), "NaN detectado en kappas con velocidad cero"
    assert kappas_flat[3] == 0.0 and kappas_flat[4] == 0.0, "Capa con v=0 debió dar kappa=0"
    print(f"  [✅ PASS] Resiliencia ante Velocidad Cero (||v|| ≈ 0): Cero NaN, kappa[3]={kappas_flat[3]:.1f}, kappa[4]={kappas_flat[4]:.1f}")

    print("\n✓ SUITE UNITARIA HITO 2.2-R1 CERTIFICADA AL 100% EN SILICIO")

if __name__ == "__main__":
    test_fact_band_router_unit()
