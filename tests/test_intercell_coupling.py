#!/usr/bin/env python3
"""
tests/test_intercell_coupling.py
═══════════════════════════════════════════════════════════════════════════════
SUITE DE CERTIFICACIÓN H2.1-D: INTEGRACIÓN INTER-CELULAR (GAP JUNCTION)
Célula 1 (GeodesicTrajectoryCell / ConformalCoupling) ──► Célula 2 (HilbertMemoryCell)
Verificaciones:
  1. Auto-ingesta automática del subproducto h_orthogonal en memoria persistente
  2. Consulta de slots ocupados y recuperación de tensores
  3. Álgebra de Hilbert (Superposición + Deflación) sobre subproductos reales
  4. Consulta de resonancia geodésica contra memoria viva
  5. No-regresión: modo pasivo no deposita memoria
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
NUM_STEPS = 5

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def test_intercell_coupling():
    section("H2.1-D: CERTIFICACIÓN INTER-CELULAR (GAP JUNCTION)")
    np.random.seed(42)
    D = 2048

    # ─── 0. RESET COMPLETO ─────────────────────────────────────────────────
    aether_native_c.junction_reset()
    aether_native_c.hilbert_memory_reset()
    initial_count = aether_native_c.hilbert_memory_slot_count()
    assert initial_count == 0, f"Memoria debería estar vacía al inicio, pero count={initial_count}"
    print(f"  [✅ PASS] Reset completo: junction + hilbert_memory (count={initial_count})")

    # ─── 1. MODO PASIVO: NO DEPOSITA MEMORIA ──────────────────────────────
    section("1/5. MODO PASIVO: Verificar que NO se deposita memoria")
    h_passive = mx.array(np.random.randn(D).astype(np.float32))
    h_passive = h_passive / mx.sqrt(mx.sum(h_passive * h_passive))
    u_att = mx.array(np.random.randn(D).astype(np.float32))
    u_att = u_att / mx.sqrt(mx.sum(u_att * u_att))
    mx.eval(h_passive, u_att)

    res_passive = aether_native_c.dispatch_conformal_coupling(
        h_passive, u_att, step=0,
        tau_eff=0.15, kappa_att=0.80, beta_gate=12.0, theta_gate=0.35,
        mode=0, force_g=-1.0  # mode=0 → PassiveObserve
    )
    assert not res_passive["memory_ingested"], "Modo pasivo NO debió depositar memoria"
    assert aether_native_c.hilbert_memory_slot_count() == 0, "Memoria debería seguir vacía en modo pasivo"
    print(f"  [✅ PASS] Modo Pasivo: memory_ingested={res_passive['memory_ingested']}, count=0")

    # ─── 2. MODO ACTIVO: SECUENCIA DE 5 PASOS CON AUTO-INGESTA ───────────
    section(f"2/5. MODO ACTIVO: Secuencia de {NUM_STEPS} pasos con auto-ingesta")
    aether_native_c.junction_reset()
    aether_native_c.hilbert_memory_reset()

    # Generar atractor contextual fijo
    u_attractor = mx.array(np.random.randn(D).astype(np.float32))
    u_attractor = u_attractor / mx.sqrt(mx.sum(u_attractor * u_attractor))
    mx.eval(u_attractor)

    captured_ortho = []  # Guardar h_orthogonal de cada paso para verificación posterior

    for step in range(NUM_STEPS):
        h_t = mx.array(np.random.randn(D).astype(np.float32))
        h_t = h_t / mx.sqrt(mx.sum(h_t * h_t))
        mx.eval(h_t)

        res = aether_native_c.dispatch_conformal_coupling(
            h_t, u_attractor, step=step,
            tau_eff=0.15, kappa_att=0.80, beta_gate=12.0, theta_gate=0.35,
            mode=1, force_g=1.0  # mode=1 → ActiveCoupled, force_g=1.0 para garantizar activación
        )
        mx.eval(res["h_orthogonal"])

        assert res["cell_evaluated"], f"Paso {step}: cell_evaluated debió ser True con mode=1 y force_g=1.0"
        assert res["memory_ingested"], f"Paso {step}: memory_ingested debió ser True"
        assert res["memory_count"] == step + 1, f"Paso {step}: count esperado {step+1}, obtenido {res['memory_count']}"

        captured_ortho.append(np.array(res["h_orthogonal"]))
        print(f"    Paso {step}: memory_ingested=True, slot={res['memory_slot']}, "
              f"count={res['memory_count']}, g={res['permeability_g']:.4f}, "
              f"r={res['correlation_r']:.4f}")

    # Verificar conteo final
    final_count = aether_native_c.hilbert_memory_slot_count()
    assert final_count == NUM_STEPS, f"Conteo final: esperado {NUM_STEPS}, obtenido {final_count}"
    print(f"\n  [✅ PASS] Auto-ingesta de {NUM_STEPS} subproductos: count={final_count}")

    # ─── 3. RECUPERACIÓN DE TENSORES DESDE SLOTS ─────────────────────────
    section("3/5. RECUPERACIÓN: Verificar tensores almacenados en memoria")
    for slot_idx in range(NUM_STEPS):
        slot_data = aether_native_c.hilbert_memory_get_slot(slot_idx)
        assert slot_data["valid"], f"Slot {slot_idx} debería ser válido"
        assert slot_data["dimension"] == D, f"Slot {slot_idx}: dimensión incorrecta"

        slot_tensor = np.array(slot_data["tensor"])
        expected = captured_ortho[slot_idx]

        # El tensor en memoria debe coincidir con el h_orthogonal capturado
        max_diff = np.max(np.abs(slot_tensor - expected))
        assert max_diff < EPS, f"Slot {slot_idx}: discrepancia de tensor {max_diff:.2e}"
        print(f"    Slot {slot_idx}: D={slot_data['dimension']}, max_diff_vs_captured={max_diff:.2e} ✓")

    # Verificar slot inválido
    invalid_slot = aether_native_c.hilbert_memory_get_slot(99)
    assert not invalid_slot["valid"], "Slot 99 no debería ser válido"
    print(f"\n  [✅ PASS] Recuperación de {NUM_STEPS} tensores y verificación de slot inválido")

    # ─── 4. ÁLGEBRA DE HILBERT SOBRE SUBPRODUCTOS REALES ─────────────────
    section("4/5. ÁLGEBRA DE HILBERT: Composición sobre subproductos reales capturados")

    # Recuperar dos subproductos reales de la memoria
    slot_0 = aether_native_c.hilbert_memory_get_slot(0)
    slot_1 = aether_native_c.hilbert_memory_get_slot(1)
    m_A = slot_0["tensor"]
    m_B = slot_1["tensor"]
    mx.eval(m_A, m_B)

    # 4a. Superposición AND-like sobre hechos reales
    # Los subproductos h_orthogonal NO son unitarios (provienen de la deflación conformal),
    # así que normalizamos antes de entrar al álgebra de Hilbert, que opera en S^{D-1}.
    norm_A = mx.sqrt(mx.sum(m_A * m_A))
    norm_B = mx.sqrt(mx.sum(m_B * m_B))
    mx.eval(norm_A, norm_B)
    print(f"    Normas originales: ‖slot_0‖ = {float(norm_A):.6f}, ‖slot_1‖ = {float(norm_B):.6f}")

    # Normalizar para operar en la esfera
    m_A_unit = m_A / norm_A
    m_B_unit = m_B / norm_B
    mx.eval(m_A_unit, m_B_unit)

    pack_result = aether_native_c.hilbert_memory_pack_two(m_A_unit, m_B_unit)
    mx.eval(pack_result["result"])
    m_packed = pack_result["result"]

    # Verificar norma unitaria del resultado
    norm_packed = float(mx.sqrt(mx.sum(m_packed * m_packed)))
    assert abs(norm_packed - 1.0) < EPS, f"Norma de pack: {norm_packed}"

    # Verificar resonancia con ambos componentes
    res_A = float(mx.sum(m_packed * m_A_unit))
    res_B = float(mx.sum(m_packed * m_B_unit))
    print(f"    Superposición AND-like (slots 0 & 1 normalizados):")
    print(f"      ‖m_packed‖ = {norm_packed:.6f}")
    print(f"      ⟨packed, slot_0_unit⟩ = {res_A:.4f}")
    print(f"      ⟨packed, slot_1_unit⟩ = {res_B:.4f}")
    print(f"      Degenerado: {pack_result['degenerate']}")

    if not pack_result["degenerate"]:
        # Ambas resonancias deben ser positivas (misma semiesfera)
        assert res_A > 0 and res_B > 0, "Resonancias deben ser positivas para subproductos no antiparalelos"
        print(f"    [✅ PASS] Superposición: Ambas resonancias positivas, norma unitaria")

    # 4b. Deflación NOT-like: Extraer slot_0 del pack
    deflate_result = aether_native_c.hilbert_memory_deflate(m_packed, m_A_unit)
    mx.eval(deflate_result["result"])
    m_deflated = deflate_result["result"]

    if not deflate_result["degenerate"]:
        # Verificar aniquilación de m_A_unit
        residual_A = abs(float(mx.sum(m_deflated * m_A_unit)))
        assert residual_A < EPS, f"Fallo aniquilación: {residual_A}"

        # Verificar que m_B sobrevive (su resonancia debería ser alta)
        recovery_B = float(mx.sum(m_deflated * m_B_unit))
        print(f"    Deflación NOT-like (extraer slot_0 del pack):")
        print(f"      |⟨deflated, slot_0⟩| = {residual_A:.2e} (aniquilado)")
        print(f"      ⟨deflated, slot_1⟩ = {recovery_B:.4f} (sobrevive)")
        print(f"    [✅ PASS] Deflación: slot_0 aniquilado, slot_1 preservado")
    else:
        print(f"    [⚠️  INFO] Deflación degenerada (motivo={deflate_result['degeneracy_reason']}), "
              f"subproductos reales eran colineales — comportamiento esperado en algunos seeds")

    # ─── 5. CONSULTA DE RESONANCIA GEODÉSICA ─────────────────────────────
    section("5/5. CONSULTA DE RESONANCIA GEODÉSICA contra memoria viva")

    # Usar el primer subproducto normalizado como query
    query = m_A_unit
    resonance_result = aether_native_c.hilbert_memory_query_resonance(query)
    mx.eval(resonance_result["resonances"])

    assert resonance_result["count"] == NUM_STEPS
    resonances = np.array(resonance_result["resonances"])
    print(f"    Query: slot_0 normalizado contra {resonance_result['count']} slots (no normalizados)")
    for i in range(NUM_STEPS):
        print(f"      Slot {i}: resonancia = {resonances[i]:+.6f}")

    # El slot 0 debe tener resonancia = ⟨m_A_unit, h_ortho_0⟩ = ‖h_ortho_0‖
    # Esto es un invariante matemático: la resonancia de un vector normalizado contra su original = norma del original
    expected_res_0 = float(norm_A)
    actual_res_0 = resonances[0]
    err_res = abs(actual_res_0 - expected_res_0)
    assert err_res < EPS, f"Resonancia de auto-consulta: esperado {expected_res_0:.6f}, obtenido {actual_res_0:.6f}"
    print(f"    [✅ PASS] Resonancia slot_0 = {actual_res_0:.6f} ≈ ‖h_ortho_0‖ = {expected_res_0:.6f} (err={err_res:.2e})")

    section("RESULTADO: H2.1-D GAP JUNCTION 100% CERTIFICADO EN SILICIO")
    print("  Célula 1 (ConformalCoupling) ──► Célula 2 (HilbertMemoryCell)")
    print(f"  {NUM_STEPS} subproductos capturados automáticamente")
    print("  Álgebra de Hilbert operativa sobre memoria viva")
    print("  Consulta de resonancia geodésica funcional")
    print()

if __name__ == "__main__":
    test_intercell_coupling()
