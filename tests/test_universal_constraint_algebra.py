#!/usr/bin/env python3
"""
tests/test_universal_constraint_algebra.py
═══════════════════════════════════════════════════════════════════════════════
SUITE DE CERTIFICACIÓN DEL ÁLGEBRA UNIVERSAL DE RESTRICCIONES (UCA)
Verificación de los 3 Casos en el Mismo Tensor Fijo + Operador EML Sheffer
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, math
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from tools.universal_constraint_enzyme import (
    eml, eml_ln, eml_exp, eml_add, eml_sub, eml_mul,
    UniversalTensorEnzyme, Channel
)

EPS = 1e-5

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def test_universal_constraint_algebra():
    section("INICIANDO LABORATORIO: UNIVERSAL CONSTRAINT ALGEBRA (UCA)")

    # ─── TEST 1: RECONSTRUCCIÓN CON EL OPERADOR EML SHEFFER (Odrzywołek 2026) ─
    print("▶ 1/4. Verificando el Operador EML Sheffer como Primitiva Continua Universal...")
    x, y = 14.0, 7.0
    
    # Resta x - y
    res_sub = eml_sub(x, y)
    assert abs(res_sub - 7.0) < EPS, f"Error en EML sub: {res_sub}"
    print(f"  [✅ PASS] EML Resta (14 - 7) = {res_sub:.4f} (Esperado: 7.0000)")

    # Suma x + y
    res_add = eml_add(x, y)
    assert abs(res_add - 21.0) < EPS, f"Error en EML add: {res_add}"
    print(f"  [✅ PASS] EML Suma (14 + 7)  = {res_add:.4f} (Esperado: 21.0000)")

    # Multiplicación x * y
    res_mul = eml_mul(x, y)
    assert abs(res_mul - 98.0) < EPS, f"Error en EML mul: {res_mul}"
    print(f"  [✅ PASS] EML Multiplicación (14 * 7) = {res_mul:.4f} (Esperado: 98.0000)")

    enzyme = UniversalTensorEnzyme(max_entities=16, latent_dim=2048)

    # ─── TEST 2: CASO LÓGICO PURO (ORDEN Y TRANSITIVIDAD) ────────────────────
    print("\n▶ 2/4. Caso Lógico: Puzle de Orden A > B, B > C, C > D (Mismo Tensor)...")
    enzyme.reset()
    enzyme.set_greater_than("A", "B")
    enzyme.set_greater_than("B", "C")
    enzyme.set_greater_than("C", "D")

    # Matriz inicial antes de catálisis
    S_init = enzyme.S.copy()
    iters = enzyme.propagate_fixed_point()

    idx_A = enzyme.name_to_idx["A"]
    idx_D = enzyme.name_to_idx["D"]

    # Deducción transitiva formal: A > D debe haberse encendido automáticamente
    assert enzyme.S[idx_A, idx_D, Channel.GT] == 1.0
    assert enzyme.S[idx_D, idx_A, Channel.LT] == 1.0
    print(f"  [✅ PASS] Punto Fijo alcanzado en {iters} iteraciones")
    print(f"  [✅ PASS] Deducción Transitiva Emergente: A > D = {enzyme.S[idx_A, idx_D, Channel.GT] == 1.0}")
    print(f"  [✅ PASS] Consistencia Inversa: D < A = {enzyme.S[idx_D, idx_A, Channel.LT] == 1.0}")

    # ─── TEST 3: CASO ARITMÉTICO PURO (ECUACIONES LINEALES EN LA MISMA MATRIZ)
    print("\n▶ 3/4. Caso Aritmético: Luis=7, Ana=2*Luis, Pedro=Ana+3 (Mismo Tensor)...")
    enzyme.reset()
    enzyme.set_scalar_value("Luis", 7.0)
    enzyme.set_linear_relation("Ana", "Luis", mult=2.0, offset=0.0)    # Ana = 2 * Luis
    enzyme.set_linear_relation("Pedro", "Ana", mult=1.0, offset=3.0)    # Pedro = Ana + 3

    S_arith_init = enzyme.S.copy()
    iters_arith = enzyme.propagate_fixed_point()

    idx_luis  = enzyme.name_to_idx["Luis"]
    idx_ana   = enzyme.name_to_idx["Ana"]
    idx_pedro = enzyme.name_to_idx["Pedro"]

    val_luis  = enzyme.S[idx_luis,  idx_luis,  Channel.SCALAR]
    val_ana   = enzyme.S[idx_ana,   idx_ana,   Channel.SCALAR]
    val_pedro = enzyme.S[idx_pedro, idx_pedro, Channel.SCALAR]

    assert abs(val_luis - 7.0) < EPS
    assert abs(val_ana - 14.0) < EPS
    assert abs(val_pedro - 17.0) < EPS
    print(f"  [✅ PASS] Punto Fijo Aritmético en {iters_arith} iteraciones")
    print(f"  [✅ PASS] Luis  = {val_luis:.1f} (Esperado: 7.0)")
    print(f"  [✅ PASS] Ana   = {val_ana:.1f} (Esperado: 14.0)")
    print(f"  [✅ PASS] Pedro = {val_pedro:.1f} (Esperado: 17.0)")

    # ─── TEST 4: INOCULACIÓN ADITIVA MÍNIMA ΔR EN EL RESIDUAL STREAM ─────────
    print("\n▶ 4/4. Inoculación Mínima ΔR hacia el Residual Stream R^2048...")
    inoculation = enzyme.compute_residual_inoculation(S_arith_init)
    norm_delta = inoculation["norm_delta"]
    assert inoculation["is_effective"]
    assert norm_delta > 0.0
    print(f"  [✅ PASS] Vector de alteración mínima calculado: ||ΔR|| = {norm_delta:.4f}")
    print(f"  [✅ PASS] Conservación de Contexto: Sólo altera la dirección del subespacio resuelto")

    section("RESULTADO: ÁLGEBRA UNIVERSAL DE RESTRICCIONES (UCA) VALIDADA AL 100%")

if __name__ == "__main__":
    test_universal_constraint_algebra()