#!/usr/bin/env python3
"""
tests/lab20_r3_purified_identifiability.py
═══════════════════════════════════════════════════════════════════════════════
LAB 20-R3 PURIFICADO: IDENTIFICABILIDAD EXACTA Y ORÁCULO DE FUERZA BRUTA
Comparación Dual:
  1. Oráculo de Fuerza Bruta Exhaustivo (5 bucles en {1..15}^5)
  2. Solver Simbólico UCA con offset dinámico k
Para todo k ∈ [1 .. 14]
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
from tests.test_analyst_challenge import UCASolverCore, OpType

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

# ── 1. ORÁCULO DE FUERZA BRUTA INDEPENDIENTE (GROUND TRUTH MATEMÁTICO) ───────
def brute_force_oracle(k_offset):
    solutions = []
    for a in range(1, 16):
        if a % 2 != 0: continue # 7. A es par
        for c in range(1, 16):
            if c % 2 == 0: continue # 8. C es impar
            e = a + c # 4. E = A + C
            if e >= 13: continue # 5. E < 13
            d = c + k_offset # 3. D = C + k
            if d < 1 or d > 15: continue
            for b in range(1, 16):
                if not (a > b): continue # 1. A > B
                if not (b > c): continue # 2. B > C
                if not (d < b): continue # 6. D < B
                
                vals = [a, b, c, d, e]
                if len(set(vals)) == 5:
                    solutions.append((a, b, c, d, e))
    return solutions

# ── 2. BÚSQUEDA DINÁMICA CON UCA CON OFFSET PARAMETRIZADO k ─────────────────
def search_uca_solutions(solver, k_offset):
    solutions = []
    dom_A = solver.get_domain_list("A")
    dom_B = solver.get_domain_list("B")
    dom_C = solver.get_domain_list("C")
    dom_D = solver.get_domain_list("D")
    dom_E = solver.get_domain_list("E")

    for a in dom_A:
        for b in dom_B:
            if a <= b: continue
            for c in dom_C:
                if b <= c: continue
                d = c + k_offset # DINÁMICO
                if d not in dom_D or d >= b: continue
                e = a + c
                if e not in dom_E or e >= 13: continue
                
                vals = [a, b, c, d, e]
                if len(set(vals)) == 5:
                    solutions.append((a, b, c, d, e))
    return solutions

def run_purified_sweep():
    section("LAB 20-R3: BARRIDO PURIFICADO DE IDENTIFICABILIDAD SIMBÓLICA (k=1..14)")
    print(f"  {'k':<4} │ {'Oráculo Bruto':<16} │ {'UCA Solver':<16} │ {'Paridad':<9} │ Valores de D*")
    print("  " + "─" * 74)

    sat_k = []
    unsat_k = []

    for k in range(1, 15):
        # 1. Oráculo
        oracle_sols = brute_force_oracle(k)
        n_oracle = len(oracle_sols)
        d_oracle = sorted(list(set([s[3] for s in oracle_sols])))

        # 2. UCA Solver
        solver = UCASolverCore(entities=["A", "B", "C", "D", "E"], val_min=1, val_max=15)
        solver.set_gt("A", "B")
        solver.set_gt("B", "C")
        solver.set_offset("D", "C", float(k))
        solver.set_compound("E", "A", "C", OpType.ADD)
        solver.set_upper_bound("E", 13)
        solver.set_gt("B", "D")
        solver.set_parity("A", 0)
        solver.set_parity("C", 1)
        solver.solve_fixed_point()

        uca_sols = search_uca_solutions(solver, k)
        n_uca = len(uca_sols)
        d_uca = sorted(list(set([s[3] for s in uca_sols])))

        # 3. Comparación como conjuntos canónicos ordenados
        match = (set(oracle_sols) == set(uca_sols))
        paridad_str = "✅ EXACTO" if match else "❌ MISMATCH"

        if n_oracle > 0:
            sat_k.append(k)
            status_str = f"SAT ({n_oracle} sols)"
        else:
            unsat_k.append(k)
            status_str = "UNSAT (0 sols)"

        d_str = str(d_oracle) if n_oracle > 0 else "∅"
        print(f"  k={k:<2d} │ {status_str:<16} │ {f'SAT ({n_uca})' if n_uca > 0 else 'UNSAT (0)':<16} │ {paridad_str:<9} │ {d_str}")
        assert match, f"Discrepancia en k={k}: Oracle={len(oracle_sols)} vs UCA={len(uca_sols)}"

    section("MAPA MATEMÁTICO VERIFICADO DE IDENTIFICABILIDAD")
    print(f"  • Casos Satisfacibles (SAT)     : {sat_k}")
    print(f"  • Casos Incompatibles (UNSAT)    : {unsat_k}")
    print("\n  DEDUCCIÓN FORMAL INAPELABLE:")
    print(f"  1. k ∈ {sat_k} son formalmente SAT.")
    print("  2. Para k=6: D tiene solución única D = {7} (Exactamente 2 soluciones).")
    print("  3. Para k=7: D tiene solución única D = {8} (Exactamente 1 solución).")
    print(f"  4. A partir de k >= {unsat_k[0]}: Colapso analítico universal UNSAT.")
    print("  5. PARIDAD ORÁCULO ↔ UCA SOLVER: 100% IDÉNTICA EN LOS 14 VALORES.")

if __name__ == "__main__":
    run_purified_sweep()
