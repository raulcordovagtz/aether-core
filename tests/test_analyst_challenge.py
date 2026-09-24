#!/usr/bin/env python3
"""
tests/test_analyst_challenge.py
═══════════════════════════════════════════════════════════════════════════════
RETO DEL ASESOR: COEXISTENCIA LÓGICA + ARITMÉTICA EN EL MISMO ESTRATO UCA
Prueba de Falsabilidad:
  1. Problema Base: E = A + C
  2. Mutación Unitaria del Operador: E = A * C (mismo tensor, solo cambia OP)
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. ESTRATO DE RESTRICCIONES UNIVERSAL (UCA TENSOR ENGINE)
# ─────────────────────────────────────────────────────────────────────────────

class OpType:
    NONE = 0
    ADD  = 1
    MUL  = 2

class UCASolverCore:
    def __init__(self, entities=["A", "B", "C", "D", "E"], val_min=1, val_max=15):
        self.entities = entities
        self.N = len(entities)
        self.val_min = val_min
        self.val_max = val_max
        self.V = val_max - val_min + 1
        self.e2i = {e: i for i, e in enumerate(entities)}

        # ── ESTRATO MULTI-CANAL ──
        # Canal 0: Dominios discretos (N x V) bitmask en [1..15]
        self.domains = np.ones((self.N, self.V), dtype=bool)

        # Canal 1: Matriz de Orden Relacional GT (N x N)
        self.rel_gt = np.zeros((self.N, self.N), dtype=bool)

        # Canal 2: Matriz Aritmética Lineal (N x N): target = source + offset
        self.arith_offset = np.full((self.N, self.N), np.nan)

        # Canal 3: Operaciones Compuestas: target = op(src1, src2)
        self.compound_ops = [] # list of (target, src1, src2, OpType)

    def set_gt(self, a, b):
        """Restricción relacional: a > b"""
        i, j = self.e2i[a], self.e2i[b]
        self.rel_gt[i, j] = True

    def set_offset(self, target, source, c):
        """Restricción aritmética de desplazamiento: target = source + c"""
        i, j = self.e2i[target], self.e2i[source]
        self.arith_offset[i, j] = c

    def set_compound(self, target, src1, src2, op_type):
        """Restricción algebraica binaria: target = op(src1, src2)"""
        self.compound_ops.append((self.e2i[target], self.e2i[src1], self.e2i[src2], op_type))

    def set_parity(self, entity, remainder):
        """Restricción de paridad: entity % 2 == remainder"""
        i = self.e2i[entity]
        for v in range(self.val_min, self.val_max + 1):
            if v % 2 != remainder:
                self.domains[i, v - self.val_min] = False

    def set_upper_bound(self, entity, bound):
        """Restricción de cota: entity < bound"""
        i = self.e2i[entity]
        for v in range(self.val_min, self.val_max + 1):
            if v >= bound:
                self.domains[i, v - self.val_min] = False

    # ── PROPAGADORES DEL ESTRATO ──

    def propagate_transitivity(self):
        """Clausura transitiva de orden: i > k and k > j ==> i > j"""
        changed = False
        for k in range(self.N):
            for i in range(self.N):
                if self.rel_gt[i, k]:
                    for j in range(self.N):
                        if self.rel_gt[k, j] and not self.rel_gt[i, j]:
                            self.rel_gt[i, j] = True
                            changed = True
        return changed

    def propagate_domain_bounds(self):
        """Consistencia de arcos en desigualdades: a > b"""
        changed = False
        for i in range(self.N):
            for j in range(self.N):
                if self.rel_gt[i, j]:
                    # Los valores de j deben ser estrictamente menores que el máximo de i
                    vals_i = np.where(self.domains[i])[0] + self.val_min
                    vals_j = np.where(self.domains[j])[0] + self.val_min
                    if len(vals_i) == 0 or len(vals_j) == 0:
                        continue
                    max_i = np.max(vals_i)
                    min_j = np.min(vals_j)

                    # Podar j >= max_i
                    mask_j = (vals_j < max_i)
                    if not np.all(mask_j):
                        for v in vals_j[~mask_j]:
                            self.domains[j, v - self.val_min] = False
                            changed = True

                    # Podar i <= min_j
                    mask_i = (vals_i > min_j)
                    if not np.all(mask_i):
                        for v in vals_i[~mask_i]:
                            self.domains[i, v - self.val_min] = False
                            changed = True
        return changed

    def propagate_arithmetic(self):
        """Propagación de restricciones aritméticas de desplazamiento D = C + 4"""
        changed = False
        for i in range(self.N):
            for j in range(self.N):
                if not np.isnan(self.arith_offset[i, j]):
                    c = int(self.arith_offset[i, j])
                    vals_i = np.where(self.domains[i])[0] + self.val_min
                    vals_j = np.where(self.domains[j])[0] + self.val_min

                    # i = j + c
                    for vi in vals_i:
                        if (vi - c) not in vals_j:
                            self.domains[i, vi - self.val_min] = False
                            changed = True

                    for vj in vals_j:
                        if (vj + c) not in vals_i:
                            self.domains[j, vj - self.val_min] = False
                            changed = True
        return changed

    def propagate_compound(self):
        """Propagación algebraica del operador (ADD o MUL)"""
        changed = False
        for target, s1, s2, op_type in self.compound_ops:
            vals_t = np.where(self.domains[target])[0] + self.val_min
            vals_1 = np.where(self.domains[s1])[0] + self.val_min
            vals_2 = np.where(self.domains[s2])[0] + self.val_min

            valid_t = set()
            valid_1 = set()
            valid_2 = set()

            for v1 in vals_1:
                for v2 in vals_2:
                    if op_type == OpType.ADD:
                        res = v1 + v2
                    elif op_type == OpType.MUL:
                        res = v1 * v2
                    else:
                        continue

                    if res in vals_t:
                        valid_t.add(res)
                        valid_1.add(v1)
                        valid_2.add(v2)

            # Podar inconsistencias
            for vt in vals_t:
                if vt not in valid_t:
                    self.domains[target, vt - self.val_min] = False
                    changed = True
            for v1 in vals_1:
                if v1 not in valid_1:
                    self.domains[s1, v1 - self.val_min] = False
                    changed = True
            for v2 in vals_2:
                if v2 not in valid_2:
                    self.domains[s2, v2 - self.val_min] = False
                    changed = True
        return changed

    def solve_fixed_point(self, max_cycles=32):
        """Catálisis hasta punto fijo"""
        for it in range(max_cycles):
            c1 = self.propagate_transitivity()
            c2 = self.propagate_domain_bounds()
            c3 = self.propagate_arithmetic()
            c4 = self.propagate_compound()
            if not (c1 or c2 or c3 or c4):
                return it + 1
        return max_cycles

    def is_consistent(self):
        return all(np.sum(self.domains[i]) > 0 for i in range(self.N))

    def get_domain_list(self, e):
        i = self.e2i[e]
        return (np.where(self.domains[i])[0] + self.val_min).tolist()

    def print_raw_state(self, title):
        print(f"\n{'═' * 78}")
        print(f"  {title}")
        print(f"{'═' * 78}")
        print("  ESTADO DE DOMINIOS ACTIVOS TRAS CATÁLISIS:")
        for e in self.entities:
            dom = self.get_domain_list(e)
            print(f"    {e} ∈ {dom if len(dom) > 0 else '∅ [VACÍO / UNSAT]'}")

        print("\n  MATRIZ DE RELACIÓN DE ORDEN (GT = 1):")
        header = "       " + "  ".join(self.entities)
        print(header)
        for i, e1 in enumerate(self.entities):
            row = f"  {e1}  [ " + "  ".join(["1" if self.rel_gt[i, j] else "·" for j in range(self.N)]) + " ]"
            print(row)

# ─────────────────────────────────────────────────────────────────────────────
# 2. ENUMERADOR DETERMINISTA CON ALDIFERENT (DESPUÉS DEL PUNTO FIJO)
# ─────────────────────────────────────────────────────────────────────────────

def search_distinct_solutions(solver, op_type):
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
                d = c + 4
                if d not in dom_D or d >= b: continue
                e = (a + c) if op_type == OpType.ADD else (a * c)
                if e not in dom_E or e >= 13: continue
                
                # Restricción AllDifferent
                vals = [a, b, c, d, e]
                if len(set(vals)) == 5:
                    solutions.append((a, b, c, d, e))
    return solutions


# ═════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN DE LOS DOS RETOS
# ═════════════════════════════════════════════════════════════════════════════

def run_challenge():
    # ── RETO 1: E = A + C ──
    s1 = UCASolverCore()
    s1.set_gt("A", "B")          # 1. A > B
    s1.set_gt("B", "C")          # 2. B > C
    s1.set_offset("D", "C", 4)   # 3. D = C + 4
    s1.set_compound("E", "A", "C", OpType.ADD) # 4. E = A + C
    s1.set_upper_bound("E", 13)  # 5. E < 13
    s1.set_gt("B", "D")          # 6. D < B
    s1.set_parity("A", 0)        # 7. A es par
    s1.set_parity("C", 1)        # 8. C es impar

    iters1 = s1.solve_fixed_point()
    s1.print_raw_state(f"RETO 1: E = A + C (PUNTO FIJO EN {iters1} CICLOS)")

    sol1 = search_distinct_solutions(s1, OpType.ADD)
    print(f"\n  SOLUCIONES DETERMINISTAS ENCONTRADAS (Total: {len(sol1)}):")
    for s in sol1:
        print(f"    (A={s[0]:2d}, B={s[1]:2d}, C={s[2]:2d}, D={s[3]:2d}, E={s[4]:2d})")

    # ── RETO 2: E = A * C (SOLO CAMBIA EL OPERADOR) ──
    s2 = UCASolverCore()
    s2.set_gt("A", "B")          # 1. A > B
    s2.set_gt("B", "C")          # 2. B > C
    s2.set_offset("D", "C", 4)   # 3. D = C + 4
    # MUTACIÓN UNITARIA: exactamente el mismo código, sólo OpType.MUL
    s2.set_compound("E", "A", "C", OpType.MUL) # 4. E = A * C
    s2.set_upper_bound("E", 13)  # 5. E < 13
    s2.set_gt("B", "D")          # 6. D < B
    s2.set_parity("A", 0)        # 7. A es par
    s2.set_parity("C", 1)        # 8. C es impar

    iters2 = s2.solve_fixed_point()
    s2.print_raw_state(f"RETO 2: E = A * C (PUNTO FIJO EN {iters2} CICLOS)")

    sol2 = search_distinct_solutions(s2, OpType.MUL)
    print(f"\n  SOLUCIONES DETERMINISTAS ENCONTRADAS (Total: {len(sol2)}):")
    if len(sol2) == 0:
        print("    ∅ NINGUNA SOLUCIÓN — SISTEMA MATEMÁTICAMENTE UNSAT / INCONSISTENTE")

if __name__ == "__main__":
    run_challenge()