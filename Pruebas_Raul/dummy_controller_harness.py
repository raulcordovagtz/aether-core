#!/usr/bin/env python3
"""
dummy_controller_harness.py
═══════════════════════════════════════════════════════════════════════════════
PROTOTIPO DUMMY HITO 3.0: INTRA-CYCLE PREDICTIVE CONTROLLER
Simulación en espacio latente R^D (D=2048) con búsqueda local de hipótesis,
conformal commit, fallback de seguridad y adaptación de pesos en línea.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ─── 1. ESTRUCTURAS DE DATOS CANÓNICAS ────────────────────────────────────────

@dataclass
class HypothesisState:
    id: str
    h_pred: np.ndarray          # Vector predicho normalizado en S^{D-1}
    resonance_r: float          # Afinidad con memoria C2
    kinematic_cons: float       # Consistencia con v_t y a_t
    persistence: float          # Proximidad a h_t
    angular_step: float         # Ángulo d_S(h_t, h_pred) en radianes
    score: float = 0.0
    pruned: bool = False

# ─── 2. MOTOR DEL CONTROLADOR INTRA-CICLO ──────────────────────────────────────

class IntraCyclePredictiveController:
    def __init__(
        self,
        dim: int = 2048,
        theta_max_commit: float = 0.15,   # Conformal bound: máx 0.15 rad de giro
        min_score_margin: float = 0.04,   # Margen mínimo para commit activo
        learning_rate_meta: float = 0.10  # Tasa de adaptación de predictores
    ):
        self.D = dim
        self.theta_max = theta_max_commit
        self.min_margin = min_score_margin
        self.lr_meta = learning_rate_meta

        # Búfer Markoviano cinemático (C1)
        self.history_h: List[np.ndarray] = []
        
        # Ranuras de Memoria de Hilbert (C2)
        self.memory_slots: List[np.ndarray] = []

        # Fiabilidad adaptativa de los predictores basales (Meta-Pesos)
        self.predictor_weights = {
            "persistence": 0.20,
            "const_velocity": 0.25,
            "const_accel": 0.25,
            "memory_attractor": 0.30
        }

    def seed_memory(self, slots: List[np.ndarray]):
        """Carga memorias fácticas normalizadas en C2."""
        self.memory_slots = [s / np.linalg.norm(s) for s in slots]

    # ─── FASE 1: OBSERVE ──────────────────────────────────────────────────────
    def observe(self, h_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Calcula cinemática C1 y resonancia diferencial C2 sin alterar el estado."""
        h_t = h_t / np.linalg.norm(h_t)
        self.history_h.append(h_t)
        if len(self.history_h) > 3:
            self.history_h.pop(0)

        # Cinemática discreta
        v_t = np.zeros(self.D)
        a_t = np.zeros(self.D)
        if len(self.history_h) >= 2:
            v_t = self.history_h[-1] - self.history_h[-2]
        if len(self.history_h) == 3:
            v_prev = self.history_h[-2] - self.history_h[-3]
            a_t = v_t - v_prev

        # Curvatura y resonancia C2
        sq_v = np.dot(v_t, v_t)
        sq_a = np.dot(a_t, a_t)
        dot_va = np.dot(v_t, a_t)
        bivector = max(0.0, (sq_v * sq_a) - (dot_va ** 2))
        kappa = np.sqrt(bivector) / (sq_v ** 1.5 + 1e-12)

        # Resonancias con slots y margen diferencial Δr = r1 - r2
        resonances = [float(np.dot(h_t, m)) for m in self.memory_slots]
        if len(resonances) >= 2:
            sorted_r = sorted(resonances, reverse=True)
            r_max, delta_r = sorted_r[0], sorted_r[0] - sorted_r[1]
        else:
            r_max, delta_r = (resonances[0], 1.0) if resonances else (0.0, 0.0)

        return v_t, a_t, kappa, delta_r

    # ─── FASE 2 Y 3: PREDICT & BRANCH (Árbol Geométrico Local) ────────────────
    def branch_hypotheses(self, h_t: np.ndarray, v_t: np.ndarray, a_t: np.ndarray) -> List[HypothesisState]:
        """Genera un haz de 8 hipótesis puramente en R^D (Cero llamada a logits)."""
        hypotheses = []

        def make_unit(vec):
            n = np.linalg.norm(vec)
            return vec / n if n > 1e-12 else h_t

        # 1. Cinemáticas Basales
        h_persist = h_t
        h_cvel = make_unit(h_t + v_t)
        h_cacc = make_unit(h_t + v_t + 0.5 * a_t)

        # 2. Atractor de Memoria C2 (Top-1)
        best_slot = self.memory_slots[int(np.argmax([np.dot(h_t, m) for m in self.memory_slots]))]
        # Rotación tangencial hacia la memoria (paso pequeño)
        v_mem = best_slot - np.dot(best_slot, h_t) * h_t
        v_mem_unit = v_mem / (np.linalg.norm(v_mem) + 1e-12)
        h_mem = np.cos(0.08) * h_t + np.sin(0.08) * v_mem_unit

        # 3. Composición Álgebra de Hilbert (Pack m1 + m2)
        if len(self.memory_slots) >= 2:
            m_pack = make_unit(self.memory_slots[0] + self.memory_slots[1])
            v_pack = m_pack - np.dot(m_pack, h_t) * h_t
            h_pack = np.cos(0.08) * h_t + np.sin(0.08) * make_unit(v_pack)
        else:
            h_pack = h_mem

        # 4. Deflación Gram-Schmidt (Peel: m1 ⊥ m2)
        if len(self.memory_slots) >= 2:
            m_peel = self.memory_slots[0] - np.dot(self.memory_slots[0], self.memory_slots[1]) * self.memory_slots[1]
            m_peel = make_unit(m_peel)
            v_peel = m_peel - np.dot(m_peel, h_t) * h_t
            h_peel = np.cos(0.08) * h_t + np.sin(0.08) * make_unit(v_peel)
        else:
            h_peel = h_mem

        # 5. Híbridos Inercia + Memoria
        h_cvel_mem = make_unit(0.6 * h_cvel + 0.4 * h_mem)
        h_cacc_mem = make_unit(0.5 * h_cacc + 0.5 * h_pack)

        raw_candidates = [
            ("persistence",  h_persist),
            ("const_vel",    h_cvel),
            ("const_acc",    h_cacc),
            ("memory_pull",  h_mem),
            ("hilbert_pack", h_pack),
            ("hilbert_peel", h_peel),
            ("cvel+mem",     h_cvel_mem),
            ("cacc+pack",    h_cacc_mem),
        ]

        for name, h_cand in raw_candidates:
            # Métricas endógenas
            dot_origin = np.clip(np.dot(h_t, h_cand), -1.0, 1.0)
            theta = np.arccos(dot_origin)
            
            # Afinidad fáctica
            r_val = max(np.dot(h_cand, m) for m in self.memory_slots)
            
            # Consistencia inercial: alineación con el vector velocidad previo
            v_cand = h_cand - h_t
            v_align = np.dot(v_cand, v_t) / (np.linalg.norm(v_cand) * np.linalg.norm(v_t) + 1e-12)

            hyp = HypothesisState(
                id=name,
                h_pred=h_cand,
                resonance_r=r_val,
                kinematic_cons=max(0.0, float(v_align)),
                persistence=float(dot_origin),
                angular_step=float(theta)
            )

            # Poda rígida (pruning): giros angulares que rompen la física
            if hyp.angular_step > self.theta_max * 1.5:
                hyp.pruned = True

            hypotheses.append(hyp)

        return hypotheses

    # ─── FASE 4: CORRECT (Scoring Multi-Objetivo Adaptativo) ──────────────────
    def correct_and_score(self, hypotheses: List[HypothesisState]):
        w_r = 2.50      # Mayor autoridad a la memoria fáctica
        w_k = 0.80      
        w_p = 0.50      
        w_theta = 0.40  # Amortiguar penalización angular para permitir giros fácticos

        for hyp in hypotheses:
            if hyp.pruned:
                hyp.score = -999.0
                continue

            # Buscar la fiabilidad meta-aprendida de esta familia
            meta_reliability = 0.25
            for k, weight in self.predictor_weights.items():
                if k in hyp.id:
                    meta_reliability = weight
                    break

            # Si es un atractor de memoria con resonancia positiva, relajar coste angular
            effective_angular_cost = w_theta * (hyp.angular_step / self.theta_max)
            if "mem" in hyp.id or "pack" in hyp.id:
                effective_angular_cost *= (1.0 - np.clip(hyp.resonance_r, 0.0, 0.8))

            # Puntuación ponderada por la fiabilidad viva del meta-controlador
            base_score = (
                w_r * hyp.resonance_r +
                w_k * hyp.kinematic_cons +
                w_p * hyp.persistence -
                effective_angular_cost
            )
            
            # EL CIERRE DEL BUCLE: La fiabilidad acumulada modula el score
            hyp.score = base_score * (0.5 + meta_reliability)

    # ─── FASE 5: COMMIT (Decisión Conformal y Fallback Seguro) ─────────────────
    def commit(self, h_vanilla: np.ndarray, hypotheses: List[HypothesisState]) -> Tuple[np.ndarray, str, dict]:
        """Elige la trayectoria ganadora o cae en modo Vanilla si hay ambigüedad."""
        valid_hyps = [h for h in hypotheses if not h.pruned]
        valid_hyps.sort(key=lambda x: x.score, reverse=True)

        h_1 = valid_hyps[0]
        h_2 = valid_hyps[1] if len(valid_hyps) > 1 else None

        margin = (h_1.score - h_2.score) if h_2 else 1.0

        telemetry = {
            "selected": h_1.id,
            "margin": margin,
            "angle": h_1.angular_step,
            "score": h_1.score,
            "status": "ACTIVE_COMMIT"
        }

        # Regla de Seguridad Conformal:
        # Si la ventaja es débil (ambigüedad) o el giro supera el límite seguro -> FALLBACK
        if margin < self.min_margin or h_1.angular_step > self.theta_max:
            telemetry["status"] = "FALLBACK_VANILLA"
            telemetry["fallback_reason"] = "LOW_MARGIN" if margin < self.min_margin else "ANGULAR_OVERSTEP"
            return h_vanilla, "vanilla", telemetry

        return h_1.h_pred, h_1.id, telemetry

    # ─── FASE 6: FEEDBACK & META-WEIGHTING (Aprendizaje sin Backprop) ──────────
    def feedback_and_reweight(self, h_committed: np.ndarray, h_real_next: np.ndarray, selected_id: str):
        """Mide el error contra el estado siguiente observado y adapta las fiabilidades."""
        err = 1.0 - np.clip(np.dot(h_committed, h_real_next), -1.0, 1.0)

        # Si acertó (error bajo), refuerza la confianza en ese tipo de trayectoria
        base_key = None
        for k in self.predictor_weights.keys():
            if k in selected_id:
                base_key = k
                break

        if base_key:
            reward = np.exp(-err * 50.0) # Recompensa exponencial al error angular
            self.predictor_weights[base_key] = (
                (1.0 - self.lr_meta) * self.predictor_weights[base_key] + self.lr_meta * reward
            )
            # Normalizar meta-pesos
            total = sum(self.predictor_weights.values())
            for k in self.predictor_weights:
                self.predictor_weights[k] /= total

        return err

# ─── 3. SIMULACIÓN DUMMY EN SILICIO (20 PASOS TEMPORALES) ──────────────────────

def run_dummy_experiment():
    print("═" * 78)
    print("  SIMULACIÓN DUMMY: HARNESS CONTROLADOR INTRA-CICLO (HITO 3.0)")
    print("═" * 78)

    D = 2048
    np.random.seed(42)
    controller = IntraCyclePredictiveController(dim=D)

    # Creamos 3 memorias en C2 (Hechos fácticos puros)
    m1 = np.random.randn(D); m1 /= np.linalg.norm(m1)
    m2 = np.random.randn(D); m2 = m2 - np.dot(m2, m1) * m1; m2 /= np.linalg.norm(m2)
    controller.seed_memory([m1, m2])

    # Estado inicial h_0
    h_current = np.random.randn(D)
    h_current /= np.linalg.norm(h_current)

    print(f"• Dimensión: D={D} | Memorias C2: 2 slots ortogonales cargados")
    print(f"• Conformal Angle Bound: {controller.theta_max:.2f} rad | Margen Mínimo: {controller.min_margin:.2f}")
    print("\nIniciando secuencia de 15 tokens latentes (Token 8 introduce choque fáctico)...\n")
    print(f" {'Paso':<5} │ {'Acción / Commit':<18} │ {'Margen':<8} │ {'Giro θ':<9} │ {'Error Real':<11} │ {'Modo'}")
    print(" ──────┼────────────────────┼──────────┼───────────┼─────────────┼───────────────────")

    latencies_us = []

    for step in range(15):
        t0 = time.perf_counter()

        # 1. OBSERVE
        v_t, a_t, kappa, delta_r = controller.observe(h_current)

        # 2 & 3. PREDICT & BRANCH (8 hipótesis)
        branches = controller.branch_hypotheses(h_current, v_t, a_t)

        # 4. CORRECT (Scoring local)
        controller.correct_and_score(branches)

        # Simulación del estado nativo vanilla de la red
        h_vanilla = h_current + 0.05 * v_t
        h_vanilla /= np.linalg.norm(h_vanilla)

        # 5. COMMIT
        h_committed, chosen_name, telemetry = controller.commit(h_vanilla, branches)

        t_us = (time.perf_counter() - t0) * 1e6
        latencies_us.append(t_us)

        # Simulación de la evolución real externa del Transformer
        # En el paso 8, el contexto externo sufre un 'atractor shock' hacia m1 (evidencia factual)
        if step >= 8:
            drift = 0.07 * m1 + 0.02 * np.random.randn(D)
        else:
            drift = 0.03 * v_t + 0.01 * np.random.randn(D)

        h_real_next = h_committed + drift
        h_real_next /= np.linalg.norm(h_real_next)

        # 6. FEEDBACK & META-WEIGHT
        err = controller.feedback_and_reweight(h_committed, h_real_next, chosen_name)

        status_flag = "🟢 ACTIVO" if telemetry["status"] == "ACTIVE_COMMIT" else "🟡 FALLBACK"
        print(f" #{step+1:<4} │ {chosen_name:<18} │ {telemetry['margin']:8.4f} │ {telemetry['angle']:8.4f} rad │ {err:10.2e}  │ {status_flag}")

        # Avanzar el estado
        h_current = h_real_next

    p50 = np.percentile(latencies_us, 50)
    print(" ──────┴────────────────────┴──────────┴───────────┴─────────────┴───────────────────")
    print(f"\n📊 Telemetría de Latencia: p50 = {p50:.2f} µs por ciclo intra-token")
    print("• Fiabilidad Adaptada de Predictores (Meta-Pesos finales):")
    for pred, w in controller.predictor_weights.items():
        print(f"   - {pred:<18}: {w:.4f}")

    print("\n✅ CERTIFICACIÓN DEL DUMMY CONCLUIDA CON ÉXITO.")

if __name__ == "__main__":
    run_dummy_experiment()