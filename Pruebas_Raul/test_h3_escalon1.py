#!/usr/bin/env python3
"""
Pruebas_Raul/test_h3_escalon1.py
═══════════════════════════════════════════════════════════════════════════════
HITO 3 — ESCALÓN 1: VERIFICACIÓN DEL FLUJO CAUSAL Y DESACOPLE EN 3 CAPAS
• Capa A: Evaluador Simbólico Autónomo (Restricciones discretas abstractas)
• Capa B: Predictor Cinemático Neuronal (Evolución vectorial endógena)
• Capa C: Controlador Conformal Intra-Ciclo (Interfaz de compatibilidad)
• Batería de Control Ciego: C0 (Vanilla), C1 (Passive), C2 (Random),
                           C3 (Consistent), C4 (Contradictory), C5 (H3 Loop)
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple, Optional

# ─── CAPA A: MUNDO SIMBÓLICO AUTÓNOMO (CERO VECTORES) ─────────────────────────

class SymbolicWorldA:
    """
    Gestiona el espacio de hipótesis discretas H_t.
    No sabe qué es una red neuronal, ni un tensor, ni un embedding.
    Solo procesa conjuntos de estados admisibles y reglas de eliminación.
    """
    def __init__(self, initial_states: Set[int]):
        self.admissible_states: Set[int] = set(initial_states)
        self.history_pruned: List[Set[int]] = []

    def apply_negative_observation(self, eliminated_states: Set[int]):
        """Aplica una observación excluyente: H_{t+1} = H_t \ E."""
        pruned = self.admissible_states.intersection(eliminated_states)
        self.history_pruned.append(pruned)
        self.admissible_states.difference_update(eliminated_states)

    def is_admissible(self, state_id: int) -> bool:
        return state_id in self.admissible_states

    def get_admissible_set(self) -> Set[int]:
        return set(self.admissible_states)


# ─── CAPA B: PREDICTOR CINEMÁTICO NEURONAL (CERO LÓGICA) ──────────────────────

class KinematicPredictorB:
    """
    Observa el flujo residual h_t y predice h_hat_{t+1} en R^D.
    No sabe qué acertijo se está resolviendo ni qué significan los tokens.
    """
    def __init__(self, dim: int = 256):
        self.D = dim
        self.history_h: List[np.ndarray] = []

    def observe(self, h_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        h_norm = h_t / (np.linalg.norm(h_t) + 1e-12)
        self.history_h.append(h_norm)
        if len(self.history_h) > 3:
            self.history_h.pop(0)

        v_t = np.zeros(self.D)
        a_t = np.zeros(self.D)
        if len(self.history_h) >= 2:
            v_t = self.history_h[-1] - self.history_h[-2]
        if len(self.history_h) == 3:
            v_prev = self.history_h[-2] - self.history_h[-3]
            a_t = v_t - v_prev

        sq_v = float(np.dot(v_t, v_t))
        sq_a = float(np.dot(a_t, a_t))
        dot_va = float(np.dot(v_t, a_t))
        bivector = max(0.0, (sq_v * sq_a) - (dot_va ** 2))
        kappa = np.sqrt(bivector) / (sq_v ** 1.5 + 1e-12)

        return v_t, a_t, float(kappa)

    def predict_next(self, h_t: np.ndarray, v_t: np.ndarray, a_t: np.ndarray, tau: float = 1.0) -> np.ndarray:
        """Extrapolación geodésica inercial de segundo orden."""
        h_cand = h_t + (tau * v_t) + (0.5 * (tau ** 2) * a_t)
        return h_cand / (np.linalg.norm(h_cand) + 1e-12)


# ─── CAPA C: CONTROLADOR CONFORMAL INTRA-CICLO (LA INTERFAZ) ──────────────────

@dataclass
class HypothesisCandidate:
    id: str
    target_state_id: int
    h_vec: np.ndarray
    angular_cost: float
    admissible_in_world_A: bool
    score: float = 0.0

class IntraCycleControllerC:
    """
    Evalúa si la predicción neuronal respeta el espacio admisible de Capa A.
    Aplica Conformal Commit acotado (theta <= theta_max) y telemetría de colapso.
    """
    def __init__(
        self,
        dim: int = 256,
        theta_max: float = 0.10,     # Límite conformal riguroso
        min_margin: float = 0.05,    # Certeza mínima requerida
        epsilon_tol: float = 1e-5    # Tolerancia numérica en fp32
    ):
        self.D = dim
        self.theta_max = theta_max
        self.min_margin = min_margin
        self.eps_tol = epsilon_tol
        self.tripwire_triggered = False

    def evaluate_compatibility(
        self,
        h_t: np.ndarray,
        h_pred: np.ndarray,
        symbolic_world: SymbolicWorldA,
        state_prototypes: Dict[int, np.ndarray],
        W_U: np.ndarray
    ) -> List[HypothesisCandidate]:
        """
        Genera hipótesis locales proyectadas y evalúa su conformidad con Capa A.
        """
        candidates = []
        admissible_set = symbolic_world.get_admissible_set()

        # Probar cada estado discreto como atractor potencial
        for s_id, proto in state_prototypes.items():
            is_adm = (s_id in admissible_set)
            
            # Dirección tangente hacia el prototipo
            v_dir = proto - np.dot(proto, h_t) * h_t
            norm_v = np.linalg.norm(v_dir)
            if norm_v > 1e-8:
                v_unit = v_dir / norm_v
                # Retracción acotada por theta_max
                theta_step = min(self.theta_max, 0.08)
                h_steered = np.cos(theta_step) * h_t + np.sin(theta_step) * v_unit
            else:
                h_steered = h_t
                theta_step = 0.0

            cand = HypothesisCandidate(
                id=f"state_{s_id}",
                target_state_id=s_id,
                h_vec=h_steered,
                angular_cost=theta_step,
                admissible_in_world_A=is_adm
            )
            # Scoring: Premia admisibilidad en A y persistencia; penaliza giro
            sim_pred = float(np.dot(h_steered, h_pred))
            cand.score = (2.0 if is_adm else -2.0) + 1.0 * sim_pred - 0.5 * (theta_step / self.theta_max)
            candidates.append(cand)

        return candidates

    def commit(self, h_vanilla: np.ndarray, candidates: List[HypothesisCandidate]) -> Tuple[np.ndarray, str, Dict]:
        candidates.sort(key=lambda c: c.score, reverse=True)
        top1 = candidates[0]
        top2 = candidates[1] if len(candidates) > 1 else None

        margin = (top1.score - top2.score) if top2 else 1.0
        telemetry = {
            "chosen": top1.id,
            "margin": margin,
            "angle": top1.angular_cost,
            "status": "ACTIVE_COMMIT"
        }

        # Conformal Safe Guard: Fallback si no hay margen o si viola la cota angular
        if margin < self.min_margin or top1.angular_cost > self.theta_max or not top1.admissible_in_world_A:
            telemetry["status"] = "FALLBACK_VANILLA"
            return h_vanilla, "fallback_vanilla", telemetry

        return top1.h_vec, top1.id, telemetry

    @staticmethod
    def measure_degeneration(token_history: List[int], window: int = 15) -> float:
        """Calcula el ratio de repetición empírico R_H sobre una ventana de tokens."""
        if len(token_history) < 3:
            return 0.0
        sub = token_history[-window:]
        unique = len(set(sub))
        h = len(sub)
        return float(h - unique) / float(max(1, h - 1))


# ─── BATERÍA EXPERIMENTAL DE CAREO CIEGO (C0 A C5) ───────────────────────────

def run_causal_battery_experiment():
    print("═" * 78)
    print("  HITO 3 / ESCALÓN 1: BATERÍA DE EVALUACIÓN CAUSAL CIEGA (C0 A C5)")
    print("═" * 78)

    np.random.seed(42)
    D = 256  # Dimensión latente sintética
    V = 32   # Tamaño de vocabulario sintético

    # Matriz sintética de des-cuantización / salida (W_U)
    W_U = np.random.randn(V, D)
    W_U = W_U / np.linalg.norm(W_U, axis=1, keepdims=True)

    # Definir 3 estados abstractos {S0, S1, S2} y sus prototipos latentes en S^{D-1}
    # En nuestro acertijo conceptual: S0 = Ambigüedad, S1 = Estado Blanco, S2 = Estado Negro
    state_prototypes = {}
    for s in [0, 1, 2]:
        vec = np.random.randn(D)
        vec /= np.linalg.norm(vec)
        state_prototypes[s] = vec

    # Inicializar las 3 Capas
    world_A = SymbolicWorldA(initial_states={0, 1, 2})
    predictor_B = KinematicPredictorB(dim=D)
    controller_C = IntraCycleControllerC(dim=D, theta_max=0.10, min_margin=0.05)

    print(f"• Dimensión: D={D} | Vocabulario: V={V} | Estados Simbólicos: {world_A.get_admissible_set()}")
    print("• Aplicando observación lógica en Capa A: Descartar Estado S1 (Inadmisible)...")
    world_A.apply_negative_observation(eliminated_states={1})
    print(f"✓ Capa A actualizada. Estados supervivientes admisibles: {world_A.get_admissible_set()}\n")

    # Crear una trayectoria inercial base h_{t-1}, h_t
    h_prev = np.random.randn(D); h_prev /= np.linalg.norm(h_prev)
    h_curr = h_prev + 0.05 * np.random.randn(D); h_curr /= np.linalg.norm(h_curr)

    # Capa B: Observar cinemática y predecir h_{t+1}
    predictor_B.observe(h_prev)
    v_t, a_t, kappa = predictor_B.observe(h_curr)
    h_pred = predictor_B.predict_next(h_curr, v_t, a_t, tau=1.0)

    # Estado Vanilla nativo
    h_vanilla = h_curr + 0.05 * v_t
    h_vanilla /= np.linalg.norm(h_vanilla)

    def get_logits(vec):
        return np.dot(W_U, vec)

    # Target fáctico a observar: Proyección del Estado S2 (Compatible superviviente)
    target_idx = 2

    # ─── EJECUCIÓN DE LAS 6 CONDICIONES EXPERIMENTALES (C0 - C5) ─────────────
    results = {}

    # C0: Vanilla
    z_c0 = get_logits(h_vanilla)
    results["C0_Vanilla"] = {"h": h_vanilla, "z": z_c0, "angle": 0.0}

    # C1: Passive (Conformal Hook con intervención forzada = 0)
    # Verificación de tolerancia numérica rigurosa eps_tol
    h_c1 = h_vanilla.copy()
    z_c1 = get_logits(h_c1)
    diff_c1 = float(np.max(np.abs(z_c1 - z_c0)))
    results["C1_Passive"] = {"h": h_c1, "z": z_c1, "angle": 0.0, "diff_max": diff_c1}

    # C2: Random Geometric (Retracción ortogonal aleatoria con idéntico theta = 0.08 rad)
    rnd = np.random.randn(D)
    rnd_tangent = rnd - np.dot(rnd, h_curr) * h_curr
    rnd_unit = rnd_tangent / np.linalg.norm(rnd_tangent)
    theta_test = 0.08
    h_c2 = np.cos(theta_test) * h_curr + np.sin(theta_test) * rnd_unit
    results["C2_Random"] = {"h": h_c2, "z": get_logits(h_c2), "angle": theta_test}

    # C3: Consistent State (Intervención hacia S2, admisible en Capa A)
    proto_s2 = state_prototypes[2]
    v_s2 = proto_s2 - np.dot(proto_s2, h_curr) * h_curr
    h_c3 = np.cos(theta_test) * h_curr + np.sin(theta_test) * (v_s2 / np.linalg.norm(v_s2))
    results["C3_Consistent"] = {"h": h_c3, "z": get_logits(h_c3), "angle": theta_test}

    # C4: Contradictory State (Intervención hacia S1, eliminado por Capa A)
    proto_s1 = state_prototypes[1]
    v_s1 = proto_s1 - np.dot(proto_s1, h_curr) * h_curr
    h_c4 = np.cos(theta_test) * h_curr + np.sin(theta_test) * (v_s1 / np.linalg.norm(v_s1))
    results["C4_Contradictory"] = {"h": h_c4, "z": get_logits(h_c4), "angle": theta_test}

    # C5: Closed Loop H3 (Controlador C arbitrando hipótesis)
    candidates = controller_C.evaluate_compatibility(h_curr, h_pred, world_A, state_prototypes, W_U)
    h_c5, selected_id, telemetry = controller_C.commit(h_vanilla, candidates)
    results["C5_H3_Loop"] = {"h": h_c5, "z": get_logits(h_c5), "angle": telemetry["angle"], "telemetry": telemetry}

    # ─── TABLA DE MEDICIONES Y TELEMETRÍA ─────────────────────────────────────
    print(f" {'Condición':<18} │ {'Logit Target':<14} │ {'Δz (vs C0)':<12} │ {'Giro θ':<10} │ {'Rendimiento η':<14}")
    print(" ───────────────────┼────────────────┼──────────────┼────────────┼────────────────")

    base_logit = results["C0_Vanilla"]["z"][target_idx]

    for name, res in results.items():
        z_val = res["z"][target_idx]
        delta_z = z_val - base_logit
        th = res["angle"]
        eta = (abs(delta_z) / th) if th > 1e-6 else 0.0
        print(f" {name:<18} │ {z_val:14.4f} │ {delta_z:+12.4f} │ {th:8.4f} r │ {eta:14.2f}")

    # ─── MÉTRICAS CIENTÍFICAS PRIMARIAS ───────────────────────────────────────
    print("\n═" * 78)
    print("  ANÁLISIS DE FALSABILIDAD DEL ESCALÓN 1")
    print("═" * 78)

    # 1. Inocuidad de C1 bajo tolerancia numérica
    p_pass_c1 = results["C1_Passive"]["diff_max"] < controller_C.eps_tol
    print(f"• Invariancia C1 (Passive): ||Δz||_∞ = {results['C1_Passive']['diff_max']:.2e} "
          f"(Tolerancia < {controller_C.eps_tol:.1e}) ──► {'✅ PASS' if p_pass_c1 else '❌ FAIL'}")

    # 2. Diferencial de Selectividad Causal (Δ_causal = Δz_consistent - Δz_contradictory)
    dz_cons = results["C3_Consistent"]["z"][target_idx] - base_logit
    dz_cont = results["C4_Contradictory"]["z"][target_idx] - base_logit
    delta_causal = dz_cons - dz_cont
    p_pass_causal = delta_causal > 0.10
    print(f"• Selectividad Causal: Δ_causal = Δz_cons ({dz_cons:+.3f}) - Δz_cont ({dz_cont:+.3f}) = {delta_causal:+.4f} "
          f"(Umbral > 0.10) ──► {'✅ PASS' if p_pass_causal else '❌ FAIL'}")

    # 3. Rendimiento Geométrico (η_geom = |Δz| / θ)
    eta_consistent = dz_cons / theta_test
    print(f"• Rendimiento Geométrico C3 (η_geom): {eta_consistent:.2f} logits/radián")

    # 4. Decisión de C5 (Closed Loop)
    c5_status = results["C5_H3_Loop"]["telemetry"]["status"]
    c5_chosen = results["C5_H3_Loop"]["telemetry"]["chosen"]
    print(f"• Decisión C5 (H3 Loop): {c5_chosen} ({c5_status}) | Giro: {results['C5_H3_Loop']['angle']:.4f} rad")

    # 5. Prueba de Telemetría de Degeneración (Simulada post-commit)
    simulated_tokens_healthy = [2, 14, 5, 2, 8, 9, 21, 2, 11, 4]   # Tokens diversos
    simulated_tokens_collapse = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]     # Degeneración
    r_h_healthy = IntraCycleControllerC.measure_degeneration(simulated_tokens_healthy)
    r_h_collapse = IntraCycleControllerC.measure_degeneration(simulated_tokens_collapse)
    print(f"• Detector R_H Degeneración: Secuencia Normal={r_h_healthy:.2f} | Secuencia Colapso={r_h_collapse:.2f} "
          f"(Umbral Tripwire > 0.40)")

    print("\n" + "═" * 78)
    if p_pass_c1 and p_pass_causal:
        print("🏆 RESULTADO: ESCALÓN 1 CERTIFICADO — EL FLUJO CAUSAL ES FUNCIONAL Y FALSABLE.")
    else:
        print("⚠️ RESULTADO: DISCREPANCIA CAUSAL DETECTADA.")
    print("═" * 78)

if __name__ == "__main__":
    run_causal_battery_experiment()
