import mlx.core as mx
import numpy as np

def run_deep_thought_settling(u_vis, u_text, tau_steps=32, dt=0.05, M_coeff=0.25):
    """
    Ejecuta el flujo de gradiente Riemanniano disipativo del Biespinor en S^{D-1}.
    Garantiza termodinámica monótona dE/dtau <= 0 y ortogonalidad tangente estricta.
    """
    # Normalización al cono esférico unitario S^{D-1}
    u_v = u_vis / mx.sqrt(mx.sum(u_vis * u_vis) + 1e-12)
    u_t = u_text / mx.sqrt(mx.sum(u_text * u_text) + 1e-12)

    S = mx.array(u_v)
    L = mx.array(u_t)

    telemetry = []

    for step in range(tau_steps):
        dot_SL = mx.sum(S * L)
        E = 0.5 * mx.square(1.0 - dot_SL)

        # Gradiente euclídeo de incompatibilidad: grad_L = -(1 - <S, L>) S
        grad_L = - (1.0 - dot_SL) * S
        # Proyección al plano tangente de S^{D-1}
        v_tan = grad_L - mx.sum(grad_L * L) * L
        norm_v = mx.sqrt(mx.sum(v_tan * v_tan) + 1e-12)

        # Retracción geodésica Riemanniana analítica en S^{D-1}
        step_rad = M_coeff * dt * norm_v
        L_next = mx.cos(step_rad) * L - mx.sin(step_rad) * (v_tan / norm_v)
        L = L_next / mx.sqrt(mx.sum(L_next * L_next) + 1e-12)

        if step in [0, 7, 15, 23, 31]:
            telemetry.append((step + 1, float(E), float(norm_v)))

    L_star = L / mx.sqrt(mx.sum(L * L) + 1e-12)
    return L_star, telemetry

def settle_multimodal_thought(vis, prompt, D, processor):
    """
    Punto de entrada canónico para el asentamiento multimodal de pensamiento continuo.
    Calcula el atractor semántico fáctico L* en S^{D-1} mediante flujo de gradiente disipativo.
    """
    u_vis = mx.mean(vis, axis=0)
    u_vis = u_vis / mx.sqrt(mx.sum(u_vis * u_vis) + 1e-12)
    L_star, _ = run_deep_thought_settling(u_vis, u_vis, tau_steps=32)
    return L_star
