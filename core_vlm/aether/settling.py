import mlx.core as mx
import numpy as np

def run_deep_thought_settling(u_vis, u_text, tau_steps=32, dt=0.05, M_coeff=0.25):
    """
    Ejecuta los 32 pasos continuos del Biespinor Phi = [S, L] en GPU.
    Dinamica Puerto-Hamiltoniana GENERIC: dPhi/dt = (J - M) grad(E).
    """
    D = u_vis.shape[0]
    
    # Base de bivectores de bajo rango R=32 para el operador simpléctico J
    U_c = mx.zeros((D, 32))
    V_c = mx.zeros((D, 32))
    
    # Inyectar atractores fácticos en el modo principal
    U_c_np = np.zeros((D, 32), dtype=np.float32)
    V_c_np = np.zeros((D, 32), dtype=np.float32)
    U_c_np[:, 0] = np.array(u_text)
    V_c_np[:, 0] = np.array(u_vis)
    U_c = mx.array(U_c_np)
    V_c = mx.array(V_c_np)

    S = u_vis
    L = u_text

    telemetry = []

    for step in range(tau_steps):
        n_s = mx.sqrt(mx.sum(S * S) + 1e-12)
        n_l = mx.sqrt(mx.sum(L * L) + 1e-12)
        s_hat = S / n_s
        l_hat = L / n_l

        cos_v = mx.sum(s_hat * u_vis)
        cos_t = mx.sum(l_hat * u_text)
        cos_vl = mx.sum(s_hat * l_hat)

        # Energía de contradicción E(Phi)
        E = 0.5 * (1.0 - cos_v)**2 + 0.5 * (1.0 - cos_t)**2 + 0.25 * (1.0 - cos_vl)**2
        
        # Gradientes tangenciales
        grad_S = - (1.0 - cos_v) * (u_vis - cos_v * s_hat) - 0.5 * (1.0 - cos_vl) * (l_hat - cos_vl * s_hat)
        grad_L = - (1.0 - cos_t) * (u_text - cos_t * l_hat) - 0.5 * (1.0 - cos_vl) * (s_hat - cos_vl * l_hat)

        # Operador de Poisson J (rotación conservativa entre S y L)
        J_S = mx.matmul(U_c, mx.matmul(V_c.T, grad_L))
        J_L = - mx.matmul(V_c, mx.matmul(U_c.T, grad_S))

        # Flujo total Puerto-Hamiltoniano: (J - M) grad(E)
        flow_S = J_S - M_coeff * grad_S
        flow_L = J_L - M_coeff * grad_L

        S = S + dt * flow_S
        L = L + dt * flow_L

        if step in [0, 7, 15, 23, 31]:
            telemetry.append((step + 1, float(E), float(mx.sqrt(mx.sum(flow_L * flow_L)))))

    # Retornar el estado semántico óptimo asentado L*
    L_star = L / mx.sqrt(mx.sum(L * L) + 1e-12)
    return L_star, telemetry
