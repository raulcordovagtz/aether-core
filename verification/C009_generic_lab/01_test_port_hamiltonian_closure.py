import numpy as np
import os

print("=================================================================================")
print(" 🔬 EXPERIMENTO C-009: DINÁMICA PUERTO-HAMILTONIANA Y CIERRE DE HORIZONTE")
print("    Ecuación: dΦ/dτ = (J - M) ∇E(Φ) con J^T = -J, M ≥ 0")
print("=================================================================================\n")

D = 5120
R = 32

# 1. Cargar parches fotónicos reales de 005.jpg
vis_path = "visual_embeddings.bin"
patches = np.frombuffer(open(vis_path, "rb").read(), dtype=np.float32).reshape(-1, D).copy()
u_O_vis = np.mean(patches, axis=0); u_O_vis /= np.linalg.norm(u_O_vis)

np.random.seed(42)
u_O_text = np.random.randn(D); u_O_text /= np.linalg.norm(u_O_text)

# 2. Operador Antisimétrico de Poisson J (r=32)
U_c = np.zeros((D, R)); V_c = np.zeros((D, R))
U_c[:, 0] = u_O_text; V_c[:, 0] = u_O_vis

def apply_J(Grad):
    """Aplica J @ Grad en O(D*r). J^T = -J exacto."""
    G_s = Grad[:D]; G_l = Grad[D:]
    J_s = U_c @ (V_c.T @ G_l)
    J_l = - V_c @ (U_c.T @ G_s)
    return np.concatenate([J_s, J_l])

# Coeficiente de disipación M (Métrica riemanniana positiva)
M_coeff = 0.25 # Escala de relajación de incoherencia

def compute_energy_and_gradient(Phi):
    """
    Funcional de Coherencia E(S, L) = 0.5*(1 - <S, u_vis>)^2 + 0.5*(1 - <L, u_txt>)^2 + 0.5*gamma*(1 - <S, L>)^2
    """
    S = Phi[:D]; L = Phi[D:]
    n_s = np.linalg.norm(S) + 1e-12
    n_l = np.linalg.norm(L) + 1e-12
    s_hat = S / n_s
    l_hat = L / n_l

    cos_v = np.dot(s_hat, u_O_vis)
    cos_t = np.dot(l_hat, u_O_text)
    cos_vl = np.dot(s_hat, l_hat)

    # Energía escalar de incompatibilidad
    E = 0.5 * (1.0 - cos_v)**2 + 0.5 * (1.0 - cos_t)**2 + 0.25 * (1.0 - cos_vl)**2

    # Gradientes euclidianos tangenciales
    grad_S = - (1.0 - cos_v) * (u_O_vis - cos_v * s_hat) - 0.5 * (1.0 - cos_vl) * (l_hat - cos_vl * s_hat)
    grad_L = - (1.0 - cos_t) * (u_O_text - cos_t * l_hat) - 0.5 * (1.0 - cos_vl) * (s_hat - cos_vl * l_hat)

    Grad_E = np.concatenate([grad_S, grad_L])
    return E, Grad_E

def generic_flow(Phi):
    E, Grad_E = compute_energy_and_gradient(Phi)
    
    # 1. Movimiento conservativo ortogonal al gradiente (J @ Grad_E)
    conserv_force = apply_J(Grad_E)
    
    # 2. Resolución disipativa pura (-M @ Grad_E)
    dissip_force = - M_coeff * Grad_E
    
    dPhi = conserv_force + dissip_force
    return dPhi, E, Grad_E

# 3. Evolución Continua
Phi_0 = np.concatenate([u_O_vis, u_O_text])
Phi = Phi_0.copy()
dt = 0.1 # Paso temporal

print("• Integrando flujo Puerto-Hamiltoniano hasta τ = 128 pasos...")
tau_records = []

for step in range(129):
    dPhi, E_val, Grad_E = generic_flow(Phi)
    vel = np.linalg.norm(dPhi)
    grad_norm = np.linalg.norm(Grad_E)
    
    if step in [0, 8, 16, 32, 64, 96, 128]:
        tau_records.append((step, E_val, grad_norm, vel))

    # Integrador Midpoint
    k1, _, _ = generic_flow(Phi)
    phi_mid = Phi + 0.5 * dt * k1
    k2, _, _ = generic_flow(phi_mid)
    Phi = Phi + dt * k2

print("✓ Simulación completada.\n")
print("=======================================================================================================")
print(f"{'Paso τ':<8} | {'Energía E(τ)':<16} | {'||∇E|| (Gradiente)':<20} | {'Velocidad ||Φ̇||':<18} | {'Estado':<20}")
print("=======================================================================================================")

for s, E_val, g_norm, v in tau_records:
    # Identificar régimen
    if s == 0:
        st = "Inicio"
    elif g_norm < 1e-3:
        st = "CIERRE NATURAL (Φ*)"
    elif g_norm < 0.1:
        st = "Asentándose (B)"
    else:
        st = "Descubrimiento (A)"
    print(f"{s:<8} | {E_val:<16.8f} | {g_norm:<20.8f} | {v:<18.8f} | {st:<20}")

print("=======================================================================================================")
