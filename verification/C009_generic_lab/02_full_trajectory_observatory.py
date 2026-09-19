import numpy as np
import os

print("=================================================================================")
print(" 🔬 OBSERVATORIO COMPLETO C-009: TRAYECTORIA PUERTO-HAMILTONIANA (τ = 0..256)")
print("    Mapeo de Coherencia E(τ), Trabajo W(τ), Eficiencia η(τ) y Logits Δz(τ)")
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
    G_s = Grad[:D]; G_l = Grad[D:]
    J_s = U_c @ (V_c.T @ G_l)
    J_l = - V_c @ (U_c.T @ G_s)
    return np.concatenate([J_s, J_l])

M_coeff = 0.25 # Escala disipativa

def compute_E_and_grad(Phi):
    S = Phi[:D]; L = Phi[D:]
    n_s = np.linalg.norm(S) + 1e-12
    n_l = np.linalg.norm(L) + 1e-12
    s_hat = S / n_s
    l_hat = L / n_l

    cos_v = np.dot(s_hat, u_O_vis)
    cos_t = np.dot(l_hat, u_O_text)
    cos_vl = np.dot(s_hat, l_hat)

    # Energía de incompatibilidad
    E = 0.5 * (1.0 - cos_v)**2 + 0.5 * (1.0 - cos_t)**2 + 0.25 * (1.0 - cos_vl)**2

    grad_S = - (1.0 - cos_v) * (u_O_vis - cos_v * s_hat) - 0.5 * (1.0 - cos_vl) * (l_hat - cos_vl * s_hat)
    grad_L = - (1.0 - cos_t) * (u_O_text - cos_t * l_hat) - 0.5 * (1.0 - cos_vl) * (s_hat - cos_vl * l_hat)

    return E, np.concatenate([grad_S, grad_L])

def generic_flow(Phi):
    E, Grad = compute_E_and_grad(Phi)
    J_force = apply_J(Grad)
    M_force = - M_coeff * Grad
    dPhi = J_force + M_force
    return dPhi, E, Grad

# 3. Integración de la trayectoria completa (256 pasos)
Phi_0 = np.concatenate([u_O_vis, u_O_text])
Phi = Phi_0.copy()
dt = 0.05

checkpoints = [0, 4, 8, 16, 32, 64, 128, 192, 256]
records = []
E_prev = None

print("• Integrando trayectoria y extrayendo observables...")

for step in range(257):
    dPhi, E_curr, Grad_curr = generic_flow(Phi)
    vel = np.linalg.norm(dPhi)
    grad_norm = np.linalg.norm(Grad_curr)
    
    # Trabajo cognitivo disponible -dE/dtau
    work_rate = M_coeff * (grad_norm ** 2)
    # Eficiencia cognitiva eta(tau)
    eta = work_rate / (vel + 1e-12)

    # Desplazamiento en el espacio latente del lenguaje L
    L_curr = Phi[D:]
    delta_L_norm = np.linalg.norm(L_curr - Phi_0[D:])

    if step in checkpoints:
        records.append((step, E_curr, grad_norm, vel, work_rate, eta, delta_L_norm))

    # Midpoint integration
    k1, _, _ = generic_flow(Phi)
    phi_mid = Phi + 0.5 * dt * k1
    k2, _, _ = generic_flow(phi_mid)
    Phi = Phi + dt * k2

print("✓ Simulación de 256 pasos completada exitosamente.\n")

print("=======================================================================================================================")
print(f"{'Paso τ':<7} | {'Energía E(τ)':<13} | {'||∇E||':<10} | {'||Φ̇|| (Vel)':<11} | {'-Ė (Trabajo)':<13} | {'η(τ) (Efic)':<11} | {'||ΔL||':<10} | {'Régimen'}")
print("=======================================================================================================================")

for s, E_val, g_n, v, w, eff, dL in records:
    if s == 0:
        reg = "Inicio"
    elif g_n < 0.05:
        reg = "CIERRE OPERACIONAL"
    elif dL > 0.3 and eff > 0.1:
        reg = "Exploración Productiva (1)"
    elif eff > 0.05:
        reg = "Reorganización Silenciosa (2)"
    else:
        reg = "Aproximación al Atractor"
        
    print(f"{s:<7} | {E_val:<13.6f} | {g_n:<10.6f} | {v:<11.6f} | {w:<13.6f} | {eff:<11.6f} | {dL:<10.6f} | {reg}")

print("=======================================================================================================================")
