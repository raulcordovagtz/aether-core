import numpy as np
import os

print("=================================================================================")
print(" 🔬 C-008 LAB: ASENTAMIENTO CRÍTICO MEDIANTE RELAJACIÓN DE LIE K(τ)")
print("=================================================================================\n")

D = 5120
R = 32

vis_path = "visual_embeddings.bin"
patches = np.frombuffer(open(vis_path, "rb").read(), dtype=np.float32).reshape(-1, D).copy()
u_O_vis = np.mean(patches, axis=0); u_O_vis /= np.linalg.norm(u_O_vis)

np.random.seed(42)
u_O_text = np.random.randn(D); u_O_text /= np.linalg.norm(u_O_text)
u_T_text = np.random.randn(D); u_T_text /= np.linalg.norm(u_T_text)

U_c = np.zeros((D, R)); V_c = np.zeros((D, R))
U_c[:, 0] = u_T_text; V_c[:, 0] = u_O_vis
U_s = np.zeros((D, R)); V_s = np.zeros((D, R))
U_l = np.zeros((D, R)); V_l = np.zeros((D, R))
U_s[:, 0] = u_O_vis; V_s[:, 0] = u_O_vis
U_l[:, 0] = u_O_text; V_l[:, 0] = u_T_text

kappa = 1.0 / np.sqrt(D)
dt = 1.0 / 64.0
mu_attractor = 0.25
tau_relax = 24.0 # Constante de tiempo de relajación de Lie

def project_perp(F, P):
    n2 = np.dot(P, P)
    return F - (np.dot(F, P) / (n2 + 1e-12)) * P

def dynamic_flow_critical(Phi, step_idx):
    S = Phi[:D]; L = Phi[D:]
    
    # 1. Torsión de Lie con relajación asintótica (Exploración -> Asentamiento)
    lie_decay = np.exp(-float(step_idx) / tau_relax)
    
    As_S = U_s @ (V_s.T @ S) - V_s @ (U_s.T @ S)
    Al_L = U_l @ (V_l.T @ L) - V_l @ (U_l.T @ L)
    C_L  = U_c @ (V_c.T @ L)
    neg_CT_S = - V_c @ (U_c.T @ S)
    
    dot_S = lie_decay * (As_S + kappa * C_L)
    dot_L = lie_decay * (kappa * neg_CT_S + Al_L)

    # 2. Pozo Atractor de Coherencia
    grad_S = mu_attractor * project_perp(u_O_vis, S)
    grad_L = mu_attractor * project_perp(u_T_text, L)

    # 3. Disipación crítica hacia el atractor
    dPhi = np.concatenate([dot_S + grad_S, dot_L + grad_L])
    return dPhi

Phi_0 = np.concatenate([u_O_vis, u_O_text])
Phi = Phi_0.copy()

print("• Integrando trayectoria con Relajación de Lie...")
trajectory_vels = []
trajectory_dists = []
alignments_L = []

for step in range(129):
    flow_vec = dynamic_flow_critical(Phi, step)
    vel = np.linalg.norm(flow_vec)
    dist = np.linalg.norm(Phi - Phi_0)
    L_curr = Phi[D:]
    cos_align = np.dot(L_curr, u_T_text) / (np.linalg.norm(L_curr) * np.linalg.norm(u_T_text))

    trajectory_vels.append(vel)
    trajectory_dists.append(dist)
    alignments_L.append(cos_align)

    k1 = dynamic_flow_critical(Phi, step)
    phi_mid = Phi + 0.5 * dt * k1
    k2 = dynamic_flow_critical(phi_mid, step + 0.5)
    Phi = Phi + dt * k2

print("✓ Simulación completada.\n")
print("=======================================================================================================")
print(f"{'Paso τ':<8} | {'Velocidad ||Φ̇||':<18} | {'Alineación Texto (Cos θ)':<25} | {'Estado Dinámico':<20}")
print("=======================================================================================================")

for s in [0, 8, 16, 32, 64, 96, 128]:
    v = trajectory_vels[s]
    cos_t = alignments_L[s]
    state = "Inicio" if s == 0 else ("CONVERGIDO (Φ*)" if v < 0.05 else "Asentándose...")
    print(f"{s:<8} | {v:<18.6f} | {cos_t:<25.6f} | {state:<20}")

print("=======================================================================================================")
