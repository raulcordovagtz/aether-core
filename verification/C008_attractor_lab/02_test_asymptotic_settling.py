import numpy as np
import os, time

print("=================================================================================")
print(" 🔬 C-008 LAB: SIMULACIÓN DE ASENTAMIENTO COGNITIVO CON POZO ATRACTOR")
print("=================================================================================\n")

D = 5120
R = 32

# 1. Cargar parches fotónicos reales de 005.jpg
vis_path = "visual_embeddings.bin"
if not os.path.exists(vis_path):
    print("❌ No se encontró visual_embeddings.bin")
    exit(1)

patches = np.frombuffer(open(vis_path, "rb").read(), dtype=np.float32).reshape(-1, D).copy()
u_O_vis = np.mean(patches, axis=0); u_O_vis /= np.linalg.norm(u_O_vis)

np.random.seed(42)
u_O_text = np.random.randn(D); u_O_text /= np.linalg.norm(u_O_text)
u_T_text = np.random.randn(D); u_T_text /= np.linalg.norm(u_T_text)

# 2. Bases y Constantes Analíticas
U_c = np.zeros((D, R)); V_c = np.zeros((D, R))
U_c[:, 0] = u_T_text; V_c[:, 0] = u_O_vis
U_s = np.zeros((D, R)); V_s = np.zeros((D, R))
U_l = np.zeros((D, R)); V_l = np.zeros((D, R))
U_s[:, 0] = u_O_vis; V_s[:, 0] = u_O_vis
U_l[:, 0] = u_O_text; V_l[:, 0] = u_T_text

kappa = 1.0 / np.sqrt(D)
alpha = kappa / 2.0
dt = 1.0 / 64.0
lam = 0.05 # Amortiguamiento crítico del pozo
mu_attractor = 0.15 # Fuerza del pozo atractor de coherencia

def project_perp(F, P):
    n2 = np.dot(P, P)
    return F - (np.dot(F, P) / (n2 + 1e-12)) * P

def dynamic_flow_with_attractor(Phi):
    S = Phi[:D]; L = Phi[D:]
    
    # 1. Rotación de Lie (Conservativa)
    As_S = U_s @ (V_s.T @ S) - V_s @ (U_s.T @ S)
    Al_L = U_l @ (V_l.T @ L) - V_l @ (U_l.T @ L)
    C_L  = U_c @ (V_c.T @ L)
    neg_CT_S = - V_c @ (U_c.T @ S)
    dot_S = As_S + kappa * C_L
    dot_L = kappa * neg_CT_S + Al_L

    # 2. Pozo Atractor de Coherencia (-∇F)
    # Empuja S hacia u_O_vis y L hacia u_T_text tangencialmente
    grad_S = mu_attractor * project_perp(u_O_vis, S)
    grad_L = mu_attractor * project_perp(u_T_text, L)

    # 3. No-linealidad EML
    safe_xs = np.clip(-S, -20.0, 20.0); safe_xl = np.clip(-L, -20.0, 20.0)
    F_eml = np.concatenate([S / (1.0 + np.exp(safe_xs)), L / (1.0 + np.exp(safe_xl))])
    F_tan = project_perp(F_eml, Phi)

    dPhi = np.concatenate([dot_S + grad_S, dot_L + grad_L]) + alpha * F_tan - lam * Phi
    return dPhi

# 3. Evolución Continua
Phi_0 = np.concatenate([u_O_vis, u_O_text])
Phi = Phi_0.copy()

print("• Integrando trayectoria con pozo atractor activo...")
trajectory_vels = []
trajectory_dists = []

for step in range(129):
    vel = np.linalg.norm(dynamic_flow_with_attractor(Phi))
    dist = np.linalg.norm(Phi - Phi_0)
    trajectory_vels.append(vel)
    trajectory_dists.append(dist)

    k1 = dynamic_flow_with_attractor(Phi)
    phi_mid = Phi + 0.5 * dt * k1
    k2 = dynamic_flow_with_attractor(phi_mid)
    Phi = Phi + dt * k2

print("✓ Simulación de asentamiento completada.\n")
print("=======================================================================================================")
print(f"{'Paso τ':<8} | {'Velocidad ||Φ̇||':<18} | {'Distancia ||ΔΦ||':<18} | {'Estado Dinámico':<20}")
print("=======================================================================================================")

for s in [0, 8, 16, 32, 64, 96, 128]:
    v = trajectory_vels[s]
    d = trajectory_dists[s]
    state = "Inicio" if s == 0 else ("CONVERGIDO (Φ*)" if v < 0.08 else "Asentándose...")
    print(f"{s:<8} | {v:<18.6f} | {d:<18.6f} | {state:<20}")

print("=======================================================================================================")
