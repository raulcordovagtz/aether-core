import numpy as np
import os, struct, time

print("=================================================================================")
print(" 🔬 EXPERIMENTO C-008: ASENTAMIENTO DE TRAYECTORIA LATENTE (τ = 0..256)")
print("=================================================================================\n")

D = 5120
R = 32

# 1. Cargar parches fotónicos reales de 005.jpg
vis_path = "visual_embeddings.bin"
if not os.path.exists(vis_path):
    print("❌ No se encontró visual_embeddings.bin")
    exit(1)

patches = np.frombuffer(open(vis_path, "rb").read(), dtype=np.float32).reshape(-1, D).copy()
num_p = patches.shape[0]

u_O_vis = np.mean(patches, axis=0)
u_O_vis /= np.linalg.norm(u_O_vis)

# 2. Vectores de Intención y Bases de Lie
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
alpha = kappa / 2.0
dt = 1.0 / 64.0
lam = kappa * (dt**2)

def flow(Phi):
    S = Phi[:D]; L = Phi[D:]
    As_S = U_s @ (V_s.T @ S) - V_s @ (U_s.T @ S)
    Al_L = U_l @ (V_l.T @ L) - V_l @ (U_l.T @ L)
    C_L  = U_c @ (V_c.T @ L)
    neg_CT_S = - V_c @ (U_c.T @ S)

    dot_S = As_S + kappa * C_L
    dot_L = kappa * neg_CT_S + Al_L

    # EML no lineal
    safe_xs = np.clip(-S, -20.0, 20.0)
    safe_xl = np.clip(-L, -20.0, 20.0)
    feml_s = S / (1.0 + np.exp(safe_xs))
    feml_l = L / (1.0 + np.exp(safe_xl))
    F_eml = np.concatenate([feml_s, feml_l])

    # Pi_perp
    norm_sq = np.dot(Phi, Phi)
    proj = np.dot(F_eml, Phi) / (norm_sq + 1e-12)
    F_tan = F_eml - proj * Phi

    dPhi = np.concatenate([dot_S, dot_L]) + alpha * F_tan - lam * Phi
    return dPhi

# 3. Simulación de la evolución continua
Phi_0 = np.concatenate([u_O_vis, u_O_text])
Phi = Phi_0.copy()

checkpoints = [0, 16, 64, 128, 256]
trajectory = {}

print("• Integrando trayectoria continua del biespinor hasta τ = 256...")
step_count = 0
for step in range(257):
    if step in checkpoints:
        trajectory[step] = Phi.copy()
    k1 = flow(Phi)
    phi_mid = Phi + 0.5 * dt * k1
    k2 = flow(phi_mid)
    Phi = Phi + dt * k2

print("✓ Trayectoria integrada exitosamente.")

# 4. Análisis de Asentamiento y Métrica C_Phi
print("\n=======================================================================================================")
print(f"{'Paso τ':<8} | {'||ΔΦ||_2 (Interno)':<18} | {'||ΔL||_2 (Lenguaje)':<18} | {'Velocidad ||Φ̇||':<16} | {'Estado':<15}")
print("=======================================================================================================")

for s in checkpoints:
    P_s = trajectory[s]
    delta_Phi = np.linalg.norm(P_s - Phi_0)
    delta_L = np.linalg.norm(P_s[D:] - Phi_0[D:])
    vel = np.linalg.norm(flow(P_s))
    
    status = "Inicio" if s == 0 else ("Asentándose" if vel < 0.05 else "Transitorio Activo")
    print(f"{s:<8} | {delta_Phi:<18.6f} | {delta_L:<18.6f} | {vel:<16.6f} | {status:<15}")

print("=======================================================================================================")
