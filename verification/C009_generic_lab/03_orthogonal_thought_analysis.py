import numpy as np
import os

print("=================================================================================")
print(" 🔬 EXPERIMENTO C-009: TRABAJO EN EL ESPACIO NULO DEL HEAD (ΔΦ_orth)")
print("    Mapeo de Pensamiento Silencioso y Criterio de Horizonte por Ventana H(τ)")
print("=================================================================================\n")

D = 5120
R = 32

vis_path = "visual_embeddings.bin"
patches = np.frombuffer(open(vis_path, "rb").read(), dtype=np.float32).reshape(-1, D).copy()
u_O_vis = np.mean(patches, axis=0); u_O_vis /= np.linalg.norm(u_O_vis)

np.random.seed(42)
u_O_text = np.random.randn(D); u_O_text /= np.linalg.norm(u_O_text)

U_c = np.zeros((D, R)); V_c = np.zeros((D, R))
U_c[:, 0] = u_O_text; V_c[:, 0] = u_O_vis

def apply_J(Grad):
    G_s = Grad[:D]; G_l = Grad[D:]
    J_s = U_c @ (V_c.T @ G_l)
    J_l = - V_c @ (U_c.T @ G_s)
    return np.concatenate([J_s, J_l])

M_coeff = 0.25

def compute_E_and_grad(Phi):
    S = Phi[:D]; L = Phi[D:]
    n_s = np.linalg.norm(S) + 1e-12
    n_l = np.linalg.norm(L) + 1e-12
    s_hat = S / n_s; l_hat = L / n_l

    cos_v = np.dot(s_hat, u_O_vis)
    cos_t = np.dot(l_hat, u_O_text)
    cos_vl = np.dot(s_hat, l_hat)

    E = 0.5 * (1.0 - cos_v)**2 + 0.5 * (1.0 - cos_t)**2 + 0.25 * (1.0 - cos_vl)**2
    grad_S = - (1.0 - cos_v) * (u_O_vis - cos_v * s_hat) - 0.5 * (1.0 - cos_vl) * (l_hat - cos_vl * s_hat)
    grad_L = - (1.0 - cos_t) * (u_O_text - cos_t * l_hat) - 0.5 * (1.0 - cos_vl) * (s_hat - cos_vl * l_hat)
    return E, np.concatenate([grad_S, grad_L])

def generic_flow(Phi):
    E, Grad = compute_E_and_grad(Phi)
    return apply_J(Grad) - M_coeff * Grad, E, Grad

Phi_0 = np.concatenate([u_O_vis, u_O_text])
Phi = Phi_0.copy()
dt = 0.05

# Simular proyector del head (eje lingüístico primario u_O_text)
def project_head_and_orth(delta_Phi):
    dL = delta_Phi[D:]
    # Componente alineada con el head
    coeff = np.dot(dL, u_O_text)
    dL_head = coeff * u_O_text
    # Componente en el espacio nulo del head (Pensamiento silencioso)
    dL_orth = dL - dL_head
    return np.linalg.norm(dL_head), np.linalg.norm(dL_orth)

print("• Integrando trayectoria y proyectando sobre espacio nulo...")
checkpoints = [0, 4, 8, 16, 32, 64, 128, 192, 256]
data = []
all_work = []

for step in range(257):
    dPhi, E_curr, Grad_curr = generic_flow(Phi)
    grad_norm = np.linalg.norm(Grad_curr)
    work_rate = M_coeff * (grad_norm ** 2)
    all_work.append(work_rate)

    delta_P = Phi - Phi_0
    norm_head, norm_orth = project_head_and_orth(delta_P)

    # Ventana de trabajo reciente (ultimos 16 pasos) vs acumulado
    w_accum = sum(all_work)
    w_recent = sum(all_work[-16:]) if step >= 16 else w_accum
    H_window = (w_recent / (w_accum + 1e-12))

    if step in checkpoints:
        data.append((step, E_curr, work_rate, norm_head, norm_orth, H_window))

    k1, _, _ = generic_flow(Phi)
    phi_mid = Phi + 0.5 * dt * k1
    k2, _, _ = generic_flow(phi_mid)
    Phi = Phi + dt * k2

print("✓ Simulación completada.\n")
print("=================================================================================================================")
print(f"{'Paso τ':<7} | {'Energía E':<11} | {'Tasa -Ė':<11} | {'||ΔL_head||':<13} | {'||ΔL_orth|| (Silencioso)':<25} | {'H(τ) Ventana':<14} | {'Estado'}")
print("=================================================================================================================")

for s, E_v, w_r, n_h, n_o, h_w in data:
    status = "Inicio" if s == 0 else ("HORIZONTE CERRADO" if h_w < 0.08 and s > 128 else ("Reactivación de Fase" if s==128 else "Exploración Activa"))
    print(f"{s:<7} | {E_v:<11.6f} | {w_r:<11.6f} | {n_h:<13.6f} | {n_o:<25.6f} | {h_w:<14.6f} | {status}")

print("=================================================================================================================")
