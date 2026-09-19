import numpy as np
import os

print("=================================================================================")
print(" 🔬 EXPERIMENTO C-018: RETRACCIÓN GEODÉSICA DE RIEMANN Y EXTINCIÓN DE FUERZA")
print("    Preservación analítica ||Φ||=1 vía mapa exponencial sin truncamiento manual")
print("=================================================================================\n")

D = 5120
R = 32
dt = 0.05
M_coeff = 0.25
BETA_STRENGTH = 4.0

vis_path = "visual_embeddings.bin"
patches = np.frombuffer(open(vis_path, "rb").read(), dtype=np.float32).reshape(-1, D).copy()
u_O_vis = np.mean(patches, axis=0); u_O_vis /= np.linalg.norm(u_O_vis)

np.random.seed(42)
u_O_text = np.random.randn(D); u_O_text /= np.linalg.norm(u_O_text)
u_exact_ALU = np.random.randn(D); u_exact_ALU /= np.linalg.norm(u_exact_ALU)

U_c = np.zeros((D, R)); V_c = np.zeros((D, R))
U_c[:, 0] = u_O_text; V_c[:, 0] = u_O_vis

def apply_J(Grad):
    G_s = Grad[:D]; G_l = Grad[D:]
    return np.concatenate([U_c @ (V_c.T @ G_l), - V_c @ (U_c.T @ G_s)])

def compute_E_dyn_and_grad(Phi):
    S = Phi[:D]; L = Phi[D:]
    n_s = np.linalg.norm(S) + 1e-12; n_l = np.linalg.norm(L) + 1e-12
    s_hat = S / n_s; l_hat = L / n_l

    cos_v = np.dot(s_hat, u_O_vis)
    cos_t = np.dot(l_hat, u_O_text)
    cos_vl = np.dot(s_hat, l_hat)

    E_dyn = 0.5 * (1.0 - cos_v)**2 + 0.5 * (1.0 - cos_t)**2 + 0.25 * (1.0 - cos_vl)**2
    grad_S = - (1.0 - cos_v) * (u_O_vis - cos_v * s_hat) - 0.5 * (1.0 - cos_vl) * (l_hat - cos_vl * s_hat)
    grad_L = - (1.0 - cos_t) * (u_O_text - cos_t * l_hat) - 0.5 * (1.0 - cos_vl) * (s_hat - cos_vl * l_hat)
    return E_dyn, np.concatenate([grad_S, grad_L])

def project_tangent(v, p):
    """Proyector Conformal exacto Pi_T = I - p p^T"""
    return v - np.dot(v, p) * p

Phi = np.concatenate([u_O_vis, u_O_text])
Phi /= np.linalg.norm(Phi)

print("• Iniciando evolución con Integración Geodésica Riemanniana (Lie Exp-Map)...")
print("=====================================================================================================================================")
print(f"{'Paso τ':<7} | {'Afinidad ALU':<14} | {'E_task':<12} | {'||B_k|| (Fuerza)':<17} | {'W_k (Trabajo Útil)':<20} | {'|‖Φ‖² - 1| (Deriva Riemann)'}")
print("=====================================================================================================================================")

for step in range(161):
    # En paso 20: Choque ortogonal
    if step == 20:
        noise = np.random.randn(D)
        noise -= np.dot(noise, u_O_text) * u_O_text
        noise /= np.linalg.norm(noise)
        Phi[D:] = 0.1 * Phi[D:] + 0.99 * noise
        Phi /= np.linalg.norm(Phi) # Reanudar condición inicial de choque en la esfera
        print(f"  ⚡ [PASO 20] Choque ortogonal inyectado.")

    L_curr = Phi[D:]
    l_hat = L_curr / (np.linalg.norm(L_curr) + 1e-12)
    cos_alu = np.dot(l_hat, u_exact_ALU)
    E_task = 0.5 * (1.0 - cos_alu) ** 2

    # Vector de corrección efectiva B_k y Trabajo W_k
    r_error = u_exact_ALU - l_hat
    B_k = project_tangent(r_error, l_hat)
    norm_Bk = np.linalg.norm(B_k)
    
    grad_task = - (1.0 - cos_alu) * (u_exact_ALU - cos_alu * l_hat)
    W_k = abs(np.dot(grad_task, B_k))

    # Compuerta dinámica de activación
    q_k = float(1.0 - cos_alu)
    g_k = 1.0 / (1.0 + np.exp(-20.0 * (q_k - 0.10)))

    # Flujo continuo
    E_dyn, Grad_dyn = compute_E_dyn_and_grad(Phi)
    flow_ph = apply_J(Grad_dyn) - M_coeff * Grad_dyn
    flow_harness = np.concatenate([np.zeros(D), g_k * BETA_STRENGTH * B_k])

    # Derivada total
    v_tangent = flow_ph + flow_harness
    # Asegurar ortogonalidad estricta con el estado actual
    v_tangent = project_tangent(v_tangent, Phi)
    v_norm = np.linalg.norm(v_tangent)

    norm_riemann_err = abs(np.dot(Phi, Phi) - 1.0)

    if step in [0, 10, 19, 20, 25, 40, 60, 80, 120, 160]:
        print(f"{step:<7} | {cos_alu:<14.6f} | {E_task:<12.6f} | {norm_Bk:<17.6f} | {W_k:<20.8f} | {norm_riemann_err:.4e}")

    # ─── RETRACCIÓN GEODÉSICA EXPONENCIAL (PRESERVACIÓN ANALÍTICA DE S^{D-1}) ─────
    if v_norm > 1e-12:
        theta = dt * v_norm
        Phi = np.cos(theta) * Phi + np.sin(theta) * (v_tangent / v_norm)

final_cos = np.dot(Phi[D:] / np.linalg.norm(Phi[D:]), u_exact_ALU)
final_Bk = norm_Bk
final_Wk = W_k
final_norm_err = abs(np.dot(Phi, Phi) - 1.0)

print("=====================================================================================================================================")
print(f"• Afinidad final con solución         : {final_cos:.6f}")
print(f"• Norma residual de corrección ||B_k||  : {final_Bk:.8f} (Extinción de fuerza)")
print(f"• Trabajo útil residual W_k            : {final_Wk:.10f} (Extinción de trabajo)")
print(f"• Deriva analítica de norma de Riemann : {final_norm_err:.4e}")

if final_Bk < 0.05 and final_Wk < 1e-4 and final_norm_err < 1e-12:
    print("\n🏆 DICTAMEN C-018: INTEGRACIÓN RIEMANNIANA Y EXTINCIÓN DE FUERZA CERTIFICADAS.")
    print("   El mapa exponencial mantuvo la norma en 1.0 sin renormalizar, y el trabajo W_k colapsó a cero.")
else:
    print("\n⚖️ ANÁLISIS DE RESIDUO FINAL.")
print("=================================================================================")
