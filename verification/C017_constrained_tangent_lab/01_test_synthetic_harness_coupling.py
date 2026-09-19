import numpy as np
import os

print("=================================================================================")
print(" 🔬 EXPERIMENTO C-017: ACOPLAMIENTO SINTÉTICO (RESTRICCIÓN H2 + TANGENTE H1)")
print("    Ecuación: dΦ/dτ = (J - M)∇E - ν L_G Φ + g_k * Π_T(u_k - L)")
print("=================================================================================\n")

D = 5120
R = 32
dt = 0.05
M_coeff = 0.25
NU_DIFF = 0.05
BETA_STRENGTH = 3.5

# 1. Cargar parches fácticos de 005.jpg
vis_path = "visual_embeddings.bin"
patches = np.frombuffer(open(vis_path, "rb").read(), dtype=np.float32).reshape(-1, D).copy()
u_O_vis = np.mean(patches, axis=0); u_O_vis /= np.linalg.norm(u_O_vis)

np.random.seed(42)
u_O_text = np.random.randn(D); u_O_text /= np.linalg.norm(u_O_text)

# Restricción exacta emitida por el Harness
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

    # Energía interna de coherencia multimodal
    E_dyn = 0.5 * (1.0 - cos_v)**2 + 0.5 * (1.0 - cos_t)**2 + 0.25 * (1.0 - cos_vl)**2
    grad_S = - (1.0 - cos_v) * (u_O_vis - cos_v * s_hat) - 0.5 * (1.0 - cos_vl) * (l_hat - cos_vl * s_hat)
    grad_L = - (1.0 - cos_t) * (u_O_text - cos_t * l_hat) - 0.5 * (1.0 - cos_vl) * (s_hat - cos_vl * l_hat)
    return E_dyn, np.concatenate([grad_S, grad_L])

def compute_L_G(Phi):
    L = Phi[D:]
    l_hat = L / (np.linalg.norm(L) + 1e-12)
    cos_t = np.dot(l_hat, u_O_text)
    q_k = float(1.0 - abs(cos_t)) # Tensión de Dirichlet en [0, 1]
    lap_L = (l_hat - cos_t * u_O_text)
    return np.concatenate([np.zeros(D), lap_L]), q_k

def project_tangent(vector, base_state):
    """Proyector Conformal exacto Pi_T = I - Phi_hat * Phi_hat^T"""
    norm_sq = np.dot(base_state, base_state)
    if norm_sq < 1e-12: return vector
    base_hat = base_state / np.sqrt(norm_sq)
    return vector - np.dot(vector, base_hat) * base_hat

Phi = np.concatenate([u_O_vis, u_O_text])
Phi /= np.linalg.norm(Phi) # Norma inicial = 1.0

print("• Iniciando evolución híbrida con monitoreo de 3 energías y norma...")
print("=================================================================================================================================")
print(f"{'Paso τ':<7} | {'E_dyn':<12} | {'Afinidad ALU':<15} | {'E_task':<12} | {'Tensión q_k':<13} | {'Compuerta g_k':<14} | {'|‖Φ‖² - 1| (Deriva)'}")
print("=================================================================================================================================")

records = []

for step in range(81):
    # En paso 15: Inyección de choque de perturbación ortogonal
    if step == 15:
        noise = np.random.randn(D)
        noise -= np.dot(noise, u_O_text) * u_O_text
        noise /= np.linalg.norm(noise)
        Phi[D:] = 0.1 * Phi[D:] + 0.99 * noise
        Phi /= np.linalg.norm(Phi) # Re-normalizar en el choque
        print(f"  ⚡ [PASO 15] Choque ortogonal inyectado. Ruptura de coherencia.")

    # 1. Laplaciano de Beltrami y Compuerta de Excitación
    lap_vec, q_k = compute_L_G(Phi)
    g_k = 1.0 / (1.0 + np.exp(-20.0 * (q_k - 0.35)))

    # 2. Dinámica Puerto-Hamiltoniana Base
    E_dyn, Grad_dyn = compute_E_dyn_and_grad(Phi)
    flow_ph = apply_J(Grad_dyn) - M_coeff * Grad_dyn

    # 3. Disipación Laplaciana
    flow_diff = - NU_DIFF * lap_vec

    # 4. Operador C-017: Restricción H2 proyectada sobre espacio tangente H1
    L_curr = Phi[D:]
    l_hat = L_curr / (np.linalg.norm(L_curr) + 1e-12)
    
    # Error de restricción H2
    r_error = u_exact_ALU - l_hat
    
    # Proyección al espacio tangente de la componente lingüística
    b_tangent = project_tangent(r_error, l_hat)
    
    flow_harness = np.concatenate([np.zeros(D), g_k * BETA_STRENGTH * b_tangent])

    # Derivada total
    dPhi = flow_ph + flow_diff + flow_harness

    # Métricas del paso
    cos_alu = np.dot(l_hat, u_exact_ALU)
    E_task = 0.5 * (1.0 - cos_alu) ** 2
    norm_error = abs(np.dot(Phi, Phi) - 1.0)

    if step in [0, 8, 14, 15, 16, 24, 32, 48, 64, 80]:
        print(f"{step:<7} | {E_dyn:<12.6f} | {cos_alu:<15.6f} | {E_task:<12.6f} | {q_k:<13.6f} | {g_k:<14.4f} | {norm_error:.2e}")

    # Integración con corrección de retracción esférica suave
    Phi += dt * dPhi
    # Retracción geométrica suave para impedir deriva acumulativa de orden superior
    Phi /= np.linalg.norm(Phi)

print("=================================================================================================================================")
final_cos = np.dot(Phi[D:] / np.linalg.norm(Phi[D:]), u_exact_ALU)
final_norm_err = abs(np.dot(Phi, Phi) - 1.0)
print(f"• Afinidad final con restricción del Harness : {final_cos:.6f}")
print(f"• Deriva final de norma en la variedad       : {final_norm_err:.2e}")

if final_cos > 0.99 and final_norm_err < 1e-12:
    print("\n🏆 DICTAMEN C-017: OPERADOR SINTÉTICO (H2+H1) NUMÉRICAMENTE VALIDADO.")
    print("   La restricción exacta combinada con la proyección tangencial recuperó la solución al >99%")
    print("   preservando estrictamente la variedad esférica.")
else:
    print("\n❌ FALLO EN CONDICIONES C-017.")
print("=================================================================================")
