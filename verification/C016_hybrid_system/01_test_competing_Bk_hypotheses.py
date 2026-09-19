import numpy as np
import os

print("=================================================================================")
print(" 🔬 EXPERIMENTO C-016: CAREO DE HIPÓTESIS PARA EL OPERADOR B_k DEL HARNESS")
print("    Comparativa: H1 (Corrección) vs H2 (Restricción) vs H3 (Bifurcación Dinámica)")
print("=================================================================================\n")

D = 5120
R = 32
dt = 0.05
M_coeff = 0.25
NU_DIFF = 0.05

# 1. Datos base
vis_path = "visual_embeddings.bin"
patches = np.frombuffer(open(vis_path, "rb").read(), dtype=np.float32).reshape(-1, D).copy()
u_O_vis = np.mean(patches, axis=0); u_O_vis /= np.linalg.norm(u_O_vis)

np.random.seed(42)
u_O_text = np.random.randn(D); u_O_text /= np.linalg.norm(u_O_text)

# Resultado exacto del Harness
u_exact_ALU = np.random.randn(D); u_exact_ALU /= np.linalg.norm(u_exact_ALU)

U_c = np.zeros((D, R)); V_c = np.zeros((D, R))
U_c[:, 0] = u_O_text; V_c[:, 0] = u_O_vis

def apply_J(Grad, J_matrix_Uc=U_c):
    G_s = Grad[:D]; G_l = Grad[D:]
    return np.concatenate([J_matrix_Uc @ (V_c.T @ G_l), - V_c @ (J_matrix_Uc.T @ G_s)])

def compute_E_and_grad(Phi):
    S = Phi[:D]; L = Phi[D:]
    n_s = np.linalg.norm(S) + 1e-12; n_l = np.linalg.norm(L) + 1e-12
    s_hat = S / n_s; l_hat = L / n_l

    cos_v = np.dot(s_hat, u_O_vis)
    cos_t = np.dot(l_hat, u_O_text)
    cos_vl = np.dot(s_hat, l_hat)

    E = 0.5 * (1.0 - cos_v)**2 + 0.5 * (1.0 - cos_t)**2 + 0.25 * (1.0 - cos_vl)**2
    grad_S = - (1.0 - cos_v) * (u_O_vis - cos_v * s_hat) - 0.5 * (1.0 - cos_vl) * (l_hat - cos_vl * s_hat)
    grad_L = - (1.0 - cos_t) * (u_O_text - cos_t * l_hat) - 0.5 * (1.0 - cos_vl) * (s_hat - cos_vl * l_hat)
    return E, np.concatenate([grad_S, grad_L])

def compute_L_G(Phi):
    # Laplaciano de Beltrami L_G >= 0
    L = Phi[D:]
    l_hat = L / (np.linalg.norm(L) + 1e-12)
    cos_t = np.dot(l_hat, u_O_text)
    # Tensión del canal
    q_arith = float(1.0 - abs(cos_t))
    lap_L = (l_hat - cos_t * u_O_text) # L_G >= 0
    lap_vec = np.concatenate([np.zeros(D), lap_L])
    return lap_vec, q_arith

# Función de simulación para cada hipótesis
def run_simulation(hypothesis_mode):
    Phi = np.concatenate([u_O_vis, u_O_text])
    
    # Inyectar choque en paso 15
    for step in range(80):
        if step == 15:
            noise = np.random.randn(D)
            noise -= np.dot(noise, u_O_text) * u_O_text
            noise /= np.linalg.norm(noise)
            Phi[D:] = 0.2 * Phi[D:] + 0.98 * noise
            Phi[D:] /= np.linalg.norm(Phi[D:])

        lap_vec, q_k = compute_L_G(Phi)
        # Compuerta sigmoidal limpia g_k
        g_k = 1.0 / (1.0 + np.exp(-20.0 * (q_k - 0.35)))

        E_curr, Grad_curr = compute_E_and_grad(Phi)
        flow_ph = apply_J(Grad_curr) - M_coeff * Grad_curr
        flow_diff = - NU_DIFF * lap_vec # Disipación -nu * L_G

        # Evaluación de las 3 hipótesis para B_k
        L_curr = Phi[D:] / (np.linalg.norm(Phi[D:]) + 1e-12)
        
        if hypothesis_mode == "H1_Correction":
            # B_k = beta * (u_exact - L) proyectado tangencial
            cos_sol = np.dot(L_curr, u_exact_ALU)
            b_k = 3.0 * (u_exact_ALU - cos_sol * L_curr)
            flow_Bk = np.concatenate([np.zeros(D), g_k * b_k])
            
        elif hypothesis_mode == "H2_Constraint":
            # B_k proyecta ortogonalmente hacia el subespacio donde <L, u_exact> = 1
            # Proyección directa de Gram-Schmidt
            b_k = (u_exact_ALU - L_curr)
            flow_Bk = np.concatenate([np.zeros(D), g_k * 4.0 * b_k])
            
        elif hypothesis_mode == "H3_Bifurcation":
            # B_k no añade fuerza aditiva; conmuta el operador de Lie J hacia la base del ALU
            if g_k > 0.5:
                # El conmutador de Lie ahora rota hacia la solución exacta
                U_c_mutated = U_c.copy()
                U_c_mutated[:, 0] = u_exact_ALU
                flow_ph = apply_J(Grad_curr, U_c_mutated) - M_coeff * Grad_curr
            flow_Bk = np.zeros(2 * D)

        dPhi = flow_ph + flow_diff + flow_Bk
        Phi += dt * dPhi

    L_final = Phi[D:] / np.linalg.norm(Phi[D:])
    final_affinity = np.dot(L_final, u_exact_ALU)
    E_final, _ = compute_E_and_grad(Phi)
    return final_affinity, E_final

print("• Ejecutando careo experimental de las 3 hipótesis...")
aff_H1, E_H1 = run_simulation("H1_Correction")
aff_H2, E_H2 = run_simulation("H2_Constraint")
aff_H3, E_H3 = run_simulation("H3_Bifurcation")

print("=======================================================================================================")
print(f"{'Hipótesis del Operador B_k':<35} | {'Afinidad con Verdad u_ALU':<28} | {'Energía Residual E':<20}")
print("=======================================================================================================")
print(f"{'H1: Corrección Aditiva Tangencial':<35} | {aff_H1:<28.6f} | {E_H1:<20.6f}")
print(f"{'H2: Restricción / Constraint Solver':<35} | {aff_H2:<28.6f} | {E_H2:<20.6f}")
print(f"{'H3: Bifurcación Dinámica (J conmutado)':<35} | {aff_H3:<28.6f} | {E_H3:<20.6f}")
print("=======================================================================================================\n")

if aff_H1 > aff_H3 and aff_H1 > aff_H2:
    print("🏆 DICTAMEN: H1 (Corrección Tangencial) es la formulación más estable y equilibrada.")
elif aff_H2 > aff_H1:
    print("🏆 DICTAMEN: H2 (Restricción Estricta) domina la convergencia.")
else:
    print("🏆 DICTAMEN: H3 (Bifurcación Dinámica) es la vía superior.")
print("=================================================================================")
