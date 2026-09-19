import numpy as np
import os

print("=================================================================================")
print(" 🔬 EXPERIMENTO C-011: TRASPLANTE SEMÁNTICO EN ESPACIO NULO")
print("    ¿El trabajo silencioso transporta información causal entre tareas?")
print("=================================================================================\n")

D = 5120
R = 32

# 1. Cargar imagen base
vis_path = "visual_embeddings.bin"
patches = np.frombuffer(open(vis_path, "rb").read(), dtype=np.float32).reshape(-1, D).copy()
u_O_vis = np.mean(patches, axis=0); u_O_vis /= np.linalg.norm(u_O_vis)

# 2. Crear DOS queries ortogonales
np.random.seed(42)
u_A = np.random.randn(D); u_A /= np.linalg.norm(u_A)
# Hacer u_B ortogonal a u_A (dot product = 0)
u_B_raw = np.random.randn(D)
u_B = u_B_raw - np.dot(u_B_raw, u_A) * u_A
u_B /= np.linalg.norm(u_B)

assert abs(np.dot(u_A, u_B)) < 1e-10, "Queries no son ortogonales"

M_coeff = 0.25
dt = 0.05

def build_operators(u_query):
    Uc = np.zeros((D, R)); Vc = np.zeros((D, R))
    Uc[:, 0] = u_query; Vc[:, 0] = u_O_vis
    return Uc, Vc

Uc_A, Vc_A = build_operators(u_A)
Uc_B, Vc_B = build_operators(u_B)

def compute_E_and_grad(Phi, u_q):
    S = Phi[:D]; L = Phi[D:]
    n_s = np.linalg.norm(S) + 1e-12; n_l = np.linalg.norm(L) + 1e-12
    s_hat = S / n_s; l_hat = L / n_l

    cos_v = np.dot(s_hat, u_O_vis)
    cos_t = np.dot(l_hat, u_q)
    cos_vl = np.dot(s_hat, l_hat)

    E = 0.5 * (1.0 - cos_v)**2 + 0.5 * (1.0 - cos_t)**2 + 0.25 * (1.0 - cos_vl)**2
    grad_S = - (1.0 - cos_v) * (u_O_vis - cos_v * s_hat) - 0.5 * (1.0 - cos_vl) * (l_hat - cos_vl * s_hat)
    grad_L = - (1.0 - cos_t) * (u_q - cos_t * l_hat) - 0.5 * (1.0 - cos_vl) * (s_hat - cos_vl * l_hat)
    return E, np.concatenate([grad_S, grad_L])

def generic_flow(Phi, u_q, Uc, Vc):
    E, Grad = compute_E_and_grad(Phi, u_q)
    G_s = Grad[:D]; G_l = Grad[D:]
    J_s = Uc @ (Vc.T @ G_l)
    J_l = - Vc @ (Uc.T @ G_s)
    J_force = np.concatenate([J_s, J_l])
    return J_force - M_coeff * Grad

def step_midpoint(P, u_q, Uc, Vc):
    k1 = generic_flow(P, u_q, Uc, Vc)
    p_mid = P + 0.5 * dt * k1
    k2 = generic_flow(p_mid, u_q, Uc, Vc)
    return P + dt * k2

def get_perp(Phi_curr, Phi_init, u_q):
    delta = Phi_curr - Phi_init
    dL = delta[D:]
    dL_par = np.dot(dL, u_q) * u_q
    dL_perp = dL - dL_par
    return np.concatenate([delta[:D], dL_perp])

# ─── FASE 1: EVOLUCIÓN AISLADA HASTA τ = 64 ──────────────────────────────────
Phi_A = np.concatenate([u_O_vis, u_A])
Phi_B = np.concatenate([u_O_vis, u_B])
Phi_0_A, Phi_0_B = Phi_A.copy(), Phi_B.copy()

print("• Evolucionando ambos queries hasta τ = 64...")
for _ in range(64):
    Phi_A = step_midpoint(Phi_A, u_A, Uc_A, Vc_A)
    Phi_B = step_midpoint(Phi_B, u_B, Uc_B, Vc_B)

perp_A = get_perp(Phi_A, Phi_0_A, u_A)
perp_B = get_perp(Phi_B, Phi_0_B, u_B)

# ─── FASE 2: INTERVENCIÓN Y TRASPLANTE EN τ = 64 ─────────────────────────────
# Al sistema A le quitamos su pensamiento (perp_A) y le inyectamos el de B (perp_B)
Phi_Hybrid = Phi_A - perp_A + perp_B

print("✓ Trasplante de estado silencioso (B -> A) completado.")

# ─── FASE 3: CONTINUACIÓN DE LA DINÁMICA A HASTA τ = 256 ─────────────────────
# El híbrido corre con la meta de A y los operadores de A
for _ in range(192):
    Phi_A = step_midpoint(Phi_A, u_A, Uc_A, Vc_A)
    Phi_Hybrid = step_midpoint(Phi_Hybrid, u_A, Uc_A, Vc_A)

# ─── FASE 4: ANÁLISIS DE LA CONTAMINACIÓN SEMÁNTICA ──────────────────────────
# Evaluamos cuánto "sabe" el híbrido sobre la meta B en comparación con la rama A pura
final_L_A = Phi_A[D:]
final_L_Hybrid = Phi_Hybrid[D:]

cos_A_pure_vs_B = np.dot(final_L_A, u_B) / (np.linalg.norm(final_L_A) * np.linalg.norm(u_B))
cos_Hybrid_vs_B = np.dot(final_L_Hybrid, u_B) / (np.linalg.norm(final_L_Hybrid) * np.linalg.norm(u_B))

E_A_pure_vs_B, _ = compute_E_and_grad(Phi_A, u_B)
E_Hybrid_vs_B, _ = compute_E_and_grad(Phi_Hybrid, u_B)

print("\n=======================================================================================================")
print(f"{'Métrica':<45} | {'Control A (Puro)':<20} | {'A Híbrido (Injerto B)':<20}")
print("=======================================================================================================")
print(f"{'Alineación (Cos θ) con la Meta B oculta':<45} | {cos_A_pure_vs_B:<20.6f} | {cos_Hybrid_vs_B:<20.6f}")
print(f"{'Energía E respecto a la Meta B oculta':<45} | {E_A_pure_vs_B:<20.6f} | {E_Hybrid_vs_B:<20.6f}")
print("=======================================================================================================\n")

if cos_Hybrid_vs_B > (cos_A_pure_vs_B + 0.05):
    print("🏆 DICTAMEN C-011: CAUSALIDAD SEMÁNTICA TRANSPORTABLE DEMOSTRADA.")
    print("   El componente silencioso inyectó información direccional del Query B en la trayectoria de A.")
else:
    print("❌ DICTAMEN: El espacio ortogonal es un artefacto geométrico sin transporte de información semántica.")
print("=================================================================================")
