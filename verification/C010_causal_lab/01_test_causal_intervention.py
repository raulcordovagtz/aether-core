import numpy as np
import os

print("=================================================================================")
print(" 🔬 EXPERIMENTO C-010: INTERVENCIÓN CONTRAFACTUAL EN τ = 64")
print("    ¿El trabajo silencioso causa la cristalización lingüística posterior?")
print("=================================================================================\n")

D = 5120
R = 32

# 1. Cargar parches reales de 005.jpg
vis_path = "visual_embeddings.bin"
patches = np.frombuffer(open(vis_path, "rb").read(), dtype=np.float32).reshape(-1, D).copy()
u_O_vis = np.mean(patches, axis=0); u_O_vis /= np.linalg.norm(u_O_vis)

np.random.seed(42)
u_O_text = np.random.randn(D); u_O_text /= np.linalg.norm(u_O_text)

U_c = np.zeros((D, R)); V_c = np.zeros((D, R))
U_c[:, 0] = u_O_text; V_c[:, 0] = u_O_vis
M_coeff = 0.25
dt = 0.05

def apply_J(Grad):
    G_s = Grad[:D]; G_l = Grad[D:]
    J_s = U_c @ (V_c.T @ G_l)
    J_l = - V_c @ (U_c.T @ G_s)
    return np.concatenate([J_s, J_l])

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

def generic_flow(Phi):
    E, Grad = compute_E_and_grad(Phi)
    return apply_J(Grad) - M_coeff * Grad, E, Grad

def step_midpoint(P):
    k1, _, _ = generic_flow(P)
    p_mid = P + 0.5 * dt * k1
    k2, _, _ = generic_flow(p_mid)
    return P + dt * k2

# Descomposición ortogonal respecto al eje lingüístico
def decompose(P, P0):
    delta = P - P0
    dL = delta[D:]
    coeff = np.dot(dL, u_O_text)
    dL_par = coeff * u_O_text
    dL_perp = dL - dL_par
    
    delta_par = np.concatenate([np.zeros(D), dL_par])
    delta_perp = np.concatenate([delta[:D], dL_perp]) # Todo S + componente ortogonal de L
    return delta_par, delta_perp, np.linalg.norm(dL_par), np.linalg.norm(dL_perp)

# ─── FASE 1: EVOLUCIÓN CONJUNTA HASTA τ = 64 ──────────────────────────────────
Phi_0 = np.concatenate([u_O_vis, u_O_text])
Phi_64 = Phi_0.copy()

print("• Evolucionando estado basal hasta τ = 64...")
for _ in range(64):
    Phi_64 = step_midpoint(Phi_64)

delta_par_64, delta_perp_64, norm_head_64, norm_orth_64 = decompose(Phi_64, Phi_0)
E_64, _ = compute_E_and_grad(Phi_64)

print(f"✓ Estado en τ = 64 alcanzado:")
print(f"  - Incoherencia E(64)       : {E_64:.6f}")
print(f"  - Componente Head ||ΔL_∥|| : {norm_head_64:.6f}")
print(f"  - Componente Orth ||ΔL_⊥|| : {norm_orth_64:.6f}\n")

# ─── FASE 2: BIFURCACIÓN EN 3 RAMAS CONTRAFACTUALES (τ = 64 -> 256) ───────────
print("• Bifurcando en 3 ramas contrafactuales hasta τ = 256 (192 pasos adicionales)...")

# Rama A: Control Natural (Sin intervención)
Phi_A = Phi_64.copy()

# Rama B: Ablación Silenciosa (Se extirpa delta_perp)
Phi_B = Phi_64 - delta_perp_64

# Rama C: Ablación Visible (Se extirpa delta_par)
Phi_C = Phi_64 - delta_par_64

STEPS_REMAINING = 192

for _ in range(STEPS_REMAINING):
    Phi_A = step_midpoint(Phi_A)
    Phi_B = step_midpoint(Phi_B)
    Phi_C = step_midpoint(Phi_C)

# ─── FASE 3: EVALUACIÓN CAUSAL COMPARATIVA EN τ = 256 ─────────────────────────
def analyze_final(P, name):
    E_fin, G_fin = compute_E_and_grad(P)
    dpar, dperp, n_head, n_orth = decompose(P, Phi_0)
    vel = np.linalg.norm(generic_flow(P)[0])
    return E_fin, n_head, n_orth, vel

E_A, h_A, o_A, v_A = analyze_final(Phi_A, "Rama A (Control)")
E_B, h_B, o_B, v_B = analyze_final(Phi_B, "Rama B (Sin Silencioso)")
E_C, h_C, o_C, v_C = analyze_final(Phi_C, "Rama C (Sin Head)")

print("\n=======================================================================================================")
print(f"{'Rama Experimental':<30} | {'E(256)':<12} | {'||ΔL_head|| (Salida)':<20} | {'||ΔL_orth|| (Interno)':<20} | {'Velocidad ||Φ̇||':<15}")
print("=======================================================================================================")
print(f"{'A: Control Natural (Completo)':<30} | {E_A:<12.6f} | {h_A:<20.6f} | {o_A:<20.6f} | {v_A:<15.6f}")
print(f"{'B: Ablación Silenciosa (-ΔΦ_⊥)':<30} | {E_B:<12.6f} | {h_B:<20.6f} | {o_B:<20.6f} | {v_B:<15.6f}")
print(f"{'C: Ablación Visible (-ΔΦ_∥)':<30} | {E_C:<12.6f} | {h_C:<20.6f} | {o_C:<20.6f} | {v_C:<15.6f}")
print("=======================================================================================================\n")

# Dictamen Causal Estricto
impacto_B = (h_A - h_B) / (h_A + 1e-12) * 100.0
impacto_C = (h_A - h_C) / (h_A + 1e-12) * 100.0

print(f"• Supresión de proyección lingüística al extirpar el trabajo silencioso (Rama B): {impacto_B:+.2f}%")
print(f"• Supresión al extirpar la componente superficial visible (Rama C): {impacto_C:+.2f}%")

if impacto_B > 50.0:
    print("\n🏆 DICTAMEN CAUSAL C-010: DEMOSTRADO.")
    print("   El trabajo en el espacio ortogonal es la CAUSA MOTRIZ de la cristalización en el head.")
else:
    print("\n⚖️ DICTAMEN: Causalidad no dominante o desacoplada.")
print("=================================================================================")
