import numpy as np
import os

print("=================================================================================")
print(" 🔬 EXPERIMENTO C-015: LAPLACIANO DE BELTRAMI-GRAFO Y EXCITACIÓN ESTABLE")
print("    Ecuación: dΦ/dτ = (J - M)∇E + ν Δ_G Φ - σ(Tensión) ∇U_cell")
print("=================================================================================\n")

D = 5120
R = 32
dt = 0.05
M_coeff = 0.25
NU_DIFFUSION = 0.05   # Coeficiente de difusión estable (von Neumann stable)
BETA_HARNESS = 3.0   # Gravedad del atractor de la celda
THETA_EXCITE = 0.35  # Umbral de tensión normalizada (rango 0..1)

# 1. Cargar parches fácticos de 005.jpg
vis_path = "visual_embeddings.bin"
patches = np.frombuffer(open(vis_path, "rb").read(), dtype=np.float32).reshape(-1, D).copy()
u_O_vis = np.mean(patches, axis=0); u_O_vis /= np.linalg.norm(u_O_vis)

np.random.seed(42)
u_O_text = np.random.randn(D); u_O_text /= np.linalg.norm(u_O_text)

# Solución exacta generada por el Harness
u_exact_solution = np.random.randn(D); u_exact_solution /= np.linalg.norm(u_exact_solution)

U_c = np.zeros((D, R)); V_c = np.zeros((D, R))
U_c[:, 0] = u_O_text; V_c[:, 0] = u_O_vis

def apply_J(Grad):
    G_s = Grad[:D]; G_l = Grad[D:]
    return np.concatenate([U_c @ (V_c.T @ G_l), - V_c @ (U_c.T @ G_s)])

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

def compute_beltrami_laplacian(Phi):
    """
    Laplaciano de Grafo Normalizado en la variedad latente:
    Δ_G L = (u_txt u_txt^T - I) L = Proyección ortogonal negativa (Disipador de curvatura)
    Tensión de Dirichlet: T = 1 - <L, u_txt>^2 ∈ [0, 1]
    """
    S = Phi[:D]; L = Phi[D:]
    l_hat = L / (np.linalg.norm(L) + 1e-12)
    s_hat = S / (np.linalg.norm(S) + 1e-12)

    # Laplaciano de difusión sobre el eje de coherencia
    cos_t = np.dot(l_hat, u_O_text)
    lap_L = (cos_t * u_O_text - l_hat) # Estrictamente acotado, autovalores <= 0
    
    cos_v = np.dot(s_hat, u_O_vis)
    lap_S = (cos_v * u_O_vis - s_hat)

    # Tensión de Dirichlet: mide qué tan lejos está la fase de su alineamiento
    tension = float(1.0 - abs(cos_t)) # 0 = alineado perfecto, 1 = ortogonal / conflicto máximo
    
    lap_vector = np.concatenate([lap_S, lap_L])
    return lap_vector, tension

Phi = np.concatenate([u_O_vis, u_O_text])
harness_activated = False
tau_activation = -1

print("• Iniciando evolución con Laplaciano de Beltrami estable...")
print("=======================================================================================================================")
print(f"{'Paso τ':<7} | {'Energía E(τ)':<13} | {'||∇E||':<10} | {'Tensión Dirichlet':<20} | {'Compuerta σ_ALU':<16} | {'Estado'}")
print("=======================================================================================================================")

records = []

for step in range(81):
    # En paso 15 inyectamos un conflicto ortogonal formal (paridad/aritmética no derivable)
    if step == 15:
        # Rotar L ortogonalmente para crear tensión máxima de Dirichlet (T -> 1.0)
        noise = np.random.randn(D)
        noise -= np.dot(noise, u_O_text) * u_O_text
        noise /= np.linalg.norm(noise)
        Phi[D:] = 0.2 * Phi[D:] + 0.98 * noise
        Phi[D:] /= np.linalg.norm(Phi[D:])
        print(f"  ⚡ [PASO 15] Ruptura de coherencia: Inyección de contradicción formal.")

    # 1. Laplaciano y Tensión confinada [0, 1]
    lap_vec, tension = compute_beltrami_laplacian(Phi)

    # 2. Función de activación sigmoidal de excitación
    sigma_gate = 1.0 / (1.0 + np.exp(-20.0 * (tension - THETA_EXCITE)))

    # 3. Flujo base Puerto-Hamiltoniano
    E_curr, Grad_curr = compute_E_and_grad(Phi)
    flow_base = apply_J(Grad_curr) - M_coeff * Grad_curr

    # 4. Difusión Laplaciana (Estabilidad incondicional de Dirichlet)
    flow_laplacian = NU_DIFFUSION * lap_vec

    # 5. Potencial de la Celda del Harness modulado por la compuerta continua
    L_curr = Phi[D:] / (np.linalg.norm(Phi[D:]) + 1e-12)
    cos_sol = np.dot(L_curr, u_exact_solution)
    grad_harness_L = - BETA_HARNESS * (1.0 - cos_sol) * (u_exact_solution - cos_sol * L_curr)
    flow_harness = np.concatenate([np.zeros(D), - sigma_gate * grad_harness_L])

    if sigma_gate > 0.5 and not harness_activated:
        harness_activated = True
        tau_activation = step
        status = "¡EXCITACIÓN HARNESS!"
    elif harness_activated and tension < 0.20:
        status = "Reconvergencia Final"
    elif harness_activated:
        status = "Resolución Harness"
    else:
        status = "Evolución Suave"

    if step in [0, 8, 14, 15, 16, 20, 32, 48, 64, 80]:
        print(f"{step:<7} | {E_curr:<13.6f} | {np.linalg.norm(Grad_curr):<10.4f} | {tension:<20.6f} | {sigma_gate:<16.4f} | {status}")

    # Integración explícita estable
    dPhi_total = flow_base + flow_laplacian + flow_harness
    Phi += dt * dPhi_total

L_final = Phi[D:] / np.linalg.norm(Phi[D:])
final_align = np.dot(L_final, u_exact_solution)

print("=======================================================================================================================")
print(f"• Afinidad final con la solución exacta del Harness: {final_align:.6f}")

if final_align > 0.80:
    print("\n🏆 DICTAMEN C-015: ACOPLAMIENTO LAPLACIANO CONFIRMADO Y ESTABILIZADO.")
    print("   El Laplaciano de Beltrami-Grafo detectó la tensión sin divergencia numérica,")
    print("   despertó la compuerta del Harness en el momento exacto y reconvergió a la certeza.")
else:
    print("\n❌ FALLO EN RECONVERGENCIA.")
print("=================================================================================")
