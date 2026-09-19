import numpy as np
import yaml

print("=================================================================================")
print(" 🔬 EXPERIMENTO C-014: BIFURCACIÓN ENDÓGENA Y RECONVERGENCIA (CALIBRADO)")
print("=================================================================================\n")

D = 5120
R = 32
dt = 0.05
M_coeff = 0.25

vis_path = "visual_embeddings.bin"
patches = np.frombuffer(open(vis_path, "rb").read(), dtype=np.float32).reshape(-1, D).copy()
u_O_vis = np.mean(patches, axis=0); u_O_vis /= np.linalg.norm(u_O_vis)

np.random.seed(42)
u_O_text = np.random.randn(D); u_O_text /= np.linalg.norm(u_O_text)

u_conflict = np.random.randn(D)
u_conflict = u_conflict - np.dot(u_conflict, u_O_text) * u_O_text
u_conflict /= np.linalg.norm(u_conflict)

u_sub_solution = np.random.randn(D); u_sub_solution /= np.linalg.norm(u_sub_solution)

U_c = np.zeros((D, R)); V_c = np.zeros((D, R))
U_c[:, 0] = u_O_text; V_c[:, 0] = u_O_vis

def apply_J(Grad):
    G_s = Grad[:D]; G_l = Grad[D:]
    return np.concatenate([U_c @ (V_c.T @ G_l), - V_c @ (U_c.T @ G_s)])

with open("spec/branching/C014_phase_branch.yaml") as f:
    spec = yaml.safe_load(f)
MIN_TAU = spec["trigger_conditions"]["min_tau_warmup"]
E_THRESH = spec["trigger_conditions"]["energy_threshold"]
BETA = spec["reconvergence"]["coupling_strength_beta"]

print(f"✓ Configuración C-014:")
print(f"  • Warmup mínimo τ : {MIN_TAU} pasos")
print(f"  • Umbral E_crit   : {E_THRESH}")
print(f"  • Gravedad β      : {BETA}\n")

def compute_E_and_grad(Phi, injected_attractor=None):
    S = Phi[:D]; L = Phi[D:]
    n_s = np.linalg.norm(S) + 1e-12; n_l = np.linalg.norm(L) + 1e-12
    s_hat = S / n_s; l_hat = L / n_l

    cos_v = np.dot(s_hat, u_O_vis)
    cos_t = np.dot(l_hat, u_O_text)
    cos_vl = np.dot(s_hat, l_hat)

    E = 0.5 * (1.0 - cos_v)**2 + 0.5 * (1.0 - cos_t)**2 + 0.25 * (1.0 - cos_vl)**2
    grad_S = - (1.0 - cos_v) * (u_O_vis - cos_v * s_hat) - 0.5 * (1.0 - cos_vl) * (l_hat - cos_vl * s_hat)
    grad_L = - (1.0 - cos_t) * (u_O_text - cos_t * l_hat) - 0.5 * (1.0 - cos_vl) * (s_hat - cos_vl * l_hat)

    if injected_attractor is not None:
        cos_sub = np.dot(l_hat, injected_attractor)
        E += 0.5 * BETA * (1.0 - cos_sub)**2
        grad_L += - BETA * (1.0 - cos_sub) * (injected_attractor - cos_sub * l_hat)

    return E, np.concatenate([grad_S, grad_L])

def generic_flow(Phi, attractor=None):
    E, Grad = compute_E_and_grad(Phi, attractor)
    return apply_J(Grad) - M_coeff * Grad, E, Grad

Phi_padre = np.concatenate([u_O_vis, u_O_text])
fork_triggered = False
tau_fork = -1

print("• [PADRE] Flujo continuo avanzando...")
for step in range(32):
    # En paso 12 inyectamos conflicto tras haber pasado el warmup
    if step == 12:
        Phi_padre[D:] = 0.5 * Phi_padre[D:] + 0.866 * u_conflict
        Phi_padre[D:] /= np.linalg.norm(Phi_padre[D:])
        print(f"  ⚠ [PASO {step}] Conflicto ortogonal inyectado en el padre.")

    dPhi, E_curr, Grad_curr = generic_flow(Phi_padre)
    
    if step >= MIN_TAU and E_curr > E_THRESH and not fork_triggered:
        fork_triggered = True
        tau_fork = step
        print(f"\n⚡ [TRIGGER C-014] Disparo confirmado en τ = {step}: E = {E_curr:.4f} > {E_THRESH}")
        print("  -> BIFURCACIÓN: Sub-Agente clonado en memoria UMA aislada...")
        break

    Phi_padre += dt * dPhi

assert fork_triggered, "Fallo en disparo post-warmup."

# ─── CLON ────────────────────────────────────────────────────────────────────
Phi_clon = Phi_padre.copy()
print(f"• [CLON] Resolviendo sub-problema con prompt inyectado ({spec['branch_configuration']['target_engine']})...")

for sub_step in range(spec["branch_configuration"]["max_sub_steps"]):
    L_c = Phi_clon[D:]
    L_c_hat = L_c / np.linalg.norm(L_c)
    grad_sub = - (u_sub_solution - np.dot(L_c_hat, u_sub_solution) * L_c_hat)
    Phi_clon[D:] += dt * (- M_coeff * grad_sub)

u_atractor_sub = Phi_clon[D:] / np.linalg.norm(Phi_clon[D:])
clon_max_align = np.dot(u_atractor_sub, u_sub_solution)
print(f"✓ [CLON] Atractor generado tras 64 pasos. Calidad de solución: {clon_max_align:.4f}")

# ─── RECONVERGENCIA ──────────────────────────────────────────────────────────
print("\n• [PADRE] Absorbiendo atractor del sub-agente (Reconvergencia)...")
print("=======================================================================================================")
print(f"{'Paso τ':<8} | {'Energía E(τ)':<15} | {'||∇E||':<12} | {'Afinidad con Clon':<25}")
print("=======================================================================================================")

records = []
for step in range(tau_fork, 65):
    dPhi, E_val, Grad_val = generic_flow(Phi_padre, attractor=u_atractor_sub)
    Phi_padre += dt * dPhi

    L_curr = Phi_padre[D:] / np.linalg.norm(Phi_padre[D:])
    cos_clon = np.dot(L_curr, u_atractor_sub)
    
    if step in [tau_fork, tau_fork + 4, tau_fork + 12, tau_fork + 24, 64]:
        records.append((step, E_val, np.linalg.norm(Grad_val), cos_clon))

for s, e, g, c in records:
    print(f"{s:<8} | {e:<15.6f} | {g:<12.6f} | {c:<25.6f}")

print("=======================================================================================================")
final_cos = records[-1][3]
ratio_absorcion = final_cos / clon_max_align

print(f"• Tasa de absorción de la solución del clon: {ratio_absorcion * 100.0:.2f}% (Final: {final_cos:.4f} / Clon: {clon_max_align:.4f})")

if ratio_absorcion >= 0.90:
    print("\n🏆 DICTAMEN C-014: BIFURCACIÓN ENDÓGENA Y RECONVERGENCIA TOTALMENTE CERTIFICADAS.")
    print("   El sub-agente resolvió la tarea y el host absorbió >90% de su orientación.")
else:
    print("\n❌ FALLO EN TASA DE ABSORCIÓN.")
print("=================================================================================")
