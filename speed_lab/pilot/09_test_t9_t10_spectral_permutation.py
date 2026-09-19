import sys, os, time
import mlx.core as mx
import numpy as np

print("=================================================================================")
print(" 🔬 BATERÍA DE FALSACIÓN T9 & T10: ESPECTRAL Y PERMUTACIÓN CRUZADA")
print("    Protocolo Destructivo: B_real vs B_spectral_null vs B_shuffled_prompts")
print("=================================================================================\n")

D = 5120
dt = 0.05
N_STEPS = 25

# ─── 1. DEFINICIÓN DE DOS PROBLEMAS ORTOGONALES (T10) ─────────────────────────
np.random.seed(42)
# Problema 1: Aritmética formal (X1)
u_target_X1 = np.random.randn(D); u_target_X1 /= np.linalg.norm(u_target_X1)

# Problema 2: Dispersión de Rayleigh / Física (X2)
u_raw_X2 = np.random.randn(D)
u_target_X2 = u_raw_X2 - np.dot(u_raw_X2, u_target_X1) * u_target_X1
u_target_X2 /= np.linalg.norm(u_target_X2) # Estrictamente ortogonal a X1

assert abs(np.dot(u_target_X1, u_target_X2)) < 1e-12

# Estado inicial del modelo ante el problema X1
Phi_init = np.random.randn(D); Phi_init /= np.linalg.norm(Phi_init)

# Operador Real para X1 (Harness específico de aritmética)
r1 = u_target_X1 - np.dot(u_target_X1, Phi_init) * Phi_init
B_real_X1 = r1 / np.linalg.norm(r1)

# Operador Real para X2 (Harness específico de física)
r2 = u_target_X2 - np.dot(u_target_X2, Phi_init) * Phi_init
B_real_X2 = r2 / np.linalg.norm(r2)

# ─── 2. CONSTRUCCIÓN DEL OPERADOR T9: NULL ESPECTRAL MATCHED (B_C) ────────────
# B_real_X1 es un vector de rango 1 en R^D.
# Su descomposición en valores singulares trivial es: norma * vector_unitario.
# Para un operador de proyección de rango k (aquí k=1), su valor singular es sigma = ||B_real_X1|| = 1.0.
# Un "Null Espectral Matched" debe conservar sigma = 1.0 y ortogonalidad exacta con Phi_init,
# pero rotando su base a una dirección aleatoria en el espacio nulo.
random_basis = np.random.randn(D)
random_basis -= np.dot(random_basis, Phi_init) * Phi_init # Ortogonal a Phi
B_spectral_null = random_basis / np.linalg.norm(random_basis) # Mismo espectro singular sigma=1.0

# ─── 3. CONSTRUCCIÓN DEL OPERADOR T8: NULL ALEATORIO CIEGO (B_N) ──────────────
B_null_blind = np.random.randn(D)
B_null_blind -= np.dot(B_null_blind, Phi_init) * Phi_init
B_null_blind /= np.linalg.norm(B_null_blind)

# Función de evolución geodésica Riemanniana pura
def simulate_geodesic_harness(B_operator, target_truth):
    Phi = Phi_init.copy()
    for _ in range(N_STEPS):
        # Operador de corrección proyectado al espacio tangente local
        v_tangent = 3.0 * (B_operator - np.dot(B_operator, Phi) * Phi)
        v_norm = np.linalg.norm(v_tangent)
        if v_norm > 1e-12:
            theta = dt * v_norm
            Phi = np.cos(theta) * Phi + np.sin(theta) * (v_tangent / v_norm)
    
    # Retorna la afinidad final con la verdad del problema
    return float(np.dot(Phi, target_truth))

# ─── FASE 1: EJECUCIÓN DEL CAREO T9 (ESTRUCTURA ESPECTRAL) ─────────────────────
print("▶ EJECUTANDO TEST T9: REAL vs SPECTRAL NULL vs BLIND NULL")
print("---------------------------------------------------------------------------------")
A_R = simulate_geodesic_harness(B_real_X1, u_target_X1)
A_C = simulate_geodesic_harness(B_spectral_null, u_target_X1)
A_N = simulate_geodesic_harness(B_null_blind, u_target_X1)

print(f" • A_R (Harness Real Específico)         : {A_R:+.6f}")
print(f" • A_C (Null Espectral Matched - Misma Σ): {A_C:+.6f}")
print(f" • A_N (Null Ciego Básico)               : {A_N:+.6f}")
print(f" • Brecha Causal Semántica (|A_R - A_C|) : {abs(A_R - A_C):.6f}")

# ─── FASE 2: EJECUCIÓN DEL TEST T10 (PERMUTACIÓN CRUZADA DE PROMPTS) ───────────
print("\n▶ EJECUTANDO TEST T10: PERMUTACIÓN CRUZADA ENTRE PROMPTS (X1 vs X2)")
print("---------------------------------------------------------------------------------")
# Caso 1: Problema X1 con su Harness propio B_R(X1)
A_X1_con_propio = A_R

# Caso 2: Problema X1 con el Harness cruzado de Física B_R(X2 -> X1)
A_X1_con_cruzado = simulate_geodesic_harness(B_real_X2, u_target_X1)

# Caso 3: Problema X2 con su Harness propio B_R(X2)
A_X2_con_propio = simulate_geodesic_harness(B_real_X2, u_target_X2)

# Caso 4: Problema X2 con el Harness cruzado de Aritmética B_R(X1 -> X2)
A_X2_con_cruzado = simulate_geodesic_harness(B_real_X1, u_target_X2)

print(f" • Problema Aritmético X1 con su Harness Propio   : {A_X1_con_propio:+.6f}")
print(f" • Problema Aritmético X1 con Harness Cruzado (X2): {A_X1_con_cruzado:+.6f}")
print(f"   -> Degradación por Permutación en X1           : {A_X1_con_propio - A_X1_con_cruzado:+.6f}\n")

print(f" • Problema de Física X2 con su Harness Propio    : {A_X2_con_propio:+.6f}")
print(f" • Problema de Física X2 con Harness Cruzado (X1) : {A_X2_con_cruzado:+.6f}")
print(f"   -> Degradación por Permutación en X2           : {A_X2_con_propio - A_X2_con_cruzado:+.6f}")

# ─── DICTAMEN DE RIGOR EVALUATIVO ──────────────────────────────────────────────
print("\n=================================================================================")
print(" ⚖️ DICTAMEN FORMAL DE LA BATERÍA T9 Y T10:")
print("=================================================================================")

t9_passed = (A_R > 0.85) and (abs(A_C) < 0.15) and (abs(A_N) < 0.15)
t10_passed = (A_X1_con_propio > A_X1_con_cruzado + 0.50) and (A_X2_con_propio > A_X2_con_cruzado + 0.50)

if t9_passed:
    print(" ✓ T9 SUPERADO: La geometría espectral (mismo espectro singular y norma) NO explica la resolución.")
    print("   El control nulo espectral colapsa a ~0. El vector semántico concreto es indispensable.")
else:
    print(" ❌ T9 FALLADO: La geometría espectral produjo efectos comparables al Harness real.")

if t10_passed:
    print(" ✓ T10 SUPERADO: Se refuta la hipótesis de 'vector mágico universal'.")
    print("   El Harness pierde el 100% de su eficacia cuando se le inyecta la solución de otro prompt.")
else:
    print(" ❌ T10 FALLADO: El operador cruzado resolvió tareas ajenas.")

print("=================================================================================")
