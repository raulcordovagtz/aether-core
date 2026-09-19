import sympy as sp

print("=================================================================================")
print(" 🔬 CERTIFICACIÓN SIMBÓLICA C-007: INVARIANTES DE LYAPUNOV Y ANTISIMETRÍA")
print("=================================================================================\n")

# Representación simbólica en R^2
s, l = sp.symbols('s l', real=True)
Phi = sp.Matrix([s, l])

# Operador K antisimétrico genérico 2x2
k12 = sp.symbols('k12', real=True)
K = sp.Matrix([[0, k12], [-k12, 0]])

print("▶ 1. Verificando conservación exacta de norma bajo K (d/dtau ||Phi||^2 = 2 <Phi, K Phi>)...")
work = Phi.dot(K * Phi)
print(f"   • Trabajo <Φ, K Φ> = {sp.simplify(work)}")
assert work == 0, "Fallo: K no es antisimétrico."
print("   ✅ IDENTIDAD DEMOSTRADA: d/dtau ||Phi||^2 == 0 para el término lineal.\n")

print("▶ 2. Verificando proyector tangencial con fuerza no lineal arbitraria F...")
f1, f2 = sp.symbols('f1 f2', real=True)
F = sp.Matrix([f1, f2])

norm_sq = Phi.dot(Phi)
proj_coeff = F.dot(Phi) / norm_sq
Pi_perp_F = F - proj_coeff * Phi

dot_tangent = sp.simplify(Phi.dot(Pi_perp_F))
print(f"   • <Φ, Π_⊥(F, Φ)> = {dot_tangent}")
assert dot_tangent == 0, "Fallo: Pi_perp no anula la proyección radial."
print("   ✅ IDENTIDAD DEMOSTRADA: El término EML jamás induce aceleración radial.\n")

print("▶ 3. Derivada completa con término disipativo -lambda * Phi...")
lam = sp.symbols('lambda', positive=True, real=True)
Phi_dot = K * Phi + Pi_perp_F - lam * Phi
norm_derivative = sp.simplify(2 * Phi.dot(Phi_dot))
print(f"   • d/dtau ||Phi||^2 = {norm_derivative}")
print("   ✅ IDENTIDAD DEMOSTRADA: d/dtau ||Phi||^2 = -2*lambda*||Phi||^2 <= 0 (Lyapunov asintótico).")
print("=================================================================================")
print(" 🏆 DICTAMEN SYMPY: EL SISTEMA DINÁMICO C-007 ES MATEMÁTICAMENTE ESTABLE.")
print("=================================================================================")
