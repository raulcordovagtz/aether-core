import subprocess, os, struct, yaml
import numpy as np

KAPPA_0 = 1.0 / np.sqrt(5120.0)
FACTORS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

print("=================================================================================")
print(" 🔬 BARRIDO PARAMÉTRICO FORMAL: CURVA DE RESPUESTA DEL SPIN (C-007)")
print(f"    κ_0 base = {KAPPA_0:.10f} | Factores: {FACTORS}")
print("=================================================================================\n")

results = []

for f in FACTORS:
    kappa_val = float(f * KAPPA_0)
    exp_name = f"EXP_K_{f:.2f}"
    yaml_path = f"spec/experiments/{exp_name}.yaml"
    out_bin = f"logits_{exp_name}.bin"

    # 1. Emitir YAML formal
    spec_data = {
        "experiment_id": exp_name,
        "description": f"Barrido de acoplamiento kappa factor {f}x",
        "parameters": {
            "D": 5120, "R": 32, "steps": 64,
            "coupling_scale": kappa_val,
            "eml_scale": float(KAPPA_0 / 2.0),
            "lyapunov_damping": float(KAPPA_0 * ((1.0/64.0)**2))
        }
    }
    with open(yaml_path, "w") as yf:
        yaml.dump(spec_data, yf)

    # 2. Transpilar AOT y compilar Metallib
    subprocess.run(["python3", "tools/transpilar_c007_aot.py", yaml_path], stdout=subprocess.DEVNULL)
    subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", "metal/aether_c007_spinor_integrator.metal", "-o", "metal/aether_c007_spinor_integrator.air"], stdout=subprocess.DEVNULL)
    subprocess.run(["xcrun", "-sdk", "macosx", "metallib", "metal/aether_c007_spinor_integrator.air", "-o", "metal/aether_c007_spinor_integrator.metallib"], stdout=subprocess.DEVNULL)
    if os.path.exists("metal/aether_c007_spinor_integrator.air"):
        os.remove("metal/aether_c007_spinor_integrator.air")

    # 3. Ejecutar sonda rápida de 3 tokens
    env = os.environ.copy()
    env["AETHER_LOGITS_OUT"] = out_bin
    subprocess.run(["./bin/aether_engine", "3"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 4. Cargar logits
    logits = np.fromfile(out_bin, dtype=np.float32)
    results.append((f, kappa_val, logits))
    print(f"✓ Factor {f:4.2f}x (κ = {kappa_val:.8f}) capturado.")

# Análisis comparativo respecto a la base (Factor 0.0)
z_base = results[0][2]
def softmax(z):
    ez = np.exp(z - np.max(z))
    return ez / np.sum(ez)

p_base = softmax(z_base)
eps = 1e-15

print("\n=======================================================================================================")
print(f"{'Factor':<8} | {'κ':<12} | {'||Δz||_2':<10} | {'||Δz||_∞':<10} | {'ρ_margin':<10} | {'D_KL (nats)':<14} | {'N_eff':<8}")
print("=======================================================================================================")

for f, k_val, z in results:
    dz = z - z_base
    l2 = np.linalg.norm(dz)
    linf = np.max(np.abs(dz))
    m = np.sort(z)[-1] - np.sort(z)[-2]
    rho = (linf / m) if m > 0 else 0.0
    p = softmax(z)
    dkl = np.sum(p * np.log(np.maximum(p, eps) / np.maximum(p_base, eps)))
    neff = (l2 / linf)**2 if linf > 0 else 0.0

    print(f"{f:6.2f}x  | {k_val:10.8f} | {l2:10.6f} | {linf:10.6f} | {rho:10.6f} | {dkl:14.4e} | {neff:8.1f}")
print("=======================================================================================================")
