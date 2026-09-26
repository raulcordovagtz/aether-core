import os, sys, subprocess, sysconfig
import mlx.core

print("═" * 78)
print(" 🔨 COMPILACIÓN DEL MOTOR NATIVO AETHER (C++20 & METAL GPU)")
print("═" * 78)

metal_shaders = [
    ("metal/hilbert_memory_cell.metal", "metal/hilbert_memory_cell.metallib", "metal/hilbert_memory_cell.air"),
    ("metal/fact_band_router.metal", "metal/fact_band_router.metallib", "metal/fact_band_router.air"),
    ("metal/tetrapolar_predictor_cell.metal", "metal/tetrapolar_predictor_cell.metallib", "metal/tetrapolar_predictor_cell.air"),
    ("metal/tetrapolar_extractor.metal", "metal/tetrapolar_extractor.metallib", "metal/tetrapolar_extractor.air")
]

for src, out, air in metal_shaders:
    if os.path.exists(src):
        print(f"• Compilando shader Metal: {src} -> {out}...")
        res_m1 = subprocess.run([
            "xcrun", "-sdk", "macosx", "metal", "-c", src, "-o", air
        ])
        if res_m1.returncode != 0:
            print(f"❌ Error compilando shader Metal {src} a bitcode AIR.")
            sys.exit(1)

        res_m2 = subprocess.run([
            "xcrun", "-sdk", "macosx", "metallib", air, "-o", out
        ])
        if os.path.exists(air):
            os.remove(air)
        if res_m2.returncode != 0:
            print(f"❌ Error enlazando {out}.")
            sys.exit(1)
        print(f"✓ Shader Metal compilado con éxito: {out}")

# 1. Headers y bibliotecas oficiales de MLX
mlx_dir = os.path.dirname(mlx.core.__file__)
mlx_inc = os.path.join(mlx_dir, "include")
mlx_lib = os.path.join(mlx_dir, "lib")
mlx_dylib = os.path.join(mlx_lib, "libmlx.dylib")

print("• Directorio base MLX:", mlx_dir)
print("• Headers de MLX:", mlx_inc)
print("• Biblioteca MLX:", mlx_dylib)
assert os.path.exists(mlx_dylib), f"No se encontró libmlx.dylib en {mlx_lib}"

# 2. Nanobind v2.15.0 compatible con la versión de MLX (ABI v21 con dominio 'mlx')
nb_dir = os.path.abspath("tools/nanobind_mlx")
if not os.path.exists(nb_dir):
    print("• Descargando nanobind v2.15.0 (versión exacta de MLX v0.32.2)...")
    subprocess.run([
        "git", "clone", "--branch", "v2.15.0", "--depth", "1",
        "https://github.com/wjakob/nanobind.git", nb_dir
    ], check=True)

nb_inc = os.path.join(nb_dir, "include")
nb_src = os.path.join(nb_dir, "src", "nb_combined.cpp")
nb_robin = os.path.join(nb_dir, "ext", "robin_map", "include")

if not os.path.exists(os.path.join(nb_robin, "tsl", "robin_map.h")):
    try:
        import nanobind as nb_pkg
        cand = os.path.join(os.path.dirname(nb_pkg.include_dir()), "ext", "robin_map", "include")
        if os.path.exists(cand):
            nb_robin = cand
    except Exception:
        pass

py_inc = sysconfig.get_path("include")
ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
out_file = f"aether_vlm/aether_native_c{ext_suffix}"

# 3. Invocación de compilación con clang++ (Objective-C++ para soporte nativo de Metal)
cmd = [
    "clang++", "-O3", "-Wall", "-shared", "-std=c++20", "-fPIC",
    "-ObjC++",
    "-Wno-c++20-extensions",
    "-Wno-unused-const-variable",
    "-DNB_DOMAIN=mlx",
    f"-I{py_inc}",
    f"-I{nb_inc}",
    f"-I{nb_robin}",
    f"-I{mlx_inc}",
    "-Iinclude",
    f"-L{mlx_lib}", "-lmlx",
    "-framework", "Metal",
    "-framework", "Foundation",
    f"-Wl,-rpath,{mlx_lib}",
    "-undefined", "dynamic_lookup",
    nb_src,
    "aether_vlm/aether_native.cpp",
    "-o", out_file
]

print("• Compilando extensión C++/Metal con clang++...")
res = subprocess.run(cmd)

if res.returncode == 0:
    print(f"✓ Compilación C++/Metal exitosa: {out_file}")
    
    print("\n• Probando enlace nativo e invocación en Python...")
    sys.path.insert(0, os.path.abspath("aether_vlm"))
    import aether_native_c
    import mlx.core as mx

    # Validación funcional de kernels previos
    h = mx.array([1.0, 2.0, 3.0])
    u = mx.array([0.0, 1.0, 0.0])
    out_step = aether_native_c.dispatch_riemannian_step(h, u, 0.5)
    mx.eval(out_step)
    print("✓ dispatch_riemannian_step:", out_step)

    z_impact = mx.array([1.0, 2.0, 3.0, 4.0])
    z_L_star = mx.array([0.5, 0.5, 0.5, 0.5])
    out_cond = aether_native_c.dispatch_vapor_condensation(z_impact, z_L_star)
    mx.eval(out_cond)
    print("✓ dispatch_vapor_condensation:", out_cond)

    # Validación funcional Hito 2.1: Célula de Memoria Geométrica (CPU y Metal GPU)
    D = 1024
    mA = mx.zeros((D,)) + (1.0 / (D ** 0.5))
    mB = mx.zeros((D,))
    mB[0] = 1.0
    pack_res = aether_native_c.hilbert_memory_pack_two(mA, mB)
    mx.eval(pack_res["result"])
    print("✓ hilbert_memory_pack_two (CPU): degenerate =", pack_res["degenerate"])

    pack_metal = aether_native_c.hilbert_memory_pack_two_metal(mA, mB)
    mx.eval(pack_metal["result"])
    print("✓ hilbert_memory_pack_two_metal (Metal GPU): degenerate =", pack_metal["degenerate"])

    # Validación funcional C-022: Predictor Geodésico Tetrapolar (CPU y Metal GPU)
    h_in = mx.zeros((D,)) + (1.0 / (D ** 0.5))
    v_tan = mx.zeros((D,))
    v_tan[1] = 0.25
    u_o = mx.zeros((D,)) + (1.0 / (D ** 0.5))
    u_t = mx.zeros((D,)); u_t[1] = 1.0
    u_a = mx.zeros((D,)); u_a[1] = -1.0
    u_e = mx.zeros((D,)); u_e[2] = 1.0

    pred_cpu = aether_native_c.tetrapolar_predictor_step(h_in, v_tan, u_o, u_t, u_a, u_e, 0.5)
    mx.eval(pred_cpu["h_star"])
    print("✓ tetrapolar_predictor_step (CPU): grad_teleo =", pred_cpu["telemetry"]["grad_teleo"])

    pred_metal = aether_native_c.tetrapolar_predictor_step_metal(h_in, v_tan, u_o, u_t, u_a, u_e, 0.5)
    mx.eval(pred_metal["h_star"])
    print("✓ tetrapolar_predictor_step_metal (Metal GPU): grad_teleo =", pred_metal["telemetry"]["grad_teleo"])

    # Validación funcional Extractor de Tetrapolos (Metal GPU)
    X_test = mx.zeros((8, D)) + 0.1
    w_eos_test = mx.zeros((D,)); w_eos_test[0] = 1.0
    poles_test = aether_native_c.extract_tetrapolar_poles_metal(X_test, w_eos_test, 4)
    mx.eval(poles_test["u_onto"], poles_test["u_teleo"])
    print("✓ extract_tetrapolar_poles_metal (Metal GPU): norm(u_onto) =", float(mx.sqrt(mx.sum(poles_test["u_onto"]**2))))

    print("\n🚀 ¡Módulo aether_native_c (C++ y Metal GPU) compilado, enlazado y ejecutado al 100% con éxito!")
else:
    print("❌ Fallo en la compilación.")
    sys.exit(1)
