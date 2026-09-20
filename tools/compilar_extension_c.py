import os, sys, subprocess, sysconfig
import mlx.core

# 1. Headers y bibliotecas oficiales de MLX (distribuidos dentro del propio paquete instalado)
mlx_dir = os.path.dirname(mlx.core.__file__)
mlx_inc = os.path.join(mlx_dir, "include")
mlx_lib = os.path.join(mlx_dir, "lib")
mlx_dylib = os.path.join(mlx_lib, "libmlx.dylib")

print("• Directorio base MLX:", mlx_dir)
print("• Headers de MLX:", mlx_inc)
print("• Biblioteca MLX:", mlx_dylib)
assert os.path.exists(mlx_dylib), f"No se encontró libmlx.dylib en {mlx_lib}"

# 2. Nanobind v2.15.0 compatible con la versión de MLX (ABI v21 con dominio 'mlx')
# MLX 0.32.2 utiliza nanobind v2.15.0 con NB_DOMAIN=mlx para el type-caster de arrays.
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

# Si robin_map no está en el submódulo local, buscar en nanobind de conda
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

# 3. Invocación de compilación con clang++
cmd = [
    "clang++", "-O3", "-Wall", "-shared", "-std=c++20", "-fPIC",
    "-Wno-c++20-extensions",
    "-Wno-unused-const-variable",
    "-DNB_DOMAIN=mlx",
    f"-I{py_inc}",
    f"-I{nb_inc}",
    f"-I{nb_robin}",
    f"-I{mlx_inc}",
    f"-L{mlx_lib}", "-lmlx",
    f"-Wl,-rpath,{mlx_lib}",
    "-undefined", "dynamic_lookup",
    nb_src,
    "aether_vlm/aether_native.cpp",
    "-o", out_file
]

print("• Compilando con clang++...")
res = subprocess.run(cmd)

if res.returncode == 0:
    print(f"✓ Compilación C++ exitosa: {out_file}")
    
    print("\n• Probando enlace nativo e invocación en Python...")
    sys.path.insert(0, os.path.abspath("aether_vlm"))
    import aether_native_c
    import mlx.core as mx

    # Validación funcional con tensores de prueba
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

    print("\n🚀 ¡Módulo aether_native_c compilado, enlazado y ejecutado al 100% con éxito!")
else:
    print("❌ Fallo en la compilación.")
    sys.exit(1)
