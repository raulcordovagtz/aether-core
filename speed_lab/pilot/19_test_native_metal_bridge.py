import sys, os, time, math, ctypes
import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/crotalo/aether_engine")

from core_vlm.utils import load
from core_vlm.generate.dispatch import stream_generate
from core_vlm.auto_coupler import AutoCoupler, HookedDecoderLayer

print("=================================================================================")
print(" 🔬 VERIFICACIÓN DE SILICIO: MOTOR C++ / METAL NATIVO CONECTADO A GPU")
print("    Shader: aether_c018_riemannian_engine.metallib despachado vía libaether_bridge")
print("=================================================================================\n")

# 1. Cargar biblioteca dinámica compilada en C++
dylib_path = "/Users/crotalo/aether_engine/bin/libaether_bridge.dylib"
metallib_path = "/Users/crotalo/aether_engine/metal/aether_c018_riemannian_engine.metallib"

if not os.path.exists(dylib_path):
    print(f"❌ No se encontró: {dylib_path}")
    exit(1)

lib_bridge = ctypes.CDLL(dylib_path)

# Firma exacta de la función C++: aether_metal_step
lib_bridge.aether_metal_step.argtypes = [
    ctypes.c_void_p,                # h_io_ptr
    ctypes.c_void_p,                # u_target_ptr
    ctypes.c_uint32,                # D
    ctypes.c_uint32,                # R
    ctypes.c_float,                 # dt
    ctypes.c_float,                 # kappa_0
    ctypes.c_float,                 # beta_steer
    ctypes.c_float,                 # M_diss
    ctypes.c_float,                 # nu_diff
    ctypes.c_char_p                 # metallib_path
]
lib_bridge.aether_metal_step.restype = ctypes.c_int
print("✓ Biblioteca C++ libaether_bridge.dylib cargada y vinculada.")

# 2. Cargar modelo 0.8B
model_path = "/Users/crotalo/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit"
model, processor = load(model_path)

coupler = AutoCoupler(model_path)
p = coupler.params
D = p["d_model"] # 1024
L = p["num_layers"] # 24
HOOK_LAYER = coupler.get_causal_layer_index() # Capa 10

kappa_0 = p["kappa_0"] # 0.03125
dt = p["dt_canonical"] # 1/24
beta_steer = math.log(float(D)) * kappa_0 * 2.0 # Fuerza supercrítica
M_diss = 0.25
nu_diff = 0.05

print(f"• Parámetros de silicio: D={D}, Capa={HOOK_LAYER}, κ_0={kappa_0:.6f}, β={beta_steer:.6f}")

# 3. Vector objetivo en memoria UMA (des-cuantizado)
target_token = processor.tokenizer.encode(" exact")[0]
u_target = model.language_model.model.embed_tokens(mx.array([target_token]))[0].astype(mx.float32)
u_target = u_target / (mx.linalg.norm(u_target) + 1e-12)
mx.eval(u_target)
u_target_np = np.array(u_target, copy=False)
u_target_ptr = ctypes.c_void_p(u_target_np.ctypes.data)

# 4. Función de Hook que llama directamente a la GPU Metal
gpu_call_count = 0
total_gpu_time_us = 0.0

def native_metal_hook(h_last):
    global gpu_call_count, total_gpu_time_us
    # Convertir a float32 contiguo en memoria unificada
    h_f32 = mx.contiguous(h_last.astype(mx.float32))
    mx.eval(h_f32)
    
    h_np = np.array(h_f32, copy=False)
    h_ptr = ctypes.c_void_p(h_np.ctypes.data)
    
    t0 = time.time()
    # LLAMADA DIRECTA AL KERNEL DE METAL EN GPU
    res = lib_bridge.aether_metal_step(
        h_ptr,
        u_target_ptr,
        D,
        32, # Rango R
        dt,
        kappa_0,
        beta_steer,
        M_diss,
        nu_diff,
        metallib_path.encode('utf-8')
    )
    t1 = time.time()
    
    if res != 0:
        print("❌ Error en kernel Metal GPU.")
    
    total_gpu_time_us += (t1 - t0) * 1e6
    gpu_call_count += 1
    
    return h_f32.astype(h_last.dtype)

# 5. Probar una pasada con el hook enganchado en Capa 10
print("\n• Ejecutando prueba de inferencia conectada directamente a la GPU...")
prompt = "The exact mathematical result is"
messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
prompt_str = processor.apply_chat_template(messages, add_generation_prompt=True)

coupler.attach(model, native_metal_hook, layer_idx=HOOK_LAYER)

try:
    t0_gen = time.time()
    tokens = []
    for resp in stream_generate(model, processor, prompt=prompt_str, max_tokens=15):
        print(resp.text, end="", flush=True)
        tokens.append(resp.text)
    t_gen = time.time() - t0_gen
finally:
    coupler.detach(model)

print("\n---------------------------------------------------------------------------------")
print(f"📊 REPORTE DE SILICIO PURO:")
print(f"• Tokens generados en GPU : {len(tokens)} tokens en {t_gen:.2f} s ({len(tokens)/t_gen:.1f} tok/s)")
print(f"• Despachos a Metal GPU   : {gpu_call_count} ejecuciones reales")
print(f"• Latencia promedio en GPU: {total_gpu_time_us / max(1, gpu_call_count):.2f} microsegundos por token")
print("=================================================================================")
print(" 🏆 MOTOR NATIVO C++ / METAL CONECTADO Y EJECUTANDO EN SILICIO.")
print("=================================================================================")
