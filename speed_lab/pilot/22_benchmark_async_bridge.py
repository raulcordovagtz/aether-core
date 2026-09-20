import sys, os, time, math, ctypes
import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/crotalo/aether_engine")

from core_vlm.utils import load
from core_vlm.generate.dispatch import stream_generate
from core_vlm.auto_coupler import AutoCoupler

print("=================================================================================")
print(" 🚀 BENCHMARK ASÍNCRONO: RECUPERACIÓN TOTAL DE VELOCIDAD EN SILICIO (0.8B)")
print("=================================================================================\n")

dylib_path = "/Users/crotalo/aether_engine/bin/libaether_bridge.dylib"
metallib_path = "/Users/crotalo/aether_engine/metal/aether_c018_riemannian_engine.metallib"

lib_bridge = ctypes.CDLL(dylib_path)

# Inicializar puente una sola vez
lib_bridge.aether_metal_init.argtypes = [ctypes.c_char_p]
lib_bridge.aether_metal_init.restype = ctypes.c_int
init_res = lib_bridge.aether_metal_init(metallib_path.encode('utf-8'))
if init_res != 0:
    print("❌ Error en aether_metal_init")
    exit(1)
print("✓ Pipeline Metal pre-compilado en GPU Apple M2 Max.")

lib_bridge.aether_metal_async_step.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float
]
lib_bridge.aether_metal_async_step.restype = ctypes.c_int

model_path = "/Users/crotalo/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit"
model, processor = load(model_path)

coupler = AutoCoupler(model_path)
p = coupler.params
D = p["d_model"]
HOOK_LAYER = coupler.get_causal_layer_index()

kappa_0 = p["kappa_0"]
dt = p["dt_canonical"]
beta_steer = math.log(float(D)) * kappa_0
M_diss = 0.25
nu_diff = 0.05

target_token = processor.tokenizer.encode(" exact")[0]
u_target = model.language_model.model.embed_tokens(mx.array([target_token]))[0].astype(mx.float32)
u_target = u_target / (mx.linalg.norm(u_target) + 1e-12)
mx.eval(u_target)
u_target_np = np.array(u_target, copy=False)
u_target_ptr = ctypes.c_void_p(u_target_np.ctypes.data)

hook_times_us = []

def async_metal_hook(h_last):
    h_f32 = mx.contiguous(h_last.astype(mx.float32))
    mx.eval(h_f32)
    h_np = np.array(h_f32, copy=False)
    h_ptr = ctypes.c_void_p(h_np.ctypes.data)

    t0 = time.time()
    # DESPACHO ASÍNCRONO EN GPU (Sin bloqueo de CPU)
    lib_bridge.aether_metal_async_step(
        h_ptr,
        u_target_ptr,
        D,
        dt,
        kappa_0,
        beta_steer,
        M_diss,
        nu_diff
    )
    t1 = time.time()
    hook_times_us.append((t1 - t0) * 1e6)
    return h_f32.astype(h_last.dtype)

prompt = "The exact mathematical result is"
messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
prompt_str = processor.apply_chat_template(messages, add_generation_prompt=True)

coupler.attach(model, async_metal_hook, layer_idx=HOOK_LAYER)

print("• Generando con motor asíncrono en silicio...")
tokens = []
t0_gen = time.time()
try:
    for resp in stream_generate(model, processor, prompt=prompt_str, max_tokens=60):
        print(resp.text, end="", flush=True)
        tokens.append(resp.text)
finally:
    coupler.detach(model)
t_gen = time.time() - t0_gen

print("\n---------------------------------------------------------------------------------")
print(f"📊 REPORTE DE VELOCIDAD ASÍNCRONA:")
print(f"• Throughput Sostenido       : {len(tokens) / t_gen:.1f} tokens/segundo")
print(f"• Latencia de llamada a Metal: {np.mean(hook_times_us):.2f} microsegundos por token")
print(f"• Latencia mínima de llamada : {np.min(hook_times_us):.2f} microsegundos")
print("=================================================================================")
