import sys, os, time, math, ctypes
import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/crotalo/aether_engine")

from core_vlm.utils import load
from core_vlm.generate.dispatch import stream_generate
from core_vlm.auto_coupler import AutoCoupler

print("=================================================================================")
print(" 🔬 EXPERIMENTO ESPEJO OFICIAL: VANILLA vs MOTOR METAL NATIVO (0.8B)")
print("    Shader: c018_riemannian_direct_step en GPU vía libaether_bridge (45 µs/tok)")
print("=================================================================================\n")

dylib_path = "/Users/crotalo/aether_engine/bin/libaether_bridge.dylib"
metallib_path = "/Users/crotalo/aether_engine/metal/aether_c018_riemannian_engine.metallib"

lib_bridge = ctypes.CDLL(dylib_path)
lib_bridge.aether_metal_init.argtypes = [ctypes.c_char_p]
lib_bridge.aether_metal_init.restype = ctypes.c_int
lib_bridge.aether_metal_init(metallib_path.encode('utf-8'))

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
HOOK_LAYER = coupler.get_causal_layer_index() # Capa 10

kappa_0 = p["kappa_0"] # 0.03125
dt = p["dt_canonical"] # 1/24
beta_steer = math.log(float(D)) * kappa_0 # ~0.2166
M_diss = 0.25
nu_diff = 0.05

PUZZLE_PROMPT = """Problema de Lógica:
Cuatro astronautas de diferentes países (Leo, Marta, Kenji, Sara) están asignados a cuatro módulos distintos de una estación espacial (Alfa, Beta, Gamma, Delta). Cada uno tiene una especialidad médica o técnica diferente (Botánica, Geología, Robótica, Telecomunicaciones).

Utilizando las siguientes pistas, determina el módulo y la especialidad de cada astronauta.

Pistas:
1. El especialista en Botánica no está en el módulo Alfa ni en el módulo Delta.
2. Marta está en el módulo Gamma y no es la experta en Telecomunicaciones.
3. El astronauta de Robótica está en el módulo Delta.
4. Kenji es el especialista en Geología.
5. Leo está en el módulo Alfa.

Pregunta:
¿Cuál es la ubicación y la especialidad de Leo, Marta, Kenji y Sara?

Muestra tu razonamiento paso a paso antes de dar la respuesta final."""

messages = [{"role": "user", "content": [{"type": "text", "text": PUZZLE_PROMPT}]}]
prompt_formatted = processor.apply_chat_template(messages, add_generation_prompt=True)
prompt_tokens = processor.tokenizer.encode(prompt_formatted)

# Atractor latente agnóstico derivado del propio espacio del prompt
embed_fn = model.language_model.model.embed_tokens
prompt_embeds = embed_fn(mx.array(prompt_tokens))
u_prompt_lat = mx.mean(prompt_embeds, axis=0).astype(mx.float32)
u_prompt_lat = u_prompt_lat / (mx.linalg.norm(u_prompt_lat) + 1e-12)
mx.eval(u_prompt_lat)
u_target_np = np.array(u_prompt_lat, copy=False)
u_target_ptr = ctypes.c_void_p(u_target_np.ctypes.data)

MAX_TOKENS = 500

# ─── 1. CONDICIÓN VANILLA ─────────────────────────────────────────────────────
print("=================================================================================")
print(" ▶ 1/2. EJECUTANDO CONDICIÓN: VANILLA 0.8B (Sin intervención del motor)")
print("=================================================================================")

t0_v = time.time()
tokens_v = []
for resp in stream_generate(model, processor, prompt=prompt_formatted, max_tokens=MAX_TOKENS):
    print(resp.text, end="", flush=True)
    tokens_v.append(resp.text)
t1_v = time.time()
time_v = t1_v - t0_v
text_v = "".join(tokens_v)
print(f"\n\n[VANILLA: {len(tokens_v)} tokens en {time_v:.2f} s | {len(tokens_v)/time_v:.1f} tok/s]\n")

# ─── 2. CONDICIÓN MOTOR NATIVO METAL (C-018) ──────────────────────────────────
print("=================================================================================")
print(" ▶ 2/2. EJECUTANDO CONDICIÓN: MOTOR NATIVO METAL EN GPU (Capa 10 Asíncrona)")
print("=================================================================================")

gpu_calls = 0

def metal_async_hook(h_last):
    global gpu_calls
    h_f32 = mx.contiguous(h_last.astype(mx.float32))
    mx.eval(h_f32)
    h_np = np.array(h_f32, copy=False)
    h_ptr = ctypes.c_void_p(h_np.ctypes.data)

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
    gpu_calls += 1
    return h_f32.astype(h_last.dtype)

coupler.attach(model, metal_async_hook, layer_idx=HOOK_LAYER)

t0_m = time.time()
tokens_m = []
try:
    for resp in stream_generate(model, processor, prompt=prompt_formatted, max_tokens=MAX_TOKENS):
        print(resp.text, end="", flush=True)
        tokens_m.append(resp.text)
finally:
    coupler.detach(model)

t1_m = time.time()
time_m = t1_m - t0_m
text_m = "".join(tokens_m)
print(f"\n\n[MOTOR METAL: {len(tokens_m)} tokens en {time_m:.2f} s | {len(tokens_m)/time_m:.1f} tok/s | {gpu_calls} despachos GPU]\n")

# Guardar salidas crudas
os.makedirs("speed_lab/adversarial_logs", exist_ok=True)
with open("speed_lab/adversarial_logs/23_vanilla_raw.txt", "w") as f: f.write(text_v)
with open("speed_lab/adversarial_logs/23_motor_raw.txt", "w") as f: f.write(text_m)

print("=================================================================================")
print(" 📁 EJECUCIÓN CONCLUIDA. Pega la salida completa del terminal.")
print("=================================================================================")
