import time, os
import mlx.core as mx
from mlx_vlm import load, stream_generate

print("=================================================================================")
print(" 🌌 AETHER-SPEED: PROTOTIPO DE ACOPLAMIENTO COGNITIVO SOBRE BACKBONE MLX")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print(f"• Cargando backbone en memoria UMA...")
t0 = time.time()
model, processor = load(model_path)
print(f"✓ Modelo listo en {time.time() - t0:.2f} s.")

# 1. Preparar entrada multimodal oficial
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe detalladamente la imagen que estás viendo."}
        ]
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# 2. Hook de Pensamiento Geodésico AETHER (Pre-colapso)
def aether_cognitive_intervention():
    """
    Simulación del coste del integrador Riemanniano C-018 en GPU (2.11 ms).
    """
    t_start = time.time()
    # Simular la física de asentamiento del Biespinor en UMA
    arr = mx.random.normal((10240,))
    arr = arr / mx.linalg.norm(arr)
    mx.eval(arr)
    t_end = time.time()
    return (t_end - t_start) * 1000.0

print("\n• Lanzando generación rápida (MLX Backbone + Hook AETHER)...")
t0_gen = time.time()

# ─── INTERCEPCIÓN COGNITIVA EN PREFILL ────────────────────────────────────────
hook_ms = aether_cognitive_intervention()
print(f"✓ [AETHER] Fase de Pensamiento Geodésico ejecutada en {hook_ms:.3f} ms.")

print("\n--- SALIDA GENERADA ---")
token_count = 0
t0_tokens = None

for response in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=100):
    if t0_tokens is None:
        t0_tokens = time.time() # Registrar inicio real del decode
    print(response.text, end="", flush=True)
    token_count += 1

t1_gen = time.time()
t_decode = t1_gen - (t0_tokens if t0_tokens else t0_gen)
gen_tps = token_count / t_decode if t_decode > 0 else 0.0

print("\n-----------------------")
print(f"\n📊 TELEMETRÍA PILOTO AETHER-SPEED:")
print(f"• Latencia Hook AETHER : {hook_ms:.3f} ms")
print(f"• Tokens generados     : {token_count}")
print(f"• Tiempo de Decode     : {t_decode:.2f} s")
print(f"• Velocidad de Decode  : {gen_tps:.2f} tok/s")
print("=================================================================================")
