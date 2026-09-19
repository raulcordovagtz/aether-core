import sys, os, time
import mlx.core as mx

# 1. Forzar que Python busque PRIMERO en nuestro directorio soberano local
sys.path.insert(0, "/Users/crotalo/aether_engine")

print("=================================================================================")
print(" 🏛️ AETHER-SOVEREIGN: PRUEBA DE MOTOR CON CÓDIGO FUENTE INTERNALIZADO")
print("=================================================================================\n")

# Importar desde el código fuente interno de aether_engine
from core_vlm.utils import load
from core_vlm.generate.dispatch import stream_generate

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print(f"• Cargando modelo mediante core_vlm interno...")
t0_load = time.time()
model, processor = load(model_path)
print(f"✓ Modelo cargado en UMA en {time.time() - t0_load:.2f} s desde código fuente soberano.\n")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe brevemente la figura central y sus colores."}
        ]
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

print("• Ejecutando inferencia soberana (Límite: 60 tokens)...")
print("\n--- SALIDA SOBERANA STREAM ---")

token_count = 0
t0_tokens = None

for response in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=60):
    if t0_tokens is None:
        t0_tokens = time.time()
    print(response.text, end="", flush=True)
    token_count += 1

t1 = time.time()
t_decode = t1 - (t0_tokens if t0_tokens else t1)
tps = token_count / t_decode if t_decode > 0 else 0.0

print("\n------------------------------")
print(f"\n📊 TELEMETRÍA DEL MOTOR SOBERANO:")
print(f"• Tokens generados : {token_count}")
print(f"• Tiempo de Decode : {t_decode:.2f} s")
print(f"• Velocidad real   : {tps:.2f} tok/s")
print("=================================================================================")
