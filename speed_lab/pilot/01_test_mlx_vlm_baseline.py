import time, os
from PIL import Image
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.utils import load_image

print("=================================================================================")
print(" 🚀 PILOTO SPEED: BENCHMARK BASAL MLX-VLM (APPLE M2 MAX)")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print(f"• Cargando modelo multimodal desde: {model_path}")
t0_load = time.time()
model, processor = load(model_path)
t1_load = time.time()
print(f"✓ Modelo cargado en UMA en {t1_load - t0_load:.2f} s.\n")

# Estructurar mensaje con el formato canónico de Qwen-VL
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
print(f"✓ Prompt con plantilla oficial generado: {len(prompt)} caracteres.\n")

print("• Ejecutando inferencia basal sobre 005.jpg (Límite: 100 tokens)...")
t0_gen = time.time()
output = generate(
    model, 
    processor, 
    prompt=prompt, 
    image=img_path, 
    max_tokens=100, 
    verbose=True
)
t1_gen = time.time()

total_time = t1_gen - t0_gen
print(f"\n✓ Inferencia completada en {total_time:.2f} s.")
print("=================================================================================")
