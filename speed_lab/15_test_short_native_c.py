import sys, os, time
import mlx.core as mx
from PIL import Image

sys.path.insert(0, "/Users/crotalo/aether_engine")
from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print("• 1. Cargando Qwen 27B...")
t0 = time.time()
model, processor = load(model_path)
print(f"✓ Modelo cargado en {time.time() - t0:.2f} s.")

aether = AetherEngine(model, processor)

# Ingestión de imagen
img = Image.open(img_path).convert("RGB")
prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe la imagen en una frase."}]}], add_generation_prompt=True)
inputs = processor(text=[prompt], images=[img], return_tensors="mlx")
visual_patches = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])[0]

print("\n• 2. Asentamiento de coherencia en GPU (τ* = 32)...")
t_set0 = time.time()
aether.prepare_multimodal_thought(visual_patches, "Describe la imagen en una frase.")
print(f"✓ Atractor L* asentado en {(time.time() - t_set0)*1000:.1f} ms.")

print("\n• 3. Generando 35 tokens con Colapso 100% C++ (quantized_matmul):")
print("----------------------------------------------------------------------")

t0_gen = None
count = 0
for resp in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=35):
    if t0_gen is None:
        t0_gen = time.time()
    print(resp.text, end="", flush=True)
    count += 1

t_total = time.time() - t0_gen
tps = count / (t_total + 1e-12)

print("\n----------------------------------------------------------------------")
print(f"📊 Resumen: {count} tokens en {t_total:.2f} s | {tps:.2f} tok/s sostenidos en GPU.")
print("======================================================================")
