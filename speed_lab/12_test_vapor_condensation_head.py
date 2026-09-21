import sys, os, time
import mlx.core as mx
from PIL import Image

sys.path.insert(0, "/Users/crotalo/aether_engine")
from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

print("=================================================================================")
print(" 💧 AETHER-VLM :: PRUEBA DE CONDENSACIÓN DE VAPOR EN SILICIO (27B)")
print("    Choque Cinético + Disipación Viscosa + Nucleación Fáctica en lm_head")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print("• 1. Cargando arquitectura Qwen 27B...")
t0 = time.time()
model, processor = load(model_path)
print(f"✓ Modelo montado en {time.time() - t0:.2f} s.")

aether = AetherEngine(model, processor)

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe detalladamente la imagen."}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

img = Image.open(img_path).convert("RGB")
inputs = processor(text=[prompt], images=[img], return_tensors="mlx")
visual_patches = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])[0]

print("\n• 2. Asentamiento de Pensamiento Profundo y pre-cálculo de nucleación...")
t_set0 = time.time()
telemetria = aether.prepare_multimodal_thought(visual_patches, "Describe detalladamente la imagen.")
print(f"✓ Núcleo condensador listo en {(time.time() - t_set0)*1000:.2f} ms.")

print("\n=================================================================================")
print(" 🚀 INFERENCIA CON CONDENSACIÓN DE FLUIDO ACTIVA (140 TOKENS):")
print("=================================================================================\n")

t0_tok = None
count = 0
for resp in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=140):
    if t0_tok is None:
        t0_tok = time.time()
    print(resp.text, end="", flush=True)
    count += 1

t_gen = time.time() - (t0_tok if t0_tok else time.time())
tps = count / (t_gen + 1e-12)

print("\n\n=================================================================================")
print(" 📊 TELEMETRÍA DE LA CONDENSACIÓN EN HARDWARE:")
print("=================================================================================")
print(f"• Tokens Emitidos              : {count}")
print(f"• Tiempo de Generación         : {t_gen:.2f} s")
print(f"• Velocidad Sostenida en GPU   : {tps:.2f} tok/s")
print(f"• Mecanismo de Colapso         : Condensación de Hertz-Knudsen (ν=0.12, γ=0.35)")
print("=================================================================================")
