import os, sys, struct
import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_vlm import load

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")

img_candidates = ["/Users/crotalo/Downloads/005.jpg", "/Users/crotalo/Downloads/005.jpeg"]
img_path = None
for p in img_candidates:
    if os.path.exists(p):
        img_path = p
        break

if not img_path:
    print("❌ Error: No se encontró 005.jpg")
    sys.exit(1)

model, processor = load(model_path)
img = Image.open(img_path).convert("RGB")
W_orig, H_orig = img.size

# Calibración canónica a factor 32 para entrar en el buffer KV
factor = 32
max_dim = 768
scale = max_dim / max(W_orig, H_orig)
W_new = int((W_orig * scale) // factor) * factor
H_new = int((H_orig * scale) // factor) * factor

img_calibrated = img.resize((W_new, H_new), Image.Resampling.BICUBIC)
print(f"✓ Imagen calibrada a: {W_new}x{H_new} (Original: {W_orig}x{H_orig})")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe detalladamente los elementos, colores, objetos y contexto que se observan en la imagen."}
        ]
    }
]

prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[prompt_text], images=[img_calibrated], return_tensors="mlx")

print("• Ejecutando Torre de Visión oficial...")
out = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])
visual_features = out[0]
mx.eval(visual_features)

n_patches = visual_features.shape[0]
print(f"✓ {n_patches} parches semánticos (Dimensión: {visual_features.shape[1]}).")

visual_np = np.array(visual_features, dtype=np.float32)
visual_np.tofile("visual_embeddings.bin")
print(f"✓ visual_embeddings.bin generado: {os.path.getsize('visual_embeddings.bin')} bytes (~9.3 MB esperado).")

input_ids = inputs["input_ids"][0].tolist()
with open("prompt_input.bin", "wb") as f:
    for t in input_ids:
        f.write(struct.pack("I", t))

print(f"✓ prompt_input.bin compilado: {len(input_ids)} tokens (Dentro del límite de buffer).")
