import os, sys, struct
import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_vlm import load

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/test.jpg"

if not os.path.exists(img_path):
    print(f"❌ Error: No existe {img_path}")
    sys.exit(1)

print(f"• Cargando modelo y Torre de Visión oficial...")
model, processor = load(model_path)

img = Image.open(img_path).convert("RGB")
print(f"✓ Imagen test.jpg cargada: {img.size[0]}x{img.size[1]} píxeles.")

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

# Preprocesar en MLX
inputs = processor(text=[prompt_text], images=[img], return_tensors="mlx")

# Pasar por la Torre de Visión oficial en GPU (ViT completo)
print("• Ejecutando Torre de Visión (ViT completo) en GPU Metal...")
out = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])
visual_features = out[0] # Tensor [N_patches, 5120]
mx.eval(visual_features)

n_patches = visual_features.shape[0]
print(f"✓ {n_patches} parches semánticos extraídos por el ViT en GPU (Dimensión: {visual_features.shape[1]}).")

# Guardar visual_embeddings.bin oficial
visual_np = np.array(visual_features, dtype=np.float32)
visual_np.tofile("visual_embeddings.bin")
print(f"✓ visual_embeddings.bin generado ({os.path.getsize('visual_embeddings.bin')} bytes).")

# Extraer tokens del prompt con la cantidad exacta de parches
input_ids = inputs["input_ids"][0].tolist()

with open("prompt_input.bin", "wb") as f:
    for t in input_ids:
        f.write(struct.pack("I", t))

print(f"✓ prompt_input.bin compilado: {len(input_ids)} tokens exactos del pipeline oficial.")
print("=================================================================================")
print(" 🏆 EXTRACCIÓN ViT CONCLUIDA: El motor ya tiene los fotones reales del buque.")
print("=================================================================================")
