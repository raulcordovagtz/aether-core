import os, struct
import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_vlm import load

print("=================================================================================")
print(" 📷 EXTRACCIÓN CANÓNICA: 27 CAPAS ViT (Qwen 3.5 en GPU Metal)")
print("=================================================================================")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print("• Cargando Torre de Visión oficial...")
model, processor = load(model_path)

img = Image.open(img_path).convert("RGB")
print(f"✓ Imagen 005.jpg cargada: {img.size[0]}x{img.size[1]} píxeles.")

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
print("✓ Prompt formateado con Chat Template oficial de Qwen.")

# Preprocesar en MLX
inputs = processor(text=[prompt_text], images=[img], return_tensors="mlx")

# Pasar por las 27 capas ViT en GPU
out = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])
visual_features = out[0] # Tensor [644, 5120]
mx.eval(visual_features)

visual_np = np.array(visual_features, dtype=np.float32)
print(f"✓ Parches ViT contextualizados: {visual_np.shape[0]} parches de dimensión {visual_np.shape[1]}.")

# 1. Guardar visual_embeddings.bin
visual_np.tofile("visual_embeddings.bin")
print(f"✓ visual_embeddings.bin generado ({os.path.getsize('visual_embeddings.bin')} bytes, 644 parches reales).")

# 2. Compilar prompt_input.bin
input_ids = np.array(inputs["input_ids"][0]).tolist()
tokens_bin = [int(t) for t in input_ids]

# Cerrar el bloque <think> para forzar la descripción inmediata
# Añadimos los tokens: <think>\n\n</think>\n
think_suffix = processor.tokenizer.encode("<think>\n\n</think>\n", add_special_tokens=False)
tokens_bin.extend(think_suffix)

with open("prompt_input.bin", "wb") as f:
    for t in tokens_bin:
        f.write(struct.pack("I", t))

print(f"✓ prompt_input.bin compilado: {len(tokens_bin)} tokens (644 parches integrados + sufijo cerrado de razonamiento).")
print("=================================================================================")
print(" 🏆 FOTONES REALES Y PROMPT CANÓNICO LISTOS PARA AETHER ENGINE.")
print("=================================================================================")
