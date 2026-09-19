import os, sys, struct
from PIL import Image
import numpy as np

img_path = "/Users/crotalo/Downloads/005.jpg"
if not os.path.exists(img_path):
    print(f"❌ Error: No existe la imagen en {img_path}")
    sys.exit(1)

print("=================================================================================")
print(" 📷 EXTRACTOR AUTÓNOMO DE VISIÓN AETHER-VL (QWEN-VL PIPELINE)")
print("=================================================================================")

# Usar el procesador nativo de Qwen-VL disponible en el entorno
try:
    from transformers import Qwen2VLProcessor, AutoProcessor
    import torch
except ImportError:
    print("• Instalando o verificando dependencias locales...")
    os.system("pip install -q torchvision pillow")
    from transformers import AutoProcessor
    import torch

# Buscar modelo local en LM Studio
model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")

try:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("✓ AutoProcessor de Qwen-VL cargado desde el modelo local.")
except Exception as e:
    print(f"• Usando inicialización por procesador Qwen base: {e}")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

img = Image.open(img_path).convert("RGB")
print(f"✓ Imagen 005.jpg cargada: {img.size[0]}x{img.size[1]} píxeles.")

# Procesar imagen a través del Image Processor oficial
inputs = processor(images=img, text="<|vision_start|><|image_pad|><|vision_end|>", return_tensors="pt")
pixel_values = inputs.get("pixel_values")
image_grid_thw = inputs.get("image_grid_thw")

print(f"✓ Píxeles procesados: tensor de forma {pixel_values.shape}, Grid THW: {image_grid_thw.tolist()}")

# Para AETHER C-007, guardamos los píxeles calibrados normalizados y empaquetados
# Los pesos de patch_embed.proj se encargan de proyectar en UMA a D=5120
