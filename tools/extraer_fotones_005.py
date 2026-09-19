import os, sys, struct
import numpy as np

# Intentar usar el procesador de imágenes del modelo local
try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor
except ImportError:
    print("❌ Requiere pillow y transformers para procesar 005.jpg.")
    sys.exit(1)

img_path = "/Users/crotalo/Downloads/005.jpg"
if not os.path.exists(img_path):
    print(f"❌ No se encontró la imagen: {img_path}")
    sys.exit(1)

img = Image.open(img_path).convert("RGB")
print(f"✓ Imagen cargada: {img.size[0]}x{img.size[1]} píxeles.")

# Directorio del modelo en LM Studio
model_dir = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
if not os.path.exists(model_dir):
    # Buscar alternativa si la ruta cambió
    alt_dirs = [d for d in os.listdir(os.path.expanduser("~/.lmstudio/models")) if "Qwen" in d]
    print(f"Rutas disponibles: {alt_dirs}")

# Guardar dimensiones crudas para validación
print(f"✓ Guardando metadatos para extracción directa...")
