import sys, os, time
import mlx.core as mx
import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/crotalo/aether_engine")
from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

print("=================================================================================")
print(" 🔬 BARRIDO DE SENSIBILIDAD EN SILICIO C++: RUPTURA DE LA BARRERA DEL RLHF")
print("    Búsqueda del punto de bifurcación fáctica (kappa, theta, nu)")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print("• 1. Cargando Qwen 27B en UMA...")
model, processor = load(model_path)
aether = AetherEngine(model, processor)

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe detalladamente la imagen."}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
img = Image.open(img_path).convert("RGB")

inputs = processor(text=[prompt], images=[img], return_tensors="mlx")
visual_patches = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])[0]

# Pensamiento Profundo tau* = 32
print("• 2. Asentamiento de coherencia en GPU...")
telemetria = aether.prepare_multimodal_thought(visual_patches, "Describe detalladamente la imagen.")
print("✓ Atractor L* asentado.")

# Matriz de Sensibilidad para quebrar el RLHF
matriz_pruebas = [
    {"nombre": "Base (Control Actual)",     "theta": 0.35, "kappa": 0.15, "nu": 0.12},
    {"nombre": "Aumento Nucleación (x3)",   "theta": 0.35, "kappa": 0.50, "nu": 0.12},
    {"nombre": "Aumento Nucleación (x7)",   "theta": 0.35, "kappa": 1.00, "nu": 0.12},
    {"nombre": "Curvatura + Nucleación",    "theta": 0.80, "kappa": 1.00, "nu": 0.12},
    {"nombre": "Alta Viscosidad Laminar",   "theta": 0.80, "kappa": 1.00, "nu": 0.30},
    {"nombre": "Bifurcación Fáctica Fuerte","theta": 1.20, "kappa": 2.00, "nu": 0.25},
    {"nombre": "Ruptura Total de RLHF",     "theta": 1.50, "kappa": 3.50, "nu": 0.35},
]

print("\n=================================================================================================================")
print(f"{'Configuración':<30} | {'Theta':<5} | {'Kappa':<5} | {'Nu':<4} | {'Primeros 35 Tokens Generados'}")
print("=================================================================================================================")

for exp in matriz_pruebas:
    th = exp["theta"]
    kp = exp["kappa"]
    nu_val = exp["nu"]
    gamma_val = 0.35

    # Inyectar parámetros en el motor C++ en caliente
    for hook in aether.hooked_layers:
        hook.theta_step = th / float(aether.num_layers)
    
    aether.hooked_head.nu = nu_val
    aether.hooked_head.kappa_n = kp
    aether.hooked_head.gamma = gamma_val
    aether.hooked_head.norm_factor = gamma_val / np.sqrt(1.0 + nu_val ** 2)

    # Generar 35 tokens para ver el inicio del discurso
    text_sample = ""
    token_count = 0
    for resp in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=35):
        text_sample += resp.text
        token_count += 1
        if token_count >= 35:
            break

    # Limpiar saltos de línea para la tabla
    text_clean = text_sample.replace("\n", " ").strip()
    if len(text_clean) > 55:
        text_clean = text_clean[:52] + "..."

    print(f"{exp['nombre']:<30} | {th:<5.2f} | {kp:<5.2f} | {nu_val:<4.2f} | {text_clean}")

print("=================================================================================================================")
