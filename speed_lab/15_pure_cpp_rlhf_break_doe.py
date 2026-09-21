import sys, os, time
import mlx.core as mx
import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/crotalo/aether_engine")
from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine
import aether_vlm.coupler as coupler

print("=================================================================================")
print(" 🔬 BARRIDO DE SENSIBILIDAD C++ NATIVO (100% EN SILICIO METAL)")
print("    Búsqueda de la Bifurcación Fáctica vs Inercia de RLHF (Qwen 27B)")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print("• 1. Cargando arquitectura Qwen 27B en UMA...")
t0 = time.time()
model, processor = load(model_path)
print(f"✓ Modelo montado en {time.time() - t0:.2f} s.")

aether = AetherEngine(model, processor)

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe detalladamente la imagen."}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
img = Image.open(img_path).convert("RGB")

inputs = processor(text=[prompt], images=[img], return_tensors="mlx")
visual_patches = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])[0]

print("\n• 2. Asentamiento de Pensamiento Profundo tau* = 32...")
t_set0 = time.time()
telemetria = aether.prepare_multimodal_thought(visual_patches, "Describe detalladamente la imagen.")
print(f"✓ Atractor L* asentado en {(time.time() - t_set0)*1000:.2f} ms.")

# Matriz de Sensibilidad Física sobre el binario C++
matriz_experimentos = [
    {"nombre": "1. Base (Control Actual)",     "theta": 0.35, "kappa": 0.15, "nu": 0.12, "gamma": 0.35},
    {"nombre": "2. Impulso de Nucleación",     "theta": 0.35, "kappa": 0.80, "nu": 0.12, "gamma": 0.35},
    {"nombre": "3. Viscosidad Laminar Alta",   "theta": 0.35, "kappa": 0.80, "nu": 0.30, "gamma": 0.35},
    {"nombre": "4. Choque Cinético Fuerte",    "theta": 0.70, "kappa": 1.20, "nu": 0.20, "gamma": 0.80},
    {"nombre": "5. Curvatura Profunda",        "theta": 1.50, "kappa": 1.50, "nu": 0.25, "gamma": 0.80},
    {"nombre": "6. Bifurcación Fáctica",       "theta": 1.80, "kappa": 3.00, "nu": 0.35, "gamma": 1.20},
    {"nombre": "7. Nucleación Crítica",        "theta": 2.00, "kappa": 5.00, "nu": 0.40, "gamma": 1.50},
]

# Asegurar que el AetherCollapseHead permite modificar los parámetros dinámicamente
head = aether.hooked_head

print("\n==================================================================================================================================")
print(f"{'Configuración Física':<28} | {'θ':<4} | {'κ':<4} | {'ν':<4} | {'γ':<4} | {'tok/s':<6} | {'Primeros 40 Tokens Emitidos en C++'}")
print("==================================================================================================================================")

for exp in matriz_experimentos:
    th = exp["theta"]
    kp = exp["kappa"]
    nu_val = exp["nu"]
    gm = exp["gamma"]

    # 1. Configurar rotación geodésica en las 64 capas
    for hook in aether.hooked_layers:
        hook.theta_step = th / float(aether.num_layers)

    # 2. Configurar la llamada C++ nativa en el Head
    def make_custom_collapse_call(nu_p, gamma_p, kappa_p):
        def custom_call(h):
            if not head.active or head.state_ref is None:
                return head.original_lm_head(h)
            v_drag = head.state_ref.get("v_drag")
            z_L_star = head.state_ref.get("z_L_star")
            if v_drag is None or z_L_star is None:
                return head.original_lm_head(h)

            return coupler.aether_native_c.dispatch_full_collapse(
                h, v_drag, z_L_star,
                head.head_w, head.head_scales, head.head_biases,
                head.group_size, head.bits,
                nu_p, gamma_p, kappa_p
            )
        return custom_call

    head.__call__ = make_custom_collapse_call(nu_val, gm, kp)

    # 3. Generar 40 tokens y medir velocidad real
    t0_gen = time.time()
    text_sample = ""
    token_count = 0

    for resp in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=40):
        text_sample += resp.text
        token_count += 1
        if token_count >= 40:
            break

    elapsed = time.time() - t0_gen
    tps = token_count / (elapsed + 1e-12)

    text_clean = text_sample.replace("\n", " ").strip()
    if len(text_clean) > 58:
        text_clean = text_clean[:55] + "..."

    print(f"{exp['nombre']:<28} | {th:<4.2f} | {kp:<4.2f} | {nu_val:<4.2f} | {gm:<4.2f} | {tps:<6.2f} | {text_clean}")

print("==================================================================================================================================")
