import sys, os, time
import mlx.core as mx
import numpy as np
from PIL import Image

# IMPORTACIÓN EXCLUSIVA DE MLX_VLM OFICIAL
import mlx_vlm
from mlx_vlm import load
from mlx_vlm.utils import generate_step, prepare_inputs
from aether_vlm import AetherEngine

print("=================================================================================")
print(" 🚀 DoE EN GPU PURA (Apple M2 Max) SOBRE mlx_vlm OFICIAL")
print(f"    Dispositivo Activo: {mx.default_device()} | mlx_vlm: {mlx_vlm.__file__}")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print("• 1. Cargando Qwen 27B en GPU Metal...")
t0 = time.time()
model, processor = load(model_path)
print(f"✓ Modelo cargado en UMA en {time.time() - t0:.2f} s.")

# Ingestión de imagen y preparación de inputs en mlx_vlm oficial
img = Image.open(img_path).convert("RGB")
prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe detalladamente la imagen."}]}], add_generation_prompt=True)

# Preparar inputs con la API canónica de mlx_vlm
inputs = prepare_inputs(processor, images=[img], prompt=prompt)
input_ids = inputs["input_ids"]
pixel_values = inputs.get("pixel_values")
mask = inputs.get("mask")

# Conectar el motor de AETHER
aether = AetherEngine(model, processor)

# Ingestión de Visión y Pensamiento Profundo tau* = 32
print("\n• 2. Procesando Torre de Visión y Pensamiento Profundo (tau* = 32 en GPU)...")
grid_thw = inputs.get("image_grid_thw")
visual_patches = model.vision_tower(pixel_values, grid_thw)[0]
mx.eval(visual_patches)

t_set0 = time.time()
telemetria = aether.prepare_multimodal_thought(visual_patches, "Describe detalladamente la imagen.")
mx.eval(aether.settled_intent)
print(f"✓ Pensamiento Profundo asentado en {(time.time() - t_set0)*1000:.2f} ms.")

# ─── PREFILL CÁLIDO (SE EJECUTA UNA SOLA VEZ EN GPU) ─────────────────────────
print("\n• 3. Ejecutando Prefill multimodal cálido en GPU...")
for hook in aether.hooked_layers: hook.active = False
aether.hooked_head.active = False

t_pref0 = time.time()
gen_base = generate_step(input_ids, model, pixel_values, mask, image_grid_thw=grid_thw)
tok_base, logprobs_base = next(gen_base)
logprobs_base = logprobs_base.astype(mx.float32)
mx.eval(logprobs_base)
probs_base = mx.exp(logprobs_base)
mx.eval(probs_base)
print(f"✓ Prefill inicial completado en {(time.time() - t_pref0):.2f} s (Token 1 base: '{processor.tokenizer.decode([tok_base])}').")

# ─── MATRIZ FACTORIAL ORTOGONAL EN GPU ───────────────────────────────────────
experimentos = [
    {"id": "EXP-01", "theta": 0.10, "topo": "Pinch_L24",  "head": 0.0},
    {"id": "EXP-02", "theta": 0.70, "topo": "Pinch_L24",  "head": 0.0},
    {"id": "EXP-03", "theta": 1.50, "topo": "Pinch_L24",  "head": 0.0},
    {"id": "EXP-04", "theta": 0.10, "topo": "Band_16_38", "head": 0.0},
    {"id": "EXP-05", "theta": 0.70, "topo": "Band_16_38", "head": 0.0},
    {"id": "EXP-06", "theta": 1.50, "topo": "Band_16_38", "head": 0.0},
    {"id": "EXP-07", "theta": 0.10, "topo": "All_64",      "head": 0.0},
    {"id": "EXP-08", "theta": 0.70, "topo": "All_64",      "head": 0.0},
    {"id": "EXP-09", "theta": 1.50, "topo": "All_64",      "head": 0.0},
    {"id": "EXP-10", "theta": 0.70, "topo": "Band_16_38", "head": 0.3},
    {"id": "EXP-11", "theta": 1.50, "topo": "Band_16_38", "head": 0.3},
]

print("\n=====================================================================================================================")
print(" 🔬 RESULTADOS EN SILICIO GPU (mlx_vlm OFICIAL): SUPERFICIE DE 248,320 LOGITS")
print("=====================================================================================================================")
print(f"{'Exp ID':<7} | {'Theta':<5} | {'Topología':<11} | {'Head':<4} | {'||Δz||_2':<10} | {'||Δz||_inf':<10} | {'D_KL (nats)':<11} | {'Top-1 Ganador'} | {'Prob':<6}")
print("---------------------------------------------------------------------------------------------------------------------")

for exp in experimentos:
    th = exp["theta"]
    topo = exp["topo"]
    hd = exp["head"]

    # Configurar topología en el motor
    for idx, hook in enumerate(aether.hooked_layers):
        hook.active = False
        if topo == "Pinch_L24" and idx == 24:
            hook.active = True
            hook.theta_step = th
        elif topo == "Band_16_38" and (16 <= idx <= 38):
            hook.active = True
            hook.theta_step = th / 23.0
        elif topo == "All_64":
            hook.active = True
            hook.theta_step = th / 64.0

    aether.hooked_head.active = (hd > 0.0)

    # Disparo en GPU Metal
    gen = generate_step(input_ids, model, pixel_values, mask, image_grid_thw=grid_thw)
    _, lp = next(gen)
    lp = lp.astype(mx.float32)
    mx.eval(lp)

    # Cálculo métrico en GPU
    dlp = lp - logprobs_base
    dlp_2 = float(mx.sqrt(mx.sum(dlp * dlp)))
    dlp_inf = float(mx.max(mx.abs(dlp)))
    
    pr = mx.exp(lp)
    kl = float(mx.sum(probs_base * (logprobs_base - lp)))
    
    top_id = int(mx.argmax(lp))
    top_tok = processor.tokenizer.decode([top_id])
    top_p = float(pr[top_id]) * 100.0

    print(f"{exp['id']:<7} | {th:<5.2f} | {topo:<11} | {hd:<4.1f} | {dlp_2:<10.2f} | {dlp_inf:<10.3f} | {kl:<11.5f} | '{top_tok}'{(' '*(12-len(top_tok)))} | {top_p:<5.1f}%")

print("=====================================================================================================================")
