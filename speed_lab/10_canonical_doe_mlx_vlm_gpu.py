import sys, os
sys.path.insert(0, "/Users/crotalo/aether_engine")
import sys, os, time
import mlx.core as mx
import numpy as np
from PIL import Image

# 100% MLX_VLM OFICIAL
from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

print("=================================================================================")
print(" 🔬 DoE & ANÁLISIS DE SENSIBILIDAD GLOBAL: 100% GPU METAL (mlx_vlm OFICIAL)")
print(f"    Dispositivo: {mx.default_device()} | Backend: Native Apple Silicon UMA")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print("• 1. Cargando Qwen 27B en GPU...")
t0 = time.time()
model, processor = load(model_path)
print(f"✓ Modelo cargado en {time.time() - t0:.2f} s.")

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe la imagen."}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# Instanciar el motor de AETHER soberano
aether = AetherEngine(model, processor)

# Ingestión de Visión y Pensamiento Profundo tau* = 32
print("\n• 2. Ingestión de fotones reales y Pensamiento Profundo en GPU...")
img = Image.open(img_path).convert("RGB")
inputs_temp = processor(text=[prompt], images=[img], return_tensors="mlx")
visual_patches = model.vision_tower(inputs_temp["pixel_values"], inputs_temp["image_grid_thw"])[0]
mx.eval(visual_patches)

t_set0 = time.time()
telemetria = aether.prepare_multimodal_thought(visual_patches, "Describe la imagen.")
mx.eval(aether.settled_intent)
print(f"✓ Pensamiento Profundo asentado en {(time.time() - t_set0)*1000:.2f} ms.")

# ─── 3. OBTENCIÓN DE LA DISTRIBUCIÓN BASAL Y_base (VANILLA PURA) ─────────────
print("\n• 3. Registrando distribución basal Y_base (Motor Apagado)...")
for hook in aether.hooked_layers: hook.active = False
aether.hooked_head.active = False

for resp in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=1):
    logprobs_base = resp.logprobs.astype(mx.float32)
    tok_base = resp.token
    break

mx.eval(logprobs_base)
probs_base = mx.exp(logprobs_base)
mx.eval(probs_base)
print(f"✓ Distribución basal capturada (Token #1: '{processor.tokenizer.decode([tok_base])}', Prob: {float(probs_base[tok_base])*100:.2f}%).")

# ─── 4. MATRIZ FACTORIAL DEL DISEÑO DE EXPERIMENTOS ──────────────────────────
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
print(" 🔬 MATRIZ DE RESULTADOS (SUPERFICIE DE RESPUESTA SOBRE 248,320 LOGITS EN GPU)")
print("=====================================================================================================================")
print(f"{'Exp ID':<7} | {'Theta':<5} | {'Topología':<11} | {'Head':<4} | {'||Δlogp||_2':<13} | {'||Δlogp||_inf':<14} | {'D_KL (nats)':<11} | {'Top-1 Ganador'} | {'Prob':<6} | {'Tiempo':<7}")
print("---------------------------------------------------------------------------------------------------------------------")

for exp in experimentos:
    t_start = time.perf_counter()
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

    # Inferencia de 1 token en GPU pura oficial
    for resp in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=1):
        lp = resp.logprobs.astype(mx.float32)
        top_id = resp.token
        break

    mx.eval(lp)
    t_eval = (time.perf_counter() - t_start) * 1000.0

    # Cálculo métrico en GPU
    dlp = lp - logprobs_base
    dlp_2 = float(mx.sqrt(mx.sum(dlp * dlp)))
    dlp_inf = float(mx.max(mx.abs(dlp)))
    
    pr = mx.exp(lp)
    kl = float(mx.sum(probs_base * (logprobs_base - lp)))
    
    top_tok = processor.tokenizer.decode([top_id])
    top_p = float(pr[top_id]) * 100.0

    print(f"{exp['id']:<7} | {th:<5.2f} | {topo:<11} | {hd:<4.1f} | {dlp_2:<13.2f} | {dlp_inf:<14.3f} | {kl:<11.5f} | '{top_tok}'{(' '*(12-len(top_tok)))} | {top_p:<5.1f}% | {t_eval:<6.1f}ms")

print("=====================================================================================================================")
