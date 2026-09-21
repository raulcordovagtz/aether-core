import sys, os, time
import mlx.core as mx
import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/crotalo/aether_engine")
from core_vlm.utils import load
from core_vlm.generate.ar import generate_step
from core_vlm.generate.dispatch import _prepare_generation_inputs
from core_vlm.aether import AetherEngine

print("=================================================================================")
print(" 🔬 AUDITORÍA DE SENSIBILIDAD: DEFORMACIÓN REAL SOBRE 248,320 LOGITS")
print("    Medición directa sin texto: Delta logp, Divergencia KL y Selección Top-1")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"
model, processor = load(model_path)
tokenizer = processor.tokenizer

prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe detalladamente la imagen."}]}], add_generation_prompt=True)

gen_kwargs = {}
input_ids, pixel_values, mask, _ = _prepare_generation_inputs(model, processor, prompt, img_path, None, None, gen_kwargs)

# Conectar el motor de AETHER
aether = AetherEngine(model, processor)

# Ingestión de Visión y Pensamiento Profundo tau* = 32
visual_patches = model.vision_tower(pixel_values, gen_kwargs["image_grid_thw"])[0]
mx.eval(visual_patches)
telemetria = aether.prepare_multimodal_thought(visual_patches, "Describe detalladamente la imagen.")

# 1. CONDICIÓN BASAL: VANILLA (MOTOR APAGADO)
print("• 1. Obteniendo distribución basal de 248,320 dimensiones (Vanilla)...")
for hook in aether.hooked_layers:
    hook.active = False
aether.hooked_head.active = False

gen_v = generate_step(input_ids, model, pixel_values, mask, **gen_kwargs)
tok_v, logprobs_v = next(gen_v)
logprobs_v = logprobs_v.astype(mx.float32)
mx.eval(logprobs_v)
probs_v = mx.exp(logprobs_v)
mx.eval(probs_v)

top5_v_idx = np.argsort(np.array(logprobs_v))[-5:][::-1]
print("\n=== TOP-5 PALABRAS BASALES (VANILLA PURA) ===")
for rank, idx in enumerate(top5_v_idx):
    tok = tokenizer.decode([int(idx)])
    print(f"  #{rank+1}: Token {idx:<6} ('{tok}') | Logprob = {float(logprobs_v[idx]):.4f} | Prob = {float(probs_v[idx])*100:.2f}%")

# 2. BARRIDO DE SENSIBILIDAD GEODÉSICA
print("\n=========================================================================================================")
print(" 📊 BARRIDO DE SENSIBILIDAD: theta -> ||Delta logp||, D_KL y Selección Top-1")
print("=========================================================================================================")
print(f"{'Theta (rad)':<12} | {'||Delta logp||_2':<16} | {'||Delta logp||_inf':<18} | {'D_KL (nats)':<12} | {'Token Top-1 Ganador'}")
print("---------------------------------------------------------------------------------------------------------")

theta_sweep = [0.05, 0.15, 0.35, 0.70, 1.20, 2.00]

for theta_val in theta_sweep:
    # Activar el motor en todas las capas con el ángulo de barrido
    for hook in aether.hooked_layers:
        hook.active = True
        hook.theta_step = theta_val / float(aether.num_layers)
    aether.hooked_head.active = True

    # Generar el primer token con el motor activo
    gen_s = generate_step(input_ids, model, pixel_values, mask, **gen_kwargs)
    tok_s, logprobs_s = next(gen_s)
    logprobs_s = logprobs_s.astype(mx.float32)
    mx.eval(logprobs_s)

    # Métricas de deformación matemática del espacio de 248,320 dimensiones
    delta_lp = logprobs_s - logprobs_v
    delta_lp_2 = float(mx.sqrt(mx.sum(delta_lp * delta_lp)))
    delta_lp_inf = float(mx.max(mx.abs(delta_lp)))

    probs_s = mx.exp(logprobs_s)
    
    # Divergencia KL exacta: sum(p_v * (logp_v - logp_s))
    kl = float(mx.sum(probs_v * (logprobs_v - logprobs_s)))
    
    top1_idx = int(mx.argmax(logprobs_s))
    top1_tok = tokenizer.decode([top1_idx])
    top1_prob = float(probs_s[top1_idx]) * 100.0

    print(f"{theta_val:<12.2f} | {delta_lp_2:<16.4f} | {delta_lp_inf:<18.4f} | {kl:<12.6f} | '{top1_tok}' ({top1_prob:.1f}%)")

print("=========================================================================================================")
