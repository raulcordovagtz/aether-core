import sys, os, time
import mlx.core as mx
import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/crotalo/aether_engine")
from core_vlm.utils import load
from core_vlm.aether import AetherEngine

print("=================================================================================")
print(" 🔬 OBSERVATORIO DE SENSIBILIDAD: IMPACTO DIRECTO SOBRE 248,320 LOGITS")
print("    Medición sin texto: Delta z, Divergencia KL y Reordenamiento de Selección")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"
model, processor = load(model_path)
tokenizer = processor.tokenizer

# Preparar entrada multimodal oficial
img = Image.open(img_path).convert("RGB")
prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe detalladamente la imagen."}]}], add_generation_prompt=True)
inputs = processor(text=[prompt], images=[img], return_tensors="mlx")

# Instanciar el motor sobre el modelo
aether = AetherEngine(model, processor)

# Extraer atractor visual u_vis y correr prefill thought tau*=32
visual_patches = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])[0]
mx.eval(visual_patches)
telemetria = aether.prepare_multimodal_thought(visual_patches, "Describe detalladamente la imagen.")

# 1. PASO BASAL: VANILLA PURA (MOTOR APAGADO)
print("• 1. Obteniendo vector de logits basal z_vanilla (Motor Apagado)...")
for hook in aether.hooked_layers:
    hook.active = False
aether.hooked_head.active = False

cache_vanilla = model.make_cache()
logits_vanilla = model(
    inputs["input_ids"],
    pixel_values=inputs.get("pixel_values"),
    image_grid_thw=inputs.get("image_grid_thw"),
    cache=cache_vanilla
)[0, -1, :] # Vector [248320]
mx.eval(logits_vanilla)

shift_v = logits_vanilla - mx.max(logits_vanilla)
p_vanilla = mx.exp(shift_v) / mx.sum(mx.exp(shift_v))
mx.eval(p_vanilla)

top5_v_idx = np.argsort(np.array(logits_vanilla))[-5:][::-1]
print("\n=== TOP-5 PALABRAS BASALES (VANILLA PURA) ===")
for rank, idx in enumerate(top5_v_idx):
    tok = tokenizer.decode([int(idx)])
    print(f"  #{rank+1}: Token {idx:<6} ('{tok}') | Logit = {float(logits_vanilla[idx]):.4f} | Prob = {float(p_vanilla[idx])*100:.2f}%")

# 2. BARRIDO DE SENSIBILIDAD CON EL MOTOR ACTIVO (VARIANDO THETA)
print("\n=========================================================================================================")
print(" 📊 BARRIDO DE SENSIBILIDAD GEODÉSICA: theta -> ||Delta z||, D_KL, Margen y Ganador")
print("=========================================================================================================")
print(f"{'Theta (rad)':<12} | {'||Delta z||_2':<14} | {'||Delta z||_inf':<15} | {'D_KL (nats)':<12} | {'Palabra Top-1 Ganadora'}")
print("---------------------------------------------------------------------------------------------------------")

theta_sweep = [0.05, 0.15, 0.35, 0.70, 1.20, 2.00]

for theta_val in theta_sweep:
    # Activar el motor con el ángulo de prueba
    for hook in aether.hooked_layers:
        hook.active = True
        hook.theta_step = theta_val / float(aether.num_layers)
    aether.hooked_head.active = True

    # Cache fresco para evitar contaminación entre condiciones
    cache_exp = model.make_cache()
    logits_steered = model(
        inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        cache=cache_exp
    )[0, -1, :]
    mx.eval(logits_steered)

    # Métricas de deformación física sobre el espacio de vocabulario
    delta_z = logits_steered - logits_vanilla
    delta_z_2 = float(mx.sqrt(mx.sum(delta_z * delta_z)))
    delta_z_inf = float(mx.max(mx.abs(delta_z)))

    shift_s = logits_steered - mx.max(logits_steered)
    p_steered = mx.exp(shift_s) / mx.sum(mx.exp(shift_s))
    
    # Divergencia KL: sum(p_v * log(p_v / p_s))
    kl = float(mx.sum(p_vanilla * (mx.log(p_vanilla + 1e-12) - mx.log(p_steered + 1e-12))))
    
    top1_idx = int(np.argmax(np.array(logits_steered)))
    top1_tok = tokenizer.decode([top1_idx])
    top1_prob = float(p_steered[top1_idx]) * 100.0

    print(f"{theta_val:<12.2f} | {delta_z_2:<14.4f} | {delta_z_inf:<15.4f} | {kl:<12.6f} | '{top1_tok}' ({top1_prob:.1f}%)")

print("=========================================================================================================")
