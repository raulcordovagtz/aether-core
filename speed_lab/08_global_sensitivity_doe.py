import sys, os, time, math
import mlx.core as mx
import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/crotalo/aether_engine")
from core_vlm.utils import load
from core_vlm.generate.ar import generate_step
from core_vlm.generate.dispatch import _prepare_generation_inputs
from core_vlm.aether import AetherEngine

print("=================================================================================")
print(" 🔬 EXPERIMENTO GLOBAL: DISEÑO DE EXPERIMENTOS (DoE) & SENSIBILIDAD EN SILICIO")
print("    Mapeo de la Superficie de Respuesta de 248,320 Logits en Qwen 27B")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print("• 1. Cargando arquitectura en UMA...")
model, processor = load(model_path)
tokenizer = processor.tokenizer

# Preparación multimodal
prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe detalladamente la imagen."}]}], add_generation_prompt=True)
gen_kwargs = {}
input_ids, pixel_values, mask, _ = _prepare_generation_inputs(model, processor, prompt, img_path, None, None, gen_kwargs)

aether = AetherEngine(model, processor)
visual_patches = model.vision_tower(pixel_values, gen_kwargs["image_grid_thw"])[0]
mx.eval(visual_patches)
telemetria = aether.prepare_multimodal_thought(visual_patches, "Describe detalladamente la imagen.")

# ─── MEDICIÓN DE CONTROL BASAL (VANILLA) ─────────────────────────────────────
print("• 2. Registrando distribución de referencia Y_base...")
for hook in aether.hooked_layers: hook.active = False
aether.hooked_head.active = False

gen_base = generate_step(input_ids, model, pixel_values, mask, **gen_kwargs)
_, logprobs_base = next(gen_base)
logprobs_base = logprobs_base.astype(mx.float32)
mx.eval(logprobs_base)
probs_base = mx.exp(logprobs_base)
mx.eval(probs_base)

# ─── MATRIZ DE DISEÑO EXPERIMENTAL (BATERÍA FACTORIAL ORTOGONAL) ─────────────
# Factores:
# F1: Torsión total Theta ∈ [0.10, 0.70, 1.50]
# F2: Topología de Capas ∈ {"Pinch_L24", "Band_16_38", "All_64"}
# F3: Modulación Head ∈ {0.0, 0.3}

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
print(" 🔬 MATRIZ DE RESULTADOS EXPERIMENTALES (SUPERFICIE DE RESPUESTA SOBRE 248,320 LOGITS)")
print("=====================================================================================================================")
print(f"{'Exp ID':<7} | {'Theta':<5} | {'Topología':<11} | {'Head':<4} | {'||Δz||_2':<10} | {'||Δz||_inf':<10} | {'D_KL (nats)':<11} | {'Top-1 Ganador'} | {'Prob':<6} | {'Tiempo':<7}")
print("---------------------------------------------------------------------------------------------------------------------")

for exp in experimentos:
    t_start = time.perf_counter()
    th = exp["theta"]
    topo = exp["topo"]
    hd = exp["head"]

    # Configuración topológica de capas
    for idx, hook in enumerate(aether.hooked_layers):
        hook.active = False
        if topo == "Pinch_L24" and idx == 24:
            hook.active = True
            hook.theta_step = th
        elif topo == "Band_16_38" and (16 <= idx <= 38):
            hook.active = True
            hook.theta_step = th / 23.0 # Distribuido en las 23 capas causales
        elif topo == "All_64":
            hook.active = True
            hook.theta_step = th / 64.0

    # Configuración de Capa de Colapso
    aether.hooked_head.active = (hd > 0.0)

    # Disparo de evaluación
    gen = generate_step(input_ids, model, pixel_values, mask, **gen_kwargs)
    _, lp = next(gen)
    lp = lp.astype(mx.float32)
    mx.eval(lp)
    t_eval = (time.perf_counter() - t_start) * 1000.0

    # Métricas de Variedad
    dlp = lp - logprobs_base
    dlp_2 = float(mx.sqrt(mx.sum(dlp * dlp)))
    dlp_inf = float(mx.max(mx.abs(dlp)))
    
    pr = mx.exp(lp)
    kl = float(mx.sum(probs_base * (logprobs_base - lp)))
    
    top_id = int(mx.argmax(lp))
    top_tok = tokenizer.decode([top_id])
    top_p = float(pr[top_id]) * 100.0

    print(f"{exp['id']:<7} | {th:<5.2f} | {topo:<11} | {hd:<4.1f} | {dlp_2:<10.2f} | {dlp_inf:<10.3f} | {kl:<11.5f} | '{top_tok}'{(' '*(12-len(top_tok)))} | {top_p:<5.1f}% | {t_eval:<6.1f}ms")

print("=====================================================================================================================")
