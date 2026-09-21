import sys, os, time
import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/crotalo/aether_engine")

from mlx_vlm import load
from mlx_vlm.generate.dispatch import _prepare_generation_inputs
from aether_vlm import AetherEngine

print("=================================================================================")
print(" ⚡ DoE EN SILICIO NATIVO: EVALUACIÓN EN CACHE CALIENTE (GPU APPLE M2 MAX)")
print("    Meta: ~80 ms por punto (Prefill ejecutado UNA SOLA VEZ en GPU)")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print("• 1. Cargando arquitectura en UMA...")
t0 = time.time()
model, processor = load(model_path)
print(f"✓ Modelo cargado en {time.time() - t0:.2f} s.")

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe detalladamente la imagen."}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

gen_kwargs = {}
input_ids, pixel_values, mask, _ = _prepare_generation_inputs(model, processor, prompt, img_path, None, None, gen_kwargs)

aether = AetherEngine(model, processor)

# Ingestión de Visión y Pensamiento Profundo
print("\n• 2. Ingestión de ViT y Pensamiento Profundo continuo...")
visual_patches = model.vision_tower(pixel_values, gen_kwargs["image_grid_thw"])[0]
mx.eval(visual_patches)
telemetria = aether.prepare_multimodal_thought(visual_patches, "Describe detalladamente la imagen.")
mx.eval(aether.settled_intent)
print("✓ Pensamiento asentado en GPU.")

# ─── PREFILL MAESTRO: SE EJECUTA UNA SOLA VEZ ────────────────────────────────
print("\n• 3. Ejecutando Prefill Maestro y congelando KV Cache en GPU...")
for hook in aether.hooked_layers: hook.active = False
aether.hooked_head.active = False

master_cache = model.language_model.make_cache()

t_pref = time.time()
logits_prefill = model(
    input_ids,
    pixel_values=pixel_values,
    mask=mask,
    cache=master_cache,
    **gen_kwargs
).logits[0, -1, :]
mx.eval(logits_prefill)
print(f"✓ Prefill Maestro completado en {time.time() - t_pref:.2f} s.")

logprobs_base = logits_prefill.astype(mx.float32)
probs_base = mx.exp(logprobs_base - mx.max(logprobs_base))
probs_base = probs_base / mx.sum(probs_base)
mx.eval(probs_base)
tok_base = int(mx.argmax(logprobs_base))
print(f"✓ Token basal de referencia: '{processor.tokenizer.decode([tok_base])}' (Prob: {float(probs_base[tok_base])*100:.2f}%)")

cache_offsets = [getattr(c, "offset", 0) for c in master_cache]

# ─── MATRIZ FACTORIAL ORTOGONAL SOBRE CACHE CALIENTE ─────────────────────────
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

next_token_input = mx.array([[tok_base]], dtype=mx.uint32)

print("\n=====================================================================================================================")
print(" ⚡ SUPERFICIE DE RESPUESTA SOBRE 248,320 LOGITS (VELOCIDAD NATIVA DE SILICIO)")
print("=====================================================================================================================")
print(f"{'Exp ID':<7} | {'Theta':<5} | {'Topología':<11} | {'Head':<4} | {'||Δlogp||_2':<13} | {'||Δlogp||_inf':<14} | {'D_KL (nats)':<11} | {'Top-1 Ganador'} | {'Prob':<6} | {'Latencia':<8}")
print("---------------------------------------------------------------------------------------------------------------------")

# Medición basal de decode (Vanilla)
for i, c in enumerate(master_cache):
    if hasattr(c, "offset"): c.offset = cache_offsets[i]

logits_decode_base = model(next_token_input, cache=master_cache).logits[0, -1, :].astype(mx.float32)
mx.eval(logits_decode_base)
lp_dec_base = logits_decode_base - mx.log(mx.sum(mx.exp(logits_decode_base - mx.max(logits_decode_base))))
p_dec_base = mx.exp(lp_dec_base)
mx.eval(p_dec_base)

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

    # Restaurar offset de cache al punto del prefill
    for i, c in enumerate(master_cache):
        if hasattr(c, "offset"): c.offset = cache_offsets[i]

    # Paso de decode puro en GPU (1 token)
    logits_step = model(next_token_input, cache=master_cache).logits[0, -1, :].astype(mx.float32)
    mx.eval(logits_step)
    t_latencia = (time.perf_counter() - t_start) * 1000.0

    # Métricas sobre el espacio de 248,320 logits
    lp_step = logits_step - mx.log(mx.sum(mx.exp(logits_step - mx.max(logits_step))))
    dlp = lp_step - lp_dec_base
    dlp_2 = float(mx.sqrt(mx.sum(dlp * dlp)))
    dlp_inf = float(mx.max(mx.abs(dlp)))
    
    pr = mx.exp(lp_step)
    kl = float(mx.sum(p_dec_base * (lp_dec_base - lp_step)))
    
    top_id = int(mx.argmax(lp_step))
    top_tok = processor.tokenizer.decode([top_id])
    top_p = float(pr[top_id]) * 100.0

    print(f"{exp['id']:<7} | {th:<5.2f} | {topo:<11} | {hd:<4.1f} | {dlp_2:<13.2f} | {dlp_inf:<14.3f} | {kl:<11.5f} | '{top_tok}'{(' '*(12-len(top_tok)))} | {top_p:<5.1f}% | {t_latencia:<6.1f} ms")

print("=====================================================================================================================")
