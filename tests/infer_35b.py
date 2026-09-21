#!/usr/bin/env python3
"""
infer_35b_moe.py
================
Inferencia Multimodal con Aether Engine sobre Qwen3.6-35B-A3B MoE (Metal GPU).
"""

import sys, os, time
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import mlx.core as mx
from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

# ─── 1. CONFIGURACIÓN DE RUTAS ───────────────────────────────────
MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-6bit")
IMG_PATH = "/Users/crotalo/Downloads/005.jpg"
PROMPT_TEXT = "Describe con detalle lo que observas en la imagen y analiza el mensaje."

print("=" * 70)
print(f"CARGANDO QWEN3.6-35B-A3B MoE (6-bit)")
print(f"Ruta: {MODEL_PATH}")
print(f"Imagen: {IMG_PATH}")
print("=" * 70)

# ─── 2. CARGA DEL MODELO EN MEMORIA UNIFICADA ────────────────────
t0_load = time.perf_counter()
model, processor = load(MODEL_PATH)
print(f"✓ Modelo cargado en {time.perf_counter() - t0_load:.2f} s")

# ─── 3. CONEXIÓN DEL MOTOR AETHER (AUTO-DETECTA PERFIL moe_sparse)
aether = AetherEngine(model, processor)
print(f"✓ Motor Aether activo en perfil: [{aether.profile_name}]")
print(f"  - Capas: {aether.num_layers} | Dimensión: {aether.hidden_dim}")
print(f"  - Parámetros: κ={aether.kappa}, θ={aether.theta_steer}, ν={aether.nu}, γ={aether.gamma}")

# ─── 4. INGESTIÓN MULTIMODAL Y ASENTAMIENTO (τ* = 32) ───────────
img = Image.open(IMG_PATH).convert("RGB")
prompt_formatted = processor.apply_chat_template([
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": PROMPT_TEXT}
    ]}
], add_generation_prompt=True)

inputs = processor(text=[prompt_formatted], images=[img], return_tensors="mlx")

t0_settle = time.perf_counter()
if "image_grid_thw" in inputs:
    visual_patches = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])[0]
else:
    visual_patches = model.vision_tower(inputs["pixel_values"])[0]

telemetria = aether.prepare_multimodal_thought(visual_patches, PROMPT_TEXT)
print(f"✓ Atractor L* asentado en GPU ({time.perf_counter() - t0_settle:.3f} s)")

# ─── 5. GENERACIÓN CON COLAPSO 100% C++ EN GPU METAL ─────────────
print("\n" + "─" * 70)
print("GENERANDO:")
print("─" * 70)

tokens = []
t0_gen = time.perf_counter()
for resp in stream_generate(
    model, processor,
    prompt=prompt_formatted,
    image=IMG_PATH,
    max_tokens=150
):
    tokens.append(resp.text)
    print(resp.text, end="", flush=True)

t_gen = time.perf_counter() - t0_gen
print("\n" + "─" * 70)
print(f"✓ Generación finalizada: {len(tokens)} tokens en {t_gen:.2f} s ({len(tokens) / t_gen:.1f} tok/s)")
print("=" * 70)