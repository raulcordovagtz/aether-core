import sys, os, time, math
import mlx.core as mx
from PIL import Image

sys.path.insert(0, "/Users/crotalo/aether_engine")
from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine
import aether_vlm.coupler as coupler

print("=================================================================================")
print(" 🔬 AUDITORÍA FORENSE EXHAUSTIVA DE SILICIO: AETHER vs MLX CONVENCIONAL")
print("=================================================================================\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. INSPECCIÓN DE CÓDIGO Y TRAZABILIDAD (SIN MONKEYPATCH)
# ─────────────────────────────────────────────────────────────────────────────
print("─── REQ 1: TRAZABILIDAD Y VERIFICACIÓN DE RUTAS ────────────────────────────────")
import inspect
head_source = inspect.getsource(coupler.AetherCollapseHead.__call__)
print("• Inspección de AetherCollapseHead.__call__ (código real en disco):")
for line in head_source.strip().splitlines():
    print("   ", line)

print("\n• Módulo aether_native_c cargado desde:", coupler.aether_native_c.__file__)
print("• Métodos expuestos por binario C++:", [m for m in dir(coupler.aether_native_c) if not m.startswith("__")])

# ─────────────────────────────────────────────────────────────────────────────
# 2. CARGA DEL MODELO E INICIALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print("\n• Cargando arquitectura Qwen 27B...")
t0 = time.time()
model, processor = load(model_path)
print(f"✓ Modelo cargado en {time.time() - t0:.2f} s.")

aether = AetherEngine(model, processor, nu=0.12, kappa=0.15, theta_steer=0.35)

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe detalladamente la imagen."}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
img = Image.open(img_path).convert("RGB")
inputs = processor(text=[prompt], images=[img], return_tensors="mlx")
visual_patches = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])[0]

# ─────────────────────────────────────────────────────────────────────────────
# 3. CORRIDA A/B: MISMA SEMILLA, MISMO PROMPT (BASE vs AETHER)
# ─────────────────────────────────────────────────────────────────────────────
NUM_TOKENS = 35

print("\n─── REQ 2: CORRIDA A/B ESTRICTA (SEED = 42, 35 TOKENS) ────────────────────────")

# Corrida A: AETHER APAGADO (MLX Puro Convencional)
print("\n[A] MLX CONVENCIONAL PURO (Aether Inactivo):")
aether.set_active(False)
aether.reset_counters()
mx.random.seed(42)

t0_a = time.time()
tokens_a = []
for resp in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=NUM_TOKENS):
    tokens_a.append(resp.text)
    print(resp.text, end="", flush=True)
text_a = "".join(tokens_a)

total_layer_calls_a = sum(h.call_count for h in aether.hooked_layers)
head_calls_a = aether.hooked_head.call_count
print(f"\n   -> Invocaciones C++ registradas con Aether apagado: Capas={total_layer_calls_a}, Head={head_calls_a}")

# Corrida B: AETHER SILICIO C++ ACTIVO
print("\n[B] AETHER NATIVE SILICIO (Aether Activo: Geodésica + Condensación):")
aether.reset_counters()
telemetria = aether.prepare_multimodal_thought(visual_patches, "Describe detalladamente la imagen.")
mx.random.seed(42)

t0_b = time.time()
tokens_b = []
for resp in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=NUM_TOKENS):
    tokens_b.append(resp.text)
    print(resp.text, end="", flush=True)
text_b = "".join(tokens_b)

total_layer_calls_b = sum(h.call_count for h in aether.hooked_layers)
head_calls_b = aether.hooked_head.call_count
print(f"\n   -> Invocaciones C++ registradas con Aether activo: Capas={total_layer_calls_b}, Head={head_calls_b}")

print("\n• Comparación Textual Directa:")
print(f"  MLX Base : \"{text_a[:120]}...\"")
print(f"  Aether   : \"{text_b[:120]}...\"")
print(f"  ¿Salidas idénticas? : {'SÍ (Sin Efecto)' if text_a == text_b else 'NO (Divergencia Real Detectada)'}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. ANÁLISIS DE LOGITS Y PROBABILIDADES PASO A PASO (TOP-10, ENTROPÍA, KL)
# ─────────────────────────────────────────────────────────────────────────────
print("\n─── REQ 3: ANÁLISIS ESPECTRAL DE LOGITS Y COLAPSO (z_raw vs z_cond) ──────────")

# Tomamos el último estado capturado en el head durante la generación
z_raw = aether.hooked_head.last_raw_logits
z_cond = aether.hooked_head.last_cond_logits

if z_raw is not None and z_cond is not None:
    # Aplanar a [Vocab] si viene en [1, 1, Vocab]
    if len(z_raw.shape) == 3:
        z_r = z_raw[0, -1, :]
        z_c = z_cond[0, -1, :]
    elif len(z_raw.shape) == 2:
        z_r = z_raw[-1, :]
        z_c = z_cond[-1, :]
    else:
        z_r = z_raw
        z_c = z_cond

    p_raw = mx.softmax(z_r, axis=-1)
    p_cond = mx.softmax(z_c, axis=-1)

    # Entropía de Shannon H = - sum(p * log(p + 1e-12))
    h_raw = -float(mx.sum(p_raw * mx.log(p_raw + 1e-12)).item())
    h_cond = -float(mx.sum(p_cond * mx.log(p_cond + 1e-12)).item())

    # Divergencia KL: KL(p_cond || p_raw) = sum(p_cond * log(p_cond / p_raw))
    kl_div = float(mx.sum(p_cond * (mx.log(p_cond + 1e-12) - mx.log(p_raw + 1e-12))).item())

    argmax_raw = int(mx.argmax(z_r).item())
    argmax_cond = int(mx.argmax(z_c).item())

    token_str_raw = processor.tokenizer.decode([argmax_raw])
    token_str_cond = processor.tokenizer.decode([argmax_cond])

    print(f"• Entropía de Shannon Base (MLX)   : {h_raw:.6f} nats")
    print(f"• Entropía Condensada (Aether)     : {h_cond:.6f} nats (ΔH = {h_cond - h_raw:+.6f})")
    print(f"• Divergencia KL (P_cond || P_raw) : {kl_div:.6f} nats")
    print(f"• Argmax MLX Base                  : Token ID {argmax_raw} ('{token_str_raw}') -> Prob: {float(p_raw[argmax_raw].item())*100:.2f}%")
    print(f"• Argmax Aether C++                : Token ID {argmax_cond} ('{token_str_cond}') -> Prob: {float(p_cond[argmax_cond].item())*100:.2f}%")

    print("\n• Top-10 Candidatos de Vocabulario (MLX Base vs Aether Condensado):")
    top_k_indices_raw = mx.argsort(-p_raw)[:10].tolist()
    top_k_indices_cond = mx.argsort(-p_cond)[:10].tolist()

    print(f"   {'Rnk':<4} | {'MLX Base Token':<20} {'Prob':<10} | {'Aether C++ Token':<20} {'Prob':<10}")
    print("   " + "-" * 70)
    for r in range(10):
        idx_r = top_k_indices_raw[r]
        idx_c = top_k_indices_cond[r]
        tok_r = repr(processor.tokenizer.decode([idx_r]))
        tok_c = repr(processor.tokenizer.decode([idx_c]))
        pr_r = float(p_raw[idx_r].item()) * 100
        pr_c = float(p_cond[idx_c].item()) * 100
        print(f"   {r+1:<4} | {tok_r:<20} {pr_r:>6.2f}%    | {tok_c:<20} {pr_c:>6.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 5. BARRIDO DE HIPERPARÁMETROS: κ (NUCLEACIÓN) Y ν (VISCOSIDAD)
# ─────────────────────────────────────────────────────────────────────────────
print("\n─── REQ 4: BARRIDO DE HIPERPARÁMETROS (κ y ν sobre logits reales) ────────────")
z_L_star = aether.state.get("z_L_star")
if z_raw is not None and z_L_star is not None:
    kappas = [0.0, 0.05, 0.15, 0.50, 1.0, 3.0]
    nus = [0.0, 0.12, 0.35, 0.70]

    print(f"Evaluando matriz de respuesta de {len(kappas)} valores de κ × {len(nus)} valores de ν:")
    print(f"{'κ (Nucleación)':<15} {'ν (Viscosidad)':<15} {'Top-1 Token':<20} {'Prob Top-1':<12} {'Entropía (H)':<12} {'KL vs Base'}")
    print("-" * 85)

    for nu in nus:
        for kap in kappas:
            z_sweep = coupler.aether_native_c.dispatch_vapor_condensation(z_r, z_L_star, nu, kap)
            p_sweep = mx.softmax(z_sweep, axis=-1)
            h_sw = -float(mx.sum(p_sweep * mx.log(p_sweep + 1e-12)).item())
            kl_sw = float(mx.sum(p_sweep * (mx.log(p_sweep + 1e-12) - mx.log(p_raw + 1e-12))).item())
            top1_id = int(mx.argmax(z_sweep).item())
            top1_tok = repr(processor.tokenizer.decode([top1_id]))
            top1_prob = float(p_sweep[top1_id].item()) * 100
            print(f"{kap:<15.2f} {nu:<15.2f} {top1_tok:<20} {top1_prob:>8.2f}%   {h_sw:>9.4f}    {kl_sw:>9.4f}")

print("\n=================================================================================")
print(" ✓ AUDITORÍA FORENSE COMPLETADA CON ÉXITO")
print("=================================================================================")
