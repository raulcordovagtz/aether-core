import sys, os, time
import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/crotalo/aether_engine")
from core_vlm.utils import load
from core_vlm.generate.dispatch import stream_generate
from PIL import Image

print("=================================================================================")
print(" 🔬 CAREO FÁCTICO: VANILLA vs TIMÓN ANCLADO A FOTONES u_O_vis (27B)")
print("    Protocolo RIGOR-EVAL: Mismo Prompt, Cero Hacks, Extracción ViT en GPU")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print("• Cargando arquitectura Qwen 27B...")
t0 = time.time()
model, processor = load(model_path)
print(f"✓ Modelo cargado en {time.time() - t0:.2f} s.")

# ─── 1. EXTRACCIÓN DEL ATRACTOR VISUAL FÁCTICO u_O_vis ───────────────────────
print("\n• Extrayendo fotones reales desde la Torre de Visión oficial...")
img = Image.open(img_path).convert("RGB")
prompt_text = "Describe detalladamente los elementos, colores, objetos y contexto que se observan en la imagen."
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(text=[prompt], images=[img], return_tensors="mlx")
visual_patches = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])[0]
mx.eval(visual_patches)

# Centroide fáctico unitario u_O_vis
u_vis_centroid = mx.mean(visual_patches, axis=0)
u_vis_centroid = u_vis_centroid / mx.sqrt(mx.sum(u_vis_centroid * u_vis_centroid) + 1e-12)
mx.eval(u_vis_centroid)
print(f"✓ Atractor Visual u_O_vis consolidado: {visual_patches.shape[0]} parches condensados en norma 1.000000.")

# ─── 2. OPERADOR RIEMANNIANO FUSIONADO EN GPU ─────────────────────────────────
@mx.compile
def riemannian_visual_operator(h, u_tgt, beta, theta_scale):
    norm_h = mx.sqrt(mx.sum(h * h, axis=-1, keepdims=True) + 1e-12)
    h_unit = h / norm_h

    # Fuerza tangencial de Noether
    proj = mx.sum(u_tgt * h_unit, axis=-1, keepdims=True)
    v = beta * (u_tgt - proj * h_unit)
    norm_v = mx.sqrt(mx.sum(v * v, axis=-1, keepdims=True) + 1e-12)

    # Exponencial de Riemann
    theta = theta_scale * mx.minimum(norm_v, 1.0)
    h_steered = norm_h * (mx.cos(theta) * h_unit + mx.sin(theta) * (v / norm_v))
    return h_steered

# ─── 3. HOOK CAUSAL EN CAPA 24 ───────────────────────────────────────────────
TARGET_LAYER = 24

class FactualRudderHook:
    def __init__(self, original_layer):
        self.original_layer = original_layer
        self.active = False
        self.beta = 0.0
        self.theta_scale = 0.15 # ~8.5 grados de torsión constante
        self.u_vis = u_vis_centroid

    def __getattr__(self, name):
        return getattr(self.original_layer, name)

    def __call__(self, x, **kwargs):
        h = self.original_layer(x, **kwargs)
        if not self.active or self.beta <= 0.0:
            return h

        if h.shape[1] == 1:
            # Decode rápido
            h_mod = riemannian_visual_operator(h[0, 0, :], self.u_vis, self.beta, self.theta_scale)
            return h_mod[None, None, :]
        else:
            # Prefill: rotamos el último token
            h_last = riemannian_visual_operator(h[0, -1, :], self.u_vis, self.beta, self.theta_scale)
            return mx.concatenate([h[:, :-1, :], h_last[None, None, :]], axis=1)

rudder_hook = FactualRudderHook(model.language_model.model.layers[TARGET_LAYER])
model.language_model.model.layers[TARGET_LAYER] = rudder_hook

# ─── 4. FUNCIÓN DE EJECUCIÓN ─────────────────────────────────────────────────
def execute_condition(name, is_active, beta_val, max_tok=60):
    rudder_hook.active = is_active
    rudder_hook.beta = beta_val

    print(f"\n▶ CORRIENDO: {name}")
    print("---------------------------------------------------------------------------------")
    
    t0_tok = None
    count = 0
    text_accum = ""

    for resp in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=max_tok):
        if t0_tok is None:
            t0_tok = time.time()
        print(resp.text, end="", flush=True)
        text_accum += resp.text
        count += 1

    t_span = time.time() - (t0_tok if t0_tok else time.time())
    tps = count / (t_span + 1e-12)
    print(f"\n\n[Rendimiento: {count} tokens en {t_span:.2f} s | {tps:.2f} tok/s]")
    return text_accum, tps

# A) Condición Control: Vanilla (Beta = 0)
text_vanilla, tps_vanilla = execute_condition("A) CONTROL: VANILLA MLX (Sin Timón)", is_active=False, beta_val=0.0)

# B) Condición Experimental: Timón Anclado a u_O_vis (Beta = 2.0)
text_factual, tps_factual = execute_condition("B) EXPERIMENTAL: TIMÓN ANCLADO A FOTONES u_O_vis (Beta = 2.0)", is_active=True, beta_val=2.0)

# ─── BALANCE DE AUDITORÍA ─────────────────────────────────────────────────────
print("\n=================================================================================")
print(" 📊 BALANCE DE AUDITORÍA C-018 EN SILICIO:")
print("=================================================================================")
print(f"• Velocidad Vanilla           : {tps_vanilla:.2f} tok/s")
print(f"• Velocidad con Timón Visual  : {tps_factual:.2f} tok/s")
diff_percent = ((tps_vanilla - tps_factual) / tps_vanilla) * 100.0 if tps_vanilla > 0 else 0.0
print(f"• Impacto en Silicio (Drop)   : {diff_percent:.2f}%")
print(f"• ¿Hubo desvío semántico?     : {'SÍ (Confirmado)' if text_vanilla != text_factual else 'NO'}")
print("=================================================================================")
