import sys, os, time
import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/crotalo/aether_engine")
from core_vlm.utils import load
from core_vlm.generate.dispatch import stream_generate
from PIL import Image

print("=================================================================================")
print(" 🔬 CAREO DE GOBERNABILIDAD DIRECCIONAL EN SILICIO (140 TOKENS EN 27B)")
print("    Protocolo RIGOR-EVAL: Vanilla vs Atractor Visual vs Desvío Ortogonal")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"
model, processor = load(model_path)

# Extraer Atractor Visual u_O_vis
img = Image.open(img_path).convert("RGB")
prompt_text = "Describe detalladamente los elementos, colores, objetos y contexto que se observan en la imagen."
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(text=[prompt], images=[img], return_tensors="mlx")
visual_patches = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])[0]
mx.eval(visual_patches)

u_vis = mx.mean(visual_patches, axis=0)
u_vis = u_vis / mx.sqrt(mx.sum(u_vis * u_vis) + 1e-12)
mx.eval(u_vis)

# Vector Ortogonal de Contraste (u_ortho) perpendicular a u_vis
D = 5120
np.random.seed(777)
rand_vec = mx.array(np.random.randn(D).astype(np.float32))
# Gram-Schmidt exacto para garantizar 90 grados exactos con la imagen
u_ortho = rand_vec - mx.sum(rand_vec * u_vis) * u_vis
u_ortho = u_ortho / mx.sqrt(mx.sum(u_ortho * u_ortho) + 1e-12)
mx.eval(u_ortho)

# Operador Riemanniano Intrinseco C-018
@mx.compile
def steer_step(h, target_vec, theta_rad):
    norm_h = mx.sqrt(mx.sum(h * h, axis=-1, keepdims=True) + 1e-12)
    h_unit = h / norm_h
    proj = mx.sum(target_vec * h_unit, axis=-1, keepdims=True)
    v = target_vec - proj * h_unit
    norm_v = mx.sqrt(mx.sum(v * v, axis=-1, keepdims=True) + 1e-12)
    v_unit = v / norm_v
    return norm_h * (mx.cos(theta_rad) * h_unit + mx.sin(theta_rad) * v_unit)

TARGET_LAYER = 24

class DynamicRudder:
    def __init__(self, original):
        self.original = original
        self.target = None
        self.theta = 0.0
        self.active = False
    def __getattr__(self, name):
        return getattr(self.original, name)
    def __call__(self, x, **kwargs):
        h = self.original(x, **kwargs)
        if not self.active or self.target is None or self.theta == 0.0:
            return h
        if h.shape[1] == 1:
            h_mod = steer_step(h[0, 0, :], self.target, self.theta)
            return h_mod[None, None, :]
        else:
            h_last = steer_step(h[0, -1, :], self.target, self.theta)
            return mx.concatenate([h[:, :-1, :], h_last[None, None, :]], axis=1)

rudder = DynamicRudder(model.language_model.model.layers[TARGET_LAYER])
model.language_model.model.layers[TARGET_LAYER] = rudder

def run_test(condition_name, target, theta_rad, max_tok=140):
    rudder.target = target
    rudder.theta = theta_rad
    rudder.active = (target is not None and theta_rad > 0.0)

    print(f"\n▶ {condition_name}")
    print("---------------------------------------------------------------------------------")
    t0 = None
    count = 0
    accum = ""
    for r in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=max_tok):
        if t0 is None: t0 = time.time()
        print(r.text, end="", flush=True)
        accum += r.text
        count += 1
    tps = count / ((time.time() - t0) + 1e-12)
    print(f"\n\n[Métricas: {count} tokens | {tps:.2f} tok/s]")
    return accum, tps

# 1. VANILLA
txt_a, tps_a = run_test("A) CONTROL BASE: VANILLA MLX (Sin Timón)", None, 0.0)

# 2. TIMÓN VISUAL FÁCTICO (Theta = 0.35 rad ~ 20°)
txt_b, tps_b = run_test("B) TIMÓN VISUAL: ANCLAJE FÁCTICO A FOTONES u_O_vis (Theta = 20°)", u_vis, 0.35)

# 3. TIMÓN ORTOGONAL DE CONTRASTE (Theta = 0.35 rad ~ 20°)
txt_c, tps_c = run_test("C) TIMÓN ORTOGONAL: DESVÍO TRANSVERSAL 90° (Theta = 20°)", u_ortho, 0.35)

print("\n=================================================================================")
print(" 📊 RESUMEN EJECUTIVO DE GOBERNABILIDAD:")
print("=================================================================================")
print(f"• Velocidad Vanilla        : {tps_a:.2f} tok/s")
print(f"• Velocidad Timón Visual   : {tps_b:.2f} tok/s")
print(f"• Velocidad Timón 90°      : {tps_c:.2f} tok/s")
print(f"• Divergencia Visual vs Vanilla : {'SÍ' if txt_a != txt_b else 'NO'}")
print(f"• Divergencia 90° vs Vanilla    : {'SÍ' if txt_a != txt_c else 'NO'}")
print("=================================================================================")
