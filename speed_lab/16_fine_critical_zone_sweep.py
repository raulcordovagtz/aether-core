import sys, os
sys.path.insert(0, "/Users/crotalo/aether_engine")

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine
import aether_vlm.coupler as coupler
from PIL import Image

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"
model, processor = load(model_path)
tokenizer = processor.tokenizer

print("=================================================================================")
print(" 🔬 BARRIDO EN LA ZONA CRÍTICA: BÚSQUEDA DEL PUNTO DULCE DE NAVEGACIÓN")
print("=================================================================================")
print("• Token 158949 decodificado:", repr(tokenizer.decode([158949])))

aether = AetherEngine(model, processor)
prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe detalladamente la imagen."}]}], add_generation_prompt=True)
img = Image.open(img_path).convert("RGB")
inputs = processor(text=[prompt], images=[img], return_tensors="mlx")
visual_patches = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])[0]
aether.prepare_multimodal_thought(visual_patches, "Describe detalladamente la imagen.")

zona_critica = [
    {"th": 0.45, "kp": 0.25, "nu": 0.15, "gm": 0.40},
    {"th": 0.55, "kp": 0.35, "nu": 0.18, "gm": 0.45},
    {"th": 0.65, "kp": 0.45, "nu": 0.20, "gm": 0.50},
    {"th": 0.75, "kp": 0.60, "nu": 0.22, "gm": 0.55},
    {"th": 0.85, "kp": 0.75, "nu": 0.25, "gm": 0.60},
]

print("\n=================================================================================================================")
print(f"{'Theta':<6} | {'Kappa':<6} | {'Nu':<5} | {'Gamma':<5} | {'Primeros 45 Tokens Generados'}")
print("-----------------------------------------------------------------------------------------------------------------")

for exp in zona_critica:
    th, kp, nu_val, gm = exp["th"], exp["kp"], exp["nu"], exp["gm"]
    for hook in aether.hooked_layers: 
        hook.theta_step = th / 64.0
    
    def make_call(nu_p, gm_p, kp_p):
        def call(h):
            if not aether.hooked_head.active or aether.hooked_head.state_ref is None:
                return aether.hooked_head.original_lm_head(h)
            v_drag = aether.hooked_head.state_ref.get("v_drag")
            z_L_star = aether.hooked_head.state_ref.get("z_L_star")
            if v_drag is None or z_L_star is None:
                return aether.hooked_head.original_lm_head(h)
            return coupler.aether_native_c.dispatch_full_collapse(
                h, v_drag, z_L_star,
                aether.hooked_head.head_w, aether.hooked_head.head_scales, aether.hooked_head.head_biases,
                aether.hooked_head.group_size, aether.hooked_head.bits,
                nu_p, gm_p, kp_p
            )
        return call

    aether.hooked_head.__call__ = make_call(nu_val, gm, kp)

    text = ""
    for resp in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=35):
        text += resp.text

    text_clean = text.replace("\n", " ").strip()
    if len(text_clean) > 60: 
        text_clean = text_clean[:57] + "..."
    print(f"{th:<6.2f} | {kp:<6.2f} | {nu_val:<5.2f} | {gm:<5.2f} | {text_clean}")

print("=================================================================================================================")
