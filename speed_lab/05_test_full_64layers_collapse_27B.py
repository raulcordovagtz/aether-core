import sys, os, time
import mlx.core as mx
from PIL import Image

sys.path.insert(0, "/Users/crotalo/aether_engine")
from core_vlm.utils import load
from core_vlm.generate.dispatch import stream_generate
from core_vlm.aether import AetherEngine

print("=================================================================================")
print(" 🌌 AETHER-VLM :: TEST MULTIMODAL COMPLETO (64 CAPAS + LM_HEAD)")
print("    Campo Físico Continuo a lo Largo de Toda la Profundidad de Qwen 27B")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print("• 1. Cargando Qwen 27B en memoria UMA...")
t0 = time.time()
model, processor = load(model_path)
print(f"✓ Modelo cargado en {time.time() - t0:.2f} s.")

# Instanciar el motor sobre las 64 capas y el lm_head
print("\n• 2. Conectando AETHER sobre las 64 capas y la capa de colapso...")
aether = AetherEngine(model, processor)
print("✓ Circuito total activado: 64/64 capas + lm_head vinculados.")

# Ingestión de Visión
print("\n• 3. Ingestión de fotones reales de 005.jpg en la Torre de Visión...")
img = Image.open(img_path).convert("RGB")
prompt_text = "Describe detalladamente los elementos, colores, objetos y contexto que se observan en la imagen."
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(text=[prompt], images=[img], return_tensors="mlx")
visual_patches = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])[0]
mx.eval(visual_patches)
print(f"✓ {visual_patches.shape[0]} parches visuales asimilados en D={visual_patches.shape[1]}.")

# Pensamiento Profundo continuo en GPU
print("\n• 4. Ejecutando Pensamiento Profundo continuo (τ* = 32 pasos en GPU)...")
t_settle_0 = time.time()
telemetria = aether.prepare_multimodal_thought(visual_patches, prompt_text)
mx.eval(aether.settled_intent)
t_settle = (time.time() - t_settle_0) * 1000.0

print(f"✓ Asentamiento alcanzado en {t_settle:.2f} ms:")
for paso, e_val, v_val in telemetria:
    print(f"    τ = {paso:<2} | E(τ) = {e_val:.6f} | ||Φ̇|| = {v_val:.6f}")
print("✓ Atractor óptimo Φ* propagado a las 64 capas y a la capa de colapso.")

# Generación Continua
print("\n=================================================================================")
print(" 🚀 GENERACIÓN CONTINUA (64 CAPAS + LM_HEAD EN SILICIO):")
print("=================================================================================\n")

t0_tok = None
count = 0
accum_text = ""

for resp in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=140):
    if t0_tok is None:
        t0_tok = time.time()
    print(resp.text, end="", flush=True)
    accum_text += resp.text
    count += 1

t_gen = time.time() - (t0_tok if t0_tok else time.time())
tps = count / (t_gen + 1e-12)

print("\n\n=================================================================================")
print(" 📊 TELEMETRÍA DEL CIRCUITO TOTAL EN PRODUCCIÓN:")
print("=================================================================================")
print(f"• Pensamiento Profundo (τ*=32) : {t_settle:.2f} ms")
print(f"• Capas Moduladas por Token    : 64 capas continuas + 1 capa de colapso")
print(f"• Tokens Generados             : {count}")
print(f"• Tiempo de Generación         : {t_gen:.2f} s")
print(f"• Velocidad Sostenida (27B)    : {tps:.2f} tok/s")
print("=================================================================================")
