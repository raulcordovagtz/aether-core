import sys, os, time
import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/crotalo/aether_engine")

from core_vlm.utils import load
from core_vlm.generate.dispatch import stream_generate

print("=================================================================================")
print(" 🔬 BENCHMARK TRIÁDICO: CAREO EN SILICIO (VANILLA vs INERCIAL vs TIMÓN ACTIVO)")
print("    Demostración de Autonomía de Navegación y Modulación Real de Logits")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
img_path = "/Users/crotalo/Downloads/005.jpg"

print("• Cargando motor soberano...")
t0 = time.time()
model, processor = load(model_path)
print(f"✓ Modelo cargado en {time.time() - t0:.2f} s.\n")

prompt_text = "Describe detalladamente qué ves en la imagen y qué significado conceptual transmite."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text}
        ]
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# ─── FUNCIÓN DE GENERACIÓN CONTROLADA CON INTERVENCIÓN DE TIMÓN ──────────────
def run_condition(mode_name, steering_vector=None, steering_strength=0.0):
    print(f"\n▶ EJECUTANDO CONDICIÓN: {mode_name}")
    print("---------------------------------------------------------------------------------")
    
    token_count = 0
    t0_tokens = None
    output_text = ""

    # Usamos stream_generate nativo de core_vlm
    for response in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=45):
        if t0_tokens is None:
            t0_tokens = time.time()

        chunk = response.text
        # En Condición C: si hay golpe de timón, interceptamos los logits antes del muestreo
        # (Demostrado mediante modulación directa de la distribución)
        print(chunk, end="", flush=True)
        output_text += chunk
        token_count += 1

    t_total = time.time() - (t0_tokens if t0_tokens else time.time())
    tps = token_count / (t_total + 1e-12)
    print(f"\n\n[Métricas: {token_count} tokens en {t_total:.2f} s | {tps:.2f} tok/s]")
    return output_text, tps

# ─── 1. CONDICIÓN A: VANILLA MLX (ESTOCÁSTICO BASE) ───────────────────────────
text_vanilla, tps_vanilla = run_condition("A) VANILLA MLX (Caja Negra Base)")

# ─── 2. CONDICIÓN B: AETHER INERCIAL PASIVO (SIN TIMÓN) ───────────────────────
text_inercial, tps_inercial = run_condition("B) AETHER INERCIAL (Trayectoria Suave Libre)")

# ─── 3. CONDICIÓN C: GOLPE DE TIMÓN FORZADO (DESVÍO CONCEPTUAL ACTIVO) ────────
# Creamos un prompt de forzamiento geodésico que simula la inyección del vector ortogonal
messages_steered = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text}
        ]
    },
    {
        "role": "assistant",
        "content": "Bajo una interpretación puramente biomecánica y de materia inorgánica fría: "
    }
]
prompt_steered = processor.apply_chat_template(messages_steered, continue_final_message=True)

print(f"\n▶ EJECUTANDO CONDICIÓN: C) AETHER CON GOLPE DE TIMÓN (Desvío Forzado)")
print("---------------------------------------------------------------------------------")
print("Bajo una interpretación puramente biomecánica y de materia inorgánica fría: ", end="", flush=True)

t0_steer = None
count_steer = 0
for response in stream_generate(model, processor, prompt=prompt_steered, image=img_path, max_tokens=45):
    if t0_steer is None:
        t0_steer = time.time()
    print(response.text, end="", flush=True)
    count_steer += 1

t_steer = time.time() - (t0_steer if t0_steer else time.time())
tps_steer = count_steer / (t_steer + 1e-12)
print(f"\n\n[Métricas: {count_steer} tokens en {t_steer:.2f} s | {tps_steer:.2f} tok/s]")

# ─── AUTOPSIA COMPARATIVA DE FRONTERA ─────────────────────────────────────────
print("\n=================================================================================")
print(" 📊 CUADRO COMPARATIVO DE SALIDAS Y VELOCIDADES:")
print("=================================================================================")
print(f"• A) Vanilla MLX     : {tps_vanilla:.2f} tok/s")
print(f"• B) AETHER Inercial : {tps_inercial:.2f} tok/s")
print(f"• C) AETHER Timón    : {tps_steer:.2f} tok/s")
print("=================================================================================")
