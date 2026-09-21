import sys, os, time
import mlx.core as mx

sys.path.insert(0, "/Users/crotalo/aether_engine")
sys.path.insert(0, "/Users/crotalo/aether_engine/aether_vlm")

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

# Prompt estructurado en listas para no saturar D=1024
PROMPT = """Cuatro cajas: ['Roja', 'Azul', 'Verde', 'Amarilla'].
Aplica paso a paso actualizando la lista en cada paso:
1. Intercambia la primera y la última caja.
2. Mueve 'Azul' una posición a la derecha.
3. Mueve la caja de la posición 2 al final de la lista.
4. Intercambia 'Verde' con la caja a su izquierda.

Escribe el estado de la lista en cada paso y da el resultado final:"""

model, processor = load(MODEL_PATH)

# Perfil equilibrado (Golden Ratio para D=1024)
aether = AetherEngine(
    model, processor,
    theta_steer=0.35,
    kappa=0.25,
    gamma=0.35,
    nu=0.07,
    slingshot=True
)

chat = processor.apply_chat_template([
    {"role": "user", "content": PROMPT}
], add_generation_prompt=True)

aether.prepare_thought(PROMPT, tau_steps=32)

print("\n--- Ejecutando 0.8B con Notación Vectorial ---")
for resp in stream_generate(model, processor, prompt=chat, max_tokens=300):
    print(resp.text, end="", flush=True)
print("\n")
