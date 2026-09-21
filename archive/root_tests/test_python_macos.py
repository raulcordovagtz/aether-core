import sys, os, time
import mlx.core as mx

sys.path.insert(0, "/Users/crotalo/aether_engine")
sys.path.insert(0, "/Users/crotalo/aether_engine/aether_vlm")

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit")

PROMPT = """Escribe un script en Python para macOS que use el módulo 'subprocess' para ejecutar un comando de AppleScript ('osascript') que busque y reproduzca la canción 'Bohemian Rhapsody' en la app Music.

Incluye el código Python completo y funcional:"""

model, processor = load(MODEL_PATH)
aether = AetherEngine(model, processor)

chat = processor.apply_chat_template([
    {"role": "user", "content": PROMPT}
], add_generation_prompt=True)

print("\n" + "═" * 75)
print("  QWEN3.5-2B: PYTHON BRIDGE PARA APPLE MUSIC (AETHER ENGINE)")
print("═" * 75)

aether.set_active(True)
aether.update_parameters(theta_steer=0.50, kappa=0.40, gamma=0.30, nu=0.06, tau_relax=35.0, slingshot=True)
aether.reset_counters()
aether.prepare_thought(PROMPT, tau_steps=32)

for resp in stream_generate(model, processor, prompt=chat, max_tokens=400):
    print(resp.text, end="", flush=True)
print("\n")
