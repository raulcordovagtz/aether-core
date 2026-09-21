import sys, os, time
import mlx.core as mx

sys.path.insert(0, "/Users/crotalo/aether_engine")
sys.path.insert(0, "/Users/crotalo/aether_engine/aether_vlm")

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit")

PROMPT = """Escribe un script en AppleScript para macOS que controle la aplicación 'Music'.
El script debe:
1. Activar la app 'Music'.
2. Buscar en la biblioteca una pista cuyo nombre contenga 'Bohemian Rhapsody'.
3. Si la encuentra, reproducir la pista.

Escribe únicamente el bloque de código AppleScript ejecutable con 'tell application "Music"'."""

model, processor = load(MODEL_PATH)
aether = AetherEngine(model, processor)

chat = processor.apply_chat_template([
    {"role": "user", "content": PROMPT}
], add_generation_prompt=True)

# ─── 1. MODO VANILLA ───────────────────────────────────────────────
print("\n" + "═" * 75)
print("  QWEN3.5-2B: MODO VANILLA (ESTÁNDAR)")
print("═" * 75)
aether.set_active(False)
for resp in stream_generate(model, processor, prompt=chat, max_tokens=350):
    print(resp.text, end="", flush=True)
print("\n")

# ─── 2. MODO AETHER CON RELAJACIÓN TEMPORAL (tau_relax = 25) ──────
print("\n" + "═" * 75)
print("  QWEN3.5-2B: AETHER SINTONIZADO PARA CÓDIGO (tau_relax=25.0)")
print("═" * 75)
aether.set_active(True)
aether.update_parameters(
    theta_steer=0.70,
    kappa=0.60,
    gamma=0.40,
    nu=0.06,
    tau_relax=25.0,  # Decaimiento temporal tras 25 tokens
    slingshot=True
)
aether.reset_counters()
aether.prepare_thought(PROMPT, tau_steps=32)

for resp in stream_generate(model, processor, prompt=chat, max_tokens=350):
    print(resp.text, end="", flush=True)
print("\n")
