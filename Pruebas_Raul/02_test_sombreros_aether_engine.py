#!/usr/bin/env python3
"""
Pruebas_Raul/02_test_sombreros_aether_engine.py
===============================================
Evaluación de Razonamiento Epistémico Puro en Campo Continuo:
El texto entra crudo y AetherEngine opera exclusivamente desde sus
invariantes intrínsecas en Qwen3.5-0.8B y 2B.
"""

import sys, os, time
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

from mlx_vlm import load, stream_generate
from aether_vlm.coupler import AetherEngine

MODELS = [
    ("0.8B", os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")),
    ("2B",   os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit")),
]

# El texto crudo y exacto de Raúl, sin modificaciones
PUZZLE_PROMPT = """Tres personas están en fila una detrás de otra.El último ve a los dos de adelante.El del medio ve al primero.El primero no ve a nadie.Se sacan cinco sombreros de una caja: tres negros y dos blancos. A cada persona se le coloca un sombrero en la cabeza a ciegas. Los dos sombreros restantes de la caja se guardan ocultos.Le preguntan al último de la fila (el que ve a los otros dos): ¿Sabes el color de tu sombrero? Dice que no.Le preguntan al del medio (que ve al primero): ¿Sabes el color de tu sombrero? Dice que no.El primero de la fila (que no ve a nadie) escucha las respuestas anteriores y dice: ¡Sí, mi sombrero es de color...¿De qué color es el sombrero del primero y por qué?"""

def test_aether_pure(alias, path):
    print("═" * 78)
    print(f" 🌌 CAMPO CONTINUO ACTIVO: Qwen3.5-{alias}")
    print("═" * 78)

    model, processor = load(path)
    # Autoconfiguración soberana sin parámetros externos
    aether = AetherEngine(model, processor)

    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": PUZZLE_PROMPT}]}
    ], add_generation_prompt=True)

    # El campo se auto-organiza desde el propio prompt
    aether.prepare_thought(PUZZLE_PROMPT)

    print("\n[EMISIÓN DEL CAMPO]:\n")
    tokens = []
    t0 = time.perf_counter()
    for resp in stream_generate(model, processor, prompt=prompt_chat, max_tokens=2200):
        tokens.append(resp.text)
        print(resp.text, end="", flush=True)
    t_gen = time.perf_counter() - t0
    
    print("\n\n" + "─" * 78)
    print(f"Velocidad: {len(tokens) / max(t_gen, 1e-5):.1f} tok/s | Tokens: {len(tokens)} en {t_gen:.2f}s")
    print("─" * 78 + "\n")

    aether.set_active(False)

if __name__ == "__main__":
    for alias, path in MODELS:
        test_aether_pure(alias, path)
