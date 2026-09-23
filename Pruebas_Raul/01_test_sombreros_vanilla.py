#!/usr/bin/env python3
"""
Pruebas_Raul/01_test_sombreros_vanilla.py
=========================================
Prueba de Razonamiento Epistémico en Modelos Desnudos (Vanilla):
Evalúa Qwen3.5-0.8B y Qwen3.5-2B en el Acertijo de los Tres Sombreros.
"""

import sys, os, time
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

from mlx_vlm import load, stream_generate

MODELS = [
    ("0.8B", os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")),
    ("2B",   os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit")),
]

PUZZLE_PROMPT = """Tres personas están en fila una detrás de otra.
- El último ve a los dos de adelante.
- El del medio ve al primero.
- El primero no ve a nadie.

Se sacan cinco sombreros de una caja: tres negros y dos blancos. A cada persona se le coloca un sombrero en la cabeza a ciegas. Los dos sombreros restantes de la caja se guardan ocultos.

1. Le preguntan al último de la fila (el que ve a los otros dos): ¿Sabes el color de tu sombrero? Dice que no.
2. Le preguntan al del medio (que ve al primero): ¿Sabes el color de tu sombrero? Dice que no.
3. El primero de la fila (que no ve a nadie) escucha las respuestas anteriores y dice: ¡Sí, ya sé de qué color es mi sombrero!

¿De qué color es el sombrero del primero y cuál es la deducción lógica exacta paso a paso que le permitió saberlo?"""

def test_model(alias, path):
    print("═" * 78)
    print(f" 🧠 EVALUANDO ACERTIJO EN: Qwen3.5-{alias} (VANILLA)")
    print(f" Ruta: {path}")
    print("═" * 78)

    t0 = time.perf_counter()
    model, processor = load(path)
    print(f"✓ Modelo cargado en {time.perf_counter() - t0:.2f} s")

    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": PUZZLE_PROMPT}]}
    ], add_generation_prompt=True)

    print("\n[RESPUESTA GENERADA]:\n")
    tokens = []
    t_gen_start = time.perf_counter()
    for resp in stream_generate(model, processor, prompt=prompt_chat, max_tokens=2200):
        tokens.append(resp.text)
        print(resp.text, end="", flush=True)
    t_gen = time.perf_counter() - t_gen_start
    print("\n\n" + "─" * 78)
    print(f"Velocidad: {len(tokens) / max(t_gen, 1e-5):.1f} tok/s | Tokens: {len(tokens)} en {t_gen:.2f}s")
    print("─" * 78 + "\n")

if __name__ == "__main__":
    for alias, path in MODELS:
        test_model(alias, path)
