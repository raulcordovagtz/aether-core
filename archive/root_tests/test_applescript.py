import sys, os, time, gc
import mlx.core as mx

sys.path.insert(0, "/Users/crotalo/aether_engine")
sys.path.insert(0, "/Users/crotalo/aether_engine/aether_vlm")

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

MODEL_REGISTRY = {
    "Qwen3.5-0.8B (Edge Compact)": os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit"),
    "Qwen3.5-2B (Compact Tied)":   os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit"),
}

PROMPT = """Escribe un script en AppleScript para macOS que controle la aplicación 'Music' (Música).
El script debe:
1. Abrir o activar la app 'Music'.
2. Buscar en la biblioteca una pista cuyo nombre sea o contenga 'Bohemian Rhapsody'.
3. Si la encuentra, darle 'play' (reproducirla).

Escribe el bloque de código de AppleScript funcional y una breve explicación."""

def ejecutar_test():
    print("=" * 75)
    print("  TEST DE PROGRAMACIÓN: APPLESCRIPT PARA REPRODUCIR MÚSICA EN MACOS")
    print("  Objetivo: Buscar y reproducir 'Bohemian Rhapsody' en Music.app")
    print("=" * 75)

    for nombre, ruta in MODEL_REGISTRY.items():
        print(f"\n" + "█" * 75)
        print(f"  EVALUANDO: {nombre}")
        print("█" * 75)

        if not os.path.exists(ruta):
            print(f"❌ Error: No se encontró el modelo en {ruta}")
            continue

        model, processor = load(ruta)
        aether = AetherEngine(model, processor)

        chat = processor.apply_chat_template([
            {"role": "user", "content": PROMPT}
        ], add_generation_prompt=True)

        aether.reset_counters()
        aether.prepare_thought(PROMPT, tau_steps=32)

        print("\n--- CÓDIGO GENERADO ---")
        t0 = time.time()
        for resp in stream_generate(model, processor, prompt=chat, max_tokens=450):
            print(resp.text, end="", flush=True)
        t = time.time() - t0
        print(f"\n\n└── [Tiempo de generación: {t:.2f}s]\n")

        del aether
        del model
        del processor
        gc.collect()

if __name__ == "__main__":
    ejecutar_test()
