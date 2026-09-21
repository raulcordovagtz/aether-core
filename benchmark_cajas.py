import sys, os, time, gc
import mlx.core as mx

sys.path.insert(0, "/Users/crotalo/aether_engine")
sys.path.insert(0, "/Users/crotalo/aether_engine/aether_vlm")

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

# ─── MODELOS A EVALUAR ──────────────────────────────────────────────
MODEL_REGISTRY = {
    "Qwen3.5-0.8B (Edge Compact)": os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit"),
    "Qwen3.5-2B (Compact Tied)":   os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit"),
}

# ─── PROMPTS DE EVALUACIÓN ──────────────────────────────────────────
PROMPT_FASE1 = """Hay cuatro cajas de colores alineadas de izquierda a derecha en una mesa: Roja, Azul, Verde y Amarilla (en ese orden original). Sigue cuidadosamente estas instrucciones paso a paso y dime cuál es el orden final de las cajas de izquierda a derecha.

1. Intercambia la caja de los extremos (la primera y la última).
2. Toma la caja Azul y muévela una posición a la derecha.
3. Toma la caja que ahora está en la segunda posición (de izquierda a derecha) y colócala al final de la fila (a la extrema derecha).
4. Intercambia la caja Verde con la caja que esté inmediatamente a su izquierda.

¿Cuál es el orden final de las cajas de izquierda a derecha? Muestra tu razonamiento paso a paso."""

PROMPT_FASE2 = """Problema de lógica:
Hay cuatro cajas de colores alineadas de izquierda a derecha en una mesa: Roja, Azul, Verde y Amarilla (en ese orden original).
1. Intercambia la caja de los extremos (la primera y la última).
2. Toma la caja Azul y muévela una posición a la derecha.
3. Toma la caja que ahora está en la segunda posición (de izquierda a derecha) y colócala al final de la fila (a la extrema derecha).
4. Intercambia la caja Verde con la caja que esté inmediatamente a su izquierda.

Afirmación: "El orden final de las cajas de izquierda a derecha es: Amarilla, Azul, Verde, Roja."

Pregunta: ¿La afirmación es Verdadera o Falsa?
Responde estrictamente con una sola palabra ("Verdadera" o "Falsa"):"""

def ejecutar_benchmark():
    print("=" * 75)
    print("  CARA A CARA: LA MUDANZA DE LAS CAJAS (AETHER ENGINE)")
    print("  Solución esperada: Amarilla, Azul, Verde, Roja")
    print("=" * 75)

    for nombre_modelo, ruta in MODEL_REGISTRY.items():
        print(f"\n" + "█" * 75)
        print(f"  CARGANDO: {nombre_modelo}")
        print(f"  Ruta: {ruta}")
        print("█" * 75)

        if not os.path.exists(ruta):
            print(f"❌ Error: No se encontró el modelo en {ruta}")
            continue

        model, processor = load(ruta)
        aether = AetherEngine(model, processor)
        print(f"✓ Motor Aether acoplado (Perfil: {aether.profile_name}, Capas: {aether.num_layers})")

        # ─── FASE 1: RAZONAMIENTO PASO A PASO ───
        print(f"\n┌── [TEST 1: RAZONAMIENTO PASO A PASO] ──────────────────────────")
        chat_fase1 = processor.apply_chat_template([
            {"role": "user", "content": PROMPT_FASE1}
        ], add_generation_prompt=True)

        aether.reset_counters()
        aether.prepare_thought(PROMPT_FASE1, tau_steps=32)

        t0 = time.time()
        for resp in stream_generate(model, processor, prompt=chat_fase1, max_tokens=550):
            print(resp.text, end="", flush=True)
        t_fase1 = time.time() - t0
        print(f"\n└── [Tiempo Fase 1: {t_fase1:.2f}s] ──────────────────────────────")

        # ─── FASE 2: DISCRIMINACIÓN BINARIA (UNA SOLA PALABRA) ───
        print(f"\n┌── [TEST 2: VALIDACIÓN DIRECTA (VERDADERA / FALSA)] ───────────")
        chat_fase2 = processor.apply_chat_template([
            {"role": "user", "content": PROMPT_FASE2}
        ], add_generation_prompt=True)

        aether.reset_counters()
        aether.prepare_thought(PROMPT_FASE2, tau_steps=32)

        print("Respuesta: ", end="", flush=True)
        t0 = time.time()
        for resp in stream_generate(model, processor, prompt=chat_fase2, max_tokens=20):
            print(resp.text, end="", flush=True)
        t_fase2 = time.time() - t0
        print(f"\n└── [Tiempo Fase 2: {t_fase2:.2f}s] ──────────────────────────────\n")

        # Liberar memoria antes del siguiente modelo
        del aether
        del model
        del processor
        gc.collect()

if __name__ == "__main__":
    ejecutar_benchmark()
