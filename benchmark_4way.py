import sys, os, time, gc
import mlx.core as mx

sys.path.insert(0, "/Users/crotalo/aether_engine")
sys.path.insert(0, "/Users/crotalo/aether_engine/aether_vlm")

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

# ─── MODELOS A EVALUAR ──────────────────────────────────────────────
MODEL_REGISTRY = {
    "Qwen3.5-0.8B": os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit"),
    "Qwen3.5-2B":   os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit"),
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

def ejecutar_test(model, processor, aether, modo_aether: bool, prompt_fase1, prompt_fase2):
    tag_modo = "AETHER ENGINE" if modo_aether else "VANILLA (ESTÁNDAR)"
    print(f"\n{'─'*75}")
    print(f"  MODO: [{tag_modo}]")
    print(f"{'─'*75}")

    if modo_aether:
        aether.set_active(True)
    else:
        aether.set_active(False)

    # ─── TEST 1: RAZONAMIENTO PASO A PASO ───
    print(f"\n┌── [TEST 1: RAZONAMIENTO PASO A PASO - {tag_modo}] ──")
    chat_fase1 = processor.apply_chat_template([
        {"role": "user", "content": prompt_fase1}
    ], add_generation_prompt=True)

    if modo_aether:
        aether.reset_counters()
        aether.prepare_thought(prompt_fase1, tau_steps=32)

    t0 = time.time()
    for resp in stream_generate(model, processor, prompt=chat_fase1, max_tokens=550):
        print(resp.text, end="", flush=True)
    t_fase1 = time.time() - t0
    print(f"\n└── [Tiempo: {t_fase1:.2f}s] ──────────────────────────")

    # ─── TEST 2: VALIDACIÓN BINARIA ───
    print(f"\n┌── [TEST 2: VALIDACIÓN DIRECTA - {tag_modo}] ──")
    chat_fase2 = processor.apply_chat_template([
        {"role": "user", "content": prompt_fase2}
    ], add_generation_prompt=True)

    if modo_aether:
        aether.reset_counters()
        aether.prepare_thought(prompt_fase2, tau_steps=32)

    print("Respuesta: ", end="", flush=True)
    t0 = time.time()
    for resp in stream_generate(model, processor, prompt=chat_fase2, max_tokens=25):
        print(resp.text, end="", flush=True)
    t_fase2 = time.time() - t0
    print(f"\n└── [Tiempo: {t_fase2:.2f}s] ──────────────────────────\n")

def ejecutar_benchmark():
    print("=" * 75)
    print("  GRAN BENCHMARK 4-WAY: VANILLA vs AETHER ENGINE")
    print("  Problema: La Mudanza de las Cajas")
    print("  Solución correcta esperada: Amarilla, Azul, Verde, Roja")
    print("=" * 75)

    for nombre_modelo, ruta in MODEL_REGISTRY.items():
        print(f"\n" + "█" * 75)
        print(f"  MODELO BASE: {nombre_modelo}")
        print(f"  Ruta: {ruta}")
        print("█" * 75)

        if not os.path.exists(ruta):
            print(f"❌ Error: No se encontró el modelo en {ruta}")
            continue

        model, processor = load(ruta)
        aether = AetherEngine(model, processor)

        # 1. EJECUCIÓN VANILLA (Aether desactivado)
        ejecutar_test(model, processor, aether, modo_aether=False, 
                      prompt_fase1=PROMPT_FASE1, prompt_fase2=PROMPT_FASE2)

        # 2. EJECUCIÓN CON AETHER ENGINE (Aether activado con colapso C++)
        ejecutar_test(model, processor, aether, modo_aether=True, 
                      prompt_fase1=PROMPT_FASE1, prompt_fase2=PROMPT_FASE2)

        del aether
        del model
        del processor
        gc.collect()

if __name__ == "__main__":
    ejecutar_benchmark()
