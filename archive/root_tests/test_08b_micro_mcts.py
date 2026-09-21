import sys, os, time, gc
import mlx.core as mx

sys.path.insert(0, "/Users/crotalo/aether_engine")
sys.path.insert(0, "/Users/crotalo/aether_engine/aether_vlm")

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

# ─── DEFINICIÓN DEL PROTOCOLO DE MICRO-PASOS ───────────────────────
PASOS = [
    {
        "num": 1,
        "instruccion": "Intercambia la primera y la última caja de la lista.",
        "pregunta": "¿Cuál es la nueva lista exacta de cajas?"
    },
    {
        "num": 2,
        "instruccion": "Toma la caja 'Azul' y muévela una posición a la derecha.",
        "pregunta": "¿Cuál es la nueva lista exacta de cajas?"
    },
    {
        "num": 3,
        "instruccion": "Toma la caja que está en la 2ª posición y muévela al final de la lista.",
        "pregunta": "¿Cuál es la nueva lista exacta de cajas?"
    },
    {
        "num": 4,
        "instruccion": "Intercambia la caja 'Verde' con la caja inmediatamente a su izquierda.",
        "pregunta": "¿Cuál es la nueva lista final exacta de cajas?"
    }
]

def ejecutar_micro_inferencia():
    print("=" * 75)
    print("  QWEN3.5-0.8B: RAZONAMIENTO MARKOVIANO CON RE-ANCLAJE DE ATRACTOR")
    print("  Objetivo: Resolver paso a paso descargando entropía latente en Metal GPU")
    print("=" * 75)

    model, processor = load(MODEL_PATH)
    aether = AetherEngine(model, processor, theta_steer=0.35, kappa=0.30, gamma=0.40, nu=0.06, slingshot=True)

    # Estado inicial anclado
    estado_actual = "Posición 1: Roja, Posición 2: Azul, Posición 3: Verde, Posición 4: Amarilla"
    print(f"\n[ESTADO INICIAL CONOCIDO]:\n  {estado_actual}\n")

    t_total_inicio = time.time()

    for paso in PASOS:
        print(f"┌── [EJECUTANDO MICRO-PASO {paso['num']}] ──────────────────────────────")
        print(f"│ Regla: {paso['instruccion']}")
        print(f"│ Contexto base: {estado_actual}")

        # Prompt atómico y focalizado (sin ruido de pasos anteriores)
        prompt_paso = f"""Estado actual de las 4 cajas:
{estado_actual}

Instrucción a aplicar:
{paso['instruccion']}

Escribe el nuevo estado de las posiciones 1, 2, 3 y 4 tras aplicar la instrucción:"""

        chat = processor.apply_chat_template([
            {"role": "user", "content": prompt_paso}
        ], add_generation_prompt=True)

        # 1. Reiniciar contadores y asentar atractor limpio para este micro-paso
        aether.reset_counters()
        aether.prepare_thought(prompt_paso, tau_steps=32)

        # 2. Generar resolución atómica rápida (< 80 tokens)
        respuesta_paso = ""
        t0 = time.time()
        for resp in stream_generate(model, processor, prompt=chat, max_tokens=100):
            respuesta_paso += resp.text
        t_paso = time.time() - t0

        print(f"│\n│ Transición generada ({t_paso:.2f}s):\n")
        for linea in respuesta_paso.strip().split("\n"):
            if linea.strip():
                print(f"│   {linea}")

        # 3. El nuevo estado se convierte en la base del siguiente paso
        estado_actual = respuesta_paso.strip()
        print(f"└── [Micro-Paso {paso['num']} consolidado en GPU] ───────────────────\n")

    t_total = time.time() - t_total_inicio
    print("=" * 75)
    print(f"  RESULTADO FINAL CONSOLIDADO (Tiempo total: {t_total:.2f}s)")
    print("=" * 75)
    print(estado_actual)
    print("=" * 75)

if __name__ == "__main__":
    ejecutar_micro_inferencia()
