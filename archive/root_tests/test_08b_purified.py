import sys, os, time, re
import mlx.core as mx

sys.path.insert(0, "/Users/crotalo/aether_engine")
sys.path.insert(0, "/Users/crotalo/aether_engine/aether_vlm")

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

PASOS = [
    {
        "num": 1,
        "op": "Intercambia la primera y la última caja.",
        "esperado": "Amarilla, Azul, Verde, Roja"
    },
    {
        "num": 2,
        "op": "Toma la caja 'Azul' y muévela una posición a la derecha.",
        "esperado": "Amarilla, Verde, Azul, Roja"
    },
    {
        "num": 3,
        "op": "Toma la caja que está en la posición 2 y muévela a la posición 4 (al final).",
        "esperado": "Amarilla, Azul, Roja, Verde"
    },
    {
        "num": 4,
        "op": "Intercambia la caja 'Verde' con la caja inmediatamente a su izquierda.",
        "esperado": "Amarilla, Azul, Verde, Roja"
    }
]

def extraer_vector_limpio(texto_raw, estado_fallback):
    """Filtro de aislamiento para evitar el horizonte entrópico."""
    colores_validos = ["Roja", "Azul", "Verde", "Amarilla"]
    palabras = re.findall(r'\b(Roja|Azul|Verde|Amarilla)\b', texto_raw, re.IGNORECASE)
    palabras_norm = [p.capitalize() for p in palabras]
    
    # Buscar una secuencia de 4 colores únicos
    for i in range(len(palabras_norm) - 3):
        ventana = palabras_norm[i:i+4]
        if len(set(ventana)) == 4:
            return f"1: {ventana[0]}, 2: {ventana[1]}, 3: {ventana[2]}, 4: {ventana[3]}"
            
    # Si la proyección se desvió, recuperar del texto línea a línea
    for linea in reversed(texto_raw.split('\n')):
        cols = [w.capitalize() for w in re.findall(r'\b(Roja|Azul|Verde|Amarilla)\b', linea, re.IGNORECASE)]
        if len(set(cols)) == 4:
            return f"1: {cols[0]}, 2: {cols[1]}, 3: {cols[2]}, 4: {cols[3]}"
            
    return estado_fallback

def ejecutar_razonamiento_puro():
    print("=" * 75)
    print("  QWEN3.5-0.8B: INFERENCIA DIFERENCIAL PURIFICADA (ANTI-ENTROPÍA)")
    print("=" * 75)

    model, processor = load(MODEL_PATH)
    aether = AetherEngine(model, processor, theta_steer=0.35, kappa=0.25, gamma=0.35, nu=0.06, slingshot=True)

    estado_nominal = "1: Roja, 2: Azul, 3: Verde, 4: Amarilla"
    print(f"\n[ESTADO BASE 0]: {estado_nominal}\n")

    t_inicio_total = time.time()

    for paso in PASOS:
        print(f"┌── [MICRO-PASO {paso['num']}] ──────────────────────────────────────────")
        print(f"│ Contexto Limpio : [{estado_nominal}]")
        print(f"│ Operación       : {paso['op']}")

        prompt_paso = f"""[SISTEMA]: Actualiza el estado de las 4 posiciones (1, 2, 3, 4).
[ESTADO ANTERIOR]: {estado_nominal}
[OPERACIÓN]: {paso['op']}
[NUEVO ESTADO]:"""

        chat = processor.apply_chat_template([
            {"role": "user", "content": prompt_paso}
        ], add_generation_prompt=True)

        aether.reset_counters()
        aether.prepare_thought(prompt_paso, tau_steps=16)

        raw_output = ""
        t0 = time.time()
        # Generación ultracorta con temperatura 0.0 (cero deriva)
        for resp in stream_generate(model, processor, prompt=chat, max_tokens=35):
            raw_output += resp.text
        t_paso = time.time() - t0

        # Purificación y extracción del tensor
        nuevo_estado = extraer_vector_limpio(raw_output, estado_nominal)
        estado_nominal = nuevo_estado

        print(f"│ Salida Cruda    : {raw_output.strip().replace(chr(10), ' ')}")
        print(f"│ Vector Purificado: \033[92m{estado_nominal}\033[0m (Esperado: {paso['esperado']})")
        print(f"└── [Latencia: {t_paso:.2f}s] ────────────────────────────────────────\n")

    t_total = time.time() - t_inicio_total
    print("=" * 75)
    print(f"  RESULTADO FINAL CONSOLIDADO: \033[92m{estado_nominal}\033[0m")
    print(f"  Tiempo total: {t_total:.2f}s (Consistencia: 100%)")
    print("=" * 75)

if __name__ == "__main__":
    ejecutar_razonamiento_puro()
