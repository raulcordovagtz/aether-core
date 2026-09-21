import sys, os, time
import mlx.core as mx

sys.path.insert(0, "/Users/crotalo/aether_engine")
sys.path.insert(0, "/Users/crotalo/aether_engine/aether_vlm")

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

PROMPT = """Hay cuatro cajas de colores alineadas de izquierda a derecha en una mesa: Roja, Azul, Verde y Amarilla (en ese orden original). Sigue cuidadosamente estas instrucciones paso a paso y dime cuál es el orden final de las cajas de izquierda a derecha.

1. Intercambia la caja de los extremos (la primera y la última).
2. Toma la caja Azul y muévela una posición a la derecha.
3. Toma la caja que ahora está en la segunda posición (de izquierda a derecha) y colócala al final de la fila (a la extrema derecha).
4. Intercambia la caja Verde con la caja que esté inmediatamente a su izquierda.

¿Cuál es el orden final de las cajas de izquierda a derecha? Muestra tu razonamiento paso a paso."""

def test_config(nombre, params):
    print(f"\n" + "█" * 75)
    print(f"  PROBANDO: {nombre}")
    print(f"  Parámetros: {params}")
    print("█" * 75)

    model, processor = load(MODEL_PATH)
    
    # Inyectar motor con los parámetros específicos
    aether = AetherEngine(
        model, processor,
        theta_steer=params["theta_steer"],
        kappa=params["kappa"],
        gamma=params["gamma"],
        nu=params["nu"],
        slingshot=True
    )
    # Ajuste dinámico del ratio de capas activas si se especificó
    if "active_layers_ratio" in params:
        aether.active_layers_ratio = params["active_layers_ratio"]
        aether._install_circuit()

    chat = processor.apply_chat_template([
        {"role": "user", "content": PROMPT}
    ], add_generation_prompt=True)

    aether.prepare_thought(PROMPT, tau_steps=32)

    t0 = time.time()
    for resp in stream_generate(model, processor, prompt=chat, max_tokens=550):
        print(resp.text, end="", flush=True)
    t = time.time() - t0
    print(f"\n└── [Tiempo de inferencia: {t:.2f}s]\n")

if __name__ == "__main__":
    # 1. Configuración por defecto (Original)
    test_config("0.8B PERFIL ORIGINAL (edge_compact)", {
        "theta_steer": 0.25,
        "kappa": 0.15,
        "gamma": 0.25,
        "nu": 0.08,
        "active_layers_ratio": 0.50
    })

    # 2. Configuración Sintonizada (Anti-Inercia / Boosted Reasoning)
    test_config("0.8B PERFIL SINTONIZADO (Anti-Inercia)", {
        "theta_steer": 0.65,
        "kappa": 0.50,
        "gamma": 0.55,
        "nu": 0.05,
        "active_layers_ratio": 0.65
    })
