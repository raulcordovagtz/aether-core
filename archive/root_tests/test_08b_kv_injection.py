import sys, os, time, re
import mlx.core as mx

sys.path.insert(0, "/Users/crotalo/aether_engine")
sys.path.insert(0, "/Users/crotalo/aether_engine/aether_vlm")

from mlx_vlm import load
from aether_vlm import AetherEngine

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

PASOS = [
    {
        "num": 1,
        "instruccion": "Intercambia la primera y la última caja de los extremos.",
        "esperado": "Amarilla, Azul, Verde, Roja"
    },
    {
        "num": 2,
        "instruccion": "Toma la caja Azul y muévela una posición a la derecha.",
        "esperado": "Amarilla, Verde, Azul, Roja"
    },
    {
        "num": 3,
        "instruccion": "Toma la caja de la segunda posición y muévela al final de la fila.",
        "esperado": "Amarilla, Azul, Roja, Verde"
    },
    {
        "num": 4,
        "instruccion": "Intercambia la caja Verde con la caja que esté inmediatamente a su izquierda.",
        "esperado": "Amarilla, Azul, Verde, Roja"
    }
]

def clonar_kv_cache(cache):
    """Clona profundamente los tensores K y V en Metal VRAM."""
    if cache is None:
        return None
    nuevo_cache = []
    for c in cache:
        if hasattr(c, "keys") and hasattr(c, "values"):
            nuevo_c = c.__class__()
            if c.keys is not None:
                nuevo_c.keys = mx.array(c.keys)
            if c.values is not None:
                nuevo_c.values = mx.array(c.values)
            nuevo_c.offset = c.offset
            nuevo_cache.append(nuevo_c)
        else:
            nuevo_cache.append(c)
    return nuevo_cache

def ejecutar_kv_injection():
    print("=" * 75)
    print("  QWEN3.5-0.8B: INYECCIÓN SELECTIVA DE KV-CACHE EN METAL GPU")
    print("  Arquitectura: Memoria Estática (Steer=0) + Delta de Riemann (Steer>0)")
    print("=" * 75)

    model, processor = load(MODEL_PATH)
    tokenizer = getattr(processor, "tokenizer", processor)

    # Inicializar motor Aether
    aether = AetherEngine(
        model, processor,
        theta_steer=0.35,
        kappa=0.25,
        gamma=0.35,
        nu=0.06,
        slingshot=True
    )

    # Intentamos cargar make_cache desde mlx_lm o mlx_vlm
    try:
        from mlx_lm.models.cache import make_cache
    except ImportError:
        try:
            from mlx_vlm.utils import make_cache
        except ImportError:
            # Fallback nativo
            def make_cache(m):
                return [None] * len(m.model.layers)

    lm_model = model.language_model
    estado_actual = "Roja, Azul, Verde, Amarilla"
    print(f"\n[ESTADO 0 - CONGELADO EN SILICIO]: \033[94m{estado_actual}\033[0m\n")

    t_inicio_total = time.time()

    for paso in PASOS:
        print(f"┌── [MICRO-PASO {paso['num']}] ──────────────────────────────────────────")
        print(f"│ Estado Entrada : \033[94m{estado_actual}\033[0m")
        print(f"│ Operación      : {paso['instruccion']}")

        # ─── FASE 1: PREFILL ESTÁTICO (KV-CACHE FREEZE EN VRAM) ───
        # Construimos el prompt de estado canónico con chat template
        prompt_estado_raw = f"Las cuatro cajas en la mesa están en el orden: {estado_actual}."
        msg_prefill = [
            {"role": "user", "content": prompt_estado_raw}
        ]
        # Obtenemos tokens sin generation_prompt (solo contexto base)
        tokens_estado = tokenizer.encode(prompt_estado_raw)
        x_estado = mx.array(tokens_estado)[None, :]

        # Desactivamos Aether: lectura 100% limpia sin distorsión
        aether.set_active(False)
        cache_estatico = make_cache(lm_model.model)
        
        # Ejecución forward de Prefill para poblar K y V en GPU
        lm_model.model(x_estado, cache=cache_estatico)
        mx.eval([c.keys for c in cache_estatico if c.keys is not None])

        # ─── FASE 2: INGESTIÓN DELTA CON AETHER ENGINE (STEERING ACTIVADO) ───
        # Clonamos el caché en Metal para que el estado previo sea inmutable
        kv_operacion = clonar_kv_cache(cache_estatico)

        prompt_delta = f"\nInstrucción: {paso['instruccion']}\nEl nuevo orden de las cuatro cajas de izquierda a derecha es:"
        tokens_delta = tokenizer.encode(prompt_delta)
        x_delta = mx.array(tokens_delta)[None, :]

        # Activamos Aether Engine sobre los tokens del cambio
        aether.set_active(True)
        aether.reset_counters()
        aether.prepare_thought(paso['instruccion'], tau_steps=16)

        t0 = time.time()
        # Ingesta del delta consumiendo el caché estático
        logits_delta = lm_model(x_delta, cache=kv_operacion)
        token_actual = mx.argmax(logits_delta[:, -1, :], axis=-1)

        # ─── FASE 3: DECODIFICACIÓN ATÓMICA DE SALIDA (< 20 TOKENS) ───
        tokens_generados = []
        
        for _ in range(25):
            tok_id = int(token_actual.item())
            tokens_generados.append(tok_id)
            texto_parcial = tokenizer.decode(tokens_generados)

            # Si detectamos salto de línea o EOS, paramos el colapso
            if tok_id == tokenizer.eos_token_id or "\n" in texto_parcial or "." in texto_parcial:
                break

            # Forward del siguiente token unitario
            x_next = token_actual[None, None]
            logits_step = lm_model(x_next, cache=kv_operacion)
            token_actual = mx.argmax(logits_step[:, -1, :], axis=-1)

        t_paso = time.time() - t0
        salida_texto = tokenizer.decode(tokens_generados).strip().strip(".").strip(":")

        # Extraer los 4 colores del tensor de salida
        colores = re.findall(r'\b(Roja|Azul|Verde|Amarilla)\b', salida_texto, re.IGNORECASE)
        colores_norm = [c.capitalize() for c in colores]

        if len(colores_norm) >= 4:
            nuevo_estado = f"{colores_norm[0]}, {colores_norm[1]}, {colores_norm[2]}, {colores_norm[3]}"
        else:
            nuevo_estado = salida_texto  # Fallback si ya vino limpio

        print(f"│ Salida Cruda   : \"{salida_texto}\"")
        print(f"│ Nuevo Estado   : \033[92m{nuevo_estado}\033[0m")
        print(f"│ Esperado       : \033[90m{paso['esperado']}\033[0m")
        print(f"└── [Latencia GPU: {t_paso:.2f}s] ────────────────────────────────────────\n")

        # El nuevo estado verificado se convierte en el ancla del siguiente paso
        estado_actual = nuevo_estado

    t_total = time.time() - t_inicio_total
    print("=" * 75)
    print(f"  RESULTADO FINAL: \033[92m{estado_actual}\033[0m")
    print(f"  Solución Teórica: Amarilla, Azul, Verde, Roja")
    print(f"  Tiempo Total: {t_total:.2f}s")
    print("=" * 75)

if __name__ == "__main__":
    ejecutar_kv_injection()
