import sys, os, time
import mlx.core as mx

sys.path.insert(0, "/Users/crotalo/aether_engine")
sys.path.insert(0, "/Users/crotalo/aether_engine/aether_vlm")

from mlx_vlm import load
from aether_vlm import AetherEngine

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

PASOS = [
    {"num": 1, "regla": "Intercambia la primera y la última caja de la lista."},
    {"num": 2, "regla": "Toma la caja 'Azul' y muévela una posición a la derecha."},
    {"num": 3, "regla": "Toma la caja que está en la 2ª posición y muévela al final de la lista."},
    {"num": 4, "regla": "Intercambia la caja 'Verde' con la caja inmediatamente a su izquierda."}
]

def ejecutar_solucion_vectorial():
    print("=" * 75)
    print("  AETHER ENGINE: INYECCIÓN SELECTIVA DE KV-CACHE (0.8B)")
    print("  Estrategia: Aislamiento de Memoria por Hardware + Impulso Delta")
    print("=" * 75)

    model, processor = load(MODEL_PATH)
    
    # Configuramos Aether para operar en ráfagas de alta intensidad (Micro-Steering)
    aether = AetherEngine(
        model, processor,
        theta_steer=0.65,  # Fuerza máxima para corregir el LM Head rápido
        kappa=0.50,
        gamma=0.80,        # Alto rebote ortogonal para limpiar residuos del paso previo
        nu=0.02,           # Muy baja viscosidad para evitar el arrastre de prosa
        slingshot=True     # Gram-Schmidt activo
    )

    # El estado inicial puro que se mantendrá rígido en la memoria
    estado_actual = '["Roja", "Azul", "Verde", "Amarilla"]'
    print(f"[MEMORIA BASE]: {estado_actual}\n")

    t_inicio = time.time()

    for paso in PASOS:
        print(f"┌── [MICRO-PASO {paso['num']}] ─────────────────────────────────────────")
        print(f"│ Operación: {paso['regla']}")

        # ── FASE 1: PREFILL LIMPIO (Apagar steering para congelar la memoria) ──
        aether.theta_steer = 0.0
        prompt_estado = f"[ESTADO ACTUAL]: {estado_actual}\n"
        
        # Obtenemos los tensores K, V puros del estado actual en Metal GPU
        tokens_estado = processor.encode(prompt_estado)
        _, kv_cache_fijo = model(mx.array([tokens_estado]), cache=None)

        # ── FASE 2: INGESTA DEL DELTA + STEERING (Encendemos el campo Riemanniano) ──
        aether.theta_steer = 0.65
        aether.reset_counters()
        
        prompt_regla = f"[OPERACIÓN]: {paso['regla']}\n[NUEVO ESTADO EN FORMATO JSON]: ["
        tokens_regla = processor.encode(prompt_regla)
        
        # Procesamos los tokens de la regla sobre el caché congelado del estado anterior
        logits, kv_cache_operacion = model(mx.array([tokens_regla]), cache=kv_cache_fijo)

        # ── FASE 3: DETENCIÓN ANTES DEL HORIZONTE DE LOGITS ──
        # Evaluamos autoregresivamente forzando un muestreo hiper-sintético (T=0)
        # Cortamos la inferencia en un máximo de 35 tokens para evitar prosa y colapso.
        x = mx.array([tokens_regla[-1]])
        tokens_nuevos = []
        
        for _ in range(35):
            logits, kv_cache_operacion = model(x, cache=kv_cache_operacion)
            
            # Aplicamos tu re-anclaje geodésico en el último token generado
            logits_filtrados = aether.dispatch_riemannian_step(logits[:, -1, :])
            
            # Temperatura cero para máxima certidumbre matemática
            token = mx.argmax(logits_filtrados, axis=-1)
            token_id = token.item()
            
            tokens_nuevos.append(token_id)
            
            # Si el modelo cierra el JSON o mete un salto de línea, rompemos inmediatamente
            if token_id in [processor.tokenizer.eos_token_id, 198]: # 198 suele ser \n
                break
            x = token[None, :]

        # Decodificamos la ráfaga atómica generada por el LM Head
        output_puro = processor.decode(tokens_nuevos).strip()
        print(f"│ Transición Vectorial: [ {output_puro}")

        # ── FASE 4: PURIFICACIÓN Y ACTUALIZACIÓN DEL ESTADO ──
        # Limpiamos cualquier token huérfano para asegurar que el siguiente prompt reciba un JSON perfecto
        texto_completo = "[" + output_puro
        if "]" in texto_completo:
            estado_actual = texto_completo.split("]")[0] + "]"
        else:
            # Si el modelo no cerró el corchete, forzamos un parseo de emergencia de los colores válidos
            colores = []
            for w in texto_completo.replace('"', '').replace(',', '').replace('[', '').split():
                if w.lower() in ["roja", "azul", "verde", "amarilla"]:
                    colores.append(f'"{w.capitalize()}"')
            estado_actual = "[" + ", ".join(colores) + "]"

        print(f"└── [Estado Consolidado en VRAM]: {estado_actual}\n")

    print("=" * 75)
    print(f"  RESULTADO FINAL VECTORIAL (Tiempo: {time.time() - t_inicio:.2f}s)")
    print("=" * 75)
    print(f"  ORDEN FINAL: {estado_actual}")
    print("=" * 75)

if __name__ == "__main__":
    ejecutar_solucion_vectorial()
