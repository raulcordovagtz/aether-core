#!/usr/bin/env python3
"""
Pruebas_Raul/04_rescate_conformal_2b.py
=======================================
Intervención en Vuelo: Rescate de Trayectoria en Qwen3.5-2B.
Aplica ConformalCouplingJunction dinámicamente cuando r < 0.15
para evitar el colapso en ciclos límite autorregresivos.
"""

import sys, os, time, math
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import numpy as np
import mlx.core as mx
from mlx_vlm import load, stream_generate
from aether_vlm.coupler import AetherEngine
import aether_native_c

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit")

PUZZLE_PROMPT = """Tres personas están en fila una detrás de otra.El último ve a los dos de adelante.El del medio ve al primero.El primero no ve a nadie.Se sacan cinco sombreros de una caja: tres negros y dos blancos. A cada persona se le coloca un sombrero en la cabeza a ciegas. Los dos sombreros restantes de la caja se guardan ocultos.Le preguntan al último de la fila (el que ve a los otros dos): ¿Sabes el color de tu sombrero? Dice que no.Le preguntan al del medio (que ve al primero): ¿Sabes el color de tu sombrero? Dice que no.El primero de la fila (que no ve a nadie) escucha las respuestas anteriores y dice: ¡Sí, mi sombrero es de color...¿De qué color es el sombrero del primero y por qué?"""

def run_rescue_experiment(max_tokens=250):
    print("═" * 84)
    print(" 🚑 INTERVENCIÓN DINÁMICA DE RESCATE (Harness Conformal en Qwen3.5-2B)")
    print(f"    Ruta: {MODEL_PATH}")
    print("═" * 84)

    model, processor = load(MODEL_PATH)
    aether = AetherEngine(model, processor)

    # 1. Asentamiento del Atractor de Coherencia
    aether.prepare_thought(PUZZLE_PROMPT)
    u_attractor = aether.state.get("L_star")
    aether_native_c.junction_reset()

    lm_model = model.language_model.model
    orig_norm = lm_model.norm

    rescue_log = []
    current_step = [0]
    intervention_count = [0]

    class AdaptiveRescueProbe:
        def __init__(self, norm_module):
            self.orig = norm_module

        def __getattr__(self, name):
            return getattr(self.orig, name)

        def __call__(self, x, **kwargs):
            if hasattr(x, "shape") and x.shape[1] == 1:
                h_raw = x[0, 0, :].astype(mx.float32)
                mx.eval(h_raw)

                norm_h = mx.sqrt(mx.sum(h_raw * h_raw)) + 1e-12
                h_unit = h_raw / norm_h
                mx.eval(h_unit)

                step = current_step[0]
                r_curr = float(mx.sum(h_unit * u_attractor))

                # Condición de activación: si la correlación cae por debajo de 0.15 tras el arranque
                if r_curr < 0.15 and step >= 15:
                    # Inyección Conformal Suave hacia L* (g=0.30, paso tau acotado)
                    routed = aether_native_c.dispatch_conformal_coupling(
                        h_unit, u_attractor, step=step,
                        tau_eff=0.10, kappa_att=0.30, mode=1, force_g=0.30
                    )
                    mx.eval(routed["h_steered"])
                    h_steered = (routed["h_steered"] * norm_h).astype(x.dtype)
                    h_out_tensor = h_steered[None, None, :]
                    
                    r_after = float(mx.sum(routed["h_steered"] * u_attractor))
                    rescue_log.append((step, r_curr, r_after, True))
                    intervention_count[0] += 1
                else:
                    h_out_tensor = x
                    rescue_log.append((step, r_curr, r_curr, False))

                current_step[0] += 1
                return self.orig(h_out_tensor, **kwargs)

            return self.orig(x, **kwargs)

    probe = AdaptiveRescueProbe(orig_norm)
    lm_model.norm = probe

    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": PUZZLE_PROMPT}]}
    ], add_generation_prompt=True)

    print("\n[GENERACIÓN CON CONTROL DE RESCATE EN SILICIO]:\n")
    tokens = []
    t0 = time.perf_counter()
    for resp in stream_generate(model, processor, prompt=prompt_chat, max_tokens=max_tokens):
        tokens.append(resp.text)
        print(resp.text, end="", flush=True)
    t_gen = time.perf_counter() - t0

    lm_model.norm = orig_norm
    aether.set_active(False)

    print("\n\n" + "─" * 84)
    print(f"Velocidad: {len(tokens) / max(t_gen, 1e-5):.1f} tok/s | Tokens emitidos: {len(tokens)} en {t_gen:.2f}s")
    print(f"Total intervenciones de rescate aplicadas: {intervention_count[0]} de {current_step[0]} pasos")
    print("─" * 84)

    print("\n[HISTORIAL DE INTERVENCIONES]:")
    for s, r_pre, r_post, applied in rescue_log:
        if applied:
            print(f"  • Paso {s:03d}: Alarma r={r_pre:+.4f} < 0.15 ──► RESCATE CONFORMAL ──► r_nuevo={r_post:+.4f}")
    print("═" * 84)

if __name__ == "__main__":
    run_rescue_experiment()
