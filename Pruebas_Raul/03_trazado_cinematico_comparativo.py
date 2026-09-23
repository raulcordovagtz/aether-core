#!/usr/bin/env python3
"""
Pruebas_Raul/03_trazado_cinematico_comparativo.py
=================================================
Trazador Cinemático de Trayectoria Latente sobre S^{D-1}:
Normalización unitaria conforme a especificación C-018 antes
del cálculo de velocidad, curvatura y tensión en 0.8B y 2B.
"""

import sys, os, time, math
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import numpy as np
import mlx.core as mx
from mlx_vlm import load, stream_generate
from aether_vlm.coupler import AetherEngine
import aether_native_c

MODELS = [
    ("0.8B", os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")),
    ("2B",   os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit")),
]

PUZZLE_PROMPT = """Tres personas están en fila una detrás de otra.El último ve a los dos de adelante.El del medio ve al primero.El primero no ve a nadie.Se sacan cinco sombreros de una caja: tres negros y dos blancos. A cada persona se le coloca un sombrero en la cabeza a ciegas. Los dos sombreros restantes de la caja se guardan ocultos.Le preguntan al último de la fila (el que ve a los otros dos): ¿Sabes el color de tu sombrero? Dice que no.Le preguntan al del medio (que ve al primero): ¿Sabes el color de tu sombrero? Dice que no.El primero de la fila (que no ve a nadie) escucha las respuestas anteriores y dice: ¡Sí, mi sombrero es de color...¿De qué color es el sombrero del primero y por qué?"""

def trace_model(alias, path, max_tokens=45):
    print("\n" + "═" * 84)
    print(f" 🛰️  TRAZADO CINEMÁTICO NORMALIZADO SOBRE S^{{D-1}}: Qwen3.5-{alias}")
    print(f"     Ruta: {path}")
    print("═" * 84)

    model, processor = load(path)
    aether = AetherEngine(model, processor)
    
    # Preparar el atractor del campo continuo
    aether.prepare_thought(PUZZLE_PROMPT)
    u_attractor = aether.state.get("L_star")
    if u_attractor is None:
        print("❌ Error: No se pudo obtener L_star")
        return

    lm_model = model.language_model.model
    orig_norm = lm_model.norm
    aether_native_c.buffer_reset()

    telemetry_records = []
    current_step = [0]

    class KinematicTraceProbe:
        def __init__(self, norm_module):
            self.orig = norm_module

        def __getattr__(self, name):
            return getattr(self.orig, name)

        def __call__(self, x, **kwargs):
            if hasattr(x, "shape") and x.shape[1] == 1:
                h_raw = x[0, 0, :].astype(mx.float32)
                mx.eval(h_raw)

                # ── CORRECCIÓN MÉTRICA: Confinamiento estricto a S^{D-1} ──
                norm_h = mx.sqrt(mx.sum(h_raw * h_raw)) + 1e-12
                h_unit = h_raw / norm_h
                mx.eval(h_unit)

                step = current_step[0]
                st = aether_native_c.buffer_push_state(h_unit, step=step)
                
                # Evaluar la Célula Proyectiva con vector unitario
                cell_out = aether_native_c.dispatch_geodesic_trajectory_cell(
                    h_unit, st["v_t"], st["a_t"], u_attractor,
                    tau=0.15, kappa_att=aether.kappa
                )
                mx.eval(cell_out["correlation_r"], cell_out["curvature_kappa"])

                v_norm = math.sqrt(max(0.0, float(st["sq_v"])))
                kappa_val = float(cell_out["curvature_kappa"])
                q_val = float(st["dirichlet_tension_q"])
                r_val = float(cell_out["correlation_r"])

                telemetry_records.append({
                    "step": step,
                    "v_norm": v_norm,
                    "kappa": kappa_val,
                    "q": q_val,
                    "r": r_val,
                })
                current_step[0] += 1

            return self.orig(x, **kwargs)

    probe = KinematicTraceProbe(orig_norm)
    lm_model.norm = probe

    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": PUZZLE_PROMPT}]}
    ], add_generation_prompt=True)

    print(f"\n{'Paso':<5} │ {'Token emitido':<16} │ {'||v||':<9} │ {'κ (curv)':<10} │ {'q_k (tensión)':<13} │ {'r (certeza)':<11} │ {'Régimen'}")
    print("─" * 84)

    token_count = 0
    for resp in stream_generate(model, processor, prompt=prompt_chat, max_tokens=max_tokens):
        tok_str = resp.text.replace("\n", "↵ ")
        if token_count < len(telemetry_records):
            m = telemetry_records[token_count]
            regime = "Laminar" if m["r"] >= 0.95 else ("Transición" if m["r"] >= 0.80 else "Turbulento")
            print(f"{m['step']:03d}   │ {tok_str[:14]:<16} │ {m['v_norm']:<9.4f} │ {m['kappa']:<10.4f} │ {m['q']:<13.4f} │ {m['r']:<+11.4f} │ {regime}")
        token_count += 1

    lm_model.norm = orig_norm
    aether.set_active(False)
    print("─" * 84)

if __name__ == "__main__":
    for alias, path in MODELS:
        trace_model(alias, path, max_tokens=45)
