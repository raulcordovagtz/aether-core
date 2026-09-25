#!/usr/bin/env python3
"""
tests/test_engine_stress_telemetry.py
═══════════════════════════════════════════════════════════════════════════════
TELEMETRÍA DE ESTRÉS DEL MOTOR PURO (SIN CÉLULA 1 / SIN HARNESS)
Captura paso a paso en wrapper de clase:
  - Momento de arrastre v_drag y descomposición de Penrose
  - Distancia geodésica d_g y varianza de logits
  - Volcado a results/engine_stress_telemetry.json
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, time, json, math, argparse
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

MODEL_REGISTRY = {
    "0.8b":    os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit"),
    "2b":      os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit"),
    "27b":     os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit"),
    "35b_moe": os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-6bit"),
}

STRESS_PROMPTS = [
    {
        "id": "ESTRES_LOGICO",
        "tipo": "Restricciones Lógicas Encadenadas",
        "text": (
            "Cinco enteros distintos A, B, C, D, E entre 1 y 15 cumplen: "
            "A > B, B > C, D = C + 6, E = A + C, E < 13, D < B. A es par y C es impar. "
            "Determina el valor exacto de cada variable explicando paso a paso."
        ),
        "max_tokens": 120
    },
    {
        "id": "ESTRES_CONTRADICCION",
        "tipo": "Conflicto de Premisas Opuestas",
        "text": (
            "Premisa 1: Todos los objetos metálicos conducen electricidad.\n"
            "Premisa 2: El polímero X es un aislante que no conduce electricidad.\n"
            "Premisa 3: Se ha demostrado que el polímero X es un metal puro.\n"
            "Pregunta: Analiza la inconsistencia entre estas premisas y explica qué regla se rompe."
        ),
        "max_tokens": 120
    }
]

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def run_stress_battery(model_key="0.8b"):
    model_path = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY["0.8b"])
    section(f"INICIANDO TELEMETRÍA DEL MOTOR PURO EN [{model_key.upper()}]")
    print(f"  Ruta: {model_path}")
    print(f"  Modo: AetherEngine Nativo (Sin Célula 1 / Sin Junction)")

    model, processor = load(model_path)
    aether = AetherEngine(model, processor)
    print(f"  ✓ Motor acoplado en perfil: [{aether.profile_name}]")
    print(f"    - Capas: {aether.num_layers} | Dimensión D: {aether.hidden_dim}")
    print(f"    - Parámetros: κ={aether.kappa}, θ={aether.theta_steer}, ν={aether.nu}, γ={aether.gamma}")

    all_cases_telemetry = []

    for case in STRESS_PROMPTS:
        section(f"CASO: [{case['id']}] — {case['tipo']}")
        prompt_text = case["text"]
        print(f"  Prompt: \"{prompt_text[:80]}...\"\n")

        # 1. Asentamiento
        t0_settle = time.perf_counter()
        telemetry_settle = aether.prepare_thought(prompt_text, tau_steps=32)
        dt_settle = (time.perf_counter() - t0_settle) * 1e3
        peak_l = aether.state.get("peak_layer", "N/A")
        print(f"  • Asentamiento L* completado en {dt_settle:.2f} ms (Cresta L* = {peak_l}/{aether.num_layers})")

        # 2. Interceptor de Telemetría sobre la Cabeza
        decode_telemetry = []
        tokens_emitted = []
        real_head = model.language_model.lm_head

        class TelemetryHeadWrapper:
            def __init__(self, inner):
                self.inner = inner
                self.step = 0
            def __getattr__(self, name):
                return getattr(self.inner, name)
            def __call__(self, h, **kwargs):
                z_out = self.inner(h, **kwargs)
                if hasattr(h, "shape") and h.shape[1] == 1:
                    v_drag = aether.state.get("v_drag")
                    L_star = aether.state.get("L_star")
                    norm_v = float(mx.sqrt(mx.sum(v_drag * v_drag))) if v_drag is not None else 0.0
                    norm_L = float(mx.sqrt(mx.sum(L_star * L_star))) if L_star is not None else 0.0

                    tok_id = int(mx.argmax(z_out[0, -1, :]))
                    w_t = self.inner.deq_W[tok_id]
                    norm_wt = self.inner.row_norms[tok_id]
                    w_t_unit = w_t / (norm_wt + 1e-12)

                    if v_drag is not None:
                        v_rad_val = float(mx.sum(v_drag * w_t_unit))
                        v_perp_sq = max(0.0, norm_v**2 - v_rad_val**2)
                        v_perp_val = math.sqrt(v_perp_sq)
                    else:
                        v_rad_val, v_perp_val = 0.0, 0.0

                    h_vec = h[0, -1, :]
                    norm_h = float(mx.sqrt(mx.sum(h_vec * h_vec)) + 1e-12)
                    cos_t = float(mx.clip(mx.sum(h_vec * w_t_unit) / norm_h, -1.0 + 1e-7, 1.0 - 1e-7))
                    d_g = math.acos(cos_t)
                    var_z = float(mx.var(z_out[0, -1, :]))

                    decode_telemetry.append({
                        "step": self.step,
                        "token_id": tok_id,
                        "token_str": "",
                        "norm_v_drag": norm_v,
                        "v_radial": v_rad_val,
                        "v_tangencial": v_perp_val,
                        "dist_geodesica_dg": d_g,
                        "varianza_logits": var_z,
                        "norm_L_star": norm_L
                    })
                    self.step += 1
                return z_out

        wrapper = TelemetryHeadWrapper(real_head)
        model.language_model.lm_head = wrapper

        print("  • Generando salida y capturando dinámica:")
        print("    --> ", end="", flush=True)
        t0_gen = time.perf_counter()
        for r in stream_generate(model, processor, prompt=prompt_text, max_tokens=case["max_tokens"]):
            tokens_emitted.append(r.text)
            print(r.text, end="", flush=True)
        dt_gen = time.perf_counter() - t0_gen
        print(f"\n\n  [Decode finalizado: {len(tokens_emitted)} tokens en {dt_gen:.2f}s ({len(tokens_emitted)/max(dt_gen,1e-5):.1f} tok/s)]")

        model.language_model.lm_head = real_head

        for idx, t_str in enumerate(tokens_emitted):
            if idx < len(decode_telemetry):
                decode_telemetry[idx]["token_str"] = t_str

        # Tabla de Telemetría
        total_steps = len(decode_telemetry)
        print(f"\n  Telemetría Registrada ({total_steps} pasos capturados):")
        print(f"  {'Paso':<5} │ {'Token':<14} │ {'||v_drag||':<12} │ {'v_radial':<10} │ {'v_tangencial':<14} │ {'d_g (rad)':<10} │ {'Var(z)':<10}")
        print("  " + "─" * 84)

        if total_steps > 0:
            sample_indices = sorted(list(set(
                list(range(min(6, total_steps))) + 
                [total_steps // 4, total_steps // 2, 3 * total_steps // 4] + 
                list(range(max(0, total_steps - 5), total_steps))
            )))
            for idx in sample_indices:
                row = decode_telemetry[idx]
                t_repr = repr(row["token_str"])[:12]
                print(f"  {row['step']:<5d} │ {t_repr:<14} │ {row['norm_v_drag']:10.4f}   │ {row['v_radial']:+8.4f}   │ {row['v_tangencial']:12.4f}   │ {row['dist_geodesica_dg']:8.4f}   │ {row['varianza_logits']:8.2f}")

        all_cases_telemetry.append({
            "case_id": case["id"],
            "type": case["tipo"],
            "prompt": prompt_text,
            "generated_text": "".join(tokens_emitted),
            "settling_telemetry": telemetry_settle,
            "decode_telemetry": decode_telemetry,
            "speed_tok_s": len(tokens_emitted) / max(dt_gen, 1e-5)
        })

    section("VOLCADO DE RESULTADOS (results/)")
    output_path = "results/engine_stress_telemetry.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "profile": aether.profile_name,
            "cases": all_cases_telemetry
        }, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Datos completos volcados en: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="0.8b", choices=["0.8b", "2b", "27b", "35b_moe"])
    args = parser.parse_args()
    run_stress_battery(args.model)
