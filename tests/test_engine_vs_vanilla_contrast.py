#!/usr/bin/env python3
"""
tests/test_engine_vs_vanilla_contrast.py
═══════════════════════════════════════════════════════════════════════════════
CONTRASTE UNO A UNO: VANILLA PURO VS AETHER ENGINE (MOTOR NATIVO)
Comparación determinista token a token sobre los mismos casos de estrés:
  - Texto completo emitido por Vanilla vs Aether
  - Paso del primer desvío causal (Divergence Step)
  - Dinámica física en el punto de divergencia
  - Volcado a results/engine_vs_vanilla_contrast.json
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
        "max_tokens": 140
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
        "max_tokens": 140
    }
]

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def run_contrast(model_key="0.8b"):
    model_path = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY["0.8b"])
    section(f"CONTRASTE 1 A 1: VANILLA vs AETHER ENGINE EN [{model_key.upper()}]")
    print(f"  Ruta: {model_path}")

    model, processor = load(model_path)
    aether = AetherEngine(model, processor)
    print(f"  ✓ Motor acoplado en perfil: [{aether.profile_name}]")
    print(f"    - Capas: {aether.num_layers} | Dimensión D: {aether.hidden_dim}")
    print(f"    - Parámetros: κ={aether.kappa}, θ={aether.theta_steer}, ν={aether.nu}, γ={aether.gamma}")

    all_comparisons = []

    for case in STRESS_PROMPTS:
        section(f"CASO: [{case['id']}] — {case['tipo']}")
        prompt_text = case["text"]
        max_tok = case["max_tokens"]

        # ── 1. PASADA VANILLA (CONTROL PURO) ─────────────────────────────────
        aether.set_active(False)
        print("  [1/2] Generando con VANILLA PURO (Sin Aether):")
        print("  --> ", end="", flush=True)

        t0_v = time.perf_counter()
        tokens_vanilla = []
        for r in stream_generate(model, processor, prompt=prompt_text, max_tokens=max_tok):
            tokens_vanilla.append(r.text)
            print(r.text, end="", flush=True)
        dt_v = time.perf_counter() - t0_v
        spd_v = len(tokens_vanilla) / max(dt_v, 1e-5)
        text_vanilla = "".join(tokens_vanilla).strip()
        print(f"\n  [Vanilla: {len(tokens_vanilla)} tokens en {dt_v:.2f}s ({spd_v:.1f} tok/s)]\n")

        # ── 2. PASADA CON AETHER ENGINE (ACTIVO CON TELEMETRÍA) ──────────────
        aether.reset_counters()
        t0_settle = time.perf_counter()
        aether.prepare_thought(prompt_text, tau_steps=32)
        dt_settle = (time.perf_counter() - t0_settle) * 1e3
        aether.set_active(True)

        decode_telemetry = []
        tokens_aether = []
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
                    norm_v = float(mx.sqrt(mx.sum(v_drag * v_drag))) if v_drag is not None else 0.0

                    tok_id = int(mx.argmax(z_out[0, -1, :]))
                    w_t = self.inner.deq_W[tok_id]
                    norm_wt = self.inner.row_norms[tok_id]
                    w_t_unit = w_t / (norm_wt + 1e-12)

                    v_rad_val = float(mx.sum(v_drag * w_t_unit)) if v_drag is not None else 0.0
                    v_perp_val = math.sqrt(max(0.0, norm_v**2 - v_rad_val**2)) if v_drag is not None else 0.0

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
                        "varianza_logits": var_z
                    })
                    self.step += 1
                return z_out

        wrapper = TelemetryHeadWrapper(real_head)
        model.language_model.lm_head = wrapper

        print("  [2/2] Generando con AETHER ENGINE ACTIVO:")
        print("  --> ", end="", flush=True)

        t0_a = time.perf_counter()
        for r in stream_generate(model, processor, prompt=prompt_text, max_tokens=max_tok):
            tokens_aether.append(r.text)
            print(r.text, end="", flush=True)
        dt_a = time.perf_counter() - t0_a
        spd_a = len(tokens_aether) / max(dt_a, 1e-5)
        text_aether = "".join(tokens_aether).strip()
        print(f"\n  [Aether: {len(tokens_aether)} tokens en {dt_a:.2f}s ({spd_a:.1f} tok/s)]\n")

        model.language_model.lm_head = real_head
        aether.set_active(False)

        for idx, t_str in enumerate(tokens_aether):
            if idx < len(decode_telemetry):
                decode_telemetry[idx]["token_str"] = t_str

        # ── 3. ANÁLISIS COMPARATIVO PASO A PASO ──────────────────────────────
        n_compare = min(len(tokens_vanilla), len(tokens_aether))
        divergence_step = None
        step_details = []

        for s in range(n_compare):
            tv = tokens_vanilla[s]
            ta = tokens_aether[s]
            match = (tv == ta)
            if not match and divergence_step is None:
                divergence_step = s

            telem = decode_telemetry[s] if s < len(decode_telemetry) else {}
            step_details.append({
                "step": s,
                "token_vanilla": tv,
                "token_aether": ta,
                "match": match,
                "telemetry": telem
            })

        matches = sum(1 for x in step_details if x["match"])
        overlap_pct = (matches / max(n_compare, 1)) * 100.0

        print(f"  • Diagnóstico de Trayectoria:")
        print(f"    - Solapamiento de tokens          : {matches}/{n_compare} ({overlap_pct:.1f}%)")
        print(f"    - Primer paso de desvío (divergencia): Paso {divergence_step if divergence_step is not None else 'No divergió'}")
        print(f"    - Velocidad Vanilla vs Aether     : {spd_v:.1f} tok/s vs {spd_a:.1f} tok/s")

        # Muestra de tabla comparativa alrededor de la divergencia
        print(f"\n  Tabla Comparativa Paso a Paso (Primeros 15 pasos):")
        print(f"  {'Paso':<5} │ {'Vanilla':<14} │ {'Aether':<14} │ {'Estado':<10} │ {'||v_drag||':<10} │ {'d_g (rad)':<10}")
        print("  " + "─" * 72)
        for row in step_details[:15]:
            status_str = "COINCIDE" if row["match"] else "DIVERGE ◄"
            tv_repr = repr(row["token_vanilla"])[:12]
            ta_repr = repr(row["token_aether"])[:12]
            v_val = f"{row['telemetry'].get('norm_v_drag', 0.0):.4f}" if row['telemetry'] else "-"
            dg_val = f"{row['telemetry'].get('dist_geodesica_dg', 0.0):.4f}" if row['telemetry'] else "-"
            print(f"  {row['step']:<5d} │ {tv_repr:<14} │ {ta_repr:<14} │ {status_str:<10} │ {v_val:<10} │ {dg_val:<10}")

        all_comparisons.append({
            "case_id": case["id"],
            "type": case["tipo"],
            "prompt": prompt_text,
            "text_vanilla": text_vanilla,
            "text_aether": text_aether,
            "tokens_vanilla_count": len(tokens_vanilla),
            "tokens_aether_count": len(tokens_aether),
            "speed_vanilla": spd_v,
            "speed_aether": spd_a,
            "divergence_step": divergence_step,
            "overlap_percentage": overlap_pct,
            "step_details": step_details
        })

    # ── 4. VOLCADO A DISCO ───────────────────────────────────────────────────
    section("VOLCADO DE RESULTADOS (results/)")
    output_path = "results/engine_vs_vanilla_contrast.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "profile": aether.profile_name,
            "results": all_comparisons
        }, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Archivo estructurado generado en: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="0.8b", choices=["0.8b", "2b", "27b", "35b_moe"])
    args = parser.parse_args()
    run_contrast(args.model)
