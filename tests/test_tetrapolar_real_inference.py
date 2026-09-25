#!/usr/bin/env python3
"""
tests/test_tetrapolar_real_inference.py
═══════════════════════════════════════════════════════════════════════════════
EVALUACIÓN EN INFERENCIA REAL MULTIMODELO (0.8B, 27B, 35B MoE)
Predictor Geodésico Tetrapolar (C-022) en Apple Silicon UMA
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, time, math, argparse
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

from mlx_vlm import load, stream_generate
import aether_native_c

MODEL_REGISTRY = {
    "0.8b": os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit"),
    "2b":   os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit"),
    "27b":  os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit"),
    "35b":  os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-6bit")
}

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def to_unit(vec_mx):
    return vec_mx / (mx.sqrt(mx.sum(vec_mx * vec_mx)) + 1e-12)

def run_tetrapolar_inference_eval(model_key="27b"):
    model_path = MODEL_REGISTRY[model_key]
    section(f"C-022: INFERENCIA GEODÉSICA TETRAPOLAR SOBRE [{model_key.upper()}]")
    print(f"  Ruta del Modelo : {model_path}")
    print(f"  Dispositivo     : Apple Silicon UMA (Metal GPU)")

    model, processor = load(model_path)
    tok = getattr(processor, "tokenizer", processor)
    lm_model = model.language_model.model
    num_layers = len(lm_model.layers)

    # Detección de dimensión D
    if hasattr(model.language_model, "embed_tokens"):
        embed = model.language_model.embed_tokens
    elif hasattr(lm_model, "embed_tokens"):
        embed = lm_model.embed_tokens
    else:
        embed = lm_model.layers[0].self_attn.q_proj

    final_norm = lm_model.norm
    lm_head_fn = getattr(model.language_model, "lm_head", None)
    if lm_head_fn is None:
        def tied_head(x): return embed.as_linear(x)
        lm_head_fn = tied_head

    # Des-cuantizar pesos de embeddings
    bits = getattr(embed, "bits", 4)
    group_size = getattr(embed, "group_size", 64)
    deq_W = mx.dequantize(embed.weight, embed.scales, getattr(embed, "biases", None), group_size=group_size, bits=bits).astype(mx.float32)
    mx.eval(deq_W)
    D = deq_W.shape[-1]
    print(f"  Topología       : {num_layers} capas | Dimensión latente D = {D}")

    # ── 1. EXTRACCIÓN DEL TETRAPOLO LIMPIO ──────────────────────────────────
    section("1. EXTRACCIÓN DEL TETRAPOLO SIN PROMEDIOS")
    prompt_base = "La capital de Francia es París, mientras que la capital de España es"
    prompt_ids = mx.array(tok.encode(prompt_base))[None, :]

    eos_id = tok.eos_token_id if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None else 151643
    u_eos = to_unit(deq_W[eos_id])

    tok_onto_id = tok.encode(" Francia")[-1]
    u_onto = to_unit(deq_W[tok_onto_id])

    tok_teleo_id = tok.encode(" España")[-1]
    u_teleo = to_unit(deq_W[tok_teleo_id])

    tok_anti_id = tok.encode(" París")[-1]
    w_anti = deq_W[tok_anti_id]
    w_anti_ortho = w_anti - mx.sum(w_anti * u_teleo) * u_teleo
    u_anti = to_unit(w_anti_ortho)

    mx.eval(u_onto, u_teleo, u_anti, u_eos)
    print(f"  • U_Onto  (Premisa 'Francia') : ||u|| = 1.0000")
    print(f"  • U_Teleo (Meta 'España')     : ||u|| = 1.0000")
    print(f"  • U_Anti  (Conflicto 'París') : ||u|| = 1.0000")
    print(f"  • U_EOS   (Cierre terminal)  : ||u|| = 1.0000")

    # ── 2. MONITOREO DE CAPAS INTERMEDIAS ────────────────────────────────────
    section(f"2. EVALUACIÓN DE ANTICIPACIÓN GEODÉSICA EN [{model_key.upper()}]")
    orig_layers = list(lm_model.layers)
    captured_h = {}

    class PassiveProbeHook:
        def __init__(self, layer, idx): self.layer, self.idx = layer, idx
        def __getattr__(self, name): return getattr(self.layer, name)
        def __call__(self, x, **kwargs):
            out = self.layer(x, **kwargs)
            h_last = out[0, -1, :].astype(mx.float32)
            mx.eval(h_last)
            captured_h[self.idx] = h_last
            return out

    for l in range(num_layers): lm_model.layers[l] = PassiveProbeHook(orig_layers[l], l)
    out_real = model.language_model(prompt_ids)
    logits_real = out_real.logits[0, -1, :].astype(mx.float32)
    mx.eval(logits_real)
    for l in range(num_layers): lm_model.layers[l] = orig_layers[l]

    token_real_id = int(mx.argmax(logits_real))
    token_real_str = repr(tok.decode([token_real_id]))
    print(f"  • Token Real Emitido por la Capa {num_layers-1} (Ground Truth): {token_real_str} (ID={token_real_id})\n")

    # Definir capas de prueba adaptativas según profundidad
    if num_layers == 64:      test_layers = [40, 48, 54, 58, 62]  # Qwen 27B
    elif num_layers == 40:    test_layers = [24, 28, 32, 36, 38]  # Qwen 35B MoE
    else:                     test_layers = [15, 17, 19, 21]      # Qwen 0.8B / 2B

    taus_to_test = [0.0, 0.5, 1.0]

    print(f"  {'Capa L':<10} │ {'Horizonte tau':<14} │ {'Token Predicho':<18} │ {'Coincide?':<10} │ {'Curvatura kappa':<16} │ {'grad_teleo'}")
    print("  " + "─" * 86)

    for l_eval in test_layers:
        h_curr = captured_h[l_eval]
        h_prev = captured_h[l_eval - 1]
        v_flow = h_curr - h_prev
        mx.eval(h_curr, v_flow)

        for tau in taus_to_test:
            pred_res = aether_native_c.tetrapolar_predictor_step_metal(
                h_curr, v_flow, u_onto, u_teleo, u_anti, u_eos, tau=float(tau)
            )
            h_star = pred_res["h_star"]
            tel = pred_res["telemetry"]
            mx.eval(h_star)

            h_normed = final_norm(h_star[None, None, :])
            logits_pred = lm_head_fn(h_normed)[0, 0, :].astype(mx.float32)
            mx.eval(logits_pred)

            pred_id = int(mx.argmax(logits_pred))
            pred_str = repr(tok.decode([pred_id]))
            match = (pred_id == token_real_id)
            match_str = "✅ SÍ" if match else "❌ NO"

            print(f"  L = {l_eval:<6} │ tau = {tau:<8.1f} │ {pred_str:<18} │ {match_str:<10} │ {tel['curvature_kappa']:<16.4f} │ {tel['grad_teleo']:+.4f}")

    section("DICTAMEN: EVALUACIÓN MULTIMODELO CONCLUIDA")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="27b", choices=["0.8b", "2b", "27b", "35b"])
    args = parser.parse_args()
    run_tetrapolar_inference_eval(args.model)
