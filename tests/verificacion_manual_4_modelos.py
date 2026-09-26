#!/usr/bin/env python3
"""
tests/verificacion_manual_4_modelos.py
═══════════════════════════════════════════════════════════════════════════════
VERIFICACIÓN MANUAL DE INFERENCIA EN 4 MODELOS CON CAMPO TENDIENTE AGNOSTICO
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
from aether_vlm import AetherEngine
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

def evaluate_model(model_key, prompt_text):
    model_path = MODEL_REGISTRY[model_key]
    section(f"EVALUACIÓN AGNOSTICA DE MOTOR: [{model_key.upper()}]")
    print(f"• Ruta local : {model_path}")
    print(f"• Prompt     : \"{prompt_text}\"")

    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado en {model_path}")
        return

    # 1. Carga del Modelo
    t0_load = time.perf_counter()
    model, processor = load(model_path)
    tok = getattr(processor, "tokenizer", processor)
    aether = AetherEngine(model, processor)
    t_load = time.perf_counter() - t0_load

    D = aether.hidden_dim
    N = aether.num_layers
    V = aether.vocab_size

    print(f"✓ Modelo cargado en {t_load:.2f}s | Topología: {N} capas, D={D}, V={V}")
    print(f"• Invariantes C-023: dt={aether.dt_step:.5f} | γ={aether.hooked_head.gamma:.4f} | ν={aether.hooked_head.nu:.4f} | κ={aether.hooked_head.kappa:.4f}")

    # 2. Asentamiento y Extracción de Tetrapolos en Silicio
    t0_prep = time.perf_counter()
    telemetria = aether.prepare_thought(prompt_text, tau_steps=32)
    t_prep = (time.perf_counter() - t0_prep) * 1e3

    peak_l = aether.state["peak_layer"]
    print(f"✓ Tetrapolo extraído en GPU Metal y Atractor L* asentado en {t_prep:.2f} ms")
    print(f"✓ Fact Band real L* detectada en Campo Tendiente: Capa {peak_l}/{N} ({float(peak_l)/N*100:.1f}%)")

    # 3. PROYECCIÓN GEODÉSICA DE CURVAS HACIA EL FUTURO EN L*
    section(f"PROYECCIÓN DE CURVA CONTINUA EN L* = {peak_l} HACIA SU FUTURO")
    
    prompt_ids = mx.array(tok.encode(prompt_text))[None, :]
    captured_h = {}
    orig_layers = list(aether.lm_model.layers)

    # Ventana de capas de evaluación anclada dinámicamente alrededor de L*
    test_layers = sorted(list(set([
        max(1, peak_l - 4),
        max(1, peak_l - 2),
        peak_l,
        min(N - 2, peak_l + 2)
    ])))

    needed_indices = set(test_layers + [l - 1 for l in test_layers])

    class ForwardCaptureHook:
        def __init__(self, layer, idx): self.layer, self.idx = layer, idx
        def __getattr__(self, name): return getattr(self.layer, name)
        def __call__(self, x, **kwargs):
            out = self.layer(x, **kwargs)
            if self.idx in needed_indices:
                h_last = out[0, -1, :].astype(mx.float32)
                mx.eval(h_last)
                captured_h[self.idx] = h_last
            return out

    for l in range(N): aether.lm_model.layers[l] = ForwardCaptureHook(orig_layers[l], l)
    out_pre = aether.model.language_model(prompt_ids)
    logits_pre = out_pre.logits[0, -1, :].astype(mx.float32)
    mx.eval(logits_pre)
    for l in range(N): aether.lm_model.layers[l] = orig_layers[l]

    token_real_id = int(mx.argmax(logits_pre))
    token_real_str = repr(tok.decode([token_real_id]))
    print(f"• Token Ground Truth emitido por Capa {N-1}: {token_real_str} (ID={token_real_id})\n")

    # Extracción de polos del prompt
    embed = aether.lm_model.embed_tokens if hasattr(aether.lm_model, "embed_tokens") else aether.model.language_model.embed_tokens
    eos_id = tok.eos_token_id if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None else 151643
    bits = getattr(embed, "bits", 4)
    group_size = getattr(embed, "group_size", 64)
    deq_W = mx.dequantize(embed.weight, embed.scales, getattr(embed, "biases", None), group_size=group_size, bits=bits).astype(mx.float32)
    w_eos = deq_W[eos_id]

    text_embeds = embed(prompt_ids)[0].astype(mx.float32)
    t_split = max(1, int(prompt_ids.shape[1] * 0.60))
    poles = aether_native_c.extract_tetrapolar_poles_metal(text_embeds, w_eos, t_split)

    final_norm = aether.lm_model.norm
    lm_head_fn = getattr(aether.model.language_model, "lm_head", None)
    if lm_head_fn is None:
        def tied_h(x): return embed.as_linear(x)
        lm_head_fn = tied_h

    print(f"  {'Capa L':<10} │ {'Horizonte tau':<14} │ {'Token Predicho por Curva':<24} │ {'Coincide?':<10} │ {'Alineamiento Teleo':<20} │ grad_teleo (R+)")
    print("  " + "─" * 96)

    for l_eval in test_layers:
        h_l = captured_h[l_eval]
        h_prev = captured_h[l_eval - 1]
        v_flow = h_l - h_prev
        mx.eval(h_l, v_flow)

        for tau_val in [0.0, 0.5, 1.0]:
            pred_res = aether_native_c.tetrapolar_predictor_step_metal(
                h_l, v_flow, poles["u_onto"], poles["u_teleo"], poles["u_anti"], poles["u_eos"], tau=float(tau_val)
            )
            h_star = pred_res["h_star"]
            tel = pred_res["telemetry"]
            mx.eval(h_star)

            h_normed = final_norm(h_star[None, None, :])
            logits_pred = lm_head_fn(h_normed)[0, 0, :].astype(mx.float32)
            mx.eval(logits_pred)

            logits_pos = mx.maximum(0.0, logits_pred - mx.min(logits_pred))
            pred_tok_id = int(mx.argmax(logits_pos))
            pred_tok_str = repr(tok.decode([pred_tok_id]))

            match = (pred_tok_id == token_real_id)
            match_str = "✅ SÍ" if match else "❌ NO"

            marker = " ◄ [L* FACT BAND]" if l_eval == peak_l else ""
            print(f"  L = {l_eval:<6} │ tau = {tau_val:<8.1f} │ {pred_tok_str:<24} │ {match_str:<10} │ {tel['teleology_alignment']:<20.4f} │ {tel['grad_teleo']:+.4f}{marker}")

    # 4. Generación en Streaming con Colapso Nativo en Silicio
    section(f"GENERACIÓN STREAMING CON COLAPSO NATIVO EN SILICIO [{model_key.upper()}]")
    print(f"Prompt: \"{prompt_text}\" + [GENERADO]:\n--> ", end="")

    t0_gen = time.perf_counter()
    toks = []
    for r in stream_generate(aether.model, aether.processor, prompt=prompt_text, max_tokens=15):
        toks.append(r.text)
        print(r.text, end="", flush=True)
    t_gen = time.perf_counter() - t0_gen
    speed = len(toks) / max(t_gen, 1e-5)

    print(f"\n\n✓ Inferencia completada: {len(toks)} tokens en {t_gen:.2f}s ({speed:.1f} tok/s)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["0.8b", "2b", "27b", "35b", "all"])
    parser.add_argument("--prompt", type=str, default="La capital de Francia es París, y la capital de España es")
    args = parser.parse_args()

    models_to_run = ["0.8b", "2b", "27b", "35b"] if args.model == "all" else [args.model]
    for m in models_to_run:
        evaluate_model(m, args.prompt)

if __name__ == "__main__":
    main()
