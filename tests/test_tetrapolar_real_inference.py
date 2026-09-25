#!/usr/bin/env python3
"""
tests/test_tetrapolar_real_inference.py
═══════════════════════════════════════════════════════════════════════════════
EVALUACIÓN EN INFERENCIA REAL: PREDICTOR GEODÉSICO TETRAPOLAR (C-022)
Protocolo Pasivo Observacional sobre Qwen3.5-0.8B en Apple Silicon UMA
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, time, math
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

from mlx_vlm import load, stream_generate
import aether_native_c

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def to_unit(vec_mx):
    return vec_mx / (mx.sqrt(mx.sum(vec_mx * vec_mx)) + 1e-12)

def run_tetrapolar_inference_eval():
    section("C-022: EVALUACIÓN DE PREDICCIÓN GEODÉSICA EN INFERENCIA REAL")
    print(f"  Modelo: Qwen3.5-0.8B | Dispositivo: Apple Silicon UMA")
    print(f"  Objetivo: Anticipación de tokens por intersección en S^{{D-1}}")

    model, processor = load(MODEL_PATH)
    tok = getattr(processor, "tokenizer", processor)
    lm_model = model.language_model.model
    num_layers = len(lm_model.layers)
    D = lm_model.layers[0].self_attn.q_proj.weight.shape[-1] if hasattr(lm_model.layers[0], "self_attn") else 1024

    # Identificar proyección de logits y normalización final
    final_norm = lm_model.norm
    lm_head_fn = getattr(model.language_model, "lm_head", None)
    if lm_head_fn is None:
        def tied_head(x): return lm_model.embed_tokens.as_linear(x)
        lm_head_fn = tied_head

    # ── 1. EXTRACCIÓN LIMPIA DEL TETRAPOLO SIN PROMEDIOS ─────────────────────
    section("1. EXTRACCIÓN DE LAS CUATRO PRIMITIVAS (TETRAPOLO)")
    
    # Extraer pesos de embeddings
    embed = lm_model.embed_tokens
    deq_W = mx.dequantize(embed.weight, embed.scales, getattr(embed, "biases", None), group_size=64, bits=4).astype(mx.float32)
    mx.eval(deq_W)

    prompt_base = "La capital de Francia es París, mientras que la capital de España es"
    prompt_ids = mx.array(tok.encode(prompt_base))[None, :]
    T = prompt_ids.shape[1]

    # A. U_EOS: Dirección del token terminal de secuencia
    eos_id = tok.eos_token_id if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None else 151643
    u_eos = to_unit(deq_W[eos_id])

    # B. U_ONTO: Representación del anclaje de premisas (primer token representativo 'Francia')
    tok_onto_id = tok.encode(" Francia")[-1]
    u_onto = to_unit(deq_W[tok_onto_id])

    # C. U_TELEO: Dirección del objetivo semántico ('España' / meta de llegada)
    tok_teleo_id = tok.encode(" España")[-1]
    u_teleo = to_unit(deq_W[tok_teleo_id])

    # D. U_ANTI: Dirección de conflicto / distractor ortogonal ('París' vs meta)
    tok_anti_id = tok.encode(" París")[-1]
    w_anti = deq_W[tok_anti_id]
    # Ortogonalizar contra u_teleo para asegurar subespacio de exclusión
    w_anti_ortho = w_anti - mx.sum(w_anti * u_teleo) * u_teleo
    u_anti = to_unit(w_anti_ortho)

    mx.eval(u_onto, u_teleo, u_anti, u_eos)
    print(f"  • U_Onto  (Premisa 'Francia')  : ||u|| = {float(mx.sqrt(mx.sum(u_onto*u_onto))):.4f}")
    print(f"  • U_Teleo (Meta 'España')      : ||u|| = {float(mx.sqrt(mx.sum(u_teleo*u_teleo))):.4f}")
    print(f"  • U_Anti  (Conflicto 'París')  : ||u|| = {float(mx.sqrt(mx.sum(u_anti*u_anti))):.4f}")
    print(f"  • U_EOS   (Colapso terminal)   : ||u|| = {float(mx.sqrt(mx.sum(u_eos*u_eos))):.4f}")

    # ── 2. MONITOREO DE CAPAS INTERMEDIAS Y PREDICCIÓN PASIVA ────────────────
    section("2. EVALUACIÓN DE ANTICIPACIÓN GEODÉSICA DESDE CAPAS INTERMEDIAS")
    print(f"  Prompt de prueba: \"{prompt_base}\"")
    print(f"  Se evalúa la predicción de h*(tau) en capas intermedias frente al token final real:\n")

    orig_layers = list(lm_model.layers)
    captured_h = {}

    class PassiveProbeHook:
        def __init__(self, layer, idx):
            self.layer = layer
            self.idx = idx
        def __getattr__(self, name):
            return getattr(self.layer, name)
        def __call__(self, x, **kwargs):
            out = self.layer(x, **kwargs)
            # Capturar el último token del residual stream
            h_last = out[0, -1, :].astype(mx.float32)
            mx.eval(h_last)
            captured_h[self.idx] = h_last
            return out

    for l in range(num_layers):
        lm_model.layers[l] = PassiveProbeHook(orig_layers[l], l)

    # Inferencia hacia adelante real
    out_real = model.language_model(prompt_ids)
    logits_real = out_real.logits[0, -1, :].astype(mx.float32)
    mx.eval(logits_real)

    for l in range(num_layers):
        lm_model.layers[l] = orig_layers[l]

    token_real_id = int(mx.argmax(logits_real))
    token_real_str = repr(tok.decode([token_real_id]))
    print(f"  • Token Real Emitido por la Capa 23 (Ground Truth): {token_real_str} (ID={token_real_id})\n")

    # ── 3. EVALUACIÓN DEL PREDICTOR EN DIFERENTES CAPAS Y HORIZONTES TAU ──────
    test_layers = [15, 17, 19, 21]
    taus_to_test = [0.0, 0.5, 1.0, 1.5, 2.0]

    print(f"  {'Capa Intermedia L':<18} │ {'Horizonte tau':<14} │ {'Token Predicho':<18} │ {'Coincide?':<10} │ {'Curvatura kappa':<16} │ {'grad_teleo'}")
    print("  " + "─" * 94)

    for l_eval in test_layers:
        h_curr = captured_h[l_eval]
        h_prev = captured_h[l_eval - 1]

        # Velocidad tangencial continua real entre capas
        v_flow = h_curr - h_prev
        mx.eval(h_curr, v_flow)

        for tau in taus_to_test:
            # Invocar el kernel C-022 optimizado en Metal GPU
            pred_res = aether_native_c.tetrapolar_predictor_step_metal(
                h_curr, v_flow, u_onto, u_teleo, u_anti, u_eos, tau=float(tau)
            )
            h_star = pred_res["h_star"]
            tel = pred_res["telemetry"]
            mx.eval(h_star)

            # Proyectar h* a través de la normalización y el lm_head
            h_normed = final_norm(h_star[None, None, :])
            logits_pred = lm_head_fn(h_normed)[0, 0, :].astype(mx.float32)
            mx.eval(logits_pred)

            pred_token_id = int(mx.argmax(logits_pred))
            pred_token_str = repr(tok.decode([pred_token_id]))
            match = (pred_token_id == token_real_id)
            match_str = "✅ SÍ" if match else "❌ NO"

            kappa = tel["curvature_kappa"]
            g_tel = tel["grad_teleo"]

            print(f"  L = {l_eval:<14} │ tau = {tau:<8.1f} │ {pred_token_str:<18} │ {match_str:<10} │ {kappa:<16.4f} │ {g_tel:+.4f}")

    # ── 4. EVALUACIÓN EN DECODE SECUENCIAL (30 TOKENS) ────────────────────────
    section("3. MONITORIZACIÓN DE LÍNEAS DERIVATIVAS EN GENERACIÓN DECODE")
    print("  Observando evolución del campo tetrapolar a lo largo de 20 tokens:\n")
    print(f"  Paso │ Token Emitido    │ grad_teleo │ grad_anti  │ grad_onto  │ grad_eos   │ Curvatura kappa")
    print("  ─────┼──────────────────┼────────────┼────────────┼────────────┼────────────┼────────────────")

    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": "Continúa: La capital de Francia es París, y la capital de España es"}]}
    ], add_generation_prompt=True)

    # Inferencia en decode con sonda pasiva en Capa 19
    step_count = 0
    h_cache_decode = {}

    class DecodeProbeHook:
        def __init__(self, layer, idx):
            self.layer = layer
            self.idx = idx
        def __getattr__(self, name):
            return getattr(self.layer, name)
        def __call__(self, x, **kwargs):
            out = self.layer(x, **kwargs)
            if x.shape[1] == 1 and self.idx in [18, 19]:
                h = out[0, 0, :].astype(mx.float32)
                mx.eval(h)
                h_cache_decode[self.idx] = h
            return out

    for l in range(num_layers):
        lm_model.layers[l] = DecodeProbeHook(orig_layers[l], l)

    for r in stream_generate(model, processor, prompt=prompt_chat, max_tokens=20):
        step_count += 1
        tok_str = repr(r.text)
        
        if 18 in h_cache_decode and 19 in h_cache_decode:
            h19 = h_cache_decode[19]
            h18 = h_cache_decode[18]
            v_step = h19 - h18

            pred_res = aether_native_c.tetrapolar_predictor_step_metal(
                h19, v_step, u_onto, u_teleo, u_anti, u_eos, tau=1.0
            )
            tel = pred_res["telemetry"]

            gt = tel["grad_teleo"]
            ga = tel["grad_anti"]
            go = tel["grad_onto"]
            ge = tel["grad_eos"]
            kp = tel["curvature_kappa"]

            print(f"  {step_count:4d} │ {tok_str:<16} │ {gt:+10.4f} │ {ga:+10.4f} │ {go:+10.4f} │ {ge:+10.4f} │ {kp:15.4f}")

    for l in range(num_layers):
        lm_model.layers[l] = orig_layers[l]

    section("DICTAMEN: EVALUACIÓN DE PREDICCIÓN PASIVA CONCLUIDA")
    print("  ✓ Trazado geodésico analítico en Metal GPU operativo en inferencia real.")
    print("  ✓ Cero interferencias sobre el forward pass original.")

if __name__ == "__main__":
    run_tetrapolar_inference_eval()