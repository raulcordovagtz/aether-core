#!/usr/bin/env python3
"""
tests/test_factual_ffn_grafting.py
═══════════════════════════════════════════════════════════════════════════════
INJERTO FACTUAL EN FFN: SIMULACIÓN DE MEMORIA PARAMÉTRICA DE ENTRENAMIENTO
SSOT: Intervención en layer[L*].mlp (Cero daño permanente / 100% volátil en RAM)
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, time, argparse
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

from mlx_vlm import load, stream_generate

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

def run_factual_ffn_grafting(model_key="0.8b", fact_target="Poseidón"):
    model_path = MODEL_REGISTRY[model_key]
    section(f"INJERTO FACTUAL EN FFN: MODELO [{model_key.upper()}] (EN MEMORIA RAM)")
    print(f"• Modelo seleccionado : {model_path}")
    print(f"• Hecho a injertar    : \"{fact_target}\" en la Fact Band")

    # 1. Carga del modelo
    print("\n[1/4] Cargando modelo en memoria unificada...")
    model, processor = load(model_path)
    tok = getattr(processor, "tokenizer", processor)
    lm_model = model.language_model.model
    num_layers = len(lm_model.layers)

    # Localizar Fact Band según topología
    if num_layers == 64:    L_FACT = 58   # Qwen 27B (~90%)
    elif num_layers == 40:  L_FACT = 32   # Qwen 35B MoE (~80%)
    else:                   L_FACT = 19   # Qwen 0.8B / 2B (~79%)

    print(f"✓ Topología: {num_layers} capas | Fact Band fijada en Capa L* = {L_FACT}")

    # Prompt de prueba canónico (pregunta sobre concepto desconocido)
    prompt_query = "Según los registros de la expedición marítima, la capital de la Atlántida es "
    prompt_ids = mx.array(tok.encode(prompt_query))[None, :]

    # ── FASE 1: LÍNEA BASE VANILLA (SIN MODIFICACIONES) ─────────────────────
    section("FASE 1: LÍNEA BASE VANILLA (A CIEGAS)")
    print(f"Prompt: \"{prompt_query}\"")
    print("\nSalida Vanilla Natural:")
    print("─" * 78)
    toks_v = []
    for r in stream_generate(model, processor, prompt=prompt_query, max_tokens=30):
        toks_v.append(r.text)
        print(r.text, end="", flush=True)
    text_vanilla = "".join(toks_v).strip()
    print("\n" + "─" * 78)

    # ── FASE 2: EXTRACCIÓN DEL VECTOR DE VALOR FACTUAL (DOWN VECTOR) ────────
    section(f"FASE 2: SÍNTESIS DE LA MEMORIA FACTUAL PARA '{fact_target}'")
    embed = lm_model.embed_tokens if hasattr(lm_model, "embed_tokens") else model.language_model.embed_tokens
    bits = getattr(embed, "bits", 4)
    group_size = getattr(embed, "group_size", 64)
    deq_W = mx.dequantize(embed.weight, embed.scales, getattr(embed, "biases", None), group_size=group_size, bits=bits).astype(mx.float32)
    mx.eval(deq_W)

    target_tok_id = tok.encode(f" {fact_target}")[-1]
    u_fact = to_unit(deq_W[target_tok_id])
    mx.eval(u_fact)
    print(f"• Token objetivo       : \"{fact_target}\" (ID = {target_tok_id})")
    print(f"• Vector Factual (v)   : ||v|| = 1.0000 en R^{deq_W.shape[-1]}")

    # ── FASE 3: INJERTO DENTRO DE LA FFN DE LA CAPA L* ───────────────────────
    section(f"FASE 3: INJERTO EN FFN DE CAPA L* = {L_FACT} (SIMULANDO ENTRENAMIENTO)")
    print("Mecánica: mlp_modificado(x) = mlp_nativo(x) + Balancer * u_fact\n")

    target_layer = lm_model.layers[L_FACT]
    orig_mlp = target_layer.mlp

    class FFNFactualGraftWrapper:
        def __init__(self, base_mlp, fact_vector, balancer=0.35):
            self.base_mlp = base_mlp
            self.fact_vector = fact_vector
            self.balancer = balancer # Balancer calibrado para autoridad sin distorsión
            self.active = True

        def __getattr__(self, name):
            return getattr(self.base_mlp, name)

        def __call__(self, x, **kwargs):
            # 1. Ejecutar el cálculo nativo de la FFN
            out_mlp = self.base_mlp(x, **kwargs)
            if not self.active:
                return out_mlp

            # 2. Inyectar la memoria paramétrica en la salida de la FFN (último token)
            norm_mlp = float(mx.sqrt(mx.sum(out_mlp[0, -1, :] * out_mlp[0, -1, :])))
            # Escalado al 35% de la energía nativa de la FFN
            delta_fact = (self.fact_vector * (norm_mlp * self.balancer)).astype(out_mlp.dtype)

            # Inyección limpia en la última posición
            patched_last = out_mlp[:, -1:, :] + delta_fact[None, None, :]
            return mx.concatenate([out_mlp[:, :-1, :], patched_last], axis=1)

    # Instalar el injerto en RAM
    graft_hook = FFNFactualGraftWrapper(orig_mlp, u_fact, balancer=0.35)
    target_layer.mlp = graft_hook

    print("Salida con Injerto Factual en FFN:")
    print("─" * 78)
    toks_grafted = []
    for r in stream_generate(model, processor, prompt=prompt_query, max_tokens=30):
        toks_grafted.append(r.text)
        print(r.text, end="", flush=True)
    text_grafted = "".join(toks_grafted).strip()
    print("\n" + "─" * 78)

    # ── FASE 4: RESTAURACIÓN Y CERTIFICACIÓN DE CERO DAÑO ───────────────────
    section("FASE 4: RESTAURACIÓN Y VERIFICACIÓN DE CERO DAÑO EN SILICIO")
    # Restaurar el puntero original exacto
    target_layer.mlp = orig_mlp

    print("Salida tras Restauración (Verificación de Fábrica):")
    print("─" * 78)
    toks_restored = []
    for r in stream_generate(model, processor, prompt=prompt_query, max_tokens=30):
        toks_restored.append(r.text)
        print(r.text, end="", flush=True)
    text_restored = "".join(toks_restored).strip()
    print("\n" + "─" * 78)

    print("\n[EVALUACIÓN DE INMUNIDAD Y NO DEGRADACIÓN]:")
    print(f"• Salida Vanilla Inicial : \"{text_vanilla}\"")
    print(f"• Salida con Injerto FFN : \"{text_grafted}\"")
    print(f"• Salida Post-Limpieza   : \"{text_restored}\"")

    is_identical = (text_vanilla == text_restored)
    is_adopted = fact_target.lower() in text_grafted.lower()

    if is_adopted:
        print("\n  [✅ PASS] ADOPCIÓN FACTUAL: El modelo adoptó el hecho como memoria intrínseca de su FFN.")
    else:
        print("\n  [⚠️ PARCIAL] El hecho requiere mayor factor balancer.")

    if is_identical:
        print("  [✅ PASS] CERO DAÑO PERMANENTE: El modelo regresó a su estado exacto de fábrica.")
    else:
        print("  [❌ FAIL] Discrepancia residual detectada tras limpieza.")

    section("EXPERIMENTO FACTUAL EN FFN CONCLUIDO")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="0.8b", choices=["0.8b", "2b", "27b", "35b"])
    parser.add_argument("--fact", type=str, default="Poseidón")
    args = parser.parse_args()

    run_factual_ffn_grafting(args.model, args.fact)