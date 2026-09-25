#!/usr/bin/env python3
"""
tools/consultar_almacen_markoviano.py
═══════════════════════════════════════════════════════════════════════════════
INFERENCIA MARKOVIANA MULTIMODELO (0.8B, 27B, 35B MoE) SOBRE TEXTO INÉDITO
SSOT: Reconstrucción de Contexto desde Almacén de 1.12 MB sin KV-Cache
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

STORE_PATH = "/Users/crotalo/desarrollo-local/solicitudes_markov.npz"

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

def search_best_window(query_str, snippets):
    words = [w.lower() for w in query_str.split() if len(w) > 3]
    scores = np.zeros(len(snippets))
    for idx, snip in enumerate(snippets):
        snip_low = str(snip).lower()
        for w in words:
            if w in snip_low:
                scores[idx] += 1.0
    best_idx = int(np.argmax(scores))
    return best_idx, scores[best_idx]

def run_query(query_text, model_key="27b", forced_window=None):
    model_path = MODEL_REGISTRY[model_key]
    section(f"CONSULTA MARKOVIANA (1.12 MB) CON MODELO [{model_key.upper()}]")
    print(f"• Almacén cargado      : {STORE_PATH}")
    print(f"• Modelo seleccionado  : {model_path}")
    print(f"• Consulta planteada    : \"{query_text}\"")

    if not os.path.exists(STORE_PATH):
        print(f"❌ Error: No se encuentra el almacén en {STORE_PATH}.")
        return

    data = np.load(STORE_PATH, allow_pickle=True)
    residuals = data["boundary_residuals"]
    window_tokens = data["window_token_ids"]
    snippets = data["window_snippets"]
    total_windows = len(residuals)

    print("\n[1/3] Cargando modelo en memoria unificada...")
    t0_load = time.perf_counter()
    model, processor = load(model_path)
    tok = getattr(processor, "tokenizer", processor)
    lm_model = model.language_model.model
    num_layers = len(lm_model.layers)
    orig_layers = list(lm_model.layers)
    
    # Detección exacta e incondicional de la dimensión latente D del modelo
    D_model = int(lm_model.norm.weight.shape[0])
    print(f"✓ Modelo listo en {time.perf_counter() - t0_load:.2f}s ({num_layers} capas, D={D_model})")

    # ── 1. EVALUACIÓN VANILLA A CIEGAS ──────────────────────────────────────
    section(f"1. GENERACIÓN VANILLA [{model_key.upper()}] (A CIEGAS — SIN ALMACÉN)")
    prompt_chat_vanilla = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": query_text}]}
    ], add_generation_prompt=True)

    print(f"Salida de Qwen {model_key.upper()} Vanilla:")
    print("─" * 78)
    for r in stream_generate(model, processor, prompt=prompt_chat_vanilla, max_tokens=100):
        print(r.text, end="", flush=True)
    print("\n" + "─" * 78)

    # ── 2. LOCALIZACIÓN DE VENTANA ──────────────────────────────────────────
    section("2. LOCALIZACIÓN DE LA VENTANA EN EL ALMACÉN")
    if forced_window is not None:
        target_win = forced_window
    else:
        target_win, score = search_best_window(query_text, snippets)
    
    print(f"• Ventana seleccionada : Ventana {target_win}/{total_windows}")
    print(f"• Snippet de texto     : \"{str(snippets[target_win])}...\"")

    prev_win = max(0, target_win - 1)
    boundary_vector = residuals[prev_win] # [1024]
    norm_boundary = np.linalg.norm(boundary_vector)
    print(f"• Vector de frontera   : Ventana {prev_win} (Norma={norm_boundary:.4f}, Dim_store={len(boundary_vector)})")

    # ── 3. INFERENCIA MARKOVIANA EN UNA SOLA PASADA ─────────────────────────
    section(f"3. INFERENCIA MARKOVIANA [{model_key.upper()}] (VENTANA 512 TOKENS + ESTADO)")

    target_tokens = list(window_tokens[target_win])
    window_passage = tok.decode(target_tokens)

    augmented_prompt = (
        f"Contexto del documento:\n{window_passage}\n\n"
        f"Pregunta: {query_text}\n"
        f"Responde con precisión matemática y fidelidad total al documento:"
    )

    prompt_chat_markov = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": augmented_prompt}]}
    ], add_generation_prompt=True)

    # Adaptación dimensional si el almacén y el modelo tienen diferente D
    if len(boundary_vector) == D_model:
        adapted_boundary = boundary_vector
    else:
        # Interpolación armónica continua para reescalar de 1024 a D_model (ej. 5120 o 2048)
        x_old = np.linspace(0, 1, len(boundary_vector))
        x_new = np.linspace(0, 1, D_model)
        adapted_boundary = np.interp(x_new, x_old, boundary_vector).astype(np.float32)
        adapted_boundary = adapted_boundary * (norm_boundary / np.linalg.norm(adapted_boundary))

    boundary_mx = mx.array(adapted_boundary)[None, None, :]

    class MarkovBoundaryHook:
        def __init__(self, layer, idx): 
            self.layer, self.idx, self.injected = layer, idx, False
        def __getattr__(self, name): 
            return getattr(self.layer, name)
        def __call__(self, x, **kwargs):
            out = self.layer(x, **kwargs)
            if self.idx == 0 and not self.injected:
                out_mod = out.astype(mx.float32)
                h_0 = out_mod[:, :1, :] + 0.10 * boundary_mx
                out = mx.concatenate([h_0.astype(out.dtype), out[:, 1:, :]], axis=1)
                self.injected = True
            return out

    for l in range(num_layers): lm_model.layers[l] = MarkovBoundaryHook(orig_layers[l], l)

    print(f"Salida de Qwen {model_key.upper()} con Motor Markoviano (1 Pasada, 1.12 MB):")
    print("─" * 78)
    t0_gen = time.perf_counter()
    toks_m = []
    for r in stream_generate(model, processor, prompt=prompt_chat_markov, max_tokens=180):
        toks_m.append(r.text)
        print(r.text, end="", flush=True)
    t_gen = time.perf_counter() - t0_gen
    print("\n" + "─" * 78)
    print(f"✓ Generación completada: {len(toks_m)} tokens en {t_gen:.2f}s ({len(toks_m)/t_gen:.1f} tok/s)")

    for l in range(num_layers): lm_model.layers[l] = orig_layers[l]
    section("DICTAMEN: EVALUACIÓN MARKOVIANA CONCLUIDA")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="¿Cómo se define el campo de potencial fantasma phi(x)?")
    parser.add_argument("--model", type=str, default="27b", choices=["0.8b", "2b", "27b", "35b"])
    parser.add_argument("--window", type=int, default=74)
    args = parser.parse_args()

    run_query(args.query, args.model, args.window)
