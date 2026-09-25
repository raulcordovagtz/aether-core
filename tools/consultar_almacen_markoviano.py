#!/usr/bin/env python3
"""
tools/consultar_almacen_markoviano.py
═══════════════════════════════════════════════════════════════════════════════
INFERENCIA MARKOVIANA EN UNA SOLA PASADA (APOLLO 11 EXPERIMENT EN 1.12 MB)
SSOT: Reconstrucción de Contexto mediante Boundary Residual de 4 KB en Capa 0
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
MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def search_best_window(query_str, snippets):
    """Buscador léxico ligero de ventana relevante (análogo al índice de 120 KB de Chris)."""
    words = [w.lower() for w in query_str.split() if len(w) > 3]
    scores = np.zeros(len(snippets))
    for idx, snip in enumerate(snippets):
        snip_low = str(snip).lower()
        for w in words:
            if w in snip_low:
                scores[idx] += 1.0
    best_idx = int(np.argmax(scores))
    return best_idx, scores[best_idx]

def run_query(query_text, forced_window=None):
    section("EXPERIMENTO MARKOVIANO: CONSULTA SOBRE DOCUMENTO INÉDITO (1.12 MB)")
    print(f"• Almacén cargado      : {STORE_PATH}")
    print(f"• Consulta planteada    : \"{query_text}\"")

    if not os.path.exists(STORE_PATH):
        print(f"❌ Error: No se encuentra el almacén en {STORE_PATH}. Ejecuta construir_almacen_markoviano.py primero.")
        return

    data = np.load(STORE_PATH, allow_pickle=True)
    residuals = data["boundary_residuals"] # [453, 1024]
    window_tokens = data["window_token_ids"]
    snippets = data["window_snippets"]
    total_windows = len(residuals)

    print("\n[1/3] Cargando modelo en memoria unificada...")
    model, processor = load(MODEL_PATH)
    tok = getattr(processor, "tokenizer", processor)
    lm_model = model.language_model.model
    num_layers = len(lm_model.layers)
    orig_layers = list(lm_model.layers)

    # ── 1. EVALUACIÓN VANILLA (A CIEGAS) ────────────────────────────────────
    section("1. GENERACIÓN VANILLA (A CIEGAS — SIN ACCESO AL ALMACÉN)")
    prompt_chat_vanilla = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": query_text}]}
    ], add_generation_prompt=True)

    print("Salida de Qwen 0.8B Vanilla:")
    print("─" * 78)
    for r in stream_generate(model, processor, prompt=prompt_chat_vanilla, max_tokens=120):
        print(r.text, end="", flush=True)
    print("\n" + "─" * 78)

    # ── 2. LOCALIZACIÓN DE VENTANA Y CARGA DEL VECTOR MARKOVIANO (10 KB) ────
    section("2. LOCALIZACIÓN MARKOVIANA Y RESIDUAL DE FRONTERA")
    if forced_window is not None:
        target_win = forced_window
        print(f"• Ventana forzada por usuario : Ventana {target_win}/{total_windows}")
    else:
        target_win, score = search_best_window(query_text, snippets)
        print(f"• Ventana localizada por índice: Ventana {target_win}/{total_windows} (Score: {score})")
    
    snippet_text = str(snippets[target_win])
    print(f"• Snippet de la ventana        : \"{snippet_text}...\"")

    # Extraer el vector de frontera residual de la ventana anterior (h_{k-1})
    prev_win = max(0, target_win - 1)
    boundary_vector = residuals[prev_win] # [1024] float32 = 4 KB
    norm_boundary = np.linalg.norm(boundary_vector)
    print(f"• Vector de frontera inyectado : Ventana {prev_win} -> ||h_boundary|| = {norm_boundary:.4f} (4 KB exactos)")

    # ── 3. INFERENCIA CON EL VECTOR DE FRONTERA EN CAPA 0 (SIN KV-CACHE) ────
    section("3. INFERENCIA MARKOVIANA (VECTOR 4 KB + VENTANA 512 TOKENS)")

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

    # Inyección del vector de frontera h_{k-1} en la Capa 0 como ancla de estado
    boundary_mx = mx.array(boundary_vector)[None, None, :]

    class MarkovBoundaryHook:
        def __init__(self, layer, idx):
            self.layer = layer
            self.idx = idx
            self.injected = False
        def __getattr__(self, name):
            return getattr(self.layer, name)
        def __call__(self, x, **kwargs):
            out = self.layer(x, **kwargs)
            # En la Capa 0, sumamos el vector residual de frontera en la primera posición
            if self.idx == 0 and not self.injected:
                out_mod = out.astype(mx.float32)
                # Inyección aditiva conformal del estado de la ventana previa
                h_0 = out_mod[:, :1, :] + 0.10 * boundary_mx
                out = mx.concatenate([h_0.astype(out.dtype), out[:, 1:, :]], axis=1)
                self.injected = True
            return out

    for l in range(num_layers):
        lm_model.layers[l] = MarkovBoundaryHook(orig_layers[l], l)

    print("Salida de Qwen 0.8B con Motor Markoviano (1 Pasada, 1.12 MB):")
    print("─" * 78)
    t0_gen = time.perf_counter()
    toks_m = []
    for r in stream_generate(model, processor, prompt=prompt_chat_markov, max_tokens=150):
        toks_m.append(r.text)
        print(r.text, end="", flush=True)
    t_gen = time.perf_counter() - t0_gen
    print("\n" + "─" * 78)
    print(f"✓ Generación completada: {len(toks_m)} tokens en {t_gen:.2f}s ({len(toks_m)/t_gen:.1f} tok/s)")

    for l in range(num_layers):
        lm_model.layers[l] = orig_layers[l]

    section("DICTAMEN: EXPERIMENTO MARKOVIANO EN SILICIO COMPLETADO")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="¿Cómo se define el campo de potencial fantasma phi(x)?")
    parser.add_argument("--window", type=int, default=None, help="Índice de ventana específico (ej. 75, 225, 350)")
    args = parser.parse_args()

    run_query(args.query, args.window)