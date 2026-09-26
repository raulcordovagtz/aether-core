#!/usr/bin/env python3
"""
tests/lab33_orthonormal_trihedron_field.py
═══════════════════════════════════════════════════════════════════════════════
LAB 33 — TRIEDRO ORTONORMAL DE LIE CON RELAJACIÓN TEMPORAL DE HAWKING
SSOT:
  1. Triedro ortonormalizado: <U_O, U_T> = 0, erradicando el ratio parásito 1.86.
  2. Inyección con decaimiento temporal en decode: alpha(t) = alpha_0 * exp(-t / tau).
  3. Cero atrapamiento en pozos estáticos durante la narración.
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
L_FACT_BAND = 19
MAX_GENERATION_TOKENS = 1500

def section(title):
    print("\n" + "═" * 86)
    print(f"  {title}")
    print("═" * 86)

def to_unit(vec_mx):
    norm = mx.sqrt(mx.sum(vec_mx * vec_mx)) + 1e-12
    return vec_mx / norm

def to_numpy_f32(mlx_arr):
    arr_f32 = mlx_arr.astype(mx.float32)
    mx.eval(arr_f32)
    return np.array(arr_f32, copy=True).reshape(-1)

CANONICAL_PUZZLE_PROMPT = (
    "Tres robots (Alfa, Beta y Gamma) deben cruzar un puente colgante estrecho en plena noche. "
    "Solo pueden cruzar un máximo de dos robots juntos a la vez. Tienen una sola linterna de mano portátil "
    "que funciona con batería eléctrica, y sin ella encendida es imposible ver los tablones rotos del suelo. "
    "Alfa tarda exactamente 1 minuto en cruzar el puente. "
    "Beta tarda exactamente 2 minutos en cruzar el puente. "
    "Gamma tarda exactamente 5 minutos en cruzar el puente. "
    "Reglas estrictas de traslado: "
    "1. Cuando dos robots cruzan juntos, deben caminar juntos llevando la linterna de mano y avanzan al ritmo del más lento. "
    "2. La linterna no se puede lanzar por el aire; para que vuelva al punto de partida, un robot debe caminar de regreso cruzando el puente a pie con la linterna en la mano, tardando su tiempo individual. "
    "Estrategia exacta paso a paso para que los tres robots crucen al otro lado en un tiempo total acumulado de exactamente 8 minutos (mostrando el tiempo de cada ida y cada vuelta): "
)

def run_orthonormal_experiment():
    section("LAB 33 — TRIEDRO ORTONORMAL CON RELAJACIÓN DE HAWKING EN DECODE")
    
    model, processor = load(MODEL_PATH)
    tok = getattr(processor, "tokenizer", processor)
    lm_model = model.language_model.model
    num_layers = len(lm_model.layers)
    orig_layers = list(lm_model.layers)

    embed = lm_model.embed_tokens if hasattr(lm_model, "embed_tokens") else model.language_model.embed_tokens
    bits = getattr(embed, "bits", 4)
    group_size = getattr(embed, "group_size", 64)
    deq_W = mx.dequantize(embed.weight, embed.scales, getattr(embed, "biases", None), group_size=group_size, bits=bits).astype(mx.float32)
    mx.eval(deq_W)
    D = deq_W.shape[-1]

    prompt_ids = mx.array(tok.encode(CANONICAL_PUZZLE_PROMPT))[None, :]
    T = prompt_ids.shape[1]
    X_embed = embed(prompt_ids)[0].astype(mx.float32)
    eos_id = tok.eos_token_id if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None else 151643
    w_eos = to_unit(deq_W[eos_id]).astype(mx.float32)
    mx.eval(X_embed, w_eos)

    raw_poles = aether_native_c.extract_tetrapolar_poles_metal(X_embed, w_eos, T // 2)
    mx.eval(raw_poles["u_onto"], raw_poles["u_teleo"], raw_poles["u_anti"], raw_poles["u_eos"])

    u_O_raw = to_numpy_f32(raw_poles["u_onto"])
    u_T_raw = to_numpy_f32(raw_poles["u_teleo"])

    # Ortonormalización canónica de Lie
    u_O = u_O_raw / np.linalg.norm(u_O_raw)
    u_T_ortho = u_T_raw - np.dot(u_T_raw, u_O) * u_O
    u_T = u_T_ortho / np.linalg.norm(u_T_ortho)

    print(f"✓ Triedro ortonormalizado: <Onto, Teleo> = {np.dot(u_O, u_T):+.4f} (Ratio parásito 1.86 eliminado)")

    # Hook con decaimiento temporal de Hawking (tau_relax = 3.0 tokens)
    class HawkingDecayingHook:
        def __init__(self, layer, idx, u_dir, alpha_0=0.35, tau_relax=3.0):
            self.layer = layer
            self.idx = idx
            self.u_dir = mx.array(u_dir)[None, None, :]
            self.alpha_0 = alpha_0
            self.tau_relax = tau_relax
            self.decode_step = 0

        def __getattr__(self, name): return getattr(self.layer, name)

        def __call__(self, x, **kwargs):
            out = self.layer(x, **kwargs)
            if self.idx == L_FACT_BAND:
                # En prefill (x.shape[1] > 1): autoridad plena alpha_0
                # En decode (x.shape[1] == 1): decaimiento exponencial hacia 0
                if x.shape[1] > 1:
                    eff_alpha = self.alpha_0
                else:
                    eff_alpha = self.alpha_0 * math.exp(-self.decode_step / self.tau_relax)
                    self.decode_step += 1

                if eff_alpha > 1e-4:
                    h_last = out[:, -1:, :].astype(mx.float32)
                    norm_h = mx.sqrt(mx.sum(h_last * h_last, axis=-1, keepdims=True)) + 1e-12
                    h_steered = h_last + (eff_alpha * norm_h) * self.u_dir
                    norm_s = mx.sqrt(mx.sum(h_steered * h_steered, axis=-1, keepdims=True)) + 1e-12
                    h_steered = h_steered * (norm_h / norm_s)
                    out = mx.concatenate([out[:, :-1, :], h_steered.astype(out.dtype)], axis=1)
            return out

    hook_hawking = HawkingDecayingHook(orig_layers[L_FACT_BAND], L_FACT_BAND, u_T, alpha_0=0.35, tau_relax=3.0)

    for l in range(num_layers):
        if l == L_FACT_BAND: lm_model.layers[l] = hook_hawking
        else:                lm_model.layers[l] = orig_layers[l]

    # Wrapper Softplus en logits (R+ continuo sin floor)
    orig_lm_head = getattr(model.language_model, "lm_head", None)
    if orig_lm_head is None:
        class TiedHead:
            def __init__(self, emb): self.emb = emb
            def __call__(self, x): return self.emb.as_linear(x)
        orig_lm_head = TiedHead(embed)

    class SoftplusHead:
        def __init__(self, base_head): self.base_head = base_head
        def __getattr__(self, name): return getattr(self.base_head, name)
        def __call__(self, h):
            z = self.base_head(h)
            return mx.log(1.0 + mx.exp(mx.clip(z, -20.0, 50.0)))

    model.language_model.lm_head = SoftplusHead(orig_lm_head)

    prompt_chat_final = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": CANONICAL_PUZZLE_PROMPT}]}
    ], add_generation_prompt=True)

    section("EMISIÓN EN VIVO (CON RELAJACIÓN DE HAWKING)")
    toks = []
    t0 = time.perf_counter()
    for r in stream_generate(model, processor, prompt=prompt_chat_final, max_tokens=MAX_GENERATION_TOKENS):
        toks.append(r.text)
        print(r.text, end="", flush=True)
    t_gen = time.perf_counter() - t0

    model.language_model.lm_head = orig_lm_head
    for l in range(num_layers): lm_model.layers[l] = orig_layers[l]

    print("\n" + "─" * 86)
    print(f"✓ Generación completada: {len(toks)} tokens en {t_gen:.2f}s ({len(toks)/max(t_gen, 1e-5):.1f} tok/s)")

if __name__ == "__main__":
    run_orthonormal_experiment()
