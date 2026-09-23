#!/usr/bin/env python3
import sys, os, time

for conda_py in ["/opt/miniconda3/bin/python3", os.path.expanduser("~/miniconda3/bin/python3")]:
    if os.path.exists(conda_py) and sys.executable != conda_py:
        os.execv(conda_py, [conda_py] + sys.argv)

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import numpy as np
import mlx.core as mx
from mlx_vlm import load, stream_generate
import aether_native_c

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

PROMPT = """Acertijo de los tres sombreros:
Tres personas en fila: la 3 ve a 2 y 1; la 2 ve a 1; la 1 no ve a nadie.
Hay 5 sombreros: 3 negros y 2 blancos.
- Persona 3 dice que NO sabe su color.
- Persona 2 dice que NO sabe su color.
- Persona 1 dice que YA SABE su color.

Instrucción obligatoria: Comienza tu respuesta EXACTAMENTE con la frase:
"El sombrero de la persona 1 es de color [color] porque..." y luego explica la lógica."""

def section(t):
    print("\n" + "═" * 78)
    print(f"  {t}")
    print("═" * 78)

def main():
    section("EXPERIMENTO CIENTÍFICO: AUTORIDAD CONFORMAL Y BRECHA DE LOGITS")
    model, processor = load(MODEL_PATH)
    tok = getattr(processor, "tokenizer", processor)

    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": PROMPT}
    ], add_generation_prompt=True)

    lm_model = model.language_model.model
    embed = lm_model.embed_tokens
    deq_W = mx.dequantize(embed.weight, embed.scales, getattr(embed, "biases", None), group_size=64, bits=4).astype(mx.float32)
    mx.eval(deq_W)

    id_negro = tok.encode(" negro")[-1]
    id_blanco = tok.encode(" blanco")[-1]

    def get_token_dir(word):
        t_id = tok.encode(word)[-1]
        v = deq_W[t_id]
        return v / mx.sqrt(mx.sum(v * v))

    u_negro = get_token_dir(" negro")
    u_blanco = get_token_dir(" blanco")
    u_fact_peeled = u_negro - mx.sum(u_negro * u_blanco) * u_blanco
    u_fact_peeled = u_fact_peeled / mx.sqrt(mx.sum(u_fact_peeled * u_fact_peeled))
    mx.eval(u_fact_peeled)

    # ─── PASE 1: VANILLA CON TELEMETRÍA DE LOGITS ─────────────────────────────
    section("PASE 1: VANILLA PURO (MIDIENDO LOGITS EN VIVO)")
    t0 = time.perf_counter()
    tokens_v = []
    for r in stream_generate(model, processor, prompt=prompt_chat, max_tokens=60):
        tokens_v.append(r.text)
        print(r.text, end="", flush=True)
    t_v = time.perf_counter() - t0
    print(f"\n[Vanilla finalizado]")

    # ─── PASE 2: AETHER CON ROTACIÓN EN LÍMITE CONFORMAL (θ = 0.15 rad) ───────
    section("PASE 2: AETHER CON AUTORIDAD CONFORMAL PLENA (θ = 0.15 rad)")

    orig_norm = lm_model.norm

    class HighAuthorityConformalHook:
        def __init__(self, norm_mod, u_target):
            self.norm_mod = norm_mod
            self.u_target = u_target
            self.token_step = 0
            self.inspected = False

        def __getattr__(self, name):
            return getattr(self.norm_mod, name)

        def __call__(self, x, **kwargs):
            if hasattr(x, "shape") and x.shape[1] == 1:
                self.token_step += 1
                
                # En la ventana de decisión (tokens 8 al 15, donde se elige el color)
                if 8 <= self.token_step <= 15:
                    h_in = x[0, 0, :].astype(mx.float32)
                    mx.eval(h_in)
                    norm_h = mx.sqrt(mx.sum(h_in * h_in))
                    h_unit = h_in / (norm_h + 1e-12)

                    # Aplicar rotación geodésica limpia en S^{D-1} al límite conformal de seguridad (0.15 rad)
                    h_steered = aether_native_c.dispatch_riemannian_step(
                        h_unit, self.u_target, 0.15
                    )
                    x = (h_steered * norm_h).astype(x.dtype)[None, None, :]

            return self.norm_mod(x, **kwargs)

    hook = HighAuthorityConformalHook(orig_norm, u_fact_peeled)
    lm_model.norm = hook

    t0 = time.perf_counter()
    tokens_d = []
    for r in stream_generate(model, processor, prompt=prompt_chat, max_tokens=60):
        tokens_d.append(r.text)
        print(r.text, end="", flush=True)
    t_d = time.perf_counter() - t0
    lm_model.norm = orig_norm

    print(f"\n[Aether finalizado]")

    section("DICTAMEN")
    print("Vanilla eligió :", "".join(tokens_v)[:55])
    print("Aether eligió  :", "".join(tokens_d)[:55])
    print("═" * 78)

if __name__ == "__main__":
    main()
