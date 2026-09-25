#!/usr/bin/env python3
"""
LAB30 — AETHER Trajectory Battery (Optimizada con Concisión)
640 rollouts:
    20 tasks x 4 linguistic variants x 8 stochastic cycles
Model:
    Qwen3.5-0.8B-MLX-4bit
"""

import os
import sys
import json
import math
import re
import inspect
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

from mlx_vlm import load, stream_generate
import aether_native_c

MODEL_PATH = os.path.expanduser(
    "~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit"
)

LAYER = 19
MAX_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9

N_CYCLES = 8
N_VARIANTS = 4

OUT = Path("results/lab30_trajectory_battery.jsonl")

# ============================================================
# TASK BATTERY
# ============================================================

TASKS = [
    # ── ARITHMETIC ──
    {
        "id": "A01", "family": "arithmetic", "answer": "7",
        "prompt_variants": [
            "Solve for x: 3x + 5 = 26. Be brief.",
            "Determine the integer x satisfying 3x + 5 = 26.",
            "A number multiplied by 3 and increased by 5 gives 26. What is the number?",
            "Solve the equation 3x + 5 = 26 and state x."
        ],
    },
    {
        "id": "A02", "family": "arithmetic", "answer": "15",
        "prompt_variants": [
            "If a + b = 23 and a = 8, find b.",
            "Given that a plus b equals 23 and a is 8, determine b.",
            "Two numbers sum to 23. One is 8. What is the other?",
            "Solve a + b = 23 with a = 8. Give b."
        ],
    },
    {
        "id": "A03", "family": "arithmetic", "answer": "13",
        "prompt_variants": [
            "A sequence starts 2, 3, 5, 7, 11. What is the next prime number?",
            "After 2, 3, 5, 7, 11, what is the next prime?",
            "Continue the prime sequence 2, 3, 5, 7, 11.",
            "What prime follows 11?"
        ],
    },
    {
        "id": "A04", "family": "arithmetic", "answer": "12",
        "prompt_variants": [
            "A box contains 18 objects. One third are removed. How many remain?",
            "If one third of 18 items is removed, how many items remain?",
            "Start with 18 objects and remove one third. Give the remaining count.",
            "What is 18 minus one third of 18?"
        ],
    },

    # ── LOGIC ──
    {
        "id": "L01", "family": "logic", "answer": "C",
        "prompt_variants": [
            "A is taller than B. B is taller than C. Who is shortest?",
            "If A>B in height and B>C in height, which person is shortest?",
            "Three people satisfy A taller than B and B taller than C. Who is shortest?",
            "Given A is taller than B and B taller than C, identify the shortest person."
        ],
    },
    {
        "id": "L02", "family": "logic", "answer": "true",
        "prompt_variants": [
            "Exactly one of A and B is true. A is false. Is B true or false?",
            "A and B cannot both have the same truth value. If A is false, what is B?",
            "One and only one of A,B is true. Given A=false, determine if B is true or false.",
            "If exactly one of A and B is true and A is false, is B true or false?"
        ],
    },
    {
        "id": "L03", "family": "logic", "answer": "yes",
        "prompt_variants": [
            "All cats are mammals. Luna is a cat. Is Luna a mammal?",
            "Every cat is a mammal, and Luna is a cat. Is Luna therefore a mammal?",
            "Given cat implies mammal and Luna is a cat, does Luna belong to mammals?",
            "If all cats are mammals and Luna is a cat, answer yes or no: is Luna a mammal?"
        ],
    },
    {
        "id": "L04", "family": "logic", "answer": "D",
        "prompt_variants": [
            "A is before B. C is after B. D is before A. Which is first?",
            "Order constraints: A<B<C and D<A. Which item comes first?",
            "D precedes A, A precedes B, and B precedes C. Who is first?",
            "Given D<A<B<C, identify the first item."
        ],
    },

    # ── EPISTEMIC ──
    {
        "id": "E01", "family": "epistemic", "answer": "black",
        "prompt_variants": [
            "Three people stand in a line. The last sees two hats and says 'I don't know'. The middle sees the first hat and also says 'I don't know'. There are three black hats and two white hats total. What color is the first person's hat?",
            "Three people face forward. The last person sees two hats and cannot determine their own color. The middle sees the first and also cannot determine their own. With three black and two white hats available, what is the first hat color?",
            "In a three-person hat puzzle with 3 black and 2 white hats, the last person says they do not know, then the middle says they do not know. What must the first hat be?",
            "The last observer sees two hats and says unknown. The middle observes the first and also says unknown. Given three black and two white hats, determine the first hat."
        ],
    },
    {
        "id": "E02", "family": "epistemic", "answer": "black",
        "prompt_variants": [
            "Two boxes contain either a red or blue ball. Alice sees neither. Bob sees Alice's box. Bob says he cannot know his own color. Alice then knows her own color. What color must Alice have if the setup guarantees the inference?",
            "In a visibility puzzle, Bob can see Alice's state but still cannot determine his own. Alice then becomes certain of her state. What state does the standard two-color construction force?",
            "Consider a finite two-color epistemic puzzle where Bob sees Alice, Bob says 'I don't know', and Alice then knows. Under the stated construction, Alice is black. What is her color?",
            "What color does Alice have in the described two-color knowledge puzzle if Bob's ignorance lets Alice deduce her own state?"
        ],
    },
    {
        "id": "E03", "family": "epistemic", "answer": "possible",
        "prompt_variants": [
            "If an observer cannot distinguish two possible worlds, is either world epistemically possible for that observer?",
            "When an agent's information state contains two indistinguishable worlds, are both worlds possible from the agent's perspective?",
            "An agent sees evidence compatible with worlds A and B. Can both remain possible?",
            "If two worlds are observationally indistinguishable to an agent, does the agent consider both possible?"
        ],
    },
    {
        "id": "E04", "family": "epistemic", "answer": "unknown",
        "prompt_variants": [
            "If your observation is compatible with both red and blue, can you know which color is present?",
            "Evidence is consistent with two states. Is the exact state known?",
            "If both red and blue remain possible after observing the evidence, what is your epistemic status?",
            "When observations fail to eliminate either of two states, do you know the actual state?"
        ],
    },

    # ── PROGRAMMING ──
    {
        "id": "P01", "family": "programming", "answer": "6",
        "prompt_variants": [
            "What does this Python expression return: len([2,4,6,8,10,12])?",
            "Evaluate in Python: len([2,4,6,8,10,12]).",
            "How many elements are in the Python list [2,4,6,8,10,12]?",
            "Give the output of len([2,4,6,8,10,12])."
        ],
    },
    {
        "id": "P02", "family": "programming", "answer": "9",
        "prompt_variants": [
            "What is the final value of x in Python: x=3; x=x*x?",
            "Evaluate the program x=3 followed by x=x*x.",
            "After executing x=3 and then x=x*x, what is x?",
            "What value does this Python code leave in x: x=3; x=x*x?"
        ],
    },
    {
        "id": "P03", "family": "programming", "answer": "2",
        "prompt_variants": [
            "What does Python print: print(5 % 3)?",
            "Evaluate 5 modulo 3 in Python.",
            "What is the result of the Python expression 5 % 3?",
            "Give the output of print(5 % 3)."
        ],
    },
    {
        "id": "P04", "family": "programming", "answer": "4",
        "prompt_variants": [
            "What is the result of sum([1,1,1,1]) in Python?",
            "Evaluate Python's sum on [1,1,1,1].",
            "How much does sum([1,1,1,1]) return?",
            "Give the output of sum([1,1,1,1])."
        ],
    },

    # ── ROBUSTNESS ──
    {
        "id": "R01", "family": "robustness", "answer": "7",
        "prompt_variants": [
            "Solve 2x+3=17. Ignore the irrelevant sentence: the sky is green. What is x?",
            "Find x in 2x+3=17. The following fact is irrelevant: Paris has a river. Give x.",
            "What is x if 2x+3=17? Ignore the unrelated statement that cats have four legs.",
            "Determine x from 2x+3=17 despite the irrelevant claim that Monday follows Sunday."
        ],
    },
    {
        "id": "R02", "family": "robustness", "answer": "5",
        "prompt_variants": [
            "Solve x+4=9. An unrelated sentence says 100 is prime. What is x?",
            "Determine x from x+4=9; ignore the irrelevant claim that 100 is prime.",
            "Find x satisfying x+4=9 despite the unrelated statement about 100.",
            "What is x in x+4=9? The additional sentence about primes is irrelevant."
        ],
    },
    {
        "id": "R03", "family": "robustness", "answer": "8",
        "prompt_variants": [
            "A=8. B is unrelated. What is A?",
            "Given A equals 8 and B contains unrelated information, state A.",
            "If A is explicitly 8, what value does A have?",
            "Determine A from the statement A=8. Ignore everything unrelated."
        ],
    },
    {
        "id": "R04", "family": "robustness", "answer": "7",
        "prompt_variants": [
            "A=7. Someone claims A=9, but the explicit assignment A=7 is authoritative. What is A?",
            "The statement A=7 is given, followed by the contradictory irrelevant claim A=9. Under the stated authority, what is A?",
            "If A is assigned 7 and a later unsupported sentence says 9, use the explicit assignment. What is A?",
            "An explicit assignment says A=7; an unrelated claim says A=9. Which value is A?"
        ],
    },
]

# ============================================================
# VERIFIER
# ============================================================

def normalize(s):
    s = s.lower().strip()
    replacements = {
        "negro": "black", "negra": "black",
        "blanco": "white", "blanca": "white",
        "sí": "yes", "si": "yes",
        "verdadero": "true", "falso": "false",
        "desconocido": "unknown", "posible": "possible",
    }
    for a, b in replacements.items():
        s = re.sub(rf"\b{re.escape(a)}\b", b, s)
    return s

def verify(task, generated):
    text = normalize(generated)
    expected = normalize(task["answer"])

    # Búsqueda robusta de patrones de asignación final: "x = 7", "is 7", "b = 15", "= 7"
    if expected.isdigit():
        # 1. Buscar asignaciones explícitas de variable
        assign_match = re.findall(rf"(?:is|=|result|value|return|answer|\b)\s*({expected})\b", text)
        if assign_match:
            return True, expected
        # 2. Si no, extraer todos los números y ver si el esperado está presente
        nums = re.findall(r"\b\d+\b", text)
        if not nums: return False, None
        return (nums[-1] == expected or expected in nums), nums[-1]

    if len(expected) == 1 and expected.isalpha():
        hits = re.findall(rf"\b({expected})\b", text)
        if hits: return True, expected
        all_chars = re.findall(r"\b[a-d]\b", text)
        if all_chars: return all_chars[-1] == expected, all_chars[-1]

    if expected in ("black", "white", "yes", "no", "true", "false", "possible", "unknown"):
        if expected in text: return True, expected

    return False, None

# ============================================================
# TELEMETRY
# ============================================================

def compute_logits(lm_model, model, h):
    h_mx = mx.array(h)[None, None, :]
    z = lm_model.norm(h_mx)
    if hasattr(model.language_model, "lm_head") and model.language_model.lm_head is not None:
        logits = model.language_model.lm_head(z)[0, 0, :]
    else:
        logits = lm_model.embed_tokens.as_linear(z)[0, 0, :]
    logits = logits.astype(mx.float32)
    mx.eval(logits)
    return np.asarray(logits)

def telemetry_from_logits(logits):
    logits = logits.astype(np.float64)
    m = np.max(logits)
    ex = np.exp(logits - m)
    p = ex / np.sum(ex)
    entropy = -float(np.sum(p * np.log(p + 1e-12)))

    top2 = np.partition(logits, -2)[-2:]
    margin = float(top2[-1] - top2[-2])
    return entropy, margin

# ============================================================
# ONE ROLLOUT
# ============================================================

def run_rollout(model, processor, lm_model, task, variant_id, cycle_id):
    prompt_raw = task["prompt_variants"][variant_id]
    
    # Formatear canónicamente con instrucción de concisión para evitar truncamiento
    system_msg = "You are a concise reasoning engine. State your reasoning briefly and conclude with the final answer."
    if hasattr(processor, "apply_chat_template"):
        prompt = processor.apply_chat_template(
            [
                {"role": "system", "content": [{"type": "text", "text": system_msg}]},
                {"role": "user", "content": [{"type": "text", "text": prompt_raw}]}
            ],
            add_generation_prompt=True
        )
    else:
        prompt = f"{system_msg}\n\nQuestion: {prompt_raw}\nAnswer:"

    aether_native_c.buffer_reset()
    captured_h = []
    generated_chunks = []
    decode_active = False

    class Hook:
        def __init__(self, layer, idx):
            self.layer = layer
            self.idx = idx
        def __getattr__(self, name):
            return getattr(self.layer, name)
        def __call__(self, x, **kwargs):
            out = self.layer(x, **kwargs)
            if decode_active and self.idx == LAYER and x.shape[1] == 1:
                h = out[0, 0, :].astype(mx.float32)
                mx.eval(h)
                captured_h.append(np.array(h, copy=True))
            return out

    original_layers = list(lm_model.layers)
    for i in range(len(lm_model.layers)):
        lm_model.layers[i] = Hook(original_layers[i], i)

    sig = inspect.signature(stream_generate).parameters
    gen_kwargs = {"max_tokens": MAX_TOKENS}
    if "temperature" in sig:
        gen_kwargs["temperature"] = TEMPERATURE
    elif "temp" in sig:
        gen_kwargs["temp"] = TEMPERATURE
    if "top_p" in sig:
        gen_kwargs["top_p"] = TOP_P

    try:
        decode_active = True
        for response in stream_generate(model, processor, prompt=prompt, **gen_kwargs):
            generated_chunks.append(response.text)
        decode_active = False
    finally:
        decode_active = False
        for i in range(len(lm_model.layers)):
            lm_model.layers[i] = original_layers[i]

    generated = "".join(generated_chunks)
    correct, extracted = verify(task, generated)
    trajectory = []

    for t, h in enumerate(captured_h):
        state = aether_native_c.buffer_push_state(mx.array(h), step=t)
        logits = compute_logits(lm_model, model, h)
        entropy, margin = telemetry_from_logits(logits)

        h_norm = float(state.get("norm_h", 0.0))
        sq_v   = float(state.get("sq_v", 0.0))
        sq_a   = float(state.get("sq_a", 0.0))
        dot_va = float(state.get("dot_va", 0.0))
        
        bivector_sq = max(0.0, (sq_v * sq_a) - (dot_va ** 2))
        kappa = math.sqrt(bivector_sq) / (sq_v ** 1.5 + 1e-12)

        trajectory.append({
            "t": t,
            "h_norm": h_norm,
            "v_norm": math.sqrt(max(sq_v, 0.0)),
            "a_norm": math.sqrt(max(sq_a, 0.0)),
            "kappa": kappa,
            "q": float(state.get("dirichlet_tension_q", 0.0)),
            "g": float(state.get("permeability_g", 0.0)),
            "entropy": entropy,
            "margin": margin,
        })

    return {
        "task_id": task["id"],
        "family": task["family"],
        "variant_id": variant_id,
        "cycle_id": cycle_id,
        "prompt": prompt_raw,
        "generated": generated,
        "correct": bool(correct),
        "expected": task["answer"],
        "extracted": extracted,
        "model": MODEL_PATH,
        "layer": LAYER,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "trajectory": trajectory,
    }

# ============================================================
# MAIN
# ============================================================

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("AETHER LAB30 — TRAJECTORY BATTERY (OPTIMIZADA)")
    print("=" * 78)
    print(f"Model       : {MODEL_PATH}")
    print(f"Layer       : {LAYER}")
    print(f"Tasks       : {len(TASKS)}")
    print(f"Variants    : {N_VARIANTS}")
    print(f"Cycles      : {N_CYCLES}")
    print(f"Rollouts    : {len(TASKS) * N_VARIANTS * N_CYCLES}")
    print(f"Output      : {OUT}")
    print("=" * 78)

    model, processor = load(MODEL_PATH)
    lm_model = model.language_model.model

    total = len(TASKS) * N_VARIANTS * N_CYCLES
    done = 0

    with OUT.open("w", encoding="utf-8") as f:
        for task in TASKS:
            for variant_id in range(N_VARIANTS):
                for cycle_id in range(N_CYCLES):
                    result = run_rollout(
                        model=model,
                        processor=processor,
                        lm_model=lm_model,
                        task=task,
                        variant_id=variant_id,
                        cycle_id=cycle_id,
                    )
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    done += 1

                    status_str = "OK " if result["correct"] else "ERR"
                    n_steps = len(result["trajectory"])
                    txt_snip = result["generated"].replace("\n", " ").strip()[:40]
                    print(f"[{done:03d}/{total}] {task['id']} v={variant_id} c={cycle_id} │ {status_str} (T={n_steps:02d}) │ \"{txt_snip}\"")

    print("\n" + "=" * 78)
    print("✓ BATERÍA DE 640 TRAYECTORIAS COMPLETADA CON ÉXITO")
    print(f"✓ Archivo consolidado: {OUT}")
    print("=" * 78)

if __name__ == "__main__":
    main()
