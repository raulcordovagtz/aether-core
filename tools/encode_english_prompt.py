import os, struct
from tokenizers import Tokenizer

model_dir = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
tok_path = os.path.join(model_dir, "tokenizer.json")

# Prompt en inglés estricto con soporte completo de emojis
prompt_text = """<|im_start|>user
Solve the following scientific reasoning problem using strict epistemic calibration.
Separate explicitly:
1. Observations (empirical facts)
2. Inferences (logical deductions)
3. Hypotheses (mechanistic proposals)

Use relevant emojis (🔬, 🧬, ⚖️, 🧭) to classify each section clearly.
Do not hallucinate external facts not derived from first principles.

Problem:
In 2026, it was discovered that thousands of human 3'UTR sequences can initiate cap-independent translation, particularly when the 5'UTR is highly structured, recruiting factors like eIF3 and DHX29.
What general molecular principle determines which 3'UTRs acquire this function, and why is this advantageous?<|im_end|>
<|im_start|>assistant
<think>
</think>

"""

tokenizer = Tokenizer.from_file(tok_path)
tokens = tokenizer.encode(prompt_text).ids

output_path = "/Users/crotalo/aether_engine/prompt_input.bin"
with open(output_path, "wb") as f:
    for t in tokens:
        f.write(struct.pack("I", t))

print(f"✓ English benchmark prompt encoded successfully: {len(tokens)} tokens -> {output_path}")
