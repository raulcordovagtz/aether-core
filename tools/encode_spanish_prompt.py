import os, struct
from tokenizers import Tokenizer

model_dir = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
tok_path = os.path.join(model_dir, "tokenizer.json")

prompt_text = """<|im_start|>user
Explica brevemente la teoría de la relatividad de Einstein en español, usando términos con acentos como física, gravitación, energía, atracción y ecuación.<|im_end|>
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

print(f"✓ Spanish prompt encoded: {len(tokens)} tokens -> {output_path}")
