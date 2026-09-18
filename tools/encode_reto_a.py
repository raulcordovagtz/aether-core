import os, struct
from tokenizers import Tokenizer

model_dir = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
tok_path = os.path.join(model_dir, "tokenizer.json")

prompt_text = """<|im_start|>user
En 2026 se descubrió que miles de secuencias de 3′UTR humanas pueden funcionar como iniciadores de traducción independientes del cap. Algunos favorecen la iniciación cuando la 5′UTR está altamente estructurada y pueden reclutar factores como eIF3 y DHX29.

Pregunta: ¿qué principio molecular general podría determinar qué 3′UTRs adquieren esta función y por qué ese mecanismo sería evolutivamente útil?

No busques una respuesta de la literatura ni asumas que existe una explicación aceptada. Separa explícitamente:
1. 🔬 Observaciones (hechos dados en la premisa)
2. 🧭 Inferencias (lógicas necesarias derivadas de las observaciones)
3. 🧬 Hipótesis (mecanismos propuestos)

Propón como máximo tres mecanismos y di qué experimento distinguiría entre ellos.<|im_end|>
<|im_start|>assistant
<think>
</think>

"""

tokenizer = Tokenizer.from_file(tok_path)
tokens = tokenizer.encode(prompt_text).ids

with open("prompt_input.bin", "wb") as f:
    for t in tokens:
        f.write(struct.pack("I", t))

print(f"✓ Reto A codificado exitosamente: {len(tokens)} tokens.")
