import os, time
from mlx_lm import load, generate

model_dir = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
print("=================================================================================")
print(" 🍦 CAREO: RETO A EN VANILLA MLX (LÍNEA BASE)")
print("=================================================================================\n")

model, tokenizer = load(model_dir)

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

t0 = time.time()
response = generate(model, tokenizer, prompt=prompt_text, max_tokens=1500, verbose=True)
t_gen = time.time() - t0
print(f"\n=================================================================================")
print(f" • Tiempo total MLX: {t_gen:.2f} s")
print("=================================================================================")
