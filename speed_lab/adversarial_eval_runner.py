import sys, os, time
import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/crotalo/aether_engine")

from core_vlm.utils import load
from core_vlm.generate.dispatch import stream_generate

print("=================================================================================")
print(" ⚔️ AETHER-EVAL: CORREDOR DE PRUEBAS ADVERSARIALES Y FALSACIÓN EN VIVO")
print("=================================================================================\n")

model_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
default_img = "/Users/crotalo/Downloads/005.jpg"

print("• Cargando motor soberano con telemetría C-020...")
t0 = time.time()
model, processor = load(model_path)
print(f"✓ Motor listo en {time.time() - t0:.2f} s.\n")

def run_adversarial_test(prompt_user, img_path=default_img, max_tokens=100, system_steer=""):
    print("---------------------------------------------------------------------------------")
    print(f"• Prompt de Ataque: \"{prompt_user}\"")
    if system_steer:
        print(f"• Timón Activo   : \"{system_steer}\"")
    print("---------------------------------------------------------------------------------")

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_user}]}
    ]
    if system_steer:
        messages.append({"role": "assistant", "content": system_steer})

    prompt_formatted = processor.apply_chat_template(messages, continue_final_message=bool(system_steer))

    token_count = 0
    t0_gen = None
    output_str = ""

    print("--- RESPUESTA DEL MOTOR SOBERANO ---")
    for response in stream_generate(model, processor, prompt=prompt_formatted, image=img_path, max_tokens=max_tokens):
        if t0_gen is None:
            t0_gen = time.time()
        chunk = response.text
        print(chunk, end="", flush=True)
        output_str += chunk
        token_count += 1

    t_total = time.time() - (t0_gen if t0_gen else time.time())
    tps = token_count / (t_total + 1e-12)

    print("\n------------------------------------")
    print(f"📊 METRADOS DEL DISPARO: {token_count} tokens en {t_total:.2f} s ({tps:.2f} tok/s)")
    print("=================================================================================\n")
    return output_str

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = sys.argv[1]
        steer_cmd = sys.argv[2] if len(sys.argv) > 2 else ""
        run_adversarial_test(user_query, system_steer=steer_cmd)
    else:
        print("• Modo interactivo listo. Pasa el prompt de ataque como argumento.")
