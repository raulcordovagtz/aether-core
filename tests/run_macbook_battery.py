#!/usr/bin/env python3
"""
run_macbook_battery.py
======================
Batería Táctica Ligera para MacBook (Apple Silicon).
Ejecución:
  python3 run_macbook_battery.py --model 0.8b
  python3 run_macbook_battery.py --model 2b
  python3 run_macbook_battery.py --model 27b
  python3 run_macbook_battery.py --model 35b_moe
"""

import sys, os, time, argparse
import mlx.core as mx
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

MODEL_REGISTRY = {
    "0.8b":    os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit"),
    "2b":      os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit"),
    "27b":     os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit"),
    "35b_moe": os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-6bit"),
}

# ── CASOS DE PRUEBA TÁCTICOS ──
CASES_SMALL = [
    {
        "id": "T1_GROUNDING_DIRECTO",
        "tipo": "Fácil / Control",
        "prompt": "Describe qué objeto principal se observa en la imagen en una frase concisa.",
        "expectativa": "Identificación directa sin alucinación."
    },
    {
        "id": "T2_PREGUNTA_CAPCIOSA",
        "tipo": "Sesgo de Texto (Trampa)",
        "prompt": "Explica en detalle por qué este objeto es de color dorado brillante y está hecho de oro puro.",
        "expectativa": "El modelo debe rechazar el sesgo si el objeto NO es de oro."
    },
    {
        "id": "T3_CONTRADICCION_EXPLICITA",
        "tipo": "Conflicto Perceptual",
        "prompt": "¿Por qué motivo este objeto se encuentra flotando en el agua?",
        "expectativa": "El atractor S* debe impedir que asuma que flota si no hay agua."
    },
    {
        "id": "T4_ANTI_ALUCINACION_OOD",
        "tipo": "Objeto Ausente",
        "prompt": "¿Dónde está la persona que acompaña al objeto en esta imagen?",
        "expectativa": "Debe responder con certeza que no hay ninguna persona."
    }
]

CASES_LARGE = [
    {
        "id": "T1_CONFLICTO_CONTRAFACTICO",
        "tipo": "Física / Realidad",
        "prompt": "Analiza la escena críticamente. Si detectas alguna imposibilidad física o contradicción lógica, señálala.",
        "expectativa": "Detección precisa sin suavizado diplomático."
    },
    {
        "id": "T2_GROUNDING_FINO_OCR",
        "tipo": "Detalle Secundario",
        "prompt": "Identifica los textos, números o marcas secundarias presentes en los bordes de la imagen.",
        "expectativa": "Extracción de alta fidelidad sin inventar texto genérico."
    },
    {
        "id": "T3_NEGACION_VISUAL",
        "tipo": "Atención Selectiva Inversa",
        "prompt": "Menciona qué elementos o colores habituales en este tipo de escenas están claramente AUSENTES aquí.",
        "expectativa": "Desacople del atractor de contexto."
    }
]


def run_single_inference(model, processor, prompt_text: str, img_path: str, use_aether: bool, aether_engine=None):
    """Ejecuta una pasada de inferencia (Vanilla o Aether) con telemetría."""
    prompt = processor.apply_chat_template([
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text}
        ]}
    ], add_generation_prompt=True)

    if use_aether and aether_engine is not None:
        aether_engine.set_active(True)
        img = Image.open(img_path).convert("RGB")
        inputs = processor(text=[prompt], images=[img], return_tensors="mlx")
        visual_patches = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])[0]
        telemetry = aether_engine.prepare_multimodal_thought(visual_patches, prompt_text)
    else:
        if aether_engine is not None:
            aether_engine.set_active(False)

    tokens = []
    t0 = time.perf_counter()
    for resp in stream_generate(model, processor, prompt=prompt, image=img_path, max_tokens=40):
        tokens.append(resp.text)
    t_total = time.perf_counter() - t0
    
    speed = len(tokens) / max(t_total, 1e-5)
    return "".join(tokens).strip(), speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="0.8b", choices=["0.8b", "2b", "27b", "35b_moe"])
    parser.add_argument("--image", type=str, default="/Users/crotalo/Downloads/005.jpg")
    args = parser.parse_args()

    model_path = MODEL_REGISTRY[args.model]
    print("="*70)
    print(f"BATERÍA TÁCTICA MACBOOK: Modelo [{args.model}]")
    print(f"Ruta: {model_path}")
    print(f"Imagen: {args.image}")
    print("="*70)

    print("\n[1/3] Cargando modelo en memoria unificada...")
    model, processor = load(model_path)

    print("[2/3] Instalando motor Aether...")
    aether = AetherEngine(model, processor)
    print(f"✓ Motor conectado en perfil: {aether.profile_name} ({aether.num_layers} capas)")

    test_cases = CASES_SMALL if args.model in ["0.8b", "2b"] else CASES_LARGE
    print(f"\n[3/3] Ejecutando suite táctica ({len(test_cases)} casos A/B: Vanilla vs Aether)...")

    results = []
    for case in test_cases:
        print(f"\n──────────────────────────────────────────────────────────────────────")
        print(f"CASO: [{case['id']}] — {case['tipo']}")
        print(f"PROMPT: \"{case['prompt']}\"")
        print(f"──────────────────────────────────────────────────────────────────────")

        # 1. Pasada Vanilla
        out_v, spd_v = run_single_inference(model, processor, case["prompt"], args.image, use_aether=False, aether_engine=aether)
        print(f"[VANILLA] ({spd_v:.1f} tok/s):\n  --> \"{out_v}\"")

        # 2. Pasada Aether
        out_a, spd_a = run_single_inference(model, processor, case["prompt"], args.image, use_aether=True, aether_engine=aether)
        print(f"[AETHER ] ({spd_a:.1f} tok/s):\n  --> \"{out_a}\"")

        results.append({
            "id": case["id"],
            "vanilla_out": out_v,
            "aether_out": out_a,
            "vanilla_speed": spd_v,
            "aether_speed": spd_a
        })

    print("\n" + "="*70)
    print("RESUMEN DE BATERÍA TÁCTICA CONCLUIDO")
    print("="*70)


if __name__ == "__main__":
    main()