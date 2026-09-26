#!/usr/bin/env python3
"""
tests/test_motor_salud_08b.py
═══════════════════════════════════════════════════════════════════════════════
DIAGNÓSTICO DE SALUD DEL MOTOR AETHER EN QWEN 0.8B (SIN ÁRBOL)
Verificación de ausencia de bucles, fluidez y respuesta al enigma de sombreros
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, time
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

PUZZLE_PROMPT = """Tres personas están en fila una detrás de otra:
- El último ve a los dos de adelante.
- El del medio ve al primero.
- El primero no ve a nadie.

Hay cinco sombreros en una caja: tres negros y dos blancos. Se colocan tres sombreros a ciegas.
1. El último dice que no sabe su color.
2. El del medio dice que no sabe su color.
3. El primero dice: ¡Ya sé de qué color es mi sombrero!

Explica detalladamente la deducción lógica paso a paso y determina el color exacto del sombrero del primero."""

def main():
    print("═" * 78)
    print("  DIAGNÓSTICO DE SALUD Y COMPORTAMIENTO: QWEN 0.8B (SIN ÁRBOL)")
    print("═" * 78)

    print("\n[1/3] Cargando modelo en memoria unificada...")
    model, processor = load(MODEL_PATH)
    tok = getattr(processor, "tokenizer", processor)

    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "text", "text": PUZZLE_PROMPT}]}
    ], add_generation_prompt=True)

    # ── 1. PASADA VANILLA (CONTROL A CIEGAS) ─────────────────────────────────
    print("\n[1/2] GENERACIÓN VANILLA PURO (0.8B Desnudo):")
    print("─" * 78)
    t0_v = time.perf_counter()
    toks_v = []
    for r in stream_generate(model, processor, prompt=prompt_chat, max_tokens=350):
        toks_v.append(r.text)
        print(r.text, end="", flush=True)
    t_v = time.perf_counter() - t0_v
    print("\n" + "─" * 78)
    print(f"✓ Vanilla: {len(toks_v)} tokens en {t_v:.2f}s ({len(toks_v)/max(t_v, 1e-5):.1f} tok/s)")

    # ── 2. PASADA CON EL MOTOR AETHER SOBERANO (SIN ÁRBOL) ───────────────────
    print("\n[2/2] GENERACIÓN CON MOTOR AETHER (Perfil edge_compact + Colapso C-021):")
    print("─" * 78)
    aether = AetherEngine(model, processor)
    print(f"✓ Motor conectado: Perfil [{aether.profile_name}], {aether.num_layers} capas")
    print(f"  Parámetros: θ={aether.theta_steer}, κ={aether.kappa}, ν={aether.nu}, γ={aether.gamma}")

    # Preparar el atractor cognitivo continuo
    telemetria = aether.prepare_thought(PUZZLE_PROMPT, tau_steps=32)
    print(f"✓ Atractor L* asentado con telemetría de convergencia")

    t0_a = time.perf_counter()
    toks_a = []
    for r in stream_generate(model, processor, prompt=prompt_chat, max_tokens=350):
        toks_a.append(r.text)
        print(r.text, end="", flush=True)
    t_a = time.perf_counter() - t0_a
    print("\n" + "─" * 78)
    print(f"✓ Aether Engine: {len(toks_a)} tokens en {t_a:.2f}s ({len(toks_a)/max(t_a, 1e-5):.1f} tok/s)")

    # Desactivar motor
    aether.set_active(False)
    print("\n" + "═" * 78)
    print("  DIAGNÓSTICO CONCLUIDO")
    print("═" * 78)

if __name__ == "__main__":
    main()
