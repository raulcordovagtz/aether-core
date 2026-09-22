#!/usr/bin/env python3
"""
tests/test_runtime_cellular_engine.py
=====================================
Prueba de Integración End-to-End de la Arquitectura Celular:
Valida el enrutamiento autónomo de Fact Band y la ingesta markoviana de 10 KB.
"""

import sys, os, time
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))
sys.path.insert(0, os.path.abspath("tools"))

from mlx_vlm import load, stream_generate
from aether_vlm.coupler import AetherEngine
from markov_context_engine import MarkovContextEngine
import aether_native_c

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

def main():
    print("=" * 78)
    print(" 🌌 BATERÍA DE INTEGRACIÓN CELULAR END-TO-END (NIVEL 1 & NIVEL 2)")
    print("=" * 78)

    print("\n[1/4] Cargando modelo base Qwen3.5-0.8B...")
    model, processor = load(MODEL_PATH)

    print("[2/4] Conectando AetherEngine (Tejido Celular Activo)...")
    engine = AetherEngine(model, processor)
    print(f"✓ Motor conectado en perfil: [{engine.profile_name}]")

    print("\n[3/4] Creando documento sintético de prueba...")
    # Cada chunk tiene ~400 tokens para encajar limpiamente en ventanas separadas
    doc_chunk_0 = "La física de campos continuos en variedades de Riemann describe el movimiento de partículas sin saltos discretos en el espacio latente. " * 20
    doc_chunk_1 = "En 1969 la misión Apolo 11 alunizó en el Mar de la Tranquilidad. John Coyle ganó el campeonato de avena consumiendo 23 tazones en Corby. Neil Armstrong fue el primer astronauta en pisar la Luna. " * 10
    doc_chunk_2 = "La arquitectura de transformadores tradicionales sufre de acantilado de atención en contextos largos más allá de mil tokens por dilución de softmax. " * 20
    
    full_doc = doc_chunk_0 + "\n\n" + doc_chunk_1 + "\n\n" + doc_chunk_2

    markov_engine = MarkovContextEngine(model, processor, dimension=engine.hidden_dim)
    markov_engine.ingest_document(full_doc, chunk_size=512)

    print("\n[4/4] Consultando con el Enrutador Híbrido...")
    query = "¿Quién ganó el campeonato de comer avena y cuántos tazones consumió?"
    best_slot, score, geo_res, lex_res = markov_engine.query_resonance_window(query)
    
    print(f"\n  Query: \"{query}\"")
    print(f"  Puntuaciones por ventana:")
    for w in range(len(geo_res)):
        marker = " ◄ [MEJOR ENLACE GEODÉSICO]" if w == best_slot else ""
        print(f"    • Ventana {w:02d}: Geodésica = {geo_res[w]:+.4f} | Léxica = {lex_res[w]:.2f}{marker}")

    print(f"\n✓ Ventana seleccionada: {best_slot} (Score Híbrido: {score:.4f})")
    
    # Verificación estricta: la ventana seleccionada debe contener a John Coyle
    recovered_text = markov_engine.get_window_text(best_slot)
    assert "John Coyle" in recovered_text, f"ERROR: La ventana {best_slot} no contiene el hecho de John Coyle"
    print("  [✅ PASS] Localización exacta de la ventana fáctica sin KV-Cache.")

    # Generación condicionada sobre la ventana recuperada
    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": [
            {"type": "text", "text": f"Contexto: {recovered_text[:600]}\n\nPregunta: {query}\nResponde en una frase concisa."}
        ]}
    ], add_generation_prompt=True)

    print("\n[5/5] Generando respuesta con AetherEngine (Fact Band activa)...")
    engine.prepare_thought(query)
    print(f"  • Fact Band detectada en capa: L* = {engine.state['peak_layer']}")
    
    print("  • Salida:")
    resp_tokens = []
    for r in stream_generate(model, processor, prompt=prompt_chat, max_tokens=40):
        resp_tokens.append(r.text)
        print(r.text, end="", flush=True)
    print("\n")

    full_resp = "".join(resp_tokens)
    assert "John Coyle" in full_resp or "23" in full_resp, "La respuesta no contiene el hecho esperado"
    print("  [✅ PASS] Respuesta fáctica correcta generada con éxito.")

    print("\n" + "=" * 78)
    print(" 🏆 CASCADA END-TO-END COMPLETADA Y CERTIFICADA CON ÉXITO")
    print("=" * 78)

if __name__ == "__main__":
    main()
