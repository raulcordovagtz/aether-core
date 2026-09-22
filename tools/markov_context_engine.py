#!/usr/bin/env python3
"""
tools/markov_context_engine.py
==============================
Motor de Ingesta y Recuperación de Contexto Markoviano (Paradigma Apolo 11).
Comprime documentos masivos en vectores de frontera de 10 KB almacenados
en el anillo UMA de HilbertMemoryCell, sin mantener KV-cache en memoria.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import mlx.core as mx
import numpy as np
import aether_native_c

class MarkovContextEngine:
    def __init__(self, model, processor, dimension=1024):
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.dimension = dimension
        self.windows = []       # Almacena los token_ids de cada ventana
        self.window_texts = []  # Texto decodificado para índice léxico
        self.num_windows = 0
        aether_native_c.hilbert_memory_reset()

    def ingest_document(self, text: str, chunk_size: int = 512):
        """
        Divide el documento en ventanas de chunk_size tokens, ejecuta un forward limpio
        por ventana, intercepta el vector de frontera residual pre-norm (10 KB) y lo almacena en UMA.
        """
        print(f"[*] Ingestando documento en ventanas de {chunk_size} tokens...")
        token_ids = self.tokenizer.encode(text)
        total_tokens = len(token_ids)
        
        aether_native_c.hilbert_memory_reset()
        self.windows = []
        self.window_texts = []
        
        num_chunks = (total_tokens + chunk_size - 1) // chunk_size
        print(f"    Total tokens: {total_tokens} | Ventanas a generar: {num_chunks}")

        lm_model = self.model.language_model.model
        orig_norm = lm_model.norm

        captured_h = None
        class NormTap:
            def __init__(self, orig):
                self.orig = orig
            def __getattr__(self, name):
                return getattr(self.orig, name)
            def __call__(self, x, **kwargs):
                nonlocal captured_h
                captured_h = x[0, -1, :].astype(mx.float32)
                return self.orig(x, **kwargs)

        lm_model.norm = NormTap(orig_norm)

        try:
            for w in range(num_chunks):
                start = w * chunk_size
                end = min(start + chunk_size, total_tokens)
                w_tokens = token_ids[start:end]
                self.windows.append(w_tokens)
                self.window_texts.append(self.tokenizer.decode(w_tokens))

                input_ids = mx.array(w_tokens)[None, :]
                _ = self.model.language_model(input_ids)
                
                mx.eval(captured_h)
                norm_h = mx.sqrt(mx.sum(captured_h * captured_h)) + 1e-12
                h_unit = captured_h / norm_h
                mx.eval(h_unit)

                # Ingestar el vector de frontera de 10 KB en la memoria de Hilbert
                slot_info = aether_native_c.hilbert_memory_ingest(h_unit, timestamp=w, energy=1.0)
                print(f"    -> Ventana {w:02d} [{start:4d}:{end:4d}] ingestada en Slot {slot_info['slot_idx']} (Norma: 1.000000)")

        finally:
            lm_model.norm = orig_norm

        self.num_windows = len(self.windows)
        print(f"✓ Documento comprimido: {self.num_windows} vectores de frontera ({self.num_windows * 10} KB en UMA). Cero KV-cache.")

    def query_resonance_window(self, query_text: str):
        """
        Ruteo Híbrido Markoviano (Paradigma Apolo 11):
        Combina la resonancia geodésica del tensor de consulta en Capa 24
        con el filtrado de coincidencia léxica para un ruteo instantáneo en < 0.5 ms.
        """
        q_ids = self.tokenizer.encode(query_text)
        q_tensor = mx.array(q_ids)[None, :]

        # 1. Proyectar la consulta a la misma variedad (Capa 24 pre-norm)
        lm_model = self.model.language_model.model
        orig_norm = lm_model.norm
        captured_q = None

        class QueryTap:
            def __init__(self, orig):
                self.orig = orig
            def __getattr__(self, name):
                return getattr(self.orig, name)
            def __call__(self, x, **kwargs):
                nonlocal captured_q
                captured_q = x[0, -1, :].astype(mx.float32)
                return self.orig(x, **kwargs)

        lm_model.norm = QueryTap(orig_norm)
        try:
            _ = self.model.language_model(q_tensor)
        finally:
            lm_model.norm = orig_norm

        mx.eval(captured_q)
        norm_q = mx.sqrt(mx.sum(captured_q * captured_q)) + 1e-12
        u_query = (captured_q / norm_q).astype(mx.float32)
        mx.eval(u_query)

        # 2. Resonancia geodésica en memoria de Hilbert
        res = aether_native_c.hilbert_memory_query_resonance(u_query)
        resonances = np.array(res["resonances"])

        # 3. Ponderación léxica de palabras clave (filtro de Chris en Apolo 11)
        query_words = [w.lower() for w in query_text.split() if len(w) > 3]
        lexical_scores = np.zeros(self.num_windows)
        for w_idx, txt in enumerate(self.window_texts):
            txt_lower = txt.lower()
            matches = sum(1 for qw in query_words if qw in txt_lower)
            lexical_scores[w_idx] = matches / max(1, len(query_words))

        # Puntuación combinada (Geodésica + Léxica)
        hybrid_scores = resonances + 2.0 * lexical_scores
        best_slot = int(np.argmax(hybrid_scores))
        best_score = float(hybrid_scores[best_slot])

        return best_slot, best_score, resonances, lexical_scores

    def get_window_text(self, window_idx: int) -> str:
        if 0 <= window_idx < len(self.window_texts):
            return self.window_texts[window_idx]
        return ""
