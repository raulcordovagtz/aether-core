#!/usr/bin/env python3
"""
tools/inspeccionar_volcado.py
"""
import sys, os

FILE_PATH = "/Users/crotalo/desarrollo-local/Volcado de solicitudes.txt"
MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

if not os.path.exists(FILE_PATH):
    print(f"❌ Error: No se encontró el archivo en {FILE_PATH}")
    sys.exit(1)

from mlx_vlm import load

print("=" * 78)
print(f"  INSPECCIÓN DE ARCHIVO INÉDITO PARA ALMACÉN MARKOVIANO")
print("=" * 78)

file_bytes = os.path.getsize(FILE_PATH)
print(f"• Tamaño físico en disco : {file_bytes:,} bytes ({file_bytes / (1024*1024):.2f} MB)")

print("• Cargando tokenizador de Qwen...")
_, processor = load(MODEL_PATH)
tok = getattr(processor, "tokenizer", processor)

print("• Leyendo y tokenizando documento...")
with open(FILE_PATH, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

tokens = tok.encode(text)
total_tokens = len(tokens)
WINDOW_SIZE = 512
num_windows = (total_tokens + WINDOW_SIZE - 1) // WINDOW_SIZE

print(f"• Total de tokens reales : {total_tokens:,} tokens")
print(f"• Tamaño de ventana      : {WINDOW_SIZE} tokens")
print(f"• Total de ventanas      : {num_windows} ventanas")

# Cálculo de almacenamiento residual
D_dim = 1024 # Qwen 0.8B
bytes_per_vector = D_dim * 4 # float32
total_residual_store_mb = (num_windows * bytes_per_vector) / (1024 * 1024)

print(f"\n[GEOMETRÍA DEL ALMACÉN MARKOVIANO]:")
print(f"  - Tamaño de cada vector de frontera : {bytes_per_vector:,} bytes (~4 KB)")
print(f"  - Tamaño TOTAL del mapa residual   : {total_residual_store_mb:.2f} MB")
print(f"  - Reducción frente a KV-Cache (56GB): ~{56000 / total_residual_store_mb:.0f}x de compresión física")
print("=" * 78)