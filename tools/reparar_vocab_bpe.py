#!/usr/bin/env python3
"""
tools/reparar_vocab_bpe.py
═══════════════════════════════════════════════════════════════════════════════
REPARADOR CANÓNICO DE VOCABULARIO BPE BYTE-LEVEL PARA AETHER ENGINE
═══════════════════════════════════════════════════════════════════════════════
Corrige la codificación de los 93 tokens monobyte en el rango 0x80-0xFF
(prefijos y continuaciones UTF-8 para vocales acentuadas, 'ñ', diéresis y signos)
eliminando la doble codificación UTF-8 (mojibake) y garantizando que cada token
emita sus bytes exactos según la especificación formal de ByteLevel BPE.
"""

import os
import struct
from tokenizers import Tokenizer

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def repair_vocab():
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    
    model_dir = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit")
    tok_path = os.path.join(model_dir, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    
    # Leer el vocabulario base de vocab.bin
    old_vocab = {}
    with open("vocab.bin", "rb") as f:
        total = struct.unpack("I", f.read(4))[0]
        for _ in range(total):
            t_id, length = struct.unpack("II", f.read(8))
            word = f.read(length) if length > 0 else b""
            old_vocab[t_id] = word
            
    print(f"Tokens totales leídos de vocab.bin: {total}")
    
    # Reparar cada token utilizando el mapeo exacto de BPE a bytes
    repaired_vocab = {}
    corrupted_count = 0
    
    for t_id in range(total):
        tok_str = tokenizer.id_to_token(t_id)
        if tok_str is None:
            repaired_vocab[t_id] = old_vocab.get(t_id, b"")
            continue
            
        # Caso especial para tokens especiales de control
        if tok_str == "<|im_start|>":
            repaired_vocab[t_id] = b"<|im_start|>"
        elif tok_str == "<|im_end|>":
            repaired_vocab[t_id] = b"<|im_end|>"
        elif tok_str == "<think>":
            repaired_vocab[t_id] = b"<think>\n"
        elif tok_str == "</think>":
            repaired_vocab[t_id] = b"\n</think>\n\n"
        else:
            # Reconstruir bytes reales aplicando el byte_decoder de BPE
            true_bytes = bytearray()
            for char in tok_str:
                if char in byte_decoder:
                    true_bytes.append(byte_decoder[char])
                else:
                    true_bytes.extend(char.encode('utf-8'))
            
            raw_bytes = bytes(true_bytes)
            if old_vocab.get(t_id) != raw_bytes:
                corrupted_count += 1
            repaired_vocab[t_id] = raw_bytes
            
    print(f"✓ Tokens corregidos con mapeo de bytes estricto: {corrupted_count}")
    
    # Escribir el nuevo archivo vocab_repaired.bin
    target_path = "vocab.bin"
    # Si es symlink, romper el symlink y crear un archivo local soberano
    if os.path.islink(target_path):
        os.unlink(target_path)
        
    with open(target_path, "wb") as f:
        f.write(struct.pack("I", total))
        for t_id in range(total):
            b_word = repaired_vocab.get(t_id, b"")
            f.write(struct.pack("II", t_id, len(b_word)))
            if len(b_word) > 0:
                f.write(b_word)
                
    print(f"✓ Vocabulario reparado exitosamente guardado en {target_path}")

if __name__ == "__main__":
    repair_vocab()
