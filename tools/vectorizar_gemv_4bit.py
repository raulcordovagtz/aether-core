with open("metal/c_field_qwen38_engine.metal", "r") as f:
    code = f.read()

# Buscar el cuerpo del bucle interno b de 0 a 8
old_inner_loop = """            for (uint32_t b_idx = 0; b_idx < 8; ++b_idx) {
                uint32_t q = (u >> (b_idx * 4)) & 0x0F;
                float w_val = float(q) * s + b;
                acc += in[base_k + b_idx] * w_val;
            }"""

# Reemplazarlo por el desenrollado vectorial directo (eliminando 8 saltos de bucle por entero)
new_inner_loop = """            // 🚀 SIMD UNROLLED (DESENROLLADO VECTORIAL COMPLETO)
            acc += in[base_k + 0] * (float((u >> 0)  & 0x0F) * s + b);
            acc += in[base_k + 1] * (float((u >> 4)  & 0x0F) * s + b);
            acc += in[base_k + 2] * (float((u >> 8)  & 0x0F) * s + b);
            acc += in[base_k + 3] * (float((u >> 12) & 0x0F) * s + b);
            acc += in[base_k + 4] * (float((u >> 16) & 0x0F) * s + b);
            acc += in[base_k + 5] * (float((u >> 20) & 0x0F) * s + b);
            acc += in[base_k + 6] * (float((u >> 24) & 0x0F) * s + b);
            acc += in[base_k + 7] * (float((u >> 28) & 0x0F) * s + b);"""

if old_inner_loop in code:
    code = code.replace(old_inner_loop, new_inner_loop)
    with open("metal/c_field_qwen38_engine.metal", "w") as f:
        f.write(code)
    print("✓ Bucle interno de GEMV sustituido por desenrollado vectorial.")
else:
    print("• Buscando variante de sintaxis alternativa en el shader...")
    # Si la variable se llama diferente, hacer reemplazo regex
    import re
    code = re.sub(
        r'for\s*\(\s*uint32_t\s+b_idx\s*=\s*0;\s*b_idx\s*<\s*8;\s*\+\+b_idx\s*\)\s*\{[^}]*\}',
        new_inner_loop,
        code
    )
    with open("metal/c_field_qwen38_engine.metal", "w") as f:
        f.write(code)
    print("✓ Regex aplicada: bucle GEMV desenrollado con éxito.")
