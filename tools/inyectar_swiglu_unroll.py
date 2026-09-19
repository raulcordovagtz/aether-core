with open("metal/c_field_qwen38_engine.metal", "r") as f:
    code = f.read()

old_loop = """        for (uint b_idx = 0; b_idx < 8; ++b_idx) {
            float z_val = Z_Norm[base_k + b_idx];
            uint g_val = (gu >> (b_idx * 4)) & 0x0F;
            uint u_val = (uu >> (b_idx * 4)) & 0x0F;
            acc_g += z_val * (float(g_val) * g_s + g_b);
            acc_u += z_val * (float(u_val) * u_s + u_b);
        }"""

new_unroll = """        // 🚀 SIMD UNROLLED DIRECTO (CERO SALTOS DE BUCLE)
        #define FMA8(off, shift) \\
            { float z = Z_Norm[base_k + off]; \\
              acc_g += z * (float((gu >> shift) & 0x0F) * g_s + g_b); \\
              acc_u += z * (float((uu >> shift) & 0x0F) * u_s + u_b); }
        FMA8(0, 0);  FMA8(1, 4);  FMA8(2, 8);  FMA8(3, 12);
        FMA8(4, 16); FMA8(5, 20); FMA8(6, 24); FMA8(7, 28);
        #undef FMA8"""

assert old_loop in code, "Fallo al localizar el bucle SwiGLU en c_field_qwen38_engine.metal"
code = code.replace(old_loop, new_unroll, 1)

with open("metal/c_field_qwen38_engine.metal", "w") as f:
    f.write(code)

print("✓ metal/c_field_qwen38_engine.metal actualizado con desenrollado SIMD FMA.")
