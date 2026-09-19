import re

with open("src/main_aether.mm", "r") as f:
    code = f.read()

# 1. Cargar biblioteca Metal y PSO
anchor_lib = 'id<MTLLibrary> libSpinorC007 = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_c007_spinor_integrator.metallib"] error:&err];'
inject_lib = anchor_lib + '\n' + \
'        id<MTLLibrary> libMambaInFused = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_fused_mamba_in.metallib"] error:&err];\n' + \
'        if (!libMambaInFused) { std::cerr << "❌ Error cargando libMambaInFused\\n"; return 1; }\n' + \
'        id<MTLComputePipelineState> psoMambaIn4Fused = [device newComputePipelineStateWithFunction:[libMambaInFused newFunctionWithName:@"gemv_4bit_mamba_in4_fused"] error:&err];'

assert anchor_lib in code
code = code.replace(anchor_lib, inject_lib, 1)

# 2. Sustituir los 4 despachos por el despacho fusionado en dispatch_layer_forward
old_mamba_in = """                [enc setComputePipelineState:psoGemv4];
                [enc setBuffer:bufZNorm1 offset:0 atIndex:0];
                [enc setBuffer:lw.ssm_qkv_w offset:0 atIndex:1];
                [enc setBuffer:lw.ssm_qkv_s offset:0 atIndex:2];
                [enc setBuffer:lw.ssm_qkv_b offset:0 atIndex:3];
                [enc setBuffer:bufQKV offset:0 atIndex:4];
                [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&QKV_DIM length:sizeof(uint32_t) atIndex:6];
                dispatch_simd_gemv(enc, QKV_DIM);

                [enc setBuffer:lw.ssm_z_w offset:0 atIndex:1];
                [enc setBuffer:lw.ssm_z_s offset:0 atIndex:2];
                [enc setBuffer:lw.ssm_z_b offset:0 atIndex:3];
                [enc setBuffer:bufZProj offset:0 atIndex:4];
                [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&Z_DIM length:sizeof(uint32_t) atIndex:6];
                dispatch_simd_gemv(enc, Z_DIM);

                [enc setBuffer:lw.ssm_b_w offset:0 atIndex:1];
                [enc setBuffer:lw.ssm_b_s offset:0 atIndex:2];
                [enc setBuffer:lw.ssm_b_b offset:0 atIndex:3];
                [enc setBuffer:bufB offset:0 atIndex:4];
                [enc setBytes:&BA_DIM length:sizeof(uint32_t) atIndex:6];
                dispatch_simd_gemv(enc, BA_DIM);

                [enc setBuffer:lw.ssm_a_w offset:0 atIndex:1];
                [enc setBuffer:lw.ssm_a_s offset:0 atIndex:2];
                [enc setBuffer:lw.ssm_a_b offset:0 atIndex:3];
                [enc setBuffer:bufA offset:0 atIndex:4];
                [enc setBytes:&BA_DIM length:sizeof(uint32_t) atIndex:6];
                dispatch_simd_gemv(enc, BA_DIM);"""

new_mamba_in = """                // FUSIÓN ATÓMICA DE SILICIO (4-EN-1): QKV (10240) + Z (6144) + B (48) + A (48) en 1 solo despacho
                [enc setComputePipelineState:psoMambaIn4Fused];
                [enc setBuffer:bufZNorm1 offset:0 atIndex:0];
                [enc setBuffer:lw.ssm_qkv_w offset:0 atIndex:1];
                [enc setBuffer:lw.ssm_qkv_s offset:0 atIndex:2];
                [enc setBuffer:lw.ssm_qkv_b offset:0 atIndex:3];
                [enc setBuffer:bufQKV offset:0 atIndex:4];
                [enc setBuffer:lw.ssm_z_w offset:0 atIndex:5];
                [enc setBuffer:lw.ssm_z_s offset:0 atIndex:6];
                [enc setBuffer:lw.ssm_z_b offset:0 atIndex:7];
                [enc setBuffer:bufZProj offset:0 atIndex:8];
                [enc setBuffer:lw.ssm_b_w offset:0 atIndex:9];
                [enc setBuffer:lw.ssm_b_s offset:0 atIndex:10];
                [enc setBuffer:lw.ssm_b_b offset:0 atIndex:11];
                [enc setBuffer:bufB offset:0 atIndex:12];
                [enc setBuffer:lw.ssm_a_w offset:0 atIndex:13];
                [enc setBuffer:lw.ssm_a_s offset:0 atIndex:14];
                [enc setBuffer:lw.ssm_a_b offset:0 atIndex:15];
                [enc setBuffer:bufA offset:0 atIndex:16];
                [enc setBytes:&D length:sizeof(uint32_t) atIndex:17];
                uint32_t total_mamba_in_rows = QKV_DIM + Z_DIM + BA_DIM + BA_DIM; // 16,480
                uint32_t num_tg_mamba_in = (total_mamba_in_rows + 3) / 4;          // 4,120
                [enc dispatchThreadgroups:MTLSizeMake(num_tg_mamba_in, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];"""

assert old_mamba_in in code, "Fallo al localizar el bloque de 4 despachos Mamba en src/main_aether.mm"
code = code.replace(old_mamba_in, new_mamba_in, 1)

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ src/main_aether.mm integrado con la fusión cuádruple de Mamba.")
