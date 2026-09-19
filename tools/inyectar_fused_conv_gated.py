import re

with open("src/main_aether.mm", "r") as f:
    code = f.read()

# 1. Cargar biblioteca Metal y PSO
anchor_lib = 'id<MTLComputePipelineState> psoMambaIn4Fused = [device newComputePipelineStateWithFunction:[libMambaInFused newFunctionWithName:@"gemv_4bit_mamba_in4_fused"] error:&err];'
inject_lib = anchor_lib + '\n' + \
'        id<MTLLibrary> libConvGatedFused = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_fused_conv_gated.metallib"] error:&err];\n' + \
'        if (!libConvGatedFused) { std::cerr << "❌ Error cargando libConvGatedFused\\n"; return 1; }\n' + \
'        id<MTLComputePipelineState> psoConvGatedFused = [device newComputePipelineStateWithFunction:[libConvGatedFused newFunctionWithName:@"ssm_conv_gated_delta_fused"] error:&err];'

assert anchor_lib in code
code = code.replace(anchor_lib, inject_lib, 1)

# 2. Sustituir psoConv y psoGated por psoConvGatedFused
old_conv_gated = """                [enc setComputePipelineState:psoConv];
                [enc setBuffer:bufQKV offset:0 atIndex:0];
                [enc setBuffer:lw.conv_state offset:0 atIndex:1];
                [enc setBuffer:lw.ssm_conv_w offset:0 atIndex:2];
                [enc setBuffer:bufQKVConv offset:0 atIndex:3];
                [enc setBytes:&QKV_DIM length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&state_head length:sizeof(uint32_t) atIndex:5];
                [enc dispatchThreads:MTLSizeMake(QKV_DIM, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setComputePipelineState:psoGated];
                [enc setBuffer:bufQKVConv offset:0 atIndex:0];
                [enc setBuffer:bufZProj offset:0 atIndex:1];
                [enc setBuffer:bufB offset:0 atIndex:2];
                [enc setBuffer:bufA offset:0 atIndex:3];
                [enc setBuffer:lw.ssm_a_log offset:0 atIndex:4];
                [enc setBuffer:lw.ssm_dt_bias offset:0 atIndex:5];
                [enc setBuffer:lw.ssm_norm_w offset:0 atIndex:6];
                [enc setBuffer:lw.s_state offset:0 atIndex:7];
                [enc setBuffer:bufSSMOut offset:0 atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(48, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];"""

new_conv_gated = """                // FUSIÓN ATÓMICA DE SILICIO: Conv1D + Gated Delta en 1 solo despacho (bufQKVConv en registros)
                [enc setComputePipelineState:psoConvGatedFused];
                [enc setBuffer:bufQKV offset:0 atIndex:0];
                [enc setBuffer:lw.conv_state offset:0 atIndex:1];
                [enc setBuffer:lw.ssm_conv_w offset:0 atIndex:2];
                [enc setBytes:&state_head length:sizeof(uint32_t) atIndex:3];
                [enc setBuffer:bufZProj offset:0 atIndex:4];
                [enc setBuffer:bufB offset:0 atIndex:5];
                [enc setBuffer:bufA offset:0 atIndex:6];
                [enc setBuffer:lw.ssm_a_log offset:0 atIndex:7];
                [enc setBuffer:lw.ssm_dt_bias offset:0 atIndex:8];
                [enc setBuffer:lw.ssm_norm_w offset:0 atIndex:9];
                [enc setBuffer:lw.s_state offset:0 atIndex:10];
                [enc setBuffer:bufSSMOut offset:0 atIndex:11];
                [enc dispatchThreadgroups:MTLSizeMake(48, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];"""

assert old_conv_gated in code, "Fallo al localizar el bloque Conv+Gated en src/main_aether.mm"
code = code.replace(old_conv_gated, new_conv_gated, 1)

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ src/main_aether.mm integrado con la fusión Conv1D + Gated Delta.")
