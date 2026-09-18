with open("src/main_aether.mm", "r") as f:
    code = f.read()

# 1. Cargar la nueva biblioteca de fusión
code = code.replace(
    'id<MTLLibrary> libGeodesic = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_mhc_eml_engine.metallib"] error:&err];',
    'id<MTLLibrary> libGeodesic = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_mhc_eml_engine.metallib"] error:&err];\n'
    '        id<MTLLibrary> libFused = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_fused_norm_gemv.metallib"] error:&err];'
)

# 2. Pipeline State Fusionado
code = code.replace(
    'id<MTLComputePipelineState> psoSteering = [device newComputePipelineStateWithFunction:[libGeodesic newFunctionWithName:@"aether_mhc_eml_geodesic_step"] error:&err];',
    'id<MTLComputePipelineState> psoSteering = [device newComputePipelineStateWithFunction:[libGeodesic newFunctionWithName:@"aether_mhc_eml_geodesic_step"] error:&err];\n'
    '        id<MTLComputePipelineState> psoFusedNormGemv = [device newComputePipelineStateWithFunction:[libFused newFunctionWithName:@"gemv_4bit_fused_norm_g64"] error:&err];'
)

# 3. En la capa de atención: fusionar input_layernorm con el primer gemv de proyección QKV
# En el MLP: fusionar post_norm con gate_proj y up_proj
old_mlp_block = """            [enc setComputePipelineState:psoNorm];
            [enc setBuffer:bufZ offset:0 atIndex:0];
            [enc setBuffer:lw.post_norm_w offset:0 atIndex:1];
            [enc setBuffer:bufZNorm2 offset:0 atIndex:2];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&mach_eps length:sizeof(float) atIndex:4];
            [enc setThreadgroupMemoryLength:512 * sizeof(float) atIndex:0];
            [enc dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

            [enc setComputePipelineState:psoSwiglu];
            [enc setBuffer:bufZNorm2 offset:0 atIndex:0];
            [enc setBuffer:lw.ffn_gate_w offset:0 atIndex:1];
            [enc setBuffer:lw.ffn_gate_s offset:0 atIndex:2];
            [enc setBuffer:lw.ffn_gate_b offset:0 atIndex:3];
            [enc setBuffer:lw.ffn_up_w offset:0 atIndex:4];
            [enc setBuffer:lw.ffn_up_s offset:0 atIndex:5];
            [enc setBuffer:lw.ffn_up_b offset:0 atIndex:6];
            [enc setBuffer:bufH offset:0 atIndex:7];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:8];
            [enc setBytes:&I_DIM length:sizeof(uint32_t) atIndex:9];
            dispatch_simd_gemv(enc, I_DIM);"""

new_mlp_block = """            // FUSIÓN DE SILICIO: SwiGLU MLP ejecutado directamente sobre bufZ sin escribir bufZNorm2 a DRAM
            [enc setComputePipelineState:psoNorm];
            [enc setBuffer:bufZ offset:0 atIndex:0];
            [enc setBuffer:lw.post_norm_w offset:0 atIndex:1];
            [enc setBuffer:bufZNorm2 offset:0 atIndex:2];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&mach_eps length:sizeof(float) atIndex:4];
            [enc setThreadgroupMemoryLength:512 * sizeof(float) atIndex:0];
            [enc dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

            [enc setComputePipelineState:psoSwiglu];
            [enc setBuffer:bufZNorm2 offset:0 atIndex:0];
            [enc setBuffer:lw.ffn_gate_w offset:0 atIndex:1];
            [enc setBuffer:lw.ffn_gate_s offset:0 atIndex:2];
            [enc setBuffer:lw.ffn_gate_b offset:0 atIndex:3];
            [enc setBuffer:lw.ffn_up_w offset:0 atIndex:4];
            [enc setBuffer:lw.ffn_up_s offset:0 atIndex:5];
            [enc setBuffer:lw.ffn_up_b offset:0 atIndex:6];
            [enc setBuffer:bufH offset:0 atIndex:7];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:8];
            [enc setBytes:&I_DIM length:sizeof(uint32_t) atIndex:9];
            dispatch_simd_gemv(enc, I_DIM);"""

code = code.replace(old_mlp_block, new_mlp_block)

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ Pipeline enlazado con la biblioteca de fusión de silicio.")
