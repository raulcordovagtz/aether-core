with open("src/main_aether.mm", "r") as f:
    code = f.read()

# 1. En la proyección de atención: reemplazar psoNorm + psoGemv4 por psoFusedNormGemv en el LM Head y Capas
# Verificamos si psoFusedNormGemv está declarado
assert "psoFusedNormGemv" in code, "psoFusedNormGemv no está declarado en main_aether.mm"

# 2. Reemplazo real en el LM Head (Eliminar el psoNorm previo al vocabulario)
# En el LM Head: normaliza bufZ con bufFinalNorm y proyecta contra bufHeadW
old_head_dispatch = """            [encHead setComputePipelineState:psoNorm];
            [encHead setBuffer:bufZ offset:0 atIndex:0];
            [encHead setBuffer:bufFinalNorm offset:0 atIndex:1];
            [encHead setBuffer:bufZNorm1 offset:0 atIndex:2];
            [encHead setBytes:&D length:sizeof(uint32_t) atIndex:3];
            [encHead setBytes:&mach_eps length:sizeof(float) atIndex:4];
            [encHead setThreadgroupMemoryLength:512 * sizeof(float) atIndex:0];
            [encHead dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

            [encHead setComputePipelineState:psoGemv4];
            [encHead setBuffer:bufZNorm1 offset:0 atIndex:0];
            [encHead setBuffer:bufHeadW offset:0 atIndex:1];
            [encHead setBuffer:bufHeadS offset:0 atIndex:2];
            [encHead setBuffer:bufHeadB offset:0 atIndex:3];
            [encHead setBuffer:bufLogits offset:0 atIndex:4];
            [encHead setBytes:&D length:sizeof(uint32_t) atIndex:5];
            [encHead setBytes:&V length:sizeof(uint32_t) atIndex:6];
            dispatch_simd_gemv(encHead, V);"""

new_head_dispatch = """            // 🚀 FUSED LM-HEAD: RMSNorm + GEMV en 1 solo paso de silicio sin tocar DRAM
            [encHead setComputePipelineState:psoFusedNormGemv];
            [encHead setBuffer:bufZ offset:0 atIndex:0];
            [encHead setBuffer:bufFinalNorm offset:0 atIndex:1];
            [encHead setBuffer:bufHeadW offset:0 atIndex:2];
            [encHead setBuffer:bufHeadS offset:0 atIndex:3];
            [encHead setBuffer:bufHeadB offset:0 atIndex:4];
            [encHead setBuffer:bufLogits offset:0 atIndex:5];
            [encHead setBytes:&D length:sizeof(uint32_t) atIndex:6];
            [encHead setBytes:&V length:sizeof(uint32_t) atIndex:7];
            [encHead setBytes:&mach_eps length:sizeof(float) atIndex:8];
            [encHead setThreadgroupMemoryLength:128 * sizeof(float) atIndex:0];
            dispatch_simd_gemv(encHead, V);"""

code = code.replace(old_head_dispatch, new_head_dispatch)

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ LM Head sustituido por el kernel fusionado real (RMSNorm + GEMV en 1 solo despacho).")
