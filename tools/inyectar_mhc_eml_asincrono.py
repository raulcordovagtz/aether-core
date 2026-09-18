with open("src/main_aether.mm", "r") as f:
    code = f.read()

# 1. Cargar la nueva biblioteca transpilada
code = code.replace(
    'id<MTLLibrary> libGeodesic = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_geodesic_engine.metallib"] error:&err];',
    'id<MTLLibrary> libGeodesic = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_mhc_eml_engine.metallib"] error:&err];'
)

code = code.replace(
    'psoSteering = [device newComputePipelineStateWithFunction:[libGeodesic newFunctionWithName:@"aether_continuous_steering_step"] error:&err];',
    'psoSteering = [device newComputePipelineStateWithFunction:[libGeodesic newFunctionWithName:@"aether_mhc_eml_geodesic_step"] error:&err];'
)

# 2. Pipelining asíncrono completo: 1 solo command buffer por token para las 64 capas + integrador
old_step_block = """            id<MTLCommandBuffer> cmdStep = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encStep = [cmdStep computeCommandEncoder];
            [encStep setComputePipelineState:psoEmbed];
            [encStep setBytes:&next_tok length:sizeof(uint32_t) atIndex:0];
            [encStep setBuffer:bufEmbedW offset:0 atIndex:1];
            [encStep setBuffer:bufEmbedS offset:0 atIndex:2];
            [encStep setBuffer:bufEmbedB offset:0 atIndex:3];
            [encStep setBuffer:bufZ offset:0 atIndex:4];
            [encStep setBytes:&D length:sizeof(uint32_t) atIndex:5];
            [encStep dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encStep endEncoding];
            [cmdStep commit];
            [cmdStep waitUntilCompleted];

            // Propagación en las 64 capas
            for (uint32_t mb = 0; mb < 16; ++mb) {
                uint32_t base_layer = mb * 4;
                id<MTLCommandBuffer> cmdL = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encL = [cmdL computeCommandEncoder];
                for (uint32_t sub = 0; sub < 4; ++sub) {
                    dispatch_layer_forward(encL, base_layer + sub, current_pos);
                }
                [encL endEncoding];
                [cmdL commit];
                [cmdL waitUntilCompleted];
            }

            // Integración de la Ecuación Maestra Geodésica al final de cada paso
            id<MTLCommandBuffer> cmdSteer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encSteer = [cmdSteer computeCommandEncoder];
            [encSteer setComputePipelineState:psoSteering];
            [encSteer setBuffer:bufZ offset:0 atIndex:0];
            [encSteer setBuffer:bufV offset:0 atIndex:1];
            [encSteer setBuffer:bufOntology offset:0 atIndex:2];
            [encSteer setBuffer:bufTeleology offset:0 atIndex:3];
            [encSteer setBuffer:bufAntithesis offset:0 atIndex:4];
            [encSteer setBuffer:bufEOS offset:0 atIndex:5];
            [encSteer setBytes:&dt length:sizeof(float) atIndex:6];
            [encSteer setBytes:&sigma_gyro length:sizeof(float) atIndex:7];
            [encSteer setBytes:&omega_dial length:sizeof(float) atIndex:8];
            [encSteer setBytes:&gamma_rayleigh length:sizeof(float) atIndex:9];
            [encSteer setBytes:&k_pozo length:sizeof(float) atIndex:10];
            [encSteer setBytes:&D length:sizeof(uint32_t) atIndex:11];
            [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:0];
            [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:1];
            [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:2];
            [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:3];
            [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:4];
            [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:5];
            [encSteer dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
            [encSteer endEncoding];
            [cmdSteer commit];
            [cmdSteer waitUntilCompleted];"""

new_step_block = """            // ─── PIPELINING ASÍNCRONO FUSIONADO (1 SOLO COMMAND BUFFER) ───
            id<MTLCommandBuffer> cmdStep = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdStep computeCommandEncoder];

            // 1. Inyección de Embedding
            [enc setComputePipelineState:psoEmbed];
            [enc setBytes:&next_tok length:sizeof(uint32_t) atIndex:0];
            [enc setBuffer:bufEmbedW offset:0 atIndex:1];
            [enc setBuffer:bufEmbedS offset:0 atIndex:2];
            [enc setBuffer:bufEmbedB offset:0 atIndex:3];
            [enc setBuffer:bufZ offset:0 atIndex:4];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            // 2. Propagación continua de las 64 capas en GPU sin retorno a CPU
            for (uint32_t l = 0; l < NUM_LAYERS; ++l) {
                dispatch_layer_forward(enc, l, current_pos);
            }

            // 3. Integrador Geodésico mHC+EML in-situ
            [enc setComputePipelineState:psoSteering];
            [enc setBuffer:bufZ offset:0 atIndex:0];
            [enc setBuffer:bufV offset:0 atIndex:1];
            [enc setBuffer:bufOntology offset:0 atIndex:2];
            [enc setBuffer:bufTeleology offset:0 atIndex:3];
            [enc setBuffer:bufAntithesis offset:0 atIndex:4];
            [enc setBuffer:bufEOS offset:0 atIndex:5];
            [enc setBytes:&dt length:sizeof(float) atIndex:6];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:7];
            [enc setThreadgroupMemoryLength:512 * sizeof(float) atIndex:0];
            [enc setThreadgroupMemoryLength:512 * sizeof(float) atIndex:1];
            [enc setThreadgroupMemoryLength:512 * sizeof(float) atIndex:2];
            [enc setThreadgroupMemoryLength:512 * sizeof(float) atIndex:3];
            [enc setThreadgroupMemoryLength:512 * sizeof(float) atIndex:4];
            [enc setThreadgroupMemoryLength:512 * sizeof(float) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

            [enc endEncoding];
            [cmdStep commit];
            [cmdStep waitUntilCompleted]; // Única sincronización por token"""

code = code.replace(old_step_block, new_step_block)

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ Pipeline asíncrono mHC+EML inyectado en src/main_aether.mm.")
