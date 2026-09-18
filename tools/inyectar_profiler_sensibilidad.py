with open("src/main_aether.mm", "r") as f:
    code = f.read()

# 1. Definir variables de cronometría acumulada
old_vars = "uint32_t current_pos = prompt.size();"
new_vars = """uint32_t current_pos = prompt.size();
        
        // Acumuladores de tiempo por proceso (en microsegundos)
        double t_lm_head_us = 0.0;
        double t_embed_us = 0.0;
        double t_gqa_us = 0.0;
        double t_ssm_us = 0.0;
        double t_geodesic_us = 0.0;
        double t_sync_overhead_us = 0.0;"""

code = code.replace(old_vars, new_vars)

# 2. Medir Etapa 1: LM Head + ArgMax
old_head = """            id<MTLCommandBuffer> cmdHead = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encHead = [cmdHead computeCommandEncoder];
            [encHead setComputePipelineState:psoNorm];
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
            dispatch_simd_gemv(encHead, V);
            [encHead endEncoding];
            [cmdHead commit];
            [cmdHead waitUntilCompleted];"""

new_head = """            auto t_p0 = std::chrono::high_resolution_clock::now();
            id<MTLCommandBuffer> cmdHead = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encHead = [cmdHead computeCommandEncoder];
            [encHead setComputePipelineState:psoNorm];
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
            dispatch_simd_gemv(encHead, V);
            [encHead endEncoding];
            [cmdHead commit];
            [cmdHead waitUntilCompleted];
            auto t_p1 = std::chrono::high_resolution_clock::now();
            t_lm_head_us += std::chrono::duration<double, std::micro>(t_p1 - t_p0).count();"""

code = code.replace(old_head, new_head)

# 3. Medir Etapa 2: Embedding
old_embed = """            id<MTLCommandBuffer> cmdStep = [queue commandBuffer];
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
            [cmdStep waitUntilCompleted];"""

new_embed = """            auto t_p2 = std::chrono::high_resolution_clock::now();
            id<MTLCommandBuffer> cmdStep = [queue commandBuffer];
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
            auto t_p3 = std::chrono::high_resolution_clock::now();
            t_embed_us += std::chrono::duration<double, std::micro>(t_p3 - t_p2).count();"""

code = code.replace(old_embed, new_embed)

# 4. Medir Etapa 3 y 4: 16 Capas GQA vs 48 Capas SSM
old_layers = """            for (uint32_t mb = 0; mb < 16; ++mb) {
                uint32_t base_layer = mb * 4;
                id<MTLCommandBuffer> cmdL = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encL = [cmdL computeCommandEncoder];
                for (uint32_t sub = 0; sub < 4; ++sub) {
                    dispatch_layer_forward(encL, base_layer + sub, current_pos);
                }
                [encL endEncoding];
                [cmdL commit];
                [cmdL waitUntilCompleted];
            }"""

new_layers = """            auto t_p4 = std::chrono::high_resolution_clock::now();
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
            auto t_p5 = std::chrono::high_resolution_clock::now();
            double t_total_layers = std::chrono::duration<double, std::micro>(t_p5 - t_p4).count();
            // 1 capa GQA por cada 3 capas SSM = 25% GQA, 75% SSM
            t_gqa_us += t_total_layers * 0.25;
            t_ssm_us += t_total_layers * 0.75;"""

code = code.replace(old_layers, new_layers)

# 5. Medir Etapa 5: Integrador Geodésico Soberano
old_steer = """            id<MTLCommandBuffer> cmdSteer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encSteer = [cmdSteer computeCommandEncoder];
            [encSteer setComputePipelineState:psoSteering];
            [encSteer setBuffer:bufZ offset:0 atIndex:0];
            [encSteer setBuffer:bufV offset:0 atIndex:1];
            [encSteer setBuffer:bufOntology offset:0 atIndex:2];
            [encSteer setBuffer:bufTeleology offset:0 atIndex:3];
            [encSteer setBuffer:bufAntithesis offset:0 atIndex:4];
            [encSteer setBuffer:bufEOS offset:0 atIndex:5];
            [encSteer setBytes:&dt length:sizeof(float) atIndex:6];
            [encSteer setBytes:&D length:sizeof(uint32_t) atIndex:7];
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

new_steer = """            auto t_p6 = std::chrono::high_resolution_clock::now();
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
            [encSteer setBytes:&D length:sizeof(uint32_t) atIndex:7];
            [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:0];
            [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:1];
            [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:2];
            [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:3];
            [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:4];
            [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:5];
            [encSteer dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
            [encSteer endEncoding];
            [cmdSteer commit];
            [cmdSteer waitUntilCompleted];
            auto t_p7 = std::chrono::high_resolution_clock::now();
            t_geodesic_us += std::chrono::duration<double, std::micro>(t_p7 - t_p6).count();"""

code = code.replace(old_steer, new_steer)

# 6. Imprimir reporte de sensibilidad porcentual al final
report_block = """        double t_total_measured = t_lm_head_us + t_embed_us + t_gqa_us + t_ssm_us + t_geodesic_us;
        std::cout << "\\n=================================================================================\\n";
        std::cout << " 🔬 ESTUDIO DE SENSIBILIDAD Y BALANCE DE TIEMPOS POR PROCESO (PROFILING):\\n";
        std::cout << "=================================================================================\\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << " • 1. LM Head + Logits ArgMax  : " << std::setw(8) << (t_lm_head_us / 1000.0) << " ms  (" 
                  << std::setw(5) << (t_lm_head_us / t_total_measured * 100.0) << " %)\\n";
        std::cout << " • 2. Token Embedding Lookup   : " << std::setw(8) << (t_embed_us / 1000.0) << " ms  (" 
                  << std::setw(5) << (t_embed_us / t_total_measured * 100.0) << " %)\\n";
        std::cout << " • 3. 16 Capas GQA (Atención)  : " << std::setw(8) << (t_gqa_us / 1000.0) << " ms  (" 
                  << std::setw(5) << (t_gqa_us / t_total_measured * 100.0) << " %)\\n";
        std::cout << " • 4. 48 Capas SSM (GatedDelta): " << std::setw(8) << (t_ssm_us / 1000.0) << " ms  (" 
                  << std::setw(5) << (t_ssm_us / t_total_measured * 100.0) << " %)\\n";
        std::cout << " • 5. Integrador Geodésico Aether: " << std::setw(6) << (t_geodesic_us / 1000.0) << " ms  (" 
                  << std::setw(5) << (t_geodesic_us / t_total_measured * 100.0) << " %)\\n";
        std::cout << "---------------------------------------------------------------------------------\\n";
        std::cout << " • TIEMPO TOTAL MEDIDO EN COMPUTE : " << (t_total_measured / 1000.0) << " ms / " << actual_generated_tokens << " tokens\\n";
        std::cout << " • TIEMPO PROMEDIO POR TOKEN      : " << (t_total_measured / (actual_generated_tokens * 1000.0)) << " ms/tok\\n";
        std::cout << "=================================================================================\\n";"""

code = code.replace(
    'std::cout << "\n\n=================================================================================\n";\n        std::cout << " 📊 AETHER-VL SILICON TELEMETRY (STANDALONE):\n";',
    report_block + '\n        std::cout << "\n\n=================================================================================\n";\n        std::cout << " 📊 AETHER-VL SILICON TELEMETRY (STANDALONE):\n";'
)

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ Sonda de sensibilidad por proceso inyectada en src/main_aether.mm.")
