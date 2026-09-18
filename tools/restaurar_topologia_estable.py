# Restaurar la estructura de despacho por macrobloques que garantiza coherencia matemática
code_replacement = """            // 1. Decodificación del token a partir de logits
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

            float* logits = (float*)[bufLogits contents];
            float best_l = -1e30f;
            uint32_t next_tok = 0;
            for (uint32_t idx = 0; idx < V; ++idx) {
                if (logits[idx] > best_l) {
                    best_l = logits[idx];
                    next_tok = idx;
                }
            }

            std::string word = "";
            if (vocab.count(next_tok) && !vocab[next_tok].empty()) {
                word = vocab[next_tok];
            } else {
                word = "[" + std::to_string(next_tok) + "]";
            }

            NSData *data = [NSData dataWithBytes:word.data() length:word.size()];
            [stdOut writeData:data];

            if (next_tok == ID_EOS || next_tok == 151643 || next_tok == 248046 ||
                word == "<|im_end|>" || word.find("<|im_end|>") != std::string::npos) {
                break;
            }

            // 2. Embedding del siguiente token
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

            // 3. Propagación por los 16 macro-bloques (1 GQA + 3 SSM por bloque)
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

            // 4. Integrador Geodésico Soberano al final de las 64 capas
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
            [cmdSteer waitUntilCompleted];"""

with open("src/main_aether.mm", "r") as f:
    full_code = f.read()

# Buscar el inicio del bucle de generación y reemplazar el cuerpo
start_idx = full_code.find("for (int step = 0; step < max_tokens; ++step) {")
end_idx = full_code.find("current_pos++;\n        }")

header = full_code[:start_idx + len("for (int step = 0; step < max_tokens; ++step) {\n            actual_generated_tokens++;\n")]
footer = full_code[end_idx:]

restored_code = header + code_replacement + "\n\n            " + footer

with open("src/main_aether.mm", "w") as f:
    f.write(restored_code)

print("✓ Topología causal restaurada a granularidad de macro-bloques estables.")
