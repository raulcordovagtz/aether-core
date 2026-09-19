import os, sys

# 1. Restaurar base limpia
os.system("cp src/main_aether.mm.bak_c007 src/main_aether.mm")

with open("src/main_aether.mm", "r") as f:
    code = f.read()

# Incluir encabezado modular
anchor_inc = '#include <iomanip>'
code = code.replace(anchor_inc, anchor_inc + '\n#include "aether_c008_dispatch.h"')

# 2. Cargar bibliotecas Metal oficiales
anchor_lib = 'id<MTLLibrary> libFused = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_fused_norm_gemv.metallib"] error:&err];'
inject_libs = anchor_lib + '\n' + \
"""        id<MTLLibrary> libSpinorC008 = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_c008_cognitive_engine.metallib"] error:&err];
        if (!libSpinorC008) { std::cerr << "❌ Error cargando libSpinorC008\\n"; return 1; }
        id<MTLComputePipelineState> psoC008Red  = [device newComputePipelineStateWithFunction:[libSpinorC008 newFunctionWithName:@"c008_project_reductions"] error:&err];
        id<MTLComputePipelineState> psoC008Step = [device newComputePipelineStateWithFunction:[libSpinorC008 newFunctionWithName:@"c008_cognitive_step"] error:&err];
        id<MTLLibrary> libMambaInFused = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_fused_mamba_in.metallib"] error:&err];
        if (!libMambaInFused) { std::cerr << "❌ Error cargando libMambaInFused\\n"; return 1; }
        id<MTLComputePipelineState> psoMambaIn4Fused = [device newComputePipelineStateWithFunction:[libMambaInFused newFunctionWithName:@"gemv_4bit_mamba_in4_fused"] error:&err];
        id<MTLLibrary> libConvGatedFused = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_fused_conv_gated.metallib"] error:&err];
        if (!libConvGatedFused) { std::cerr << "❌ Error cargando libConvGatedFused\\n"; return 1; }
        id<MTLComputePipelineState> psoConvGatedFused = [device newComputePipelineStateWithFunction:[libConvGatedFused newFunctionWithName:@"ssm_conv_gated_delta_fused"] error:&err];"""
assert anchor_lib in code
code = code.replace(anchor_lib, inject_libs, 1)

# Enlazar psoSwiglu a swiglu_dense_4bit_forward
old_swi = 'id<MTLComputePipelineState> psoSwiglu   = [device newComputePipelineStateWithFunction:[libCert newFunctionWithName:@"eml_swiglu_certified_forward"] error:&err];'
new_swi = 'id<MTLComputePipelineState> psoSwiglu   = [device newComputePipelineStateWithFunction:[libBase newFunctionWithName:@"swiglu_dense_4bit_forward"] error:&err];'
code = code.replace(old_swi, new_swi, 1)

# 3. Búferes globales y carga de fotones PREVIO a prefill
anchor_buf = 'id<MTLBuffer> bufEOS = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];'
inject_bufs = anchor_buf + '\n\n' + \
"""        // Búferes del Biespinor Multimodal C-008 (2D = 10240, r = 32)
        const uint32_t R_C007 = 32;
        const uint32_t TWO_D_C007 = 2 * D;
        id<MTLBuffer> bufPhiC007 = [device newBufferWithLength:TWO_D_C007 * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPhiMidC007 = [device newBufferWithLength:TWO_D_C007 * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bufRProjC007 = [device newBufferWithLength:(6 * R_C007 + 2) * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bufUcC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufVcC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufUsC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufVsC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufUlC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufVlC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufVisualPatches = nil;
        uint32_t num_visual_patches = 0;

        // Cargar evidencia visual fáctica ANTES del prefill
        {
            std::ifstream vf("visual_embeddings.bin", std::ios::binary);
            if (vf.is_open()) {
                vf.seekg(0, std::ios::end);
                size_t sz = vf.tellg();
                vf.seekg(0, std::ios::beg);
                std::vector<float> vdata(sz / sizeof(float));
                vf.read(reinterpret_cast<char*>(vdata.data()), sz);
                num_visual_patches = uint32_t(vdata.size() / D);
                bufVisualPatches = [device newBufferWithBytes:vdata.data() length:sz options:MTLResourceStorageModeShared];
                std::cout << " ✓ Evidencia visual fáctica vinculada: " << num_visual_patches << " parches (previo a prefill).\\n";
            }
        }"""
assert anchor_buf in code
code = code.replace(anchor_buf, inject_bufs, 1)

# 4. Inyección en dispatch_layer_forward (Mamba Fused 4-in-1 + ConvGated Fused)
old_mamba_block = """                [enc setComputePipelineState:psoGemv4];
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
                dispatch_simd_gemv(enc, BA_DIM);

                [enc setComputePipelineState:psoConv];
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

new_mamba_block = """                // FUSIÓN ATÓMICA DE SILICIO (4-EN-1): QKV + Z + B + A
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
                dispatch_simd_gemv(enc, 16480);

                // FUSIÓN ATÓMICA DE SILICIO: Conv1D + Gated Delta en 1 solo despacho
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
assert old_mamba_block in code
code = code.replace(old_mamba_block, new_mamba_block, 1)

# 5. Prefill con inyección fáctica
old_prefill_loop = """        for (size_t pos = 0; pos < prompt.size(); ++pos) {
            uint32_t tid = prompt[pos];
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:psoEmbed];
            [enc setBytes:&tid length:sizeof(uint32_t) atIndex:0];
            [enc setBuffer:bufEmbedW offset:0 atIndex:1];
            [enc setBuffer:bufEmbedS offset:0 atIndex:2];
            [enc setBuffer:bufEmbedB offset:0 atIndex:3];
            [enc setBuffer:bufZ offset:0 atIndex:4];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];"""

new_prefill_loop = """        uint32_t visual_patch_cursor = 0;
        for (size_t pos = 0; pos < prompt.size(); ++pos) {
            uint32_t tid = prompt[pos];
            if (tid == 248056 && bufVisualPatches != nil && visual_patch_cursor < num_visual_patches) {
                float* z_ptr = (float*)[bufZ contents];
                const float* p_ptr = (const float*)[bufVisualPatches contents] + (visual_patch_cursor * D);
                std::memcpy(z_ptr, p_ptr, D * sizeof(float));
                visual_patch_cursor++;
            } else {
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:psoEmbed];
                [enc setBytes:&tid length:sizeof(uint32_t) atIndex:0];
                [enc setBuffer:bufEmbedW offset:0 atIndex:1];
                [enc setBuffer:bufEmbedS offset:0 atIndex:2];
                [enc setBuffer:bufEmbedB offset:0 atIndex:3];
                [enc setBuffer:bufZ offset:0 atIndex:4];
                [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
                [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }"""
assert old_prefill_loop in code
code = code.replace(old_prefill_loop, new_prefill_loop, 1)

old_snap = """            float* z_curr = (float*)[bufZ contents];
            std::vector<float> snap(z_curr, z_curr + D);
            prefill_snaps.push_back(snap);"""
new_snap = """            if (tid != 248056) {
                float* z_curr = (float*)[bufZ contents];
                std::vector<float> snap(z_curr, z_curr + D);
                prefill_snaps.push_back(snap);
            }"""
assert old_snap in code
code = code.replace(old_snap, new_snap, 1)

# 6. Pensamiento Profundo C-008 con Meta Ontológica u_O_text
anchor_prefill_tps = "float prefill_tps = float(prompt.size()) / float(prefill_sec);"
thought_block = anchor_prefill_tps + "\n\n" + \
"""        // 🧠 FASE DE PENSAMIENTO PROFUNDO C-008 (Meta Ontológica Semántica u_O)
        if (bufVisualPatches != nil && num_visual_patches > 0) {
            float* p_raw = (float*)[bufVisualPatches contents];
            float* uc_ptr = (float*)[bufUcC007 contents];
            float* vc_ptr = (float*)[bufVcC007 contents];
            float* us_ptr = (float*)[bufUsC007 contents];
            float* vs_ptr = (float*)[bufVsC007 contents];
            float* ul_ptr = (float*)[bufUlC007 contents];
            float* vl_ptr = (float*)[bufVlC007 contents];
            float* phi_thought = (float*)[bufPhiC007 contents];

            std::vector<float> u_O_vis(D, 0.0f);
            for (uint32_t p = 0; p < num_visual_patches; ++p) {
                for (uint32_t i = 0; i < D; ++i) u_O_vis[i] += p_raw[p * D + i];
            }
            float inv_p = 1.0f / float(num_visual_patches);
            float n_ov = 0.0f;
            for (uint32_t i = 0; i < D; ++i) { u_O_vis[i] *= inv_p; n_ov += u_O_vis[i] * u_O_vis[i]; }
            n_ov = std::sqrt(n_ov) + 1e-12f;
            for (uint32_t i = 0; i < D; ++i) { u_O_vis[i] /= n_ov; phi_thought[i] = u_O_vis[i]; }

            float* u_O_text = (float*)[bufOntology contents];
            float* u_T_text = (float*)[bufTeleology contents];

            for (uint32_t i = 0; i < D; ++i) {
                uc_ptr[0 * D + i] = u_T_text[i];
                vc_ptr[0 * D + i] = u_O_vis[i];
                us_ptr[0 * D + i] = u_O_vis[i];
                vs_ptr[0 * D + i] = u_O_vis[i];
                ul_ptr[0 * D + i] = u_O_text[i]; // u_O_text en Ul
                vl_ptr[0 * D + i] = u_T_text[i]; // u_T_text en Vl
            }

            float* z_prefill_last = (float*)[bufZ contents];
            for (uint32_t i = 0; i < D; ++i) phi_thought[D + i] = z_prefill_last[i];

            std::cout << " • Ejecutando Pensamiento Profundo C-008 en GPU (τ* = " << C008EngineConfig::PREFILL_STEPS << " pasos de convergencia)...\\n";
            auto t_th_0 = std::chrono::high_resolution_clock::now();

            id<MTLCommandBuffer> cmdThought = [queue commandBuffer];
            for (uint32_t tau = 0; tau < C008EngineConfig::PREFILL_STEPS; ++tau) {
                float lie_decay = std::exp(-float(tau) / C008EngineConfig::TAU_RELAX);
                // bufUlC007 contiene u_O_text (atractor semántico)
                aether_c008_step_dispatch(cmdThought, psoC008Red, psoC008Step, bufPhiC007, bufPhiMidC007, bufRProjC007,
                                          bufUcC007, bufVcC007, bufUsC007, bufVsC007, bufUlC007, bufUlC007,
                                          C008EngineConfig::DT, lie_decay);
            }
            [cmdThought commit];
            [cmdThought waitUntilCompleted];
            auto t_th_1 = std::chrono::high_resolution_clock::now();
            double thought_ms = std::chrono::duration<double, std::milli>(t_th_1 - t_th_0).count();

            for (uint32_t i = 0; i < D; ++i) z_prefill_last[i] = phi_thought[D + i];
            std::cout << " ✓ Asentamiento alcanzado en " << std::fixed << std::setprecision(2) << thought_ms << " ms. Estado óptimo Φ* proyectado a Decode.\\n";
        }"""
assert anchor_prefill_tps in code
code = code.replace(anchor_prefill_tps, thought_block, 1)

# 7. Decode modular suave
old_decode_steer = """            // 4. Integrador Geodésico Soberano al final de las 64 capas
            auto t_p6 = std::chrono::high_resolution_clock::now();
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

new_decode_steer = """            // 4. Integrador Multimodal C-008 (Decode modular derivado de YAML)
            auto t_p6 = std::chrono::high_resolution_clock::now();
            if (bufVisualPatches != nil && num_visual_patches > 0) {
                float* phi_thought = (float*)[bufPhiC007 contents];
                float* z_raw = (float*)[bufZ contents];
                for (uint32_t i = 0; i < D; ++i) phi_thought[D + i] = z_raw[i];

                id<MTLCommandBuffer> cmdC008 = [queue commandBuffer];
                for (uint32_t s = 0; s < C008EngineConfig::DECODE_STEPS; ++s) {
                    aether_c008_step_dispatch(cmdC008, psoC008Red, psoC008Step, bufPhiC007, bufPhiMidC007, bufRProjC007,
                                              bufUcC007, bufVcC007, bufUsC007, bufVsC007, bufUlC007, bufUlC007,
                                              C008EngineConfig::DT, 0.05f);
                }
                [cmdC008 commit];
                [cmdC008 waitUntilCompleted];
                for (uint32_t i = 0; i < D; ++i) z_raw[i] = phi_thought[D + i];
            } else {
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
            }"""
assert old_decode_steer in code
code = code.replace(old_decode_steer, new_decode_steer, 1)

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ src/main_aether.mm integrado limpiamente con atractor semántico u_O.")
