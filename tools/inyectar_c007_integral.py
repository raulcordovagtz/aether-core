import sys

with open("src/main_aether.mm", "r") as f:
    code = f.read()

# ─── 1. CARGA DE PIPELINES METAL C-007 (L56) ──────────────────────────────────
anchor_lib = 'id<MTLLibrary> libFused = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_fused_norm_gemv.metallib"] error:&err];'
inject_lib = anchor_lib + '\n' + \
'        id<MTLLibrary> libSpinorC007 = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_c007_spinor_integrator.metallib"] error:&err];\n' + \
'        if (!libSpinorC007) { std::cerr << "❌ Error cargando metallib C-007\\n"; return 1; }\n' + \
'        id<MTLComputePipelineState> psoC007Red  = [device newComputePipelineStateWithFunction:[libSpinorC007 newFunctionWithName:@"c007_project_reductions"] error:&err];\n' + \
'        id<MTLComputePipelineState> psoC007Step = [device newComputePipelineStateWithFunction:[libSpinorC007 newFunctionWithName:@"c007_reconstruct_and_step"] error:&err];'

assert anchor_lib in code
code = code.replace(anchor_lib, inject_lib, 1)

# ─── 2. DECLARACIÓN DE BÚFERES Y CARGA DE FOTONES PREVIO A PREFILL ────────────
anchor_buf = 'id<MTLBuffer> bufEOS = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];'
inject_buf = anchor_buf + '\n\n' + \
'        // Búferes del Biespinor Multimodal C-007 (2D = 10240, r = 32)\n' + \
'        const uint32_t R_C007 = 32;\n' + \
'        const uint32_t TWO_D_C007 = 2 * D;\n' + \
'        id<MTLBuffer> bufPhiC007 = [device newBufferWithLength:TWO_D_C007 * sizeof(float) options:MTLResourceStorageModeShared];\n' + \
'        id<MTLBuffer> bufPhiMidC007 = [device newBufferWithLength:TWO_D_C007 * sizeof(float) options:MTLResourceStorageModePrivate];\n' + \
'        id<MTLBuffer> bufRProjC007 = [device newBufferWithLength:(6 * R_C007 + 2) * sizeof(float) options:MTLResourceStorageModePrivate];\n' + \
'        id<MTLBuffer> bufUcC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];\n' + \
'        id<MTLBuffer> bufVcC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];\n' + \
'        id<MTLBuffer> bufUsC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];\n' + \
'        id<MTLBuffer> bufVsC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];\n' + \
'        id<MTLBuffer> bufUlC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];\n' + \
'        id<MTLBuffer> bufVlC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];\n' + \
'        id<MTLBuffer> bufVisualPatches = nil;\n' + \
'        uint32_t num_visual_patches = 0;\n\n' + \
'        // Cargar evidencia visual fáctica ANTES de iniciar el prefill\n' + \
'        {\n' + \
'            std::ifstream vf("visual_embeddings.bin", std::ios::binary);\n' + \
'            if (vf.is_open()) {\n' + \
'                vf.seekg(0, std::ios::end);\n' + \
'                size_t sz = vf.tellg();\n' + \
'                vf.seekg(0, std::ios::beg);\n' + \
'                std::vector<float> vdata(sz / sizeof(float));\n' + \
'                vf.read(reinterpret_cast<char*>(vdata.data()), sz);\n' + \
'                num_visual_patches = uint32_t(vdata.size() / D);\n' + \
'                bufVisualPatches = [device newBufferWithBytes:vdata.data() length:sz options:MTLResourceStorageModeShared];\n' + \
'                std::cout << " ✓ Evidencia visual fáctica vinculada: " << num_visual_patches << " parches (previo a prefill).\\n";\n' + \
'            }\n' + \
'        }'

assert anchor_buf in code
code = code.replace(anchor_buf, inject_buf, 1)

# ─── 3. PREFILL CON INYECCIÓN FÁCTICA DE FOTONES Y FILTRADO COGNITIVO ────────
anchor_prefill_loop = """        for (size_t pos = 0; pos < prompt.size(); ++pos) {
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

inject_prefill_loop = """        uint32_t visual_patch_cursor = 0;
        for (size_t pos = 0; pos < prompt.size(); ++pos) {
            uint32_t tid = prompt[pos];

            // Inyección Directa de Fotones: Si es token de imagen (248056), copiar directamente el parche fáctico
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

assert anchor_prefill_loop in code
code = code.replace(anchor_prefill_loop, inject_prefill_loop, 1)

# Filtrar tokens de imagen del cálculo de covariables
anchor_snap = """            float* z_curr = (float*)[bufZ contents];
            std::vector<float> snap(z_curr, z_curr + D);
            prefill_snaps.push_back(snap);"""

inject_snap = """            // Aislamiento Epistemológico: Solo registrar tokens de lenguaje humano en prefill_snaps
            if (tid != 248056) {
                float* z_curr = (float*)[bufZ contents];
                std::vector<float> snap(z_curr, z_curr + D);
                prefill_snaps.push_back(snap);
            }"""

assert anchor_snap in code
code = code.replace(anchor_snap, inject_snap, 1)

# ─── 4. INICIALIZACIÓN DETERMINISTA DE BASES C-007 TRAS PREFILL ───────────────
anchor_prefill_end = 'float prefill_tps = float(prompt.size()) / float(prefill_sec);'
inject_c007_setup = anchor_prefill_end + '\n\n' + \
'        // Parametrización Determinista de Bases C-007 a partir de fotones y texto\n' + \
'        if (bufVisualPatches != nil && num_visual_patches > 0) {\n' + \
'            float* p_raw = (float*)[bufVisualPatches contents];\n' + \
'            float* uc_ptr = (float*)[bufUcC007 contents];\n' + \
'            float* vc_ptr = (float*)[bufVcC007 contents];\n' + \
'            float* us_ptr = (float*)[bufUsC007 contents];\n' + \
'            float* vs_ptr = (float*)[bufVsC007 contents];\n' + \
'            float* ul_ptr = (float*)[bufUlC007 contents];\n' + \
'            float* vl_ptr = (float*)[bufVlC007 contents];\n' + \
'            float* phi_init = (float*)[bufPhiC007 contents];\n\n' + \
'            std::vector<float> u_O_vis(D, 0.0f);\n' + \
'            for (uint32_t p = 0; p < num_visual_patches; ++p) {\n' + \
'                for (uint32_t i = 0; i < D; ++i) u_O_vis[i] += p_raw[p * D + i];\n' + \
'            }\n' + \
'            float inv_p = 1.0f / float(num_visual_patches);\n' + \
'            float n_ov = 0.0f;\n' + \
'            for (uint32_t i = 0; i < D; ++i) { u_O_vis[i] *= inv_p; n_ov += u_O_vis[i] * u_O_vis[i]; }\n' + \
'            n_ov = std::sqrt(n_ov) + 1e-12f;\n' + \
'            for (uint32_t i = 0; i < D; ++i) { u_O_vis[i] /= n_ov; phi_init[i] = u_O_vis[i]; } // S inicializado con el centroide\n\n' + \
'            float* u_O_text = (float*)[bufOntology contents];\n' + \
'            float* u_T_text = (float*)[bufTeleology contents];\n\n' + \
'            for (uint32_t i = 0; i < D; ++i) {\n' + \
'                uc_ptr[0 * D + i] = u_T_text[i];\n' + \
'                vc_ptr[0 * D + i] = u_O_vis[i];\n' + \
'                us_ptr[0 * D + i] = u_O_vis[i];\n' + \
'                vs_ptr[0 * D + i] = u_O_vis[i];\n' + \
'                ul_ptr[0 * D + i] = u_O_text[i];\n' + \
'                vl_ptr[0 * D + i] = u_T_text[i];\n' + \
'            }\n' + \
'            std::cout << " ✓ Biespinor C-007 calibrado: Componente S anclada a fotones fácticos.\\n";\n' + \
'        }'

assert anchor_prefill_end in code
code = code.replace(anchor_prefill_end, inject_c007_setup, 1)

# ─── 5. INTEGRACIÓN CONTINUA EN DECODE (L598) ─────────────────────────────────
anchor_steer = """            // 4. Integrador Geodésico Soberano al final de las 64 capas
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

inject_steer = """            // 4. Integrador Multimodal Continuo C-007 (Evolución de Biespinor en Silicio)
            auto t_p6 = std::chrono::high_resolution_clock::now();
            if (bufVisualPatches != nil && num_visual_patches > 0) {
                float* phi_raw = (float*)[bufPhiC007 contents];
                float* z_raw = (float*)[bufZ contents];
                for (uint32_t i = 0; i < D; ++i) phi_raw[D + i] = z_raw[i];

                id<MTLCommandBuffer> cmdC007 = [queue commandBuffer];
                float d_tau_c007 = 0.05f;
                float half_dt_c007 = 0.5f * d_tau_c007;
                float coupling_scale = 0.05f;
                float alpha_eml = 0.02f;
                float lambda_diss = 0.0001f; // Disipación suave preservadora de norma

                for (uint32_t s = 0; s < 64; ++s) {
                    id<MTLComputeCommandEncoder> enc1 = [cmdC007 computeCommandEncoder];
                    [enc1 setComputePipelineState:psoC007Red];
                    [enc1 setBuffer:bufPhiC007 offset:0 atIndex:0];
                    [enc1 setBuffer:bufUcC007 offset:0 atIndex:1];
                    [enc1 setBuffer:bufVcC007 offset:0 atIndex:2];
                    [enc1 setBuffer:bufUsC007 offset:0 atIndex:3];
                    [enc1 setBuffer:bufVsC007 offset:0 atIndex:4];
                    [enc1 setBuffer:bufUlC007 offset:0 atIndex:5];
                    [enc1 setBuffer:bufVlC007 offset:0 atIndex:6];
                    [enc1 setBuffer:bufRProjC007 offset:0 atIndex:7];
                    [enc1 setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
                    [enc1 dispatchThreadgroups:MTLSizeMake(R_C007, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc1 endEncoding];

                    id<MTLComputeCommandEncoder> enc2 = [cmdC007 computeCommandEncoder];
                    [enc2 setComputePipelineState:psoC007Step];
                    [enc2 setBuffer:bufPhiC007 offset:0 atIndex:0];
                    [enc2 setBuffer:bufPhiC007 offset:0 atIndex:1];
                    [enc2 setBuffer:bufPhiMidC007 offset:0 atIndex:2];
                    [enc2 setBuffer:bufUcC007 offset:0 atIndex:3];
                    [enc2 setBuffer:bufVcC007 offset:0 atIndex:4];
                    [enc2 setBuffer:bufUsC007 offset:0 atIndex:5];
                    [enc2 setBuffer:bufVsC007 offset:0 atIndex:6];
                    [enc2 setBuffer:bufUlC007 offset:0 atIndex:7];
                    [enc2 setBuffer:bufVlC007 offset:0 atIndex:8];
                    [enc2 setBuffer:bufRProjC007 offset:0 atIndex:9];
                    [enc2 setBytes:&half_dt_c007 length:sizeof(float) atIndex:10];
                    [enc2 setBytes:&coupling_scale length:sizeof(float) atIndex:11];
                    [enc2 setBytes:&alpha_eml length:sizeof(float) atIndex:12];
                    [enc2 setBytes:&lambda_diss length:sizeof(float) atIndex:13];
                    [enc2 dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc2 endEncoding];

                    id<MTLComputeCommandEncoder> enc3 = [cmdC007 computeCommandEncoder];
                    [enc3 setComputePipelineState:psoC007Red];
                    [enc3 setBuffer:bufPhiMidC007 offset:0 atIndex:0];
                    [enc3 setBuffer:bufUcC007 offset:0 atIndex:1];
                    [enc3 setBuffer:bufVcC007 offset:0 atIndex:2];
                    [enc3 setBuffer:bufUsC007 offset:0 atIndex:3];
                    [enc3 setBuffer:bufVsC007 offset:0 atIndex:4];
                    [enc3 setBuffer:bufUlC007 offset:0 atIndex:5];
                    [enc3 setBuffer:bufVlC007 offset:0 atIndex:6];
                    [enc3 setBuffer:bufRProjC007 offset:0 atIndex:7];
                    [enc3 setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
                    [enc3 dispatchThreadgroups:MTLSizeMake(R_C007, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc3 endEncoding];

                    id<MTLComputeCommandEncoder> enc4 = [cmdC007 computeCommandEncoder];
                    [enc4 setComputePipelineState:psoC007Step];
                    [enc4 setBuffer:bufPhiMidC007 offset:0 atIndex:0];
                    [enc4 setBuffer:bufPhiC007 offset:0 atIndex:1];
                    [enc4 setBuffer:bufPhiC007 offset:0 atIndex:2];
                    [enc4 setBuffer:bufUcC007 offset:0 atIndex:3];
                    [enc4 setBuffer:bufVcC007 offset:0 atIndex:4];
                    [enc4 setBuffer:bufUsC007 offset:0 atIndex:5];
                    [enc4 setBuffer:bufVsC007 offset:0 atIndex:6];
                    [enc4 setBuffer:bufUlC007 offset:0 atIndex:7];
                    [enc4 setBuffer:bufVlC007 offset:0 atIndex:8];
                    [enc4 setBuffer:bufRProjC007 offset:0 atIndex:9];
                    [enc4 setBytes:&d_tau_c007 length:sizeof(float) atIndex:10];
                    [enc4 setBytes:&coupling_scale length:sizeof(float) atIndex:11];
                    [enc4 setBytes:&alpha_eml length:sizeof(float) atIndex:12];
                    [enc4 setBytes:&lambda_diss length:sizeof(float) atIndex:13];
                    [enc4 dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc4 endEncoding];
                }
                [cmdC007 commit];
                [cmdC007 waitUntilCompleted];

                for (uint32_t i = 0; i < D; ++i) z_raw[i] = phi_raw[D + i];
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

assert anchor_steer in code
code = code.replace(anchor_steer, inject_steer, 1)

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ Inyección integral de fontanería multimodal y biespinor C-007 completada.")
