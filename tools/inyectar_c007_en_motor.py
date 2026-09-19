import re

with open("src/main_aether.mm", "r") as f:
    code = f.read()

# 1. Incluir la carga de la biblioteca de silicio C-007
search_lib = 'id<MTLLibrary> libGeodesic = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_mhc_eml_engine.metallib"] error:&err];'
replace_lib = search_lib + '\n' + \
'        id<MTLLibrary> libSpinorC007 = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_c007_spinor_integrator.metallib"] error:&err];\n' + \
'        id<MTLFunction> fnC007Red = [libSpinorC007 newFunctionWithName:@"c007_project_reductions"];\n' + \
'        id<MTLFunction> fnC007Step = [libSpinorC007 newFunctionWithName:@"c007_reconstruct_and_step"];\n' + \
'        id<MTLComputePipelineState> psoC007Red = [device newComputePipelineStateWithFunction:fnC007Red error:&err];\n' + \
'        id<MTLComputePipelineState> psoC007Step = [device newComputePipelineStateWithFunction:fnC007Step error:&err];'

if search_lib in code:
    code = code.replace(search_lib, replace_lib)
    print("✓ Pipeline states de C-007 inyectados.")
else:
    print("⚠ No se encontró el ancla para libGeodesic.")

# 2. Declaración de búferes persistentes para el Biespinor C-007 (r=32)
search_buf = 'uint32_t current_pos = prompt.size();'
c007_buffers_init = """        // Inicialización de Búferes del Biespinor Multimodal C-007 (Dimensión 2D = 10240, r = 32)
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

        // Inicializar bases deterministas si hay parches visuales
        if (bufVisualPatches != nil && num_visual_patches > 0) {
            float* p_raw = (float*)[bufVisualPatches contents];
            float* uc_ptr = (float*)[bufUcC007 contents];
            float* vc_ptr = (float*)[bufVcC007 contents];
            float* us_ptr = (float*)[bufUsC007 contents];
            float* vs_ptr = (float*)[bufVsC007 contents];
            float* ul_ptr = (float*)[bufUlC007 contents];
            float* vl_ptr = (float*)[bufVlC007 contents];

            // Centroide visual en u_O_vis y foco en u_T_vis
            std::vector<float> u_O_vis(D, 0.0f);
            for (uint32_t p = 0; p < num_visual_patches; ++p) {
                for (uint32_t i = 0; i < D; ++i) u_O_vis[i] += p_raw[p * D + i];
            }
            float inv_p = 1.0f / float(num_visual_patches);
            float n_ov = 0.0f;
            for (uint32_t i = 0; i < D; ++i) { u_O_vis[i] *= inv_p; n_ov += u_O_vis[i] * u_O_vis[i]; }
            n_ov = std::sqrt(n_ov) + 1e-12f;
            for (uint32_t i = 0; i < D; ++i) u_O_vis[i] /= n_ov;

            // Inyectar en modos de rango 0 y 1
            for (uint32_t i = 0; i < D; ++i) {
                vc_ptr[0 * D + i] = u_O_vis[i];
                vs_ptr[0 * D + i] = u_O_vis[i];
                us_ptr[0 * D + i] = u_O_vis[i];
            }
            std::cout << " ✓ Operador C-007 parametrizado con centroide óptico fáctico.\\n";
        }
"""
code = code.replace(search_buf, c007_buffers_init + '\n        ' + search_buf)

# 3. Sustituir la llamada de inyección geodésica por el integrador continuo C-007
search_exec = """            // Inyección del Operador Multimodal Certificado C-005
            if (bufVisualPatches != nil && num_visual_patches > 0) {
                id<MTLCommandBuffer> cmdVis = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encVis = [cmdVis computeCommandEncoder];
                [encVis setComputePipelineState:psoVisualTide];
                [encVis setBuffer:bufZ offset:0 atIndex:0];
                [encVis setBuffer:bufVisualPatches offset:0 atIndex:1];
                [encVis setBytes:&num_visual_patches length:sizeof(uint32_t) atIndex:2];
                [encVis setBytes:&D length:sizeof(uint32_t) atIndex:3];
                float tide_scale = 0.02f; // Escala de marea suave certificada
                [encVis setBytes:&tide_scale length:sizeof(float) atIndex:4];
                [encVis setThreadgroupMemoryLength:512 * sizeof(float) atIndex:0];
                [encVis setThreadgroupMemoryLength:512 * sizeof(float) atIndex:1];
                [encVis dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
                [encVis endEncoding];
                [cmdVis commit];
                [cmdVis waitUntilCompleted];
            }"""

c007_step_call = """            // Inyección del Biespinor Multimodal Continuo C-007 (Apple M2 Max)
            if (bufVisualPatches != nil && num_visual_patches > 0) {
                // Sincronizar el estado latente bufZ con la componente simbólica L de Phi
                float* phi_raw = (float*)[bufPhiC007 contents];
                float* z_raw = (float*)[bufZ contents];
                for (uint32_t i = 0; i < D; ++i) phi_raw[D + i] = z_raw[i];

                id<MTLCommandBuffer> cmdC007 = [queue commandBuffer];
                float d_tau = 0.05f;
                float half_dt = 0.5f * d_tau;
                float coupling_scale = 0.05f;
                float alpha_eml = 0.02f;
                float lambda_diss = 0.001f;

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
                    [enc2 setBytes:&half_dt length:sizeof(float) atIndex:10];
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
                    [enc4 setBytes:&d_tau length:sizeof(float) atIndex:10];
                    [enc4 setBytes:&coupling_scale length:sizeof(float) atIndex:11];
                    [enc4 setBytes:&alpha_eml length:sizeof(float) atIndex:12];
                    [enc4 setBytes:&lambda_diss length:sizeof(float) atIndex:13];
                    [enc4 dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc4 endEncoding];
                }
                [cmdC007 commit];
                [cmdC007 waitUntilCompleted];

                // Devolver el estado proyectado del componente simbólico L a bufZ
                for (uint32_t i = 0; i < D; ++i) z_raw[i] = phi_raw[D + i];
            }"""

if search_exec in code:
    code = code.replace(search_exec, c007_step_call)
    print("✓ Bucle de ejecución geodésica sustituido por C-007 continuo.")
else:
    print("⚠ No se encontró el bloque de inyección C-005.")

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ src/main_aether.mm actualizado con el integrador persistente C-007.")
