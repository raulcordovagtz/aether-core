with open("src/main_aether.mm", "r") as f:
    code = f.read()

# 1. Cargar biblioteca C-008
old_lib = 'id<MTLLibrary> libSpinorC007 = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_c007_spinor_integrator.metallib"] error:&err];'
new_lib = 'id<MTLLibrary> libSpinorC008 = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_c008_cognitive_engine.metallib"] error:&err];\n' + \
'        if (!libSpinorC008) { std::cerr << "❌ Error cargando libSpinorC008\\n"; return 1; }\n' + \
'        id<MTLComputePipelineState> psoC008Red  = [device newComputePipelineStateWithFunction:[libSpinorC008 newFunctionWithName:@"c008_project_reductions"] error:&err];\n' + \
'        id<MTLComputePipelineState> psoC008Step = [device newComputePipelineStateWithFunction:[libSpinorC008 newFunctionWithName:@"c008_cognitive_step"] error:&err];'

if old_lib in code:
    code = code.replace(old_lib, new_lib, 1)

# Reemplazar psoC007 por psoC008 en declaraciones
code = code.replace("psoC007Red", "psoC008Red")
code = code.replace("psoC007Step", "psoC008Step")

# 2. Inyectar la fase de Pensamiento Profundo (τ* = 32) al final del prefill
prefill_thought_phase = """
            // 🧠 FASE DE PENSAMIENTO PROFUNDO EN SILICIO (τ* = 32 pasos de convergencia cognitiva)
            std::cout << " • Ejecutando Pensamiento Profundo en GPU (τ* = 32 pasos de convergencia)...\\n";
            auto t_thought_0 = std::chrono::high_resolution_clock::now();
            
            // Inicializar Phi con S = u_O_vis, L = z_last (salida del prefill)
            float* z_prefill_last = (float*)[bufZ contents];
            for (uint32_t i = 0; i < D; ++i) phi_init[D + i] = z_prefill_last[i];

            id<MTLCommandBuffer> cmdThought = [queue commandBuffer];
            const float dt_p = 0.015625f;
            const float half_dt_p = 0.0078125f;

            for (uint32_t tau = 0; tau < 32; ++tau) {
                float lie_decay = std::exp(-float(tau) / 24.0f);

                id<MTLComputeCommandEncoder> enc1 = [cmdThought computeCommandEncoder];
                [enc1 setComputePipelineState:psoC008Red];
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

                id<MTLComputeCommandEncoder> enc2 = [cmdThought computeCommandEncoder];
                [enc2 setComputePipelineState:psoC008Step];
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
                [enc2 setBytes:&half_dt_p length:sizeof(float) atIndex:10];
                [enc2 setBytes:&lie_decay length:sizeof(float) atIndex:11];
                [enc2 setBuffer:bufVsC007 offset:0 atIndex:12]; // u_vis
                [enc2 setBuffer:bufVlC007 offset:0 atIndex:13]; // u_txt
                [enc2 dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc2 endEncoding];

                id<MTLComputeCommandEncoder> enc3 = [cmdThought computeCommandEncoder];
                [enc3 setComputePipelineState:psoC008Red];
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

                id<MTLComputeCommandEncoder> enc4 = [cmdThought computeCommandEncoder];
                [enc4 setComputePipelineState:psoC008Step];
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
                [enc4 setBytes:&dt_p length:sizeof(float) atIndex:10];
                [enc4 setBytes:&lie_decay length:sizeof(float) atIndex:11];
                [enc4 setBuffer:bufVsC007 offset:0 atIndex:12]; // u_vis
                [enc4 setBuffer:bufVlC007 offset:0 atIndex:13]; // u_txt
                [enc4 dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc4 endEncoding];
            }
            [cmdThought commit];
            [cmdThought waitUntilCompleted];
            auto t_thought_1 = std::chrono::high_resolution_clock::now();
            double thought_ms = std::chrono::duration<double, std::milli>(t_thought_1 - t_thought_0).count();
            
            // Asentar el estado lingüístico óptimo L* en bufZ para el inicio de decode
            for (uint32_t i = 0; i < D; ++i) z_prefill_last[i] = phi_init[D + i];
            std::cout << " ✓ Asentamiento alcanzado en " << std::fixed << std::setprecision(2) << thought_ms << " ms. Estado óptimo Φ* proyectado a Decode.\\n";
"""

anchor_init_end = 'std::cout << " ✓ Biespinor C-007 calibrado: Componente S anclada a fotones fácticos.\\n";\n        }'
assert anchor_init_end in code
code = code.replace(anchor_init_end, anchor_init_end + prefill_thought_phase, 1)

# 3. Adaptar el paso de decode para pasar los buffers de u_vis y u_txt
decode_old_args = """                    [enc2 setBytes:&half_dt length:sizeof(float) atIndex:10];
                    [enc2 dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];"""
decode_new_args = """                    float lie_decay_dec = 0.05f; // Torsión suave residual
                    [enc2 setBytes:&half_dt length:sizeof(float) atIndex:10];
                    [enc2 setBytes:&lie_decay_dec length:sizeof(float) atIndex:11];
                    [enc2 setBuffer:bufVsC007 offset:0 atIndex:12];
                    [enc2 setBuffer:bufVlC007 offset:0 atIndex:13];
                    [enc2 dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];"""
code = code.replace(decode_old_args, decode_new_args)

decode_old_args4 = """                    [enc4 setBytes:&dt_step length:sizeof(float) atIndex:10];
                    [enc4 dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];"""
decode_new_args4 = """                    float lie_decay_dec4 = 0.05f;
                    [enc4 setBytes:&dt_step length:sizeof(float) atIndex:10];
                    [enc4 setBytes:&lie_decay_dec4 length:sizeof(float) atIndex:11];
                    [enc4 setBuffer:bufVsC007 offset:0 atIndex:12];
                    [enc4 setBuffer:bufVlC007 offset:0 atIndex:13];
                    [enc4 dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];"""
code = code.replace(decode_old_args4, decode_new_args4)

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ src/main_aether.mm actualizado con Fase de Pensamiento Profundo C-008.")
