import yaml
import re

with open("spec/C07_dirac_eml_spinor.yaml", "r") as f:
    spec = yaml.safe_load(f)

params = spec["parameters_derivation"]
STEPS = params["step_size"]["steps"]
DT = 1.0 / float(STEPS)
HALF_DT = DT / 2.0

with open("src/main_aether.mm", "r") as f:
    code = f.read()

# Bloque C++ canónico generado automáticamente
canonical_cpp_dispatch = f"""            // 4. Integrador Multimodal Continuo C-007 (Derivado analíticamente vía AOT)
            auto t_p6 = std::chrono::high_resolution_clock::now();
            if (bufVisualPatches != nil && num_visual_patches > 0) {{
                float* phi_raw = (float*)[bufPhiC007 contents];
                float* z_raw = (float*)[bufZ contents];
                for (uint32_t i = 0; i < D; ++i) phi_raw[D + i] = z_raw[i];

                id<MTLCommandBuffer> cmdC007 = [queue commandBuffer];
                const float dt_step = {DT:.10f}f;
                const float half_dt = {HALF_DT:.10f}f;

                for (uint32_t s = 0; s < {STEPS}; ++s) {{
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
                    [enc4 setBytes:&dt_step length:sizeof(float) atIndex:10];
                    [enc4 dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc4 endEncoding];
                }}
                [cmdC007 commit];
                [cmdC007 waitUntilCompleted];

                for (uint32_t i = 0; i < D; ++i) z_raw[i] = phi_raw[D + i];
            }} else {{
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
            }}"""

# Sustituir la sección del integrador en el bucle decode
pattern = r"            // 4\. Integrador Multimodal Continuo C-007.*?\n            \}"
match = re.search(pattern, code, re.DOTALL)
assert match, "Fallo al localizar el bloque de integrador previo en src/main_aether.mm"

code = code[:match.start()] + canonical_cpp_dispatch + code[match.end():]

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ src/main_aether.mm sincronizado automáticamente con la especificación canónica AOT.")
