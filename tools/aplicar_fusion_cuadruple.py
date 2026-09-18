with open("src/main_aether.mm", "r") as f:
    code = f.read()

# 1. Empaquetar los 16 macrobloques en 4 etapas cuádruples (16 capas por buffer)
# Esto reduce de 16 sincronizaciones a solo 4 por token
old_mb_loop = """            // 3. Propagación por los 16 macro-bloques (1 GQA + 3 SSM por bloque)
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
            }"""

new_mb_loop = """            // 3. Propagación por 4 Super-Bloques de 16 capas (Fusión Cuádruple)
            for (uint32_t sb = 0; sb < 4; ++sb) {
                id<MTLCommandBuffer> cmdL = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encL = [cmdL computeCommandEncoder];
                uint32_t start_layer = sb * 16;
                for (uint32_t l = 0; l < 16; ++l) {
                    dispatch_layer_forward(encL, start_layer + l, current_pos);
                }
                [encL endEncoding];
                [cmdL commit];
                [cmdL waitUntilCompleted];
            }"""

code = code.replace(old_mb_loop, new_mb_loop)

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ Fusión Cuádruple inyectada: sincronizaciones reducidas de 16 a 4 por token.")
