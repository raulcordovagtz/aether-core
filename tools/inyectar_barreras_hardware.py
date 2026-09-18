with open("src/main_aether.mm", "r") as f:
    code = f.read()

# Inyectar barrera de memoria nativa de Metal entre capas
old_layer_loop = """            // 2. Propagación continua de las 64 capas en GPU sin retorno a CPU
            for (uint32_t l = 0; l < NUM_LAYERS; ++l) {
                dispatch_layer_forward(enc, l, current_pos);
            }"""

new_layer_loop = """            // 2. Propagación continua de las 64 capas con barreras de silicio en GPU
            for (uint32_t l = 0; l < NUM_LAYERS; ++l) {
                dispatch_layer_forward(enc, l, current_pos);
                // Barrera de memoria hardware en GPU: asegura coherencia de bufZ entre capas sin pausar la CPU
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }"""

code = code.replace(old_layer_loop, new_layer_loop)

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ Barrera de memoria MTLBarrierScopeBuffers inyectada entre capas.")
