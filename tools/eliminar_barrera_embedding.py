with open("src/main_aether.mm", "r") as f:
    code = f.read()

# Enlazar psoSwiglu a swiglu_dense_4bit_forward de libBase
old_pso = 'id<MTLComputePipelineState> psoSwiglu   = [device newComputePipelineStateWithFunction:[libCert newFunctionWithName:@"eml_swiglu_certified_forward"] error:&err];'
new_pso = 'id<MTLComputePipelineState> psoSwiglu   = [device newComputePipelineStateWithFunction:[libBase newFunctionWithName:@"swiglu_dense_4bit_forward"] error:&err];'

if old_pso in code:
    code = code.replace(old_pso, new_pso, 1)
    print("✓ psoSwiglu enlazado a swiglu_dense_4bit_forward nativo acelerado.")

# Eliminar waitUntilCompleted tras el embedding
old_embed_barrier = """            [encStep endEncoding];
            [cmdStep commit];
            [cmdStep waitUntilCompleted];
            auto t_p3 = std::chrono::high_resolution_clock::now();"""

new_embed_barrier = """            [encStep endEncoding];
            [cmdStep commit];
            // Barrera síncrona eliminada: GPU encadena el embedding directamente a las capas
            auto t_p3 = std::chrono::high_resolution_clock::now();"""

if old_embed_barrier in code:
    code = code.replace(old_embed_barrier, new_embed_barrier, 1)
    print("✓ Barrera síncrona de embedding eliminada de decode.")

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ src/main_aether.mm optimizado.")
