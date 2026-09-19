with open("src/main_aether.mm", "r") as f:
    code = f.read()

# 1. Cargar biblioteca C-018 Riemanniana
old_lib = 'id<MTLLibrary> libSpinorC008 = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_c008_cognitive_engine.metallib"] error:&err];'
new_lib = 'id<MTLLibrary> libSpinorC018 = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_c018_riemannian_engine.metallib"] error:&err];\n' + \
'        if (!libSpinorC018) { std::cerr << "❌ Error cargando libSpinorC018\\n"; return 1; }\n' + \
'        id<MTLComputePipelineState> psoC018Red  = [device newComputePipelineStateWithFunction:[libSpinorC018 newFunctionWithName:@"c018_project_reductions"] error:&err];\n' + \
'        id<MTLComputePipelineState> psoC018Step = [device newComputePipelineStateWithFunction:[libSpinorC018 newFunctionWithName:@"c018_riemannian_step"] error:&err];'

assert old_lib in code, "Fallo: No se encontró old_lib C-008"
code = code.replace(old_lib, new_lib, 1)

code = code.replace("psoC008Red", "psoC018Red")
code = code.replace("psoC008Step", "psoC018Step")

# 2. Agregar búfer para restricción del Harness
anchor_buf = 'id<MTLBuffer> bufVisualPatches = nil;'
new_buf = anchor_buf + '\n        id<MTLBuffer> bufExactALU = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];\n        std::memset([bufExactALU contents], 0, D * sizeof(float));'
code = code.replace(anchor_buf, new_buf, 1)

# 3. Adaptar llamada del paso de pensamiento geodésico
old_thought_call = '[enc2 setBuffer:bufVlC007 offset:0 atIndex:13]; // u_txt'
new_thought_call = '[enc2 setBuffer:bufVlC007 offset:0 atIndex:10]; // u_txt\n' + \
'                [enc2 setBuffer:bufExactALU offset:0 atIndex:11];\n' + \
'                [enc2 setBytes:&dt_p length:sizeof(float) atIndex:12];\n' + \
'                float gate_prefill = 0.0f; // Compuerta latente pura en prefill\n' + \
'                [enc2 setBytes:&gate_prefill length:sizeof(float) atIndex:13];'
code = code.replace(old_thought_call, new_thought_call)

with open("src/main_aether.mm", "w") as f:
    f.write(code)

print("✓ src/main_aether.mm enlazado con el Motor Geodésico Riemanniano C-018.")
