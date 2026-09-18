with open("src/main_aether.mm", "r") as f:
    code = f.read()

# Usar el kernel lineal verificado en lugar de markov w512
code = code.replace(
    'psoGqaExact = [device newComputePipelineStateWithFunction:[libMarkov newFunctionWithName:@"gqa_attention_markov_w512"] error:&err];',
    'psoGqaExact = [device newComputePipelineStateWithFunction:[libGeodesic newFunctionWithName:@"aether_gqa_attention_linear_exact"] error:&err];'
)

with open("src/main_aether.mm", "w") as f:
    f.write(code)
print("✓ src/main_aether.mm enlazado con los kernels verificados.")
