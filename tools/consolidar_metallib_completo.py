# 1. Leer el kernel de atención lineal
with open("metal/aether_geodesic_engine.metal", "r") as f:
    code_attn = f.read()

# Extraer el kernel aether_gqa_attention_linear_exact
idx_start = code_attn.find("kernel void aether_gqa_attention_linear_exact")
attn_kernel_str = code_attn[idx_start:]

# 2. Leer el kernel mhc+eml actual
with open("metal/aether_mhc_eml_engine.metal", "r") as f:
    code_mhc = f.read()

# Unir ambos kernels en aether_mhc_eml_engine.metal
full_metal = code_mhc + "\n\n" + attn_kernel_str

with open("metal/aether_mhc_eml_engine.metal", "w") as f:
    f.write(full_metal)

print("✓ Ambos kernels consolidados en metal/aether_mhc_eml_engine.metal.")
