#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>

namespace c_field {

// Configuración estructural de cualquier modelo VL (Leído de config.json / Safetensors)
struct ModelTopology {
    uint32_t hidden_dim;       // D (e.g. 2048, 4096, 8192)
    uint32_t num_heads;        // H (e.g. 16, 32)
    uint32_t head_dim;         // d_k = D / H (e.g. 64, 128)
    uint32_t vocab_size;       // |V| (e.g. 32000, 151936)
    uint32_t max_context_len;  // L_ctx (e.g. 4096, 32768)
    
    // Visión Multimodal
    uint32_t patch_size;       // P (e.g. 14 o 16 px)
    uint32_t visual_dim;       // D_vis
};

// Invariantes deducidos matemáticamente desde primeros principios
struct ManifoldInvariants {
    float beta_inverse_temp;   // Temperatura de campo: sqrt(d_k)
    float fermi_potential_mu;  // Umbral de corte de vacío térmico: -ln(L_ctx)
    float r_manifold;          // Invariante de Shannon-Boltzmann: sqrt(2 / ln|V|)
    float fermat_scale;        // Factor conforme: 1 / sqrt(d_k)
    float mach_epsilon;        // Límite físico de precisión (IEEE 754)

    static ManifoldInvariants compute(const ModelTopology& topo) {
        ManifoldInvariants inv;
        
        // 1. Temperatura de la hipersfera local S^(d_k - 1)
        inv.beta_inverse_temp = std::sqrt(static_cast<float>(topo.head_dim));
        
        // 2. Umbral de corte del vacío térmico: P_vacio = 1 / L_ctx -> mu = ln(P_vacio)
        float ctx = static_cast<float>(std::max(topo.max_context_len, 2u));
        inv.fermi_potential_mu = -std::log(ctx);
        
        // 3. Radio intrínseco del horizonte de Gibbs (Shannon-Boltzmann)
        float log_V = std::log(static_cast<float>(std::max(topo.vocab_size, 2u)));
        inv.r_manifold = std::sqrt(2.0f / log_V);
        
        // 4. Factor de escala conforme
        inv.fermat_scale = 1.0f / std::sqrt(static_cast<float>(topo.head_dim));
        
        // 5. Precisión de máquina estricta
        inv.mach_epsilon = std::numeric_limits<float>::epsilon();
        
        return inv;
    }
};

} // namespace c_field
