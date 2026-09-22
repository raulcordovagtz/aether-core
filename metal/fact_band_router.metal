// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: FACT BAND ROUTER GPU KERNEL (HITO 2.2-R1)
// Búsqueda Asociativa 1×K Paralela en SRAM y Compuerta Rectificada en Registros
// ═════════════════════════════════════════════════════════════════════════════
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

struct alignas(16) RouterDecisionGPU {
    uint  selected_slot;
    float max_resonance_r;
    float second_resonance_r;
    float resonance_margin;
    float rectified_gate_g;
    uint  is_active;
};

kernel void dispatch_fact_band_batch_resonance(
    device const float*        h_layer       [[buffer(0)]],  // [D]
    device const float*        memory_slots  [[buffer(1)]],  // [K * D]
    device RouterDecisionGPU*  decision_out  [[buffer(2)]],  // Salida
    constant uint&             K             [[buffer(3)]],  // Número de slots activos
    constant uint&             D             [[buffer(4)]],  // Dimensión
    constant float&            theta_assoc   [[buffer(5)]],  // Umbral
    constant float&            beta          [[buffer(6)]],  // Sensibilidad
    threadgroup float*         shared_dots   [[threadgroup(0)]], // [K * t_per_group]
    uint tid                                 [[thread_index_in_threadgroup]],
    uint t_per_group                         [[threads_per_threadgroup]]
) {
    // Cada hilo acumula el producto punto para cada uno de los K slots
    for (uint k = 0; k < K; ++k) {
        float l_dot = 0.0f;
        device const float* slot_k = memory_slots + (k * D);
        for (uint i = tid; i < D; i += t_per_group) {
            l_dot += h_layer[i] * slot_k[i];
        }
        shared_dots[k * t_per_group + tid] = l_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reducción cooperativa para los K productos punto
    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (uint k = 0; k < K; ++k) {
                shared_dots[k * t_per_group + tid] += shared_dots[k * t_per_group + (tid + s)];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // El hilo 0 determina argmax, segundo mejor, margen y evalúa la compuerta rectificada (CERO ARCCOS)
    if (tid == 0) {
        float max_r = -2.0f;
        float second_r = -2.0f;
        uint best_k = 0;

        for (uint k = 0; k < K; ++k) {
            float r_k = shared_dots[k * t_per_group];
            if (r_k > max_r) {
                second_r = max_r;
                max_r = r_k;
                best_k = k;
            } else if (r_k > second_r) {
                second_r = r_k;
            }
        }

        if (K == 1) {
            second_r = -1.0f;
        }

        // Rectificación estricta: cero absoluto por debajo del umbral
        float g = 0.0f;
        if (max_r >= theta_assoc) {
            g = 1.0f / (1.0f + exp(-beta * (max_r - theta_assoc)));
        }

        decision_out->selected_slot      = best_k;
        decision_out->max_resonance_r    = max_r;
        decision_out->second_resonance_r = second_r;
        decision_out->resonance_margin   = (K > 1) ? (max_r - second_r) : 1.0f;
        decision_out->rectified_gate_g   = g;
        decision_out->is_active          = (g > 0.0f) ? 1 : 0;
    }
}
