#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ═════════════════════════════════════════════════════════════════════════════
// 🚀 FUSED RMSNORM + GEMV 4-BIT (ELIMINACIÓN DE TRÁFICO DRAM INTERMEDIO)
// ═════════════════════════════════════════════════════════════════════════════
kernel void gemv_4bit_fused_norm_g64(
    device const float* Z_Raw               [[buffer(0)]],
    device const ushort* Gamma_Weights      [[buffer(1)]],
    device const uint32_t* Weights_4bit     [[buffer(2)]],
    device const ushort* Scales_Fp16        [[buffer(3)]],
    device const ushort* Biases_Fp16        [[buffer(4)]],
    device float* Out_Vec                   [[buffer(5)]],
    constant uint32_t& D_In                 [[buffer(6)]],
    constant uint32_t& D_Out                [[buffer(7)]],
    constant float& Mach_Eps                [[buffer(8)]],
    threadgroup float* shared_sq_sum        [[threadgroup(0)]],
    uint tid                                [[thread_index_in_threadgroup]],
    uint gid_group                          [[threadgroup_position_in_grid]],
    uint t_per_group                        [[threads_per_threadgroup]])
{
    // 1. Reducción cooperativa de RMSNorm en SRAM local
    float local_sq = 0.0f;
    for (uint i = tid; i < D_In; i += t_per_group) {
        float z = Z_Raw[i];
        local_sq += z * z;
    }
    shared_sq_sum[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt((shared_sq_sum[0] / float(D_In)) + Mach_Eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. GEMV 4-bit multiplicando directamente el valor normalizado en registros
    uint row = gid_group * 4; // Cada threadgroup procesa 4 filas
    if (row >= D_Out) return;

    uint u32_per_row = D_In >> 3;
    uint scales_per_row = D_In >> 6;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (uint w = tid; w < u32_per_row; w += t_per_group) {
        uint base_k = w << 3;
        uint g_idx = base_k >> 6;

        // Cargar y normalizar en caliente los 8 floats en registros SIMD
        float in_norm[8];
        for (uint b = 0; b < 8; ++b) {
            float raw_z = Z_Raw[base_k + b];
            float g_val = as_type<float>(uint(Gamma_Weights[base_k + b]) << 16);
            in_norm[b] = raw_z * inv_rms * g_val;
        }

        // Fila 0
        if (row < D_Out) {
            uint32_t pack = Weights_4bit[row * u32_per_row + w];
            float s = as_type<float>(uint(Scales_Fp16[row * scales_per_row + g_idx]) << 16);
            float b_val = as_type<float>(uint(Biases_Fp16[row * scales_per_row + g_idx]) << 16);
            for (uint b = 0; b < 8; ++b) {
                float w_val = float((pack >> (b * 4)) & 0x0F) * s + b_val;
                acc0 += in_norm[b] * w_val;
            }
        }
        // Fila 1
        if (row + 1 < D_Out) {
            uint32_t pack = Weights_4bit[(row + 1) * u32_per_row + w];
            float s = as_type<float>(uint(Scales_Fp16[(row + 1) * scales_per_row + g_idx]) << 16);
            float b_val = as_type<float>(uint(Biases_Fp16[(row + 1) * scales_per_row + g_idx]) << 16);
            for (uint b = 0; b < 8; ++b) {
                float w_val = float((pack >> (b * 4)) & 0x0F) * s + b_val;
                acc1 += in_norm[b] * w_val;
            }
        }
        // Fila 2
        if (row + 2 < D_Out) {
            uint32_t pack = Weights_4bit[(row + 2) * u32_per_row + w];
            float s = as_type<float>(uint(Scales_Fp16[(row + 2) * scales_per_row + g_idx]) << 16);
            float b_val = as_type<float>(uint(Biases_Fp16[(row + 2) * scales_per_row + g_idx]) << 16);
            for (uint b = 0; b < 8; ++b) {
                float w_val = float((pack >> (b * 4)) & 0x0F) * s + b_val;
                acc2 += in_norm[b] * w_val;
            }
        }
        // Fila 3
        if (row + 3 < D_Out) {
            uint32_t pack = Weights_4bit[(row + 3) * u32_per_row + w];
            float s = as_type<float>(uint(Scales_Fp16[(row + 3) * scales_per_row + g_idx]) << 16);
            float b_val = as_type<float>(uint(Biases_Fp16[(row + 3) * scales_per_row + g_idx]) << 16);
            for (uint b = 0; b < 8; ++b) {
                float w_val = float((pack >> (b * 4)) & 0x0F) * s + b_val;
                acc3 += in_norm[b] * w_val;
            }
        }
    }

    // Reducción SIMD final por hilo
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (tid == 0) {
        if (row < D_Out) Out_Vec[row] = acc0;
        if (row + 1 < D_Out) Out_Vec[row + 1] = acc1;
        if (row + 2 < D_Out) Out_Vec[row + 2] = acc2;
        if (row + 3 < D_Out) Out_Vec[row + 3] = acc3;
    }
}
