#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

static inline float bf16_to_f32(ushort val) {
    uint u32 = uint(val) << 16;
    return as_type<float>(u32);
}

// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER-VL :: KERNEL FUSIONADO DE ENTRADA MAMBA (4-EN-1 ATÓMICO)
// Fusión de QKV (10240) + Z (6144) + B (48) + A (48) = 16,480 filas
// ═════════════════════════════════════════════════════════════════════════════
kernel void gemv_4bit_mamba_in4_fused(
    device const float* Z_In              [[buffer(0)]],
    // QKV
    device const uint32_t* QKV_W          [[buffer(1)]],
    device const ushort*   QKV_S          [[buffer(2)]],
    device const ushort*   QKV_B          [[buffer(3)]],
    device float*          Out_QKV        [[buffer(4)]],
    // Z
    device const uint32_t* Z_W            [[buffer(5)]],
    device const ushort*   Z_S            [[buffer(6)]],
    device const ushort*   Z_B            [[buffer(7)]],
    device float*          Out_Z          [[buffer(8)]],
    // B
    device const uint32_t* B_W            [[buffer(9)]],
    device const ushort*   B_S            [[buffer(10)]],
    device const ushort*   B_B            [[buffer(11)]],
    device float*          Out_B          [[buffer(12)]],
    // A
    device const uint32_t* A_W            [[buffer(13)]],
    device const ushort*   A_S            [[buffer(14)]],
    device const ushort*   A_B            [[buffer(15)]],
    device float*          Out_A          [[buffer(16)]],
    constant uint&         D_In           [[buffer(17)]],
    uint2 tid_in_tg                       [[thread_position_in_threadgroup]],
    uint2 tg_id                           [[threadgroup_position_in_grid]],
    uint  simd_lane                       [[thread_index_in_simdgroup]])
{
    uint global_row = tg_id.x * 4 + (tid_in_tg.x / 32);
    if (global_row >= 16480) return;

    device const uint32_t* W_ptr = nullptr;
    device const ushort*   S_ptr = nullptr;
    device const ushort*   B_ptr = nullptr;
    device float*          Out_ptr = nullptr;
    uint row_in_mat = 0;

    if (global_row < 10240) {
        // Matriz QKV (10,240 filas)
        row_in_mat = global_row;
        W_ptr = QKV_W;
        S_ptr = QKV_S;
        B_ptr = QKV_B;
        Out_ptr = Out_QKV;
    } else if (global_row < 16384) {
        // Matriz Z (6,144 filas)
        row_in_mat = global_row - 10240;
        W_ptr = Z_W;
        S_ptr = Z_S;
        B_ptr = Z_B;
        Out_ptr = Out_Z;
    } else if (global_row < 16432) {
        // Matriz B (48 filas)
        row_in_mat = global_row - 16384;
        W_ptr = B_W;
        S_ptr = B_S;
        B_ptr = B_B;
        Out_ptr = Out_B;
    } else {
        // Matriz A (48 filas)
        row_in_mat = global_row - 16432;
        W_ptr = A_W;
        S_ptr = A_S;
        B_ptr = A_B;
        Out_ptr = Out_A;
    }

    uint u32_per_row = D_In >> 3; // 5120 >> 3 = 640
    uint row_offset = row_in_mat * u32_per_row;
    uint scale_base = row_in_mat * (D_In >> 6); // 5120 >> 6 = 80

    device const uint32_t* row_w = W_ptr + row_offset;
    device const ushort*   row_s = S_ptr + scale_base;
    device const ushort*   row_b = B_ptr + scale_base;

    float thread_accum = 0.0f;

    for (uint w = simd_lane; w < u32_per_row; w += 32) {
        uint32_t u = row_w[w];
        uint base_k = w << 3;
        uint g_idx = base_k >> 6;
        float s = bf16_to_f32(row_s[g_idx]);
        float b = bf16_to_f32(row_b[g_idx]);

        thread_accum += Z_In[base_k + 0] * (float((u >>  0) & 0x0F) * s + b);
        thread_accum += Z_In[base_k + 1] * (float((u >>  4) & 0x0F) * s + b);
        thread_accum += Z_In[base_k + 2] * (float((u >>  8) & 0x0F) * s + b);
        thread_accum += Z_In[base_k + 3] * (float((u >> 12) & 0x0F) * s + b);
        thread_accum += Z_In[base_k + 4] * (float((u >> 16) & 0x0F) * s + b);
        thread_accum += Z_In[base_k + 5] * (float((u >> 20) & 0x0F) * s + b);
        thread_accum += Z_In[base_k + 6] * (float((u >> 24) & 0x0F) * s + b);
        thread_accum += Z_In[base_k + 7] * (float((u >> 28) & 0x0F) * s + b);
    }

    float row_sum = simd_sum(thread_accum);
    if (simd_lane == 0) {
        Out_ptr[row_in_mat] = row_sum;
    }
}
