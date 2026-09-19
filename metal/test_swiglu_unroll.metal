#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

static inline float bf16_to_f32(ushort val) {
    return as_type<float>(uint(val) << 16);
}

static inline float safe_silu(float x) {
    return x / (1.0f + exp(-x));
}

// ─── KERNEL 1: CON BUCLE DINÁMICO (ACTUAL) ───────────────────────────────────
kernel void swiglu_dynamic_loop(
    device const float* Z_Norm            [[buffer(0)]],
    device const uint32_t* Gate_W         [[buffer(1)]],
    device const ushort* Gate_S           [[buffer(2)]],
    device const ushort* Gate_B           [[buffer(3)]],
    device const uint32_t* Up_W           [[buffer(4)]],
    device const ushort* Up_S             [[buffer(5)]],
    device const ushort* Up_B             [[buffer(6)]],
    device float* H_Out                   [[buffer(7)]],
    constant uint& D                      [[buffer(8)]],
    constant uint& I_Dim                  [[buffer(9)]],
    uint2 tid_in_tg                       [[thread_position_in_threadgroup]],
    uint2 tg_id                           [[threadgroup_position_in_grid]],
    uint simd_lane                        [[thread_index_in_simdgroup]])
{
    uint row = tg_id.x * 4 + (tid_in_tg.x / 32);
    if (row >= I_Dim) return;

    uint u32_per_row = D >> 3;
    device const uint32_t* gw = Gate_W + row * u32_per_row;
    device const ushort* gs = Gate_S + row * (D >> 6);
    device const ushort* gb = Gate_B + row * (D >> 6);
    device const uint32_t* uw = Up_W + row * u32_per_row;
    device const ushort* us_row = Up_S + row * (D >> 6);
    device const ushort* ub = Up_B + row * (D >> 6);

    float acc_g = 0.0f, acc_u = 0.0f;

    for (uint w = simd_lane; w < u32_per_row; w += 32) {
        uint32_t gu = gw[w];
        uint32_t uu = uw[w];
        uint base_k = w << 3;
        uint g_idx = base_k >> 6;
        float g_s = bf16_to_f32(gs[g_idx]), g_b = bf16_to_f32(gb[g_idx]);
        float u_s = bf16_to_f32(us_row[g_idx]), u_b = bf16_to_f32(ub[g_idx]);

        for (uint b_idx = 0; b_idx < 8; ++b_idx) {
            float z_val = Z_Norm[base_k + b_idx];
            uint g_val = (gu >> (b_idx * 4)) & 0x0F;
            uint u_val = (uu >> (b_idx * 4)) & 0x0F;
            acc_g += z_val * (float(g_val) * g_s + g_b);
            acc_u += z_val * (float(u_val) * u_s + u_b);
        }
    }

    float g_sum = simd_sum(acc_g);
    float u_sum = simd_sum(acc_u);
    if (simd_lane == 0) {
        H_Out[row] = safe_silu(g_sum) * u_sum;
    }
}

// ─── KERNEL 2: TOTALMENTE DESENROLLADO (SIMD FMA DIRECTO) ────────────────────
kernel void swiglu_unrolled_fma(
    device const float* Z_Norm            [[buffer(0)]],
    device const uint32_t* Gate_W         [[buffer(1)]],
    device const ushort* Gate_S           [[buffer(2)]],
    device const ushort* Gate_B           [[buffer(3)]],
    device const uint32_t* Up_W           [[buffer(4)]],
    device const ushort* Up_S             [[buffer(5)]],
    device const ushort* Up_B             [[buffer(6)]],
    device float* H_Out                   [[buffer(7)]],
    constant uint& D                      [[buffer(8)]],
    constant uint& I_Dim                  [[buffer(9)]],
    uint2 tid_in_tg                       [[thread_position_in_threadgroup]],
    uint2 tg_id                           [[threadgroup_position_in_grid]],
    uint simd_lane                        [[thread_index_in_simdgroup]])
{
    uint row = tg_id.x * 4 + (tid_in_tg.x / 32);
    if (row >= I_Dim) return;

    uint u32_per_row = D >> 3;
    device const uint32_t* gw = Gate_W + row * u32_per_row;
    device const ushort* gs = Gate_S + row * (D >> 6);
    device const ushort* gb = Gate_B + row * (D >> 6);
    device const uint32_t* uw = Up_W + row * u32_per_row;
    device const ushort* us_row = Up_S + row * (D >> 6);
    device const ushort* ub = Up_B + row * (D >> 6);

    float acc_g = 0.0f, acc_u = 0.0f;

    for (uint w = simd_lane; w < u32_per_row; w += 32) {
        uint32_t gu = gw[w];
        uint32_t uu = uw[w];
        uint base_k = w << 3;
        uint g_idx = base_k >> 6;
        float g_s = bf16_to_f32(gs[g_idx]), g_b = bf16_to_f32(gb[g_idx]);
        float u_s = bf16_to_f32(us_row[g_idx]), u_b = bf16_to_f32(ub[g_idx]);

        #define FMA8(off, shift) \
            { float z = Z_Norm[base_k + off]; \
              acc_g += z * (float((gu >> shift) & 0x0F) * g_s + g_b); \
              acc_u += z * (float((uu >> shift) & 0x0F) * u_s + u_b); }

        FMA8(0, 0);  FMA8(1, 4);  FMA8(2, 8);  FMA8(3, 12);
        FMA8(4, 16); FMA8(5, 20); FMA8(6, 24); FMA8(7, 28);
        #undef FMA8
    }

    float g_sum = simd_sum(acc_g);
    float u_sum = simd_sum(acc_u);
    if (simd_lane == 0) {
        H_Out[row] = safe_silu(g_sum) * u_sum;
    }
}
