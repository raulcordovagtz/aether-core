#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

static inline float bf16_to_f32(ushort val) {
    return as_type<float>(uint(val) << 16);
}

// ─── KERNEL 1: ESCALAR BASE (uint32_t, 32-bit loads) ─────────────────────────
kernel void gemv_scalar_u32(
    device const float* Z_In              [[buffer(0)]],
    device const uint32_t* W_U32          [[buffer(1)]],
    device const ushort* Scales           [[buffer(2)]],
    device const ushort* Biases           [[buffer(3)]],
    device float* Out_Vec                 [[buffer(4)]],
    constant uint& D_In                   [[buffer(5)]],
    constant uint& D_Out                  [[buffer(6)]],
    uint2 tid_in_tg                       [[thread_position_in_threadgroup]],
    uint2 tg_id                           [[threadgroup_position_in_grid]],
    uint simd_lane                        [[thread_index_in_simdgroup]])
{
    uint row = tg_id.x * 4 + (tid_in_tg.x / 32);
    if (row >= D_Out) return;

    uint u32_per_row = D_In >> 3;
    device const uint32_t* row_w = W_U32 + row * u32_per_row;
    device const ushort* row_s = Scales + row * (D_In >> 6);
    device const ushort* row_b = Biases + row * (D_In >> 6);

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
    if (simd_lane == 0) Out_Vec[row] = row_sum;
}

// ─── KERNEL 2: VECTORIZADO TIPO MLX (uint4, 128-bit loads) ────────────────────
kernel void gemv_vector_u4(
    device const float* Z_In              [[buffer(0)]],
    device const uint4* W_U4              [[buffer(1)]], // 128-bit loads
    device const ushort* Scales           [[buffer(2)]],
    device const ushort* Biases           [[buffer(3)]],
    device float* Out_Vec                 [[buffer(4)]],
    constant uint& D_In                   [[buffer(5)]],
    constant uint& D_Out                  [[buffer(6)]],
    uint2 tid_in_tg                       [[thread_position_in_threadgroup]],
    uint2 tg_id                           [[threadgroup_position_in_grid]],
    uint simd_lane                        [[thread_index_in_simdgroup]])
{
    uint row = tg_id.x * 4 + (tid_in_tg.x / 32);
    if (row >= D_Out) return;

    uint u4_per_row = D_In >> 5; // D_In / 32 = 160 uint4s por fila
    device const uint4* row_w = W_U4 + row * u4_per_row;
    device const ushort* row_s = Scales + row * (D_In >> 6);
    device const ushort* row_b = Biases + row * (D_In >> 6);

    float thread_accum = 0.0f;

    for (uint w = simd_lane; w < u4_per_row; w += 32) {
        uint4 u4_val = row_w[w];
        uint base_k = w << 5; // Cada uint4 abarca 32 floats
        uint g_idx = base_k >> 6;
        float s = bf16_to_f32(row_s[g_idx]);
        float b = bf16_to_f32(row_b[g_idx]);

        #define UNPACK_AND_MADD(u, offset) \
            thread_accum += Z_In[base_k + offset + 0] * (float((u >>  0) & 0x0F) * s + b); \
            thread_accum += Z_In[base_k + offset + 1] * (float((u >>  4) & 0x0F) * s + b); \
            thread_accum += Z_In[base_k + offset + 2] * (float((u >>  8) & 0x0F) * s + b); \
            thread_accum += Z_In[base_k + offset + 3] * (float((u >> 12) & 0x0F) * s + b); \
            thread_accum += Z_In[base_k + offset + 4] * (float((u >> 16) & 0x0F) * s + b); \
            thread_accum += Z_In[base_k + offset + 5] * (float((u >> 20) & 0x0F) * s + b); \
            thread_accum += Z_In[base_k + offset + 6] * (float((u >> 24) & 0x0F) * s + b); \
            thread_accum += Z_In[base_k + offset + 7] * (float((u >> 28) & 0x0F) * s + b);

        UNPACK_AND_MADD(u4_val.x, 0);
        UNPACK_AND_MADD(u4_val.y, 8);
        UNPACK_AND_MADD(u4_val.z, 16);
        UNPACK_AND_MADD(u4_val.w, 24);
        #undef UNPACK_AND_MADD
    }

    float row_sum = simd_sum(thread_accum);
    if (simd_lane == 0) Out_Vec[row] = row_sum;
}

// ─── KERNEL 3: CACHÉ SRAM COMPARTIDA EN THREADGROUP (20.48 KB) ────────────────
kernel void gemv_sram_cached(
    device const float* Z_In              [[buffer(0)]],
    device const uint4* W_U4              [[buffer(1)]],
    device const ushort* Scales           [[buffer(2)]],
    device const ushort* Biases           [[buffer(3)]],
    device float* Out_Vec                 [[buffer(4)]],
    constant uint& D_In                   [[buffer(5)]],
    constant uint& D_Out                  [[buffer(6)]],
    threadgroup float* shared_Z           [[threadgroup(0)]], // 5120 floats en SRAM local
    uint tid_in_tg                        [[thread_index_in_threadgroup]],
    uint2 tg_id                           [[threadgroup_position_in_grid]],
    uint simd_id                          [[simdgroup_index_in_threadgroup]],
    uint simd_lane                        [[thread_index_in_simdgroup]])
{
    // 1. Carga cooperativa de Z_In en SRAM local (128 hilos cargan 5120 floats = 40 por hilo)
    for (uint i = tid_in_tg; i < D_In; i += 128) {
        shared_Z[i] = Z_In[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Cada uno de los 4 simdgroups procesa 1 fila leyendo Z desde SRAM
    uint row = tg_id.x * 4 + simd_id;
    if (row >= D_Out) return;

    uint u4_per_row = D_In >> 5; // 160 uint4s
    device const uint4* row_w = W_U4 + row * u4_per_row;
    device const ushort* row_s = Scales + row * (D_In >> 6);
    device const ushort* row_b = Biases + row * (D_In >> 6);

    float thread_accum = 0.0f;

    for (uint w = simd_lane; w < u4_per_row; w += 32) {
        uint4 u4_val = row_w[w];
        uint base_k = w << 5;
        uint g_idx = base_k >> 6;
        float s = bf16_to_f32(row_s[g_idx]);
        float b = bf16_to_f32(row_b[g_idx]);

        #define MADD_SRAM(u, off) \
            thread_accum += shared_Z[base_k + off + 0] * (float((u >>  0) & 0x0F) * s + b); \
            thread_accum += shared_Z[base_k + off + 1] * (float((u >>  4) & 0x0F) * s + b); \
            thread_accum += shared_Z[base_k + off + 2] * (float((u >>  8) & 0x0F) * s + b); \
            thread_accum += shared_Z[base_k + off + 3] * (float((u >> 12) & 0x0F) * s + b); \
            thread_accum += shared_Z[base_k + off + 4] * (float((u >> 16) & 0x0F) * s + b); \
            thread_accum += shared_Z[base_k + off + 5] * (float((u >> 20) & 0x0F) * s + b); \
            thread_accum += shared_Z[base_k + off + 6] * (float((u >> 24) & 0x0F) * s + b); \
            thread_accum += shared_Z[base_k + off + 7] * (float((u >> 28) & 0x0F) * s + b);

        MADD_SRAM(u4_val.x, 0);
        MADD_SRAM(u4_val.y, 8);
        MADD_SRAM(u4_val.z, 16);
        MADD_SRAM(u4_val.w, 24);
        #undef MADD_SRAM
    }

    float row_sum = simd_sum(thread_accum);
    if (simd_lane == 0) Out_Vec[row] = row_sum;
}
