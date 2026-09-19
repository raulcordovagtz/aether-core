#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

static inline float bf16_to_f32(ushort val) {
    return as_type<float>(uint(val) << 16);
}

static inline float safe_silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return exp(x);
    return log(1.0f + exp(x));
}

// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER-VL :: KERNEL FUSIONADO MAMBA (CONV1D + GATED DELTA ATÓMICO)
// Elimina la materialización de bufQKVConv y fusiona 2 dispatches en 1
// ═════════════════════════════════════════════════════════════════════════════
kernel void ssm_conv_gated_delta_fused(
    device const float* qkv_in          [[buffer(0)]],
    device float* conv_state            [[buffer(1)]],
    device const ushort* conv_weights   [[buffer(2)]],
    constant uint& state_head           [[buffer(3)]],
    device const float* z_proj          [[buffer(4)]],
    device const float* b_proj          [[buffer(5)]],
    device const float* a_proj          [[buffer(6)]],
    device const ushort* A_log_bf16     [[buffer(7)]],
    device const ushort* dt_bias_bf16   [[buffer(8)]],
    device const ushort* norm_w         [[buffer(9)]],
    device float* S_state               [[buffer(10)]],
    device float* out_state             [[buffer(11)]],
    uint head_v_idx                     [[threadgroup_position_in_grid]], // 0..47
    uint i                              [[thread_position_in_threadgroup]], // 0..127
    uint simd_id                        [[simdgroup_index_in_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]])
{
    if (head_v_idx >= 48) return;
    uint head_k_idx = head_v_idx / 3;

    threadgroup float q_norm[128];
    threadgroup float k_norm[128];
    threadgroup float shared_v[128];
    threadgroup float shared_z[128];
    threadgroup float decay;
    threadgroup float beta;

    threadgroup float simd_reduction_q[4];
    threadgroup float simd_reduction_k[4];
    threadgroup float simd_reduction_out[4];

    // 1. Convolución 1D evaluada directamente en registros locales
    uint h0 = state_head, h1 = (state_head + 1) % 3, h2 = (state_head + 2) % 3;
    const uint D_conv = 10240;

    auto eval_conv = [&](uint gid) -> float {
        uint w_off = gid * 4;
        float w0 = bf16_to_f32(conv_weights[w_off + 0]), w1 = bf16_to_f32(conv_weights[w_off + 1]);
        float w2 = bf16_to_f32(conv_weights[w_off + 2]), w3 = bf16_to_f32(conv_weights[w_off + 3]);
        float x_c = qkv_in[gid];
        float res = conv_state[h0 * D_conv + gid] * w0 + 
                    conv_state[h1 * D_conv + gid] * w1 + 
                    conv_state[h2 * D_conv + gid] * w2 + x_c * w3;
        conv_state[h0 * D_conv + gid] = x_c;
        return safe_silu(res);
    };

    uint gid_q = head_k_idx * 128 + i;
    uint gid_k = 2048 + head_k_idx * 128 + i;
    uint gid_v = 4096 + head_v_idx * 128 + i;

    float qi = eval_conv(gid_q);
    float ki = eval_conv(gid_k);
    float vi = eval_conv(gid_v);

    shared_v[i] = vi;
    shared_z[i] = z_proj[head_v_idx * 128 + i];

    // 2. Reducción cuadrática de normas Q y K
    float sq_qi = qi * qi;
    float sq_ki = ki * ki;

    float s_q = simd_sum(sq_qi);
    float s_k = simd_sum(sq_ki);
    if (simd_lane == 0) {
        simd_reduction_q[simd_id] = s_q;
        simd_reduction_k[simd_id] = s_k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq_q = simd_reduction_q[0] + simd_reduction_q[1] + simd_reduction_q[2] + simd_reduction_q[3];
    float total_sq_k = simd_reduction_k[0] + simd_reduction_k[1] + simd_reduction_k[2] + simd_reduction_k[3];

    if (i == 0) {
        float b_val = b_proj[head_v_idx], a_val = a_proj[head_v_idx];
        float A_log = bf16_to_f32(A_log_bf16[head_v_idx]);
        float dt_b  = bf16_to_f32(dt_bias_bf16[head_v_idx]);
        float g_t = -exp(A_log) * softplus(dt_b + a_val);
        decay = exp(g_t);
        beta = 1.0f / (1.0f + exp(-b_val));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_scale = 1.0f / sqrt(128.0f);
    float rms_q = rsqrt((total_sq_q / 128.0f) + 1e-6f);
    float rms_k = rsqrt((total_sq_k / 128.0f) + 1e-6f);
    q_norm[i] = qi * (inv_scale * inv_scale) * rms_q;
    k_norm[i] = ki * inv_scale * rms_k;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Actualización de matriz de estado S_state
    device float* S_row = S_state + (head_v_idx * 128 * 128) + (i * 128);
    float kv_mem_i = 0.0f;
    for (uint j = 0; j < 128; ++j) {
        float s_val = S_row[j] * decay;
        S_row[j] = s_val;
        kv_mem_i += s_val * k_norm[j];
    }

    float delta_i = (shared_v[i] - kv_mem_i) * beta;
    float out_raw_i = 0.0f;

    for (uint j = 0; j < 128; ++j) {
        float s_new = S_row[j] + k_norm[j] * delta_i;
        S_row[j] = s_new;
        out_raw_i += s_new * q_norm[j];
    }

    float sq_out_i = out_raw_i * out_raw_i;
    float s_out = simd_sum(sq_out_i);
    if (simd_lane == 0) {
        simd_reduction_out[simd_id] = s_out;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq_out = simd_reduction_out[0] + simd_reduction_out[1] + simd_reduction_out[2] + simd_reduction_out[3];
    float inv_rms_out = rsqrt((total_sq_out / 128.0f) + 1e-6f);

    float n_w = bf16_to_f32(norm_w[i]);
    out_state[head_v_idx * 128 + i] = out_raw_i * inv_rms_out * n_w * safe_silu(shared_z[i]);
}
