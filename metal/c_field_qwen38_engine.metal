#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant float ROPE_THETA = 10000000.0f;

inline float bf16_to_f32(ushort val) { return as_type<float>(uint(val) << 16); }
inline float safe_silu(float x) { if (x < -20.0f) return 0.0f; if (x > 20.0f) return x; return x / (1.0f + exp(-x)); }
inline float softplus(float x) { if (x > 20.0f) return x; return log(1.0f + exp(x)); }

// ─── 1. EMBEDDINGS 4-BIT ──────────────────────────────────────────────────────
kernel void project_token_embedding_4bit(
    device const uint32_t& Token_ID      [[buffer(0)]],
    device const uint32_t* Embed_W        [[buffer(1)]],
    device const ushort* Embed_S          [[buffer(2)]],
    device const ushort* Embed_B          [[buffer(3)]],
    device float* Z_Out                   [[buffer(4)]],
    constant uint& D                      [[buffer(5)]],
    uint gid                              [[thread_position_in_grid]])
{
    if (gid >= D) return;
    uint row_u32 = D >> 3;
    uint scales_per_row = D >> 6;
    uint row_offset = Token_ID * row_u32;
    uint scale_base = Token_ID * scales_per_row;
    uint word_idx = gid >> 3;
    uint bit_offset = (gid & 7) << 2;
    uint quant_val = (Embed_W[row_offset + word_idx] >> bit_offset) & 0x0F;
    uint g_idx = gid >> 6;
    float s = bf16_to_f32(Embed_S[scale_base + g_idx]);
    float b = bf16_to_f32(Embed_B[scale_base + g_idx]);
    Z_Out[gid] = float(quant_val) * s + b;
}

// ─── 2. GEMV VECTORIZADO TIPO MLX ─────────────────────────────────────────────
kernel void gemv_4bit_proj_g64(
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
    uint row_offset = row * u32_per_row;
    uint scale_base = row * (D_In >> 6);

    device const uint32_t* row_w = W_U32 + row_offset;
    device const ushort* row_s = Scales + scale_base;
    device const ushort* row_b = Biases + scale_base;

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
        Out_Vec[row] = row_sum;
    }
}

// ─── 3. SWIGLU VECTORIZADO COOPERATIVO ────────────────────────────────────────
kernel void swiglu_dense_4bit_forward(
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
    uint row_offset = row * u32_per_row;
    uint scale_base = row * (D >> 6);

    device const uint32_t* gw = Gate_W + row_offset;
    device const ushort* gs = Gate_S + scale_base;
    device const ushort* gb = Gate_B + scale_base;
    device const uint32_t* uw = Up_W + row_offset;
    device const ushort* us_row = Up_S + scale_base;
    device const ushort* ub = Up_B + scale_base;

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

    float sum_g = simd_sum(acc_g);
    float sum_u = simd_sum(acc_u);

    if (simd_lane == 0) {
        H_Out[row] = safe_silu(sum_g) * sum_u;
    }
}

// ─── 4. CONVOLUCIÓN 1D CAUSAL ───────────────────────────────────────────────
kernel void ssm_conv1d_step_qwen38(
    device const float* qkv_in [[buffer(0)]], device float* conv_state [[buffer(1)]],
    device const ushort* conv_weights [[buffer(2)]], device float* qkv_out [[buffer(3)]],
    constant uint& D_conv [[buffer(4)]], constant uint& state_head [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= D_conv) return;
    uint w_off = gid * 4;
    float w0 = bf16_to_f32(conv_weights[w_off + 0]), w1 = bf16_to_f32(conv_weights[w_off + 1]);
    float w2 = bf16_to_f32(conv_weights[w_off + 2]), w3 = bf16_to_f32(conv_weights[w_off + 3]);
    uint h0 = state_head, h1 = (state_head + 1) % 3, h2 = (state_head + 2) % 3;
    float x_c = qkv_in[gid];
    float res = conv_state[h0 * D_conv + gid] * w0 + conv_state[h1 * D_conv + gid] * w1 + conv_state[h2 * D_conv + gid] * w2 + x_c * w3;
    qkv_out[gid] = safe_silu(res);
    conv_state[h0 * D_conv + gid] = x_c;
}

// ─── 5. GATED DELTA UPDATE COOPERATIVO (PARIDAD DE BIT CERTIFICADA) ───────────
kernel void ssm_gated_delta_update_qwen38(
    device const float* qkv_conv [[buffer(0)]], device const float* z_proj [[buffer(1)]],
    device const float* b_proj [[buffer(2)]], device const float* a_proj [[buffer(3)]],
    device const ushort* A_log_bf16 [[buffer(4)]], device const ushort* dt_bias_bf16 [[buffer(5)]],
    device const ushort* norm_w [[buffer(6)]], device float* S_state [[buffer(7)]],
    device float* out_state [[buffer(8)]],
    uint head_v_idx [[threadgroup_position_in_grid]],
    uint i [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    if (head_v_idx >= 48) return;
    uint head_k_idx = head_v_idx / 3;

    threadgroup float q_norm[128];
    threadgroup float k_norm[128];
    threadgroup float shared_v[128];
    threadgroup float shared_z[128];
    threadgroup float decay;
    threadgroup float beta;
    threadgroup float shared_sq_out[128];

    // Buffers para reducir a través de los 4 SIMDgroups (128 hilos / 32 = 4 grupos)
    threadgroup float simd_reduction_q[4];
    threadgroup float simd_reduction_k[4];
    threadgroup float simd_reduction_out[4];

    device const float* q = qkv_conv + (head_k_idx * 128);
    device const float* k = qkv_conv + 2048 + (head_k_idx * 128);
    device const float* v = qkv_conv + 4096 + (head_v_idx * 128);
    device const float* z = z_proj + (head_v_idx * 128);

    shared_v[i] = v[i];
    shared_z[i] = z[i];

    float qi = q[i];
    float ki = k[i];
    float sq_qi = qi * qi;
    float sq_ki = ki * ki;

    // Reducción SIMD local (32 hilos)
    float s_q = simd_sum(sq_qi);
    float s_k = simd_sum(sq_ki);
    if (simd_lane == 0) {
        simd_reduction_q[simd_id] = s_q;
        simd_reduction_k[simd_id] = s_k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Suma completa exacta de los 128 elementos
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

    // ─── ACTUALIZACIÓN PARALELA POR FILA i DE LA MATRIZ S_t ──────────────────
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

// ─── 6. RMSNORM ─────────────────────────────────────────────────────────────
kernel void rmsnorm_conformal_d5120(
    device const float* Z_In              [[buffer(0)]],
    device const ushort* Gamma_Weight     [[buffer(1)]],
    device float* Z_Out                   [[buffer(2)]],
    constant uint& D                      [[buffer(3)]],
    constant float& Mach_Eps              [[buffer(4)]],
    threadgroup float* shared_sq          [[threadgroup(0)]],
    uint tid                              [[thread_index_in_threadgroup]],
    uint t_per_group                      [[threads_per_threadgroup]])
{
    float local_sq = 0.0f;
    for (uint i = 0; i < 10; ++i) {
        uint idx = tid * 10 + i;
        float val = Z_In[idx];
        local_sq += val * val;
    }
    shared_sq[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) shared_sq[tid] += shared_sq[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total_mean_sq = shared_sq[0] / float(D);
    float inv_rms = rsqrt(total_mean_sq + Mach_Eps);
    for (uint i = 0; i < 10; ++i) {
        uint idx = tid * 10 + i;
        float gamma = bf16_to_f32(Gamma_Weight[idx]);
        Z_Out[idx] = Z_In[idx] * inv_rms * gamma;
    }
}

// ─── 7. GQA EXACTO ───────────────────────────────────────────────────────────
kernel void gqa_attention_exact_qwen38(
    device const float* Q_Raw [[buffer(0)]],
    device const float* K_Raw [[buffer(1)]],
    device const float* V_Raw [[buffer(2)]],
    device const ushort* Q_Norm_W [[buffer(3)]],
    device const ushort* K_Norm_W [[buffer(4)]],
    device float* K_Hist [[buffer(5)]],
    device float* V_Hist [[buffer(6)]],
    device float* Attn_Out [[buffer(7)]],
    constant uint& Pos [[buffer(8)]],
    constant uint& L_Hist [[buffer(9)]],
    uint head_q [[thread_position_in_grid]])
{
    if (head_q >= 24) return;
    uint head_kv = head_q / 6;
    uint q_raw_base = head_q * 512;
    float q_vec[256], gate_vec[256], sq_q = 0.0f;
    for (uint i = 0; i < 256; ++i) {
        float q_val = Q_Raw[q_raw_base + i];
        float g_val = Q_Raw[q_raw_base + 256 + i];
        q_vec[i] = q_val;
        gate_vec[i] = g_val;
        sq_q += q_val * q_val;
    }
    float inv_rms_q = rsqrt((sq_q / 256.0f) + 1e-6f);
    for (uint i = 0; i < 256; ++i) q_vec[i] = q_vec[i] * inv_rms_q * bf16_to_f32(Q_Norm_W[i]);

    for (uint p = 0; p < 32; ++p) {
        float exponent = -2.0f * float(p) / 64.0f;
        float angle = float(Pos) * pow(ROPE_THETA, exponent);
        float cos_m = cos(angle), sin_m = sin(angle);
        float q1 = q_vec[p], q2 = q_vec[32 + p];
        q_vec[p] = q1 * cos_m - q2 * sin_m;
        q_vec[32 + p] = q1 * sin_m + q2 * cos_m;
    }

    if (head_q % 6 == 0) {
        uint kv_idx = head_kv;
        uint k_raw_base = kv_idx * 256;
        float k_vec[256], sq_k = 0.0f;
        for (uint i = 0; i < 256; ++i) {
            float k_val = K_Raw[k_raw_base + i];
            k_vec[i] = k_val;
            sq_k += k_val * k_val;
        }
        float inv_rms_k = rsqrt((sq_k / 256.0f) + 1e-6f);
        for (uint i = 0; i < 256; ++i) k_vec[i] = k_vec[i] * inv_rms_k * bf16_to_f32(K_Norm_W[i]);
        for (uint p = 0; p < 32; ++p) {
            float exponent = -2.0f * float(p) / 64.0f;
            float angle = float(Pos) * pow(ROPE_THETA, exponent);
            float cos_m = cos(angle), sin_m = sin(angle);
            float k1 = k_vec[p], k2 = k_vec[32 + p];
            k_vec[p] = k1 * cos_m - k2 * sin_m;
            k_vec[32 + p] = k1 * sin_m + k2 * cos_m;
        }
        uint hist_offset = Pos * 1024 + kv_idx * 256;
        for (uint i = 0; i < 256; ++i) {
            K_Hist[hist_offset + i] = k_vec[i];
            V_Hist[hist_offset + i] = V_Raw[k_raw_base + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    float scores[4096], max_s = -1e30f;
    for (uint t = 0; t < L_Hist; ++t) {
        uint k_off = t * 1024 + head_kv * 256;
        float dot = 0.0f;
        for (uint d = 0; d < 256; ++d) dot += q_vec[d] * K_Hist[k_off + d];
        float s = dot * 0.0625f;
        scores[t] = s;
        if (s > max_s) max_s = s;
    }
    float sum_exp = 0.0f;
    for (uint t = 0; t < L_Hist; ++t) {
        float w = exp(scores[t] - max_s);
        scores[t] = w;
        sum_exp += w;
    }
    float inv_sum = 1.0f / (sum_exp + 1e-8f);
    float ctx[256];
    for (uint d = 0; d < 256; ++d) ctx[d] = 0.0f;
    for (uint t = 0; t < L_Hist; ++t) {
        float alpha = scores[t] * inv_sum;
        uint v_off = t * 1024 + head_kv * 256;
        for (uint d = 0; d < 256; ++d) ctx[d] += alpha * V_Hist[v_off + d];
    }
    uint out_base = head_q * 256;
    for (uint d = 0; d < 256; ++d) {
        Attn_Out[out_base + d] = ctx[d] * (1.0f / (1.0f + exp(-gate_vec[d])));
    }
}

// ─── 8. KERNEL RESIDUAL IN-PLACE ─────────────────────────────────────────────
kernel void vector_add_inplace(
    device float* a             [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    constant uint& dim          [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]])
{
    if (gid >= dim) return;
    a[gid] += b[gid];
}

// ─── 9. KERNEL FUSIONADO QKV PARA GQA (ALTA OCUPACIÓN DE HARDWARE) ───────────
// Calcula Q (12288), K (1024) y V (1024) de forma contigua en un solo kernel
kernel void gemv_qkv_fused_gqa_4bit(
    device const float* Z_In              [[buffer(0)]],
    device const uint32_t* Q_W            [[buffer(1)]],
    device const ushort* Q_S              [[buffer(2)]],
    device const ushort* Q_B              [[buffer(3)]],
    device float* Q_Out                   [[buffer(4)]],
    device const uint32_t* K_W            [[buffer(5)]],
    device const ushort* K_S              [[buffer(6)]],
    device const ushort* K_B              [[buffer(7)]],
    device float* K_Out                   [[buffer(8)]],
    device const uint32_t* V_W            [[buffer(9)]],
    device const ushort* V_S              [[buffer(10)]],
    device const ushort* V_B              [[buffer(11)]],
    device float* V_Out                   [[buffer(12)]],
    constant uint& D_In                   [[buffer(13)]],
    uint2 tid_in_tg                       [[thread_position_in_threadgroup]],
    uint2 tg_id                           [[threadgroup_position_in_grid]],
    uint simd_lane                        [[thread_index_in_simdgroup]])
{
    // Rango total combinado: Q (12288) + K (1024) + V (1024) = 14336 filas
    uint global_row = tg_id.x * 4 + (tid_in_tg.x / 32);
    if (global_row >= 14336) return;

    device const uint32_t* row_w;
    device const ushort* row_s;
    device const ushort* row_b;
    device float* target_out;
    uint target_row;

    if (global_row < 12288) {
        target_row = global_row;
        row_w = Q_W; row_s = Q_S; row_b = Q_B;
        target_out = Q_Out;
    } else if (global_row < 13312) {
        target_row = global_row - 12288;
        row_w = K_W; row_s = K_S; row_b = K_B;
        target_out = K_Out;
    } else {
        target_row = global_row - 13312;
        row_w = V_W; row_s = V_S; row_b = V_B;
        target_out = V_Out;
    }

    uint u32_per_row = D_In >> 3;
    uint row_offset = target_row * u32_per_row;
    uint scale_base = target_row * (D_In >> 6);

    device const uint32_t* w_ptr = row_w + row_offset;
    device const ushort* s_ptr = row_s + scale_base;
    device const ushort* b_ptr = row_b + scale_base;

    float thread_accum = 0.0f;

    for (uint w = simd_lane; w < u32_per_row; w += 32) {
        uint32_t u = w_ptr[w];
        uint base_k = w << 3;
        uint g_idx = base_k >> 6;
        float s = bf16_to_f32(s_ptr[g_idx]);
        float b = bf16_to_f32(b_ptr[g_idx]);

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
        target_out[target_row] = row_sum;
    }
}
