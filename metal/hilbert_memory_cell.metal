// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: HILBERT MEMORY CELL GPU KERNEL (HITO 2.1-B)
// Álgebra de Hilbert y Transporte de Estilo con Reducción SRAM en Silicio
// ═════════════════════════════════════════════════════════════════════════════
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Motivos de degeneración mapeados a enum C++
// 0: NONE, 1: ZERO_SUPERPOSITION, 2: COLLINEAR_TARGET, 3: ZERO_TANGENT, 4: INVALID_NORM

// 1. Kernel Metal: Empaquetamiento de Superposición de 2 Hechos (AND-like)
kernel void dispatch_hilbert_pack_two(
    device const float* m_A          [[buffer(0)]],
    device const float* m_B          [[buffer(1)]],
    device float*       m_out        [[buffer(2)]],
    device uint*        degen_reason [[buffer(3)]],
    constant uint&      D            [[buffer(4)]],
    threadgroup float*  sh_acc       [[threadgroup(0)]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint t_per_group                 [[threads_per_threadgroup]]
) {
    float l_sq = 0.0f;
    for (uint i = tid; i < D; i += t_per_group) {
        float val = m_A[i] + m_B[i];
        m_out[i] = val;
        l_sq += val * val;
    }
    sh_acc[tid] = l_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) sh_acc[tid] += sh_acc[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sq_sum = sh_acc[0];
    if (sq_sum < 1e-10f) {
        for (uint i = tid; i < D; i += t_per_group) m_out[i] = 0.0f;
        if (tid == 0) degen_reason[0] = 1; // ZERO_SUPERPOSITION
        return;
    }

    float inv_norm = rsqrt(sq_sum);
    for (uint i = tid; i < D; i += t_per_group) {
        m_out[i] *= inv_norm;
    }
    if (tid == 0) degen_reason[0] = 0; // NONE
}

// 2. Kernel Metal: Deflación Ortogonal (NOT-like / Match & Peel)
kernel void dispatch_hilbert_deflation(
    device const float* m_pack       [[buffer(0)]],
    device const float* m_target     [[buffer(1)]],
    device float*       m_out        [[buffer(2)]],
    device uint*        degen_reason [[buffer(3)]],
    constant uint&      D            [[buffer(4)]],
    threadgroup float*  sh_acc       [[threadgroup(0)]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint t_per_group                 [[threads_per_threadgroup]]
) {
    float l_dot = 0.0f;
    for (uint i = tid; i < D; i += t_per_group) {
        l_dot += m_pack[i] * m_target[i];
    }
    sh_acc[tid] = l_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) sh_acc[tid] += sh_acc[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot_proj = sh_acc[0];

    float l_sq = 0.0f;
    for (uint i = tid; i < D; i += t_per_group) {
        float val = m_pack[i] - dot_proj * m_target[i];
        m_out[i] = val;
        l_sq += val * val;
    }
    sh_acc[tid] = l_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) sh_acc[tid] += sh_acc[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sq_rem = sh_acc[0];
    if (sq_rem < 1e-10f) {
        for (uint i = tid; i < D; i += t_per_group) m_out[i] = 0.0f;
        if (tid == 0) degen_reason[0] = 2; // COLLINEAR_TARGET
        return;
    }

    float inv_norm = rsqrt(sq_rem);
    for (uint i = tid; i < D; i += t_per_group) {
        m_out[i] *= inv_norm;
    }
    if (tid == 0) degen_reason[0] = 0; // NONE
}

// 3. Kernel Metal: Transporte Paralelo de Estilo
kernel void dispatch_hilbert_style_transport(
    device const float* h_truth      [[buffer(0)]],
    device const float* u_style      [[buffer(1)]],
    device float*       h_out        [[buffer(2)]],
    device uint*        degen_reason [[buffer(3)]],
    constant float&     theta_s      [[buffer(4)]],
    constant uint&      D            [[buffer(5)]],
    threadgroup float*  sh_acc       [[threadgroup(0)]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint t_per_group                 [[threads_per_threadgroup]]
) {
    float l_dot = 0.0f;
    for (uint i = tid; i < D; i += t_per_group) {
        l_dot += h_truth[i] * u_style[i];
    }
    sh_acc[tid] = l_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) sh_acc[tid] += sh_acc[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot_ts = sh_acc[0];

    float l_sq = 0.0f;
    for (uint i = tid; i < D; i += t_per_group) {
        float v = u_style[i] - dot_ts * h_truth[i];
        h_out[i] = v;
        l_sq += v * v;
    }
    sh_acc[tid] = l_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) sh_acc[tid] += sh_acc[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sq_v = sh_acc[0];
    if (sq_v < 1e-10f) {
        for (uint i = tid; i < D; i += t_per_group) h_out[i] = h_truth[i];
        if (tid == 0) degen_reason[0] = 3; // ZERO_TANGENT
        return;
    }

    float inv_norm_v = rsqrt(sq_v);
    float cos_s = cos(theta_s);
    float sin_s = sin(theta_s);

    for (uint i = tid; i < D; i += t_per_group) {
        h_out[i] = cos_s * h_truth[i] + sin_s * (h_out[i] * inv_norm_v);
    }
    if (tid == 0) degen_reason[0] = 0; // NONE
}
