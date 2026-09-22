// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: GEODESIC PROJECTIVE TRAJECTORY CELL
// HITO 1.1 — METAL GPU KERNEL (REDUCCIÓN DETERMINISTA EN 2 ETAPAS)
// ═════════════════════════════════════════════════════════════════════════════
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

struct CellMetricsGPU {
    float correlation_r;
    float curvature_kappa;
    float kinetic_energy;
    float angular_displacement;
    float dirichlet_tension;
    float permeability_gate;
    uint  regime;
    uint  active_mode;
};

kernel void dispatch_geodesic_trajectory_cell_step(
    device const float*       h_in         [[buffer(0)]],
    device const float*       v_drag       [[buffer(1)]],
    device const float*       a_flow       [[buffer(2)]],
    device const float*       u_attractor  [[buffer(3)]],
    device float*             h_projected  [[buffer(4)]],
    device float*             h_deflated   [[buffer(5)]],
    device CellMetricsGPU*    metrics_out  [[buffer(6)]],
    constant float&           tau          [[buffer(7)]],
    constant float&           kappa_att    [[buffer(8)]],
    constant float&           beta_perm    [[buffer(9)]],
    constant float&           theta_perm   [[buffer(10)]],
    constant uint&            mode         [[buffer(11)]],
    constant uint&            D            [[buffer(12)]],
    threadgroup float*        shared_acc   [[threadgroup(0)]],
    uint tid                               [[thread_index_in_threadgroup]],
    uint t_per_group                       [[threads_per_threadgroup]]
) {
    // 1. Reducciones de entrada
    float l_sq_h   = 0.0f, l_sq_v   = 0.0f, l_sq_a   = 0.0f;
    float l_dot_hv = 0.0f, l_dot_va = 0.0f, l_dot_hu = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {
        const float h = h_in[i];
        const float v = v_drag[i];
        const float a = a_flow[i];
        const float u = u_attractor[i];

        l_sq_h   += h * h;
        l_sq_v   += v * v;
        l_sq_a   += a * a;
        l_dot_hv += h * v;
        l_dot_va += v * a;
        l_dot_hu += h * u;
    }

    const uint base = tid * 6;
    shared_acc[base + 0] = l_sq_h;
    shared_acc[base + 1] = l_sq_v;
    shared_acc[base + 2] = l_sq_a;
    shared_acc[base + 3] = l_dot_hv;
    shared_acc[base + 4] = l_dot_va;
    shared_acc[base + 5] = l_dot_hu;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const uint dst = tid * 6;
            const uint src = (tid + s) * 6;
            for (uint k = 0; k < 6; ++k) {
                shared_acc[dst + k] += shared_acc[src + k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float sq_h   = shared_acc[0];
    const float sq_v   = shared_acc[1];
    const float sq_a   = shared_acc[2];
    const float dot_hv = shared_acc[3];
    const float dot_va = shared_acc[4];
    const float dot_hu = shared_acc[5];

    // 2. Descomposición tangencial y curvatura de Lagrange
    const float inv_norm_h = rsqrt(max(sq_h, 0.0f) + 1e-12f);
    const float coeff_hv   = dot_hv * inv_norm_h * inv_norm_h;
    const float sq_v_perp  = max(0.0f, sq_v - (dot_hv * dot_hv * inv_norm_h * inv_norm_h));
    const float norm_v_perp = sqrt(sq_v_perp + 1e-12f);
    const float inv_norm_vp = rsqrt(sq_v_perp + 1e-12f);

    const float bivector_sq   = max(0.0f, (sq_v * sq_a) - (dot_va * dot_va));
    const float bivector_norm = sqrt(bivector_sq);
    const float kappa_kin     = bivector_norm / (pow(max(sq_v, 0.0f), 1.5f) + 1e-12f);

    const float omega = norm_v_perp * inv_norm_h;
    const float theta = omega * tau;
    const float cos_t = cos(theta);
    const float sin_t = sin(theta);

    const float q_tension = sq_v_perp / (sq_h + 1e-12f);
    const float g_perm    = 1.0f / (1.0f + exp(-beta_perm * (q_tension - theta_perm)));

    // 3. Proyección Geodésica con aceleración tangencial
    float l_sq_out    = 0.0f;
    float l_dot_out_u = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {
        const float h = h_in[i];
        const float v = v_drag[i];
        const float u = u_attractor[i];

        const float v_perp = v - coeff_hv * h;
        const float v_hat  = v_perp * inv_norm_vp;
        const float a_attractor = kappa_att * (u - (dot_hu * inv_norm_h * inv_norm_h) * h);

        const float h_star = (cos_t * h) + (sin_t * v_hat) + (0.5f * tau * tau * a_attractor);
        h_projected[i] = h_star;

        l_sq_out    += h_star * h_star;
        l_dot_out_u += h_star * u;
    }

    // 4. Reducción de norma de h_projected
    shared_acc[tid * 2 + 0] = l_sq_out;
    shared_acc[tid * 2 + 1] = l_dot_out_u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const uint dst = tid * 2;
            const uint src = (tid + s) * 2;
            shared_acc[dst + 0] += shared_acc[src + 0];
            shared_acc[dst + 1] += shared_acc[src + 1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float final_sq_out  = shared_acc[0];
    const float raw_dot_out   = shared_acc[1];
    const float inv_norm_star = rsqrt(final_sq_out + 1e-12f);
    const float correlation_r = raw_dot_out * inv_norm_star;

    // Normalizar h_projected a h*
    for (uint i = tid; i < D; i += t_per_group) {
        h_projected[i] *= inv_norm_star;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. Segunda Reducción: dot(h_in, h_star) para deflación exacta <h*, h_deflated> = 0
    float l_dot_h_hstar = 0.0f;
    for (uint i = tid; i < D; i += t_per_group) {
        l_dot_h_hstar += h_in[i] * h_projected[i];
    }

    shared_acc[tid] = l_dot_h_hstar;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid] += shared_acc[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float dot_h_hstar = shared_acc[0];

    // 6. Deflación Gram-Schmidt ortogonal
    for (uint i = tid; i < D; i += t_per_group) {
        const float h_star = h_projected[i];
        h_deflated[i] = h_in[i] - dot_h_hstar * h_star;
    }

    // 7. Telemetría de salida
    if (tid == 0) {
        const uint flow_regime = (correlation_r >= 0.95f) ? 0 : ((correlation_r >= 0.80f) ? 1 : 2);
        metrics_out->correlation_r        = correlation_r;
        metrics_out->curvature_kappa      = kappa_kin;
        metrics_out->kinetic_energy       = 0.5f * sq_v_perp;
        metrics_out->angular_displacement = theta;
        metrics_out->dirichlet_tension    = q_tension;
        metrics_out->permeability_gate    = g_perm;
        metrics_out->regime               = flow_regime;
        metrics_out->active_mode          = mode;
    }
}
