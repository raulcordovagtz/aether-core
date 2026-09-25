// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: TETRAPOLAR PREDICTOR CELL GPU KERNEL
// Trazado Geodésico en S^{D-1} y Reducción de 4 Líneas Derivativas en SRAM
// ═════════════════════════════════════════════════════════════════════════════
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

struct TetrapolarTelemetryGPU {
    float omega_angular_velocity;
    float curvature_kappa;
    float grad_onto;
    float grad_teleo;
    float grad_anti;
    float grad_eos;
    float teleology_alignment;
};

kernel void dispatch_tetrapolar_predictor_step(
    device const float*            h_in          [[buffer(0)]],
    device const float*            v_tangent     [[buffer(1)]],
    device const float*            u_onto        [[buffer(2)]],
    device const float*            u_teleo       [[buffer(3)]],
    device const float*            u_anti        [[buffer(4)]],
    device const float*            u_eos         [[buffer(5)]],
    device float*                  h_star_out    [[buffer(6)]],
    device TetrapolarTelemetryGPU* telemetry_out [[buffer(7)]],
    constant float&                tau           [[buffer(8)]],
    constant uint&                 D             [[buffer(9)]],
    threadgroup float*             sh_acc        [[threadgroup(0)]],
    uint tid                                     [[thread_index_in_threadgroup]],
    uint t_per_group                             [[threads_per_threadgroup]]
) {
    // 1. Reducción en paralelo de normas y producto escalar inicial
    float l_sq_h = 0.0f;
    float l_sq_v = 0.0f;
    float l_dot_hv = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {
        float h = h_in[i];
        float v = v_tangent[i];
        l_sq_h   += h * h;
        l_sq_v   += v * v;
        l_dot_hv += h * v;
    }

    sh_acc[tid * 3 + 0] = l_sq_h;
    sh_acc[tid * 3 + 1] = l_sq_v;
    sh_acc[tid * 3 + 2] = l_dot_hv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_acc[tid * 3 + 0] += sh_acc[(tid + s) * 3 + 0];
            sh_acc[tid * 3 + 1] += sh_acc[(tid + s) * 3 + 1];
            sh_acc[tid * 3 + 2] += sh_acc[(tid + s) * 3 + 2];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sq_h   = sh_acc[0];
    float sq_v   = sh_acc[1];
    float dot_hv = sh_acc[2];

    float norm_h = sqrt(max(sq_h, 0.0f) + 1e-12f);
    float inv_norm_h = 1.0f / norm_h;

    float coeff_hv = dot_hv * inv_norm_h * inv_norm_h;
    float sq_v_perp = max(0.0f, sq_v - (dot_hv * dot_hv * inv_norm_h * inv_norm_h));
    float norm_v_perp = sqrt(sq_v_perp + 1e-12f);
    float inv_norm_vp = 1.0f / norm_v_perp;

    float omega = norm_v_perp * inv_norm_h;
    float theta = omega * tau;
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    // 2. Proyección Geodésica y evaluación simultánea de las 4 líneas derivativas
    float l_g_onto = 0.0f, l_g_teleo = 0.0f, l_g_anti = 0.0f, l_g_eos = 0.0f;
    float l_dot_teleo_proj = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {
        float h_u = h_in[i] * inv_norm_h;
        float vp = v_tangent[i] - coeff_hv * h_in[i];
        float v_hat = (norm_v_perp > 1e-8f) ? (vp * inv_norm_vp) : 0.0f;

        float star = cos_t * h_u + sin_t * v_hat;
        h_star_out[i] = star;

        l_g_onto  += v_hat * u_onto[i];
        l_g_teleo += v_hat * u_teleo[i];
        l_g_anti  += v_hat * u_anti[i];
        l_g_eos   += v_hat * u_eos[i];
        l_dot_teleo_proj += star * u_teleo[i];
    }

    // 3. Reducción de derivadas
    sh_acc[tid * 5 + 0] = l_g_onto;
    sh_acc[tid * 5 + 1] = l_g_teleo;
    sh_acc[tid * 5 + 2] = l_g_anti;
    sh_acc[tid * 5 + 3] = l_g_eos;
    sh_acc[tid * 5 + 4] = l_dot_teleo_proj;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (uint k = 0; k < 5; ++k) {
                sh_acc[tid * 5 + k] += sh_acc[(tid + s) * 5 + k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        telemetry_out->omega_angular_velocity = omega;
        telemetry_out->curvature_kappa        = (norm_v_perp > 1e-6f) ? (omega / norm_h) : 0.0f;
        telemetry_out->grad_onto              = sh_acc[0];
        telemetry_out->grad_teleo             = sh_acc[1];
        telemetry_out->grad_anti              = sh_acc[2];
        telemetry_out->grad_eos               = sh_acc[3];
        telemetry_out->teleology_alignment    = sh_acc[4];
    }
}
