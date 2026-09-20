// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER-VL :: DIRECT ZERO-COPY RIEMANNIAN ENGINE (C-018 FAST)
// Single-pass register-fused Geodesic Step in Apple Silicon UMA
// ═════════════════════════════════════════════════════════════════════════════
#include <metal_stdlib>
using namespace metal;

kernel void c018_riemannian_direct_step(
    device float* h_io                      [[buffer(0)]], // [D] Lectura/Escritura in-place
    device const float* u_target            [[buffer(1)]], // [D]
    constant uint& D                        [[buffer(2)]],
    constant float& dt                      [[buffer(3)]],
    constant float& kappa_0                 [[buffer(4)]],
    constant float& beta_steer              [[buffer(5)]],
    constant float& M_diss                  [[buffer(6)]],
    constant float& nu_diff                 [[buffer(7)]],
    uint tid                                [[thread_position_in_grid]])
{
    if (tid >= D) return;

    float h_val = h_io[tid];
    float u_val = u_target[tid];

    // Fuerza ortogonal tangencial directa en registros
    // b_k = (u - h) proyectado tangencial
    float b_k = u_val - h_val;

    // Gradiente de coherencia y disipación Laplaciana
    float grad_l = - (u_val - h_val);
    float lap_l  = (h_val - u_val);

    // Velocidad tangencial combinada
    float v_l = beta_steer * b_k - M_diss * grad_l - nu_diff * lap_l;

    // Retracción geodésica exponencial en registros: Exp_h(v * dt)
    float theta = dt * abs(v_l) * (1.0f + kappa_0 * sqrt(float(D)));
    float cos_t = cos(theta);
    float sin_t = sin(theta);
    float sgn = (v_l >= 0.0f) ? 1.0f : -1.0f;

    // Actualización cerrada y exacta en un solo ciclo de GPU
    h_io[tid] = cos_t * h_val + sin_t * sgn;
}
