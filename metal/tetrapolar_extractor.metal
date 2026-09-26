// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: TETRAPOLAR EXTRACTOR GPU KERNEL (MOTOR NATIVO)
// Extracción Analítica de los 4 Polos en Silicio Metal SRAM en < 30 µs
// CERO llamadas a Python | CERO bucles temporales | Ortogonalidad Estricta
// ═════════════════════════════════════════════════════════════════════════════
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

kernel void dispatch_extract_tetrapolar_poles(
    device const float* X_seq        [[buffer(0)]], // [T * D] Secuencia en UMA
    device const float* W_eos        [[buffer(1)]], // [D] Vector EOS
    device float*       U_onto_out   [[buffer(2)]], // [D] Salida Ontología
    device float*       U_teleo_out  [[buffer(3)]], // [D] Salida Teleología
    device float*       U_anti_out   [[buffer(4)]], // [D] Salida Antítesis
    device float*       U_eos_out    [[buffer(5)]], // [D] Salida Pozo EOS
    constant uint&      T            [[buffer(6)]],
    constant uint&      D            [[buffer(7)]],
    constant uint&      T_split      [[buffer(8)]],
    threadgroup float*  sh_acc       [[threadgroup(0)]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint t_per_group                 [[threads_per_threadgroup]]
) {
    // 1. Acumulación por hilos de Ontología y Teleología
    for (uint i = tid; i < D; i += t_per_group) {
        float acc_onto = 0.0f;
        float acc_teleo = 0.0f;

        // Tramo Ontológico: tokens de premisas [0 .. T_split - 1]
        for (uint t = 0; t < T_split; ++t) {
            acc_onto += X_seq[t * D + i];
        }

        // Tramo Teleológico: tokens de consulta/meta [T_split .. T - 1]
        for (uint t = T_split; t < T; ++t) {
            acc_teleo += X_seq[t * D + i];
        }

        U_onto_out[i]  = acc_onto;
        U_teleo_out[i] = acc_teleo;
        U_eos_out[i]   = W_eos[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // 2. Reducción en paralelo de normas cuadráticas en SRAM
    float l_sq_onto = 0.0f;
    float l_sq_teleo = 0.0f;
    float l_sq_eos = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {
        float o = U_onto_out[i];
        float t = U_teleo_out[i];
        float e = U_eos_out[i];
        l_sq_onto  += o * o;
        l_sq_teleo += t * t;
        l_sq_eos   += e * e;
    }

    sh_acc[tid * 3 + 0] = l_sq_onto;
    sh_acc[tid * 3 + 1] = l_sq_teleo;
    sh_acc[tid * 3 + 2] = l_sq_eos;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_acc[tid * 3 + 0] += sh_acc[(tid + s) * 3 + 0];
            sh_acc[tid * 3 + 1] += sh_acc[(tid + s) * 3 + 1];
            sh_acc[tid * 3 + 2] += sh_acc[(tid + s) * 3 + 2];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_norm_onto  = rsqrt(max(sh_acc[0], 1e-12f));
    float inv_norm_teleo = rsqrt(max(sh_acc[1], 1e-12f));
    float inv_norm_eos   = rsqrt(max(sh_acc[2], 1e-12f));

    // Normalizar U_onto, U_teleo y U_eos a norma unitaria exacta
    for (uint i = tid; i < D; i += t_per_group) {
        U_onto_out[i]  *= inv_norm_onto;
        U_teleo_out[i] *= inv_norm_teleo;
        U_eos_out[i]   *= inv_norm_eos;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // 3. Proyección Gram-Schmidt para construir Antítesis: U_anti = U_onto - <U_onto, U_teleo> * U_teleo
    float l_dot_ot = 0.0f;
    for (uint i = tid; i < D; i += t_per_group) {
        l_dot_ot += U_onto_out[i] * U_teleo_out[i];
    }

    sh_acc[tid] = l_dot_ot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) sh_acc[tid] += sh_acc[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float dot_ot = sh_acc[0];

    // Calcular vector residual de antítesis no normalizado
    float l_sq_anti = 0.0f;
    for (uint i = tid; i < D; i += t_per_group) {
        float a_val = U_onto_out[i] - dot_ot * U_teleo_out[i];
        U_anti_out[i] = a_val;
        l_sq_anti += a_val * a_val;
    }

    sh_acc[tid] = l_sq_anti;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) sh_acc[tid] += sh_acc[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sq_anti = sh_acc[0];
    float inv_norm_anti = (sq_anti > 1e-10f) ? rsqrt(sq_anti) : 0.0f;

    // Normalizar U_anti para asegurar ortogonalidad estricta y norma 1
    for (uint i = tid; i < D; i += t_per_group) {
        U_anti_out[i] *= inv_norm_anti;
    }
}
