import sys, os, yaml

if len(sys.argv) < 2:
    print("Uso: python3 tools/transpilar_c007_aot.py <ruta_config.yaml>")
    sys.exit(1)

cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

exp_id = cfg["experiment_id"]
p = cfg["parameters"]

D = p["D"]
R = p["R"]
STEPS = p["steps"]
DT = 1.0 / float(STEPS)
HALF_DT = DT / 2.0
KAPPA = p["coupling_scale"]
LAMBDA = p["lyapunov_damping"]
ALPHA = p["eml_scale"]

print(f"=================================================================================")
print(f" ⚙️ TRANSPILADOR AOT AUTOMÁTICO :: APLICANDO {exp_id}")
print(f"=================================================================================")
print(f"• Parámetros del Experimento:")
print(f"  - Dimensión D = {D}, Rango r = {R}, Pasos = {STEPS}")
print(f"  - κ (Acoplamiento C) : {KAPPA:.12f}")
print(f"  - λ (Disipación)    : {LAMBDA:.12e}")
print(f"  - α (EML)           : {ALPHA:.12f}")

# 1. Emitir Metal Shader Canónico
metal_code = f"""// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER-VL :: CANONICAL C-007 KERNEL [{exp_id}]
// AUTO-GENERATED FROM {cfg_path} - ZERO MANUAL EDITS
// ═════════════════════════════════════════════════════════════════════════════
#include <metal_stdlib>
using namespace metal;

constant uint D = {D};
constant uint R = {R};

constant float CANONICAL_KAPPA  = {KAPPA:.12f}f;
constant float CANONICAL_LAMBDA = {LAMBDA:.12e}f;
constant float CANONICAL_ALPHA  = {ALPHA:.12f}f;

kernel void c007_project_reductions(
    device const float* Phi                 [[buffer(0)]],
    device const float* Uc                  [[buffer(1)]],
    device const float* Vc                  [[buffer(2)]],
    device const float* Us                  [[buffer(3)]],
    device const float* Vs                  [[buffer(4)]],
    device const float* Ul                  [[buffer(5)]],
    device const float* Vl                  [[buffer(6)]],
    device float* r_proj                    [[buffer(7)]],
    threadgroup float* shared_acc           [[threadgroup(0)]],
    uint r_idx                              [[threadgroup_position_in_grid]],
    uint tid                                [[thread_index_in_threadgroup]],
    uint t_per_group                        [[threads_per_threadgroup]])
{{
    device const float* S = Phi;
    device const float* L = Phi + D;
    uint base_r = r_idx * D;

    float acc_vs_s = 0.0f, acc_us_s = 0.0f;
    float acc_vl_l = 0.0f, acc_ul_l = 0.0f;
    float acc_vc_l = 0.0f, acc_uc_s = 0.0f;
    float local_dot_f_phi = 0.0f;
    float local_norm_sq = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {{
        float s_val = S[i];
        float l_val = L[i];
        acc_vs_s += Vs[base_r + i] * s_val;
        acc_us_s += Us[base_r + i] * s_val;
        acc_vl_l += Vl[base_r + i] * l_val;
        acc_ul_l += Ul[base_r + i] * l_val;
        acc_vc_l += Vc[base_r + i] * l_val;
        acc_uc_s += Uc[base_r + i] * s_val;

        if (r_idx == 0) {{
            float safe_xs = clamp(-s_val, -20.0f, 20.0f);
            float safe_xl = clamp(-l_val, -20.0f, 20.0f);
            float feml_s = s_val / (1.0f + exp(safe_xs));
            float feml_l = l_val / (1.0f + exp(safe_xl));

            local_dot_f_phi += feml_s * s_val + feml_l * l_val;
            local_norm_sq   += s_val * s_val + l_val * l_val;
        }}
    }}

    #define REDUCE_STORE(var, offset) \\
        shared_acc[tid] = var; \\
        threadgroup_barrier(mem_flags::mem_threadgroup); \\
        for (uint s = t_per_group / 2; s > 0; s >>= 1) {{ \\
            if (tid < s) shared_acc[tid] += shared_acc[tid + s]; \\
            threadgroup_barrier(mem_flags::mem_threadgroup); \\
        }} \\
        if (tid == 0) r_proj[offset * R + r_idx] = shared_acc[0]; \\
        threadgroup_barrier(mem_flags::mem_threadgroup);

    REDUCE_STORE(acc_vs_s, 0);
    REDUCE_STORE(acc_us_s, 1);
    REDUCE_STORE(acc_vl_l, 2);
    REDUCE_STORE(acc_ul_l, 3);
    REDUCE_STORE(acc_vc_l, 4);
    REDUCE_STORE(acc_uc_s, 5);

    if (r_idx == 0) {{
        shared_acc[tid] = local_dot_f_phi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = t_per_group / 2; s > 0; s >>= 1) {{
            if (tid < s) shared_acc[tid] += shared_acc[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0) r_proj[6 * R] = shared_acc[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        shared_acc[tid] = local_norm_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = t_per_group / 2; s > 0; s >>= 1) {{
            if (tid < s) shared_acc[tid] += shared_acc[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0) r_proj[6 * R + 1] = shared_acc[0];
    }}
}}

kernel void c007_reconstruct_and_step(
    device const float* Phi_in              [[buffer(0)]],
    device const float* Phi_base            [[buffer(1)]],
    device float* Phi_out                   [[buffer(2)]],
    device const float* Uc                  [[buffer(3)]],
    device const float* Vc                  [[buffer(4)]],
    device const float* Us                  [[buffer(5)]],
    device const float* Vs                  [[buffer(6)]],
    device const float* Ul                  [[buffer(7)]],
    device const float* Vl                  [[buffer(8)]],
    device const float* r_proj              [[buffer(9)]],
    constant float& dt                      [[buffer(10)]],
    uint tid                                [[thread_position_in_grid]])
{{
    if (tid >= D) return;

    device const float* Vs_S = r_proj + 0 * R;
    device const float* Us_S = r_proj + 1 * R;
    device const float* Vl_L = r_proj + 2 * R;
    device const float* Ul_L = r_proj + 3 * R;
    device const float* Vc_L = r_proj + 4 * R;
    device const float* Uc_S = r_proj + 5 * R;

    float total_dot_f = r_proj[6 * R];
    float total_norm_sq = r_proj[6 * R + 1];
    float proj_factor = (total_norm_sq > 1e-12f) ? (total_dot_f / total_norm_sq) : 0.0f;

    float As_S = 0.0f, Al_L = 0.0f, C_L = 0.0f, neg_CT_S = 0.0f;
    for (uint r = 0; r < R; ++r) {{
        uint idx = r * D + tid;
        As_S     += Us[idx] * Vs_S[r] - Vs[idx] * Us_S[r];
        Al_L     += Ul[idx] * Vl_L[r] - Vl[idx] * Ul_L[r];
        C_L      += Uc[idx] * Vc_L[r];
        neg_CT_S -= Vc[idx] * Uc_S[r];
    }}

    float dot_S = As_S + CANONICAL_KAPPA * C_L;
    float dot_L = CANONICAL_KAPPA * neg_CT_S + Al_L;

    float s_in = Phi_in[tid];
    float l_in = Phi_in[tid + D];

    float safe_xs = clamp(-s_in, -20.0f, 20.0f);
    float safe_xl = clamp(-l_in, -20.0f, 20.0f);

    float f_eml_s = s_in / (1.0f + exp(safe_xs));
    float f_eml_l = l_in / (1.0f + exp(safe_xl));

    float f_tangent_s = f_eml_s - proj_factor * s_in;
    float f_tangent_l = f_eml_l - proj_factor * l_in;

    float dS = dot_S + CANONICAL_ALPHA * f_tangent_s - CANONICAL_LAMBDA * s_in;
    float dL = dot_L + CANONICAL_ALPHA * f_tangent_l - CANONICAL_LAMBDA * l_in;

    Phi_out[tid]     = Phi_base[tid]     + dt * dS;
    Phi_out[tid + D] = Phi_base[tid + D] + dt * dL;
}}
"""

with open("metal/aether_c007_spinor_integrator.metal", "w") as f:
    f.write(metal_code)

print(f"✓ metal/aether_c007_spinor_integrator.metal transpilado para {exp_id}.")
