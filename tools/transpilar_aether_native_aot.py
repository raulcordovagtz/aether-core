import yaml, os, sys

print("=================================================================================")
print(" ⚙️ TRANSPILADOR AOT CANÓNICO :: EMISIÓN C++ END-TO-END")
print("=================================================================================\n")

# 1. Leer parámetros canónicos desde los contratos YAML
with open("spec/collapse/C021_vapor_condensation_collapse.yaml", "r") as f:
    c021_spec = yaml.safe_load(f)
p_c021 = c021_spec["physical_parameters"]

NU_VISCOSITY     = float(p_c021["nu_viscosity"])
GAMMA_SHOCK      = float(p_c021["gamma_shock"])
KAPPA_NUCLEATION = float(p_c021["kappa_nucleation"])

# 2. Generar el código C++ nativo puro
cpp_code = f"""// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER-VLM NATIVE ENGINE :: END-TO-END C++ PIPELINE
// Generado automáticamente por tools/transpilar_aether_native_aot.py
// SSOT: spec/collapse/C021_vapor_condensation_collapse.yaml
// ═════════════════════════════════════════════════════════════════════════════
#include <mlx/mlx.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <cmath>
#include <optional>

namespace nb = nanobind;
using namespace mlx::core;

// Constantes físicas inmutables emanadas del YAML C-021
constexpr float CANONICAL_NU_VISCOSITY     = {NU_VISCOSITY}f;
constexpr float CANONICAL_GAMMA_SHOCK      = {GAMMA_SHOCK}f;
constexpr float CANONICAL_KAPPA_NUCLEATION = {KAPPA_NUCLEATION}f;

// ─── 1. KERNEL C++: PASO GEODÉSICO RIEMANNIANO EN S^{{D-1}} (C-018) ─────────────
array riemannian_step_cpp(const array& h, const array& u_target, float theta_step) {{
    auto norm_h = sqrt(sum(h * h, -1, true) + 1e-12f);
    auto h_unit = h / norm_h;

    auto norm_u = sqrt(sum(u_target * u_target, -1, true) + 1e-12f);
    auto u_unit = u_target / norm_u;

    auto proj = sum(u_unit * h_unit, -1, true);
    auto v_vec = u_unit - proj * h_unit;
    auto norm_v = sqrt(sum(v_vec * v_vec, -1, true) + 1e-12f);

    auto is_collinear = norm_v < 1e-6f;
    auto v_unit = where(is_collinear, zeros_like(v_vec), v_vec / norm_v);

    auto cos_t = cos(array(theta_step));
    auto sin_t = sin(array(theta_step));

    auto steered = where(is_collinear, h_unit, (cos_t * h_unit) + (sin_t * v_unit));
    auto norm_steered = sqrt(sum(steered * steered, -1, true) + 1e-12f);
    auto steered_unit = steered / norm_steered;

    return steered_unit * norm_h;
}}

// ─── 2. KERNEL C++: CONDENSACIÓN DE VAPOR DE FLUIDOS (C-021) ─────────────────
array vapor_condensation_cpp(
    const array& z_impact,
    const array& delta_G,
    float nu = CANONICAL_NU_VISCOSITY
) {{
    auto z_mean = mean(z_impact, -1, true);
    auto z_std  = sqrt(var(z_impact, -1, true) + 1e-6f);

    // Amortiguamiento Viscoso Laminar
    auto z_visc = (1.0f - nu) * z_impact + nu * z_mean;

    // Confinamiento Covariante al Cono Positivo de Gibbs (H+): z_visc - Delta_G
    auto z_cond = z_visc - delta_G;

    // Normalización de Entalpía
    auto z_norm = (z_cond - mean(z_cond, -1, true)) / sqrt(var(z_cond, -1, true) + 1e-6f);
    return (z_norm * z_std) + z_mean;
}}

// ─── 3. PASO DE COLAPSO TOTAL EN C++ (CHOQUE + LM_HEAD CUANTIZADO + CONDENSACIÓN) ────────
array collapse_and_condense_cpp(
    const array& h_final,
    const array& v_drag,
    const array& delta_G,
    const array& head_w,
    const array& head_scales,
    const array& head_biases,
    int group_size = 64,
    int bits = 4,
    float nu = CANONICAL_NU_VISCOSITY,
    float gamma = CANONICAL_GAMMA_SHOCK
) {{
    // 1. Choque cinético en el espacio latente
    float norm_factor = gamma / std::sqrt(1.0f + nu * nu);
    array h_impact = h_final + norm_factor * v_drag;

    // 2. Proyección matricial en C++ PURO sobre Metal GPU (Cero Python / Cero GIL)
    array z_impact = quantized_matmul(
        h_impact, head_w, head_scales,
        std::optional<array>(head_biases),
        /* transpose = */ true,
        std::optional<int>(group_size),
        std::optional<int>(bits),
        /* mode = */ "affine"
    );

    // 3. Condensación de vapor con barrera de Gibbs covariante en H+
    return vapor_condensation_cpp(z_impact, delta_G, nu);
}}

// ─── ENLACE DEL MÓDULO NANOBIND ──────────────────────────────────────────────
NB_MODULE(aether_native_c, m) {{
    m.def("dispatch_riemannian_step", &riemannian_step_cpp, "Exp-Map Riemanniano en C++ nativo",
          nb::arg("h"), nb::arg("u_target"), nb::arg("theta_step"));
    m.def("dispatch_vapor_condensation", &vapor_condensation_cpp, "Condensación de vapor C-021 en C++ nativo",
          nb::arg("z_impact"), nb::arg("delta_G"), nb::arg("nu") = CANONICAL_NU_VISCOSITY);
    m.def("dispatch_full_collapse", &collapse_and_condense_cpp, "Pipeline de Colapso Total en C++ nativo puro",
          nb::arg("h_final"), nb::arg("v_drag"), nb::arg("delta_G"),
          nb::arg("head_w"), nb::arg("head_scales"), nb::arg("head_biases"),
          nb::arg("group_size") = 64, nb::arg("bits") = 4,
          nb::arg("nu") = CANONICAL_NU_VISCOSITY, nb::arg("gamma") = CANONICAL_GAMMA_SHOCK);
}}
"""

with open("aether_vlm/aether_native.cpp", "w") as f:
    f.write(cpp_code)

print("✓ aether_vlm/aether_native.cpp emitido automáticamente desde los contratos YAML.")
