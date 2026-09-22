// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER-VLM NATIVE ENGINE :: END-TO-END C++ / METAL PIPELINE
// SSOT: spec/collapse/C021_vapor_condensation_collapse.yaml
// HITO 1.1: Célula Proyectiva Geodésica Autónoma (C++/MLX y Metal GPU)
// ═════════════════════════════════════════════════════════════════════════════
#include <mlx/mlx.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <cmath>
#include <optional>
#include <iostream>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "../include/geodesic_trajectory_cell.h"
#include "../include/intracycle_state_buffer.h"
#include "../include/permeability_gate.h"

namespace nb = nanobind;
using namespace mlx::core;

// Constantes físicas inmutables emanadas del YAML C-021
constexpr float CANONICAL_NU_VISCOSITY     = 0.12f;
constexpr float CANONICAL_GAMMA_SHOCK      = 0.35f;
constexpr float CANONICAL_KAPPA_NUCLEATION = 0.15f;

// ─── 1. KERNEL C++: PASO GEODÉSICO RIEMANNIANO EN S^{D-1} (C-018) ─────────────
array riemannian_step_cpp(const array& h, const array& u_target, float theta_step) {
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
}

// ─── 2. KERNEL C++: CONDENSACIÓN DE VAPOR DE FLUIDOS (C-021) ─────────────────
array vapor_condensation_cpp(
    const array& z_impact,
    const array& delta_G,
    float nu = CANONICAL_NU_VISCOSITY
) {
    auto z_mean = mean(z_impact, -1, true);
    auto z_std  = sqrt(var(z_impact, -1, true) + 1e-6f);

    // Amortiguamiento Viscoso Laminar
    auto z_visc = (1.0f - nu) * z_impact + nu * z_mean;

    // Confinamiento Covariante al Cono Positivo de Gibbs (H+): z_visc - Delta_G
    auto z_cond = z_visc - delta_G;

    // Normalización de Entalpía
    auto z_norm = (z_cond - mean(z_cond, -1, true)) / sqrt(var(z_cond, -1, true) + 1e-6f);
    return (z_norm * z_std) + z_mean;
}

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
) {
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
}

// ─── 4. KERNEL C++/MLX: CÉLULA PROYECTIVA GEODÉSICA AUTÓNOMA (HITO 1.1) ───────
nb::dict dispatch_geodesic_trajectory_cell_cpp(
    const array& h_in,
    const array& v_drag,
    const array& a_flow,
    const array& u_attractor,
    float tau = 1.0f,
    float kappa_att = 1.20f,
    float beta_perm = 12.0f,
    float theta_perm = 0.50f,
    uint32_t mode = 0
) {
    constexpr float EPS = 1e-12f;

    auto sq_h = sum(h_in * h_in, -1, true);
    auto inv_norm_h = rsqrt(sq_h + EPS);
    auto h_unit = h_in * inv_norm_h;

    auto sq_v = sum(v_drag * v_drag, -1, true);
    auto sq_a = sum(a_flow * a_flow, -1, true);
    auto dot_hv = sum(h_unit * v_drag, -1, true);
    auto dot_va = sum(v_drag * a_flow, -1, true);
    auto dot_hu = sum(h_unit * u_attractor, -1, true);

    auto v_perp = v_drag - dot_hv * h_unit;
    auto sq_v_perp = sum(v_perp * v_perp, -1, true);
    auto norm_v_perp = sqrt(sq_v_perp + EPS);
    auto v_hat = v_perp / norm_v_perp;

    auto bivector_sq = maximum(array(0.0f), (sq_v * sq_a) - (dot_va * dot_va));
    auto kappa_kin = sqrt(bivector_sq) / (power(sq_v, array(1.5f)) + EPS);

    auto omega = norm_v_perp * inv_norm_h;
    auto theta = omega * tau;
    auto cos_t = cos(theta);
    auto sin_t = sin(theta);

    auto a_attractor = kappa_att * (u_attractor - dot_hu * h_unit);
    auto h_projected = (cos_t * h_unit) + (sin_t * v_hat) + (0.5f * tau * tau * a_attractor);
    auto norm_proj = sqrt(sum(h_projected * h_projected, -1, true) + EPS);
    auto h_star = h_projected / norm_proj;

    auto r_val = sum(h_star * u_attractor, -1, true);
    auto dot_h_hstar = sum(h_unit * h_star, -1, true);
    auto h_deflated = h_unit - dot_h_hstar * h_star;

    auto q_tension = sq_v_perp / (sq_h + EPS);
    auto g_perm = 1.0f / (1.0f + exp(-beta_perm * (q_tension - theta_perm)));

    nb::dict result;
    result["h_star"]               = h_star;
    result["h_deflated"]           = h_deflated;
    result["correlation_r"]        = r_val;
    result["curvature_kappa"]      = kappa_kin;
    result["kinetic_energy"]       = 0.5f * sq_v_perp;
    result["angular_displacement"] = theta;
    result["dirichlet_tension"]    = q_tension;
    result["permeability_gate"]    = g_perm;
    result["mode"]                 = array(static_cast<int>(mode));
    return result;
}

// ─── 5. KERNEL METAL GPU: CÉLULA PROYECTIVA GEODÉSICA AUTÓNOMA (HITO 1.1) ─────
struct CellMetricsGPU {
    float correlation_r;
    float curvature_kappa;
    float kinetic_energy;
    float angular_displacement;
    float dirichlet_tension;
    float permeability_gate;
    uint32_t regime;
    uint32_t active_mode;
};

static id<MTLDevice> g_metal_device = nil;
static id<MTLCommandQueue> g_metal_queue = nil;
static id<MTLComputePipelineState> g_metal_pso = nil;

static id<MTLBuffer> g_buf_h = nil;
static id<MTLBuffer> g_buf_v = nil;
static id<MTLBuffer> g_buf_a = nil;
static id<MTLBuffer> g_buf_u = nil;
static id<MTLBuffer> g_buf_h_proj = nil;
static id<MTLBuffer> g_buf_h_defl = nil;
static id<MTLBuffer> g_buf_metrics = nil;
static size_t g_buf_capacity = 0;

static void init_metal_trajectory_cell() {
    if (g_metal_pso != nil) return;
    @autoreleasepool {
        g_metal_device = MTLCreateSystemDefaultDevice();
        if (!g_metal_device) {
            throw std::runtime_error("Metal GPU no disponible en este sistema");
        }
        g_metal_queue = [g_metal_device newCommandQueue];

        NSError* err = nil;
        NSString* path = @"metal/geodesic_trajectory_cell.metallib";
        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            path = @"/Users/crotalo/aether_engine/metal/geodesic_trajectory_cell.metallib";
        }
        NSURL* libURL = [NSURL fileURLWithPath:path];
        id<MTLLibrary> lib = [g_metal_device newLibraryWithURL:libURL error:&err];
        if (!lib) {
            throw std::runtime_error("No se pudo cargar geodesic_trajectory_cell.metallib: " +
                                     std::string(err ? [[err localizedDescription] UTF8String] : "unknown"));
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"dispatch_geodesic_trajectory_cell_step"];
        if (!fn) {
            throw std::runtime_error("Función dispatch_geodesic_trajectory_cell_step no encontrada en metallib");
        }
        g_metal_pso = [g_metal_device newComputePipelineStateWithFunction:fn error:&err];
        if (!g_metal_pso) {
            throw std::runtime_error("Error creando pipeline state Metal: " +
                                     std::string(err ? [[err localizedDescription] UTF8String] : "unknown"));
        }
    }
}

static void ensure_metal_buffers(size_t required_bytes) {
    if (g_buf_h != nil && g_buf_capacity >= required_bytes) return;
    size_t cap = std::max(required_bytes, static_cast<size_t>(8192 * sizeof(float)));
    g_buf_h      = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    g_buf_v      = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    g_buf_a      = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    g_buf_u      = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    g_buf_h_proj = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    g_buf_h_defl = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    if (!g_buf_metrics) {
        g_buf_metrics = [g_metal_device newBufferWithLength:sizeof(CellMetricsGPU) options:MTLResourceStorageModeShared];
    }
    g_buf_capacity = cap;
}

nb::dict dispatch_geodesic_trajectory_cell_metal(
    const array& h_in,
    const array& v_drag,
    const array& a_flow,
    const array& u_attractor,
    float tau = 1.0f,
    float kappa_att = 1.20f,
    float beta_perm = 12.0f,
    float theta_perm = 0.50f,
    uint32_t mode = 0
) {
    init_metal_trajectory_cell();

    eval({h_in, v_drag, a_flow, u_attractor});

    uint32_t D = static_cast<uint32_t>(h_in.size());
    size_t bytes_vec = D * sizeof(float);
    ensure_metal_buffers(bytes_vec);

    @autoreleasepool {
        std::memcpy([g_buf_h contents], h_in.data<float>(), bytes_vec);
        std::memcpy([g_buf_v contents], v_drag.data<float>(), bytes_vec);
        std::memcpy([g_buf_a contents], a_flow.data<float>(), bytes_vec);
        std::memcpy([g_buf_u contents], u_attractor.data<float>(), bytes_vec);

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_metal_pso];
        [enc setBuffer:g_buf_h offset:0 atIndex:0];
        [enc setBuffer:g_buf_v offset:0 atIndex:1];
        [enc setBuffer:g_buf_a offset:0 atIndex:2];
        [enc setBuffer:g_buf_u offset:0 atIndex:3];
        [enc setBuffer:g_buf_h_proj offset:0 atIndex:4];
        [enc setBuffer:g_buf_h_defl offset:0 atIndex:5];
        [enc setBuffer:g_buf_metrics offset:0 atIndex:6];

        [enc setBytes:&tau length:sizeof(float) atIndex:7];
        [enc setBytes:&kappa_att length:sizeof(float) atIndex:8];
        [enc setBytes:&beta_perm length:sizeof(float) atIndex:9];
        [enc setBytes:&theta_perm length:sizeof(float) atIndex:10];
        [enc setBytes:&mode length:sizeof(uint32_t) atIndex:11];
        [enc setBytes:&D length:sizeof(uint32_t) atIndex:12];

        // Memoria compartida threadgroup: 256 hilos * 6 floats = 6144 bytes
        [enc setThreadgroupMemoryLength:256 * 6 * sizeof(float) atIndex:0];

        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        float* p_proj = (float*)[g_buf_h_proj contents];
        float* p_defl = (float*)[g_buf_h_defl contents];
        CellMetricsGPU* p_m = (CellMetricsGPU*)[g_buf_metrics contents];

        array h_star(p_proj, {static_cast<int>(D)}, float32);
        array h_deflated(p_defl, {static_cast<int>(D)}, float32);

        nb::dict result;
        result["h_star"]               = h_star;
        result["h_deflated"]           = h_deflated;
        result["correlation_r"]        = array(p_m->correlation_r);
        result["curvature_kappa"]      = array(p_m->curvature_kappa);
        result["kinetic_energy"]       = array(p_m->kinetic_energy);
        result["angular_displacement"] = array(p_m->angular_displacement);
        result["dirichlet_tension"]    = array(p_m->dirichlet_tension);
        result["permeability_gate"]    = array(p_m->permeability_gate);
        result["regime"]               = array(static_cast<int>(p_m->regime));
        result["mode"]                 = array(static_cast<int>(p_m->active_mode));
        return result;
    }
}

// ─── 5. PUENTE C++: BÚFER CINEMÁTICO INTRACICLO (CERO ALOCACIONES) ───────────
static std::unique_ptr<aether::IntracycleStateBuffer> g_state_buffer = nullptr;
static aether::PermeabilityGate g_permeability_gate(12.0f, 0.50f, aether::GateInterventionMode::PassiveObserve);

nb::dict buffer_push_state_cpp(const array& h_t, uint32_t step) {
    uint32_t D = h_t.shape(-1);
    if (!g_state_buffer || g_state_buffer->dimension() != D) {
        g_state_buffer = std::make_unique<aether::IntracycleStateBuffer>(D);
    }

    // Ingestión directa de puntero contiguo UMA (CERO copia / CERO vector dinámico)
    aether::KinematicState k = g_state_buffer->push_state_zero_copy(h_t.data<float>(), step);
    aether::GateState g = g_permeability_gate.evaluate(k.dirichlet_tension_q);

    array v_arr = array(g_state_buffer->current_v(), {static_cast<int>(D)}, float32);
    array a_arr = array(g_state_buffer->current_a(), {static_cast<int>(D)}, float32);

    nb::dict d;
    d["norm_h"]               = k.norm_h;
    d["sq_v"]                 = k.sq_v;
    d["sq_a"]                 = k.sq_a;
    d["dot_hv"]               = k.dot_hv;
    d["dot_va"]               = k.dot_va;
    d["dirichlet_tension_q"]  = k.dirichlet_tension_q;
    d["permeability_g"]       = g.permeability_g;
    d["gate_is_open"]         = g.is_open;
    d["v_t"]                  = v_arr;
    d["a_t"]                  = a_arr;
    d["count"]                = g_state_buffer->count();
    return d;
}

void buffer_reset_cpp() {
    if (g_state_buffer) g_state_buffer->reset();
}

void gate_set_mode_cpp(uint32_t mode) {
    g_permeability_gate.set_mode(
        (mode == 0) ? aether::GateInterventionMode::PassiveObserve 
                    : aether::GateInterventionMode::ActiveCoupled
    );
}

// ─── ENLACE DEL MÓDULO NANOBIND ──────────────────────────────────────────────
NB_MODULE(aether_native_c, m) {
    m.def("dispatch_riemannian_step", &riemannian_step_cpp, "Exp-Map Riemanniano en C++ nativo",
          nb::arg("h"), nb::arg("u_target"), nb::arg("theta_step"));
    m.def("dispatch_vapor_condensation", &vapor_condensation_cpp, "Condensación de vapor C-021 en C++ nativo",
          nb::arg("z_impact"), nb::arg("delta_G"), nb::arg("nu") = CANONICAL_NU_VISCOSITY);
    m.def("dispatch_full_collapse", &collapse_and_condense_cpp, "Pipeline de Colapso Total en C++ nativo puro",
          nb::arg("h_final"), nb::arg("v_drag"), nb::arg("delta_G"),
          nb::arg("head_w"), nb::arg("head_scales"), nb::arg("head_biases"),
          nb::arg("group_size") = 64, nb::arg("bits") = 4,
          nb::arg("nu") = CANONICAL_NU_VISCOSITY, nb::arg("gamma") = CANONICAL_GAMMA_SHOCK);

    // Hito 1.1: Célula Proyectiva Geodésica Autónoma
    m.def("dispatch_geodesic_trajectory_cell", &dispatch_geodesic_trajectory_cell_cpp,
          "Célula Proyectiva Geodésica Autónoma (Backend C++/MLX) — Hito 1.1",
          nb::arg("h_in"), nb::arg("v_drag"), nb::arg("a_flow"), nb::arg("u_attractor"),
          nb::arg("tau") = 1.0f, nb::arg("kappa_att") = 1.20f,
          nb::arg("beta_perm") = 12.0f, nb::arg("theta_perm") = 0.50f,
          nb::arg("mode") = 0);

    m.def("dispatch_geodesic_trajectory_cell_metal", &dispatch_geodesic_trajectory_cell_metal,
          "Célula Proyectiva Geodésica Autónoma (Backend Metal GPU Puro) — Hito 1.1",
          nb::arg("h_in"), nb::arg("v_drag"), nb::arg("a_flow"), nb::arg("u_attractor"),
          nb::arg("tau") = 1.0f, nb::arg("kappa_att") = 1.20f,
          nb::arg("beta_perm") = 12.0f, nb::arg("theta_perm") = 0.50f,
          nb::arg("mode") = 0);

    // Hito 1.2: Búfer Intraciclo Cinemático y Compuerta Dual
    m.def("buffer_push_state", &buffer_push_state_cpp, "Registra h_t y calcula cinematica en C++ sin alocaciones",
          nb::arg("h_t"), nb::arg("step"));
    m.def("buffer_reset", &buffer_reset_cpp, "Reinicia el buffer intraciclo");
    m.def("gate_set_mode", &gate_set_mode_cpp, "Configura modo de la compuerta: 0=Pasivo, 1=Activo",
          nb::arg("mode"));
}

