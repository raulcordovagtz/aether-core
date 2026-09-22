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
#include "../include/conformal_coupling_junction.h"
#include "../include/hilbert_memory_cell.h"
#include "../include/fact_band_router.h"

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

// ─── 6. PUENTE C++: UNIÓN CONFORMAL UMA DIRECT-POINTER (HITO 1.3-R1) ─────────
static std::unique_ptr<aether::ConformalCouplingJunction> g_junction = nullptr;

// Forward: Célula de Memoria Geométrica (necesario para Gap Junction en coupling)
static std::unique_ptr<aether::HilbertMemoryCell> g_hilbert_memory = nullptr;

nb::dict dispatch_conformal_coupling_cpp(
    const array& h_state,
    const array& u_attractor,
    uint32_t step,
    float tau_eff = 0.15f,
    float kappa_att = 0.80f,
    float beta_gate = 12.0f,
    float theta_gate = 0.35f,
    uint32_t mode = 1,
    float force_g = -1.0f
) {
    eval({h_state, u_attractor});
    uint32_t D = h_state.shape(-1);
    if (!g_junction) {
        g_junction = std::make_unique<aether::ConformalCouplingJunction>(D);
    }
    g_junction->set_mode(
        (mode == 0) ? aether::GateInterventionMode::PassiveObserve 
                    : aether::GateInterventionMode::ActiveCoupled
    );
    g_junction->set_hyperparameters(tau_eff, kappa_att, beta_gate, theta_gate);

    // Ingestión Directa por Puntero UMA (CERO vector intermedio / CERO copia previa)
    aether::CouplingMetrics m = g_junction->couple_step(
        h_state.data<float>(), u_attractor.data<float>(), step, force_g
    );

    // ─── GAP JUNCTION: Auto-ingesta del subproducto en Célula 2 ─────────────
    // Cuando el acoplamiento está activo (cell_evaluated), el subproducto ortogonal
    // se deposita automáticamente en el anillo de memoria de HilbertMemoryCell
    // sin retorno a Python y sin frenar el macro-reloj de inferencia.
    bool memory_ingested = false;
    uint32_t memory_slot = 0;
    uint32_t memory_count = 0;
    if (m.cell_evaluated) {
        if (!g_hilbert_memory || g_hilbert_memory->dimension() != D) {
            g_hilbert_memory = std::make_unique<aether::HilbertMemoryCell>(D, 32);
        }
        memory_slot = g_hilbert_memory->ingest_residual_subproduct(
            g_junction->get_orthogonal_subproduct(), D, step, m.dirichlet_tension_q
        );
        memory_count = g_hilbert_memory->count();
        memory_ingested = true;
    }

    // Salida sin copias intermedias
    array h_steered = array(g_junction->get_steered_output(), {static_cast<int>(D)}, float32);
    array h_proj    = array(g_junction->get_projected_state(), {static_cast<int>(D)}, float32);
    array h_ortho   = array(g_junction->get_orthogonal_subproduct(), {static_cast<int>(D)}, float32);

    nb::dict d;
    d["h_steered"]             = h_steered;
    d["h_projected"]           = h_proj;
    d["h_orthogonal"]          = h_ortho;
    d["correlation_r"]         = m.correlation_r;
    d["curvature_kappa"]       = m.curvature_kappa;
    d["dirichlet_tension_q"]   = m.dirichlet_tension_q;
    d["permeability_g"]        = m.permeability_g;
    d["angular_displacement"]  = m.angular_displacement;
    d["gate_open"]             = m.gate_open;
    d["cell_evaluated"]        = m.cell_evaluated;
    d["intervention_applied"]  = m.intervention_applied;
    d["active_regime"]         = m.active_regime;
    d["memory_ingested"]       = memory_ingested;
    d["memory_slot"]           = memory_slot;
    d["memory_count"]          = memory_count;
    return d;
}

void junction_reset_cpp() {
    if (g_junction) g_junction->reset();
}

// ─── 7. PUENTE C++ & METAL: CÉLULA DE MEMORIA GEOMÉTRICA (HITO 2.1) ───────────
// (g_hilbert_memory declarado antes de dispatch_conformal_coupling_cpp para Gap Junction)

// Metal pipeline states y buffers para HilbertMemoryCell
static id<MTLComputePipelineState> g_metal_pso_pack = nil;
static id<MTLComputePipelineState> g_metal_pso_deflate = nil;
static id<MTLComputePipelineState> g_metal_pso_style = nil;
static id<MTLBuffer> g_buf_mem_mA = nil;
static id<MTLBuffer> g_buf_mem_mB = nil;
static id<MTLBuffer> g_buf_mem_out = nil;
static id<MTLBuffer> g_buf_mem_degen = nil;
static size_t g_buf_mem_capacity = 0;

static void init_metal_hilbert_memory() {
    if (g_metal_pso_pack != nil) return;
    @autoreleasepool {
        if (!g_metal_device) {
            g_metal_device = MTLCreateSystemDefaultDevice();
        }
        if (!g_metal_queue) {
            g_metal_queue = [g_metal_device newCommandQueue];
        }

        NSError* err = nil;
        NSString* path = @"metal/hilbert_memory_cell.metallib";
        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            path = @"/Users/crotalo/aether_engine/metal/hilbert_memory_cell.metallib";
        }
        NSURL* libURL = [NSURL fileURLWithPath:path];
        id<MTLLibrary> lib = [g_metal_device newLibraryWithURL:libURL error:&err];
        if (!lib) {
            throw std::runtime_error("No se pudo cargar hilbert_memory_cell.metallib: " +
                                     std::string(err ? [[err localizedDescription] UTF8String] : "unknown"));
        }

        id<MTLFunction> fn_pack = [lib newFunctionWithName:@"dispatch_hilbert_pack_two"];
        id<MTLFunction> fn_defl = [lib newFunctionWithName:@"dispatch_hilbert_deflation"];
        id<MTLFunction> fn_style = [lib newFunctionWithName:@"dispatch_hilbert_style_transport"];

        if (!fn_pack || !fn_defl || !fn_style) {
            throw std::runtime_error("Funciones Metal no encontradas en hilbert_memory_cell.metallib");
        }

        g_metal_pso_pack = [g_metal_device newComputePipelineStateWithFunction:fn_pack error:&err];
        g_metal_pso_deflate = [g_metal_device newComputePipelineStateWithFunction:fn_defl error:&err];
        g_metal_pso_style = [g_metal_device newComputePipelineStateWithFunction:fn_style error:&err];
    }
}

static void ensure_metal_memory_buffers(size_t required_bytes) {
    if (g_buf_mem_mA != nil && g_buf_mem_capacity >= required_bytes) return;
    size_t cap = std::max(required_bytes, static_cast<size_t>(8192 * sizeof(float)));
    g_buf_mem_mA    = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    g_buf_mem_mB    = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    g_buf_mem_out   = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    if (!g_buf_mem_degen) {
        g_buf_mem_degen = [g_metal_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    }
    g_buf_mem_capacity = cap;
}

nb::dict hilbert_memory_ingest_cpp(const array& h_subprod, uint32_t timestamp, float energy) {
    eval({h_subprod});
    uint32_t D = h_subprod.shape(-1);
    if (!g_hilbert_memory || g_hilbert_memory->dimension() != D) {
        g_hilbert_memory = std::make_unique<aether::HilbertMemoryCell>(D, 32);
    }
    uint32_t slot = g_hilbert_memory->ingest_residual_subproduct(
        h_subprod.data<float>(), D, timestamp, energy
    );
    nb::dict d;
    d["slot_idx"] = slot;
    d["count"]    = g_hilbert_memory->count();
    d["dimension"]= D;
    return d;
}

static std::vector<float> g_mem_temp_out;

nb::dict hilbert_memory_pack_two_cpp(const array& m_A, const array& m_B) {
    eval({m_A, m_B});
    uint32_t D = m_A.shape(-1);
    if (!g_hilbert_memory || g_hilbert_memory->dimension() != D) {
        g_hilbert_memory = std::make_unique<aether::HilbertMemoryCell>(D, 32);
    }
    if (g_mem_temp_out.size() < D) g_mem_temp_out.resize(D);

    const float* facts[2] = { m_A.data<float>(), m_B.data<float>() };
    aether::DegeneracyReason reason = aether::DegeneracyReason::NONE;
    bool ok = g_hilbert_memory->execute_superposition_packing(g_mem_temp_out.data(), facts, 2, reason);

    array m_out = array(g_mem_temp_out.data(), {static_cast<int>(D)}, float32);
    nb::dict d;
    d["result"]            = m_out;
    d["degenerate"]        = !ok;
    d["degeneracy_reason"] = static_cast<uint32_t>(reason);
    return d;
}

nb::dict hilbert_memory_deflate_cpp(const array& m_pack, const array& m_target) {
    eval({m_pack, m_target});
    uint32_t D = m_pack.shape(-1);
    if (!g_hilbert_memory || g_hilbert_memory->dimension() != D) {
        g_hilbert_memory = std::make_unique<aether::HilbertMemoryCell>(D, 32);
    }
    if (g_mem_temp_out.size() < D) g_mem_temp_out.resize(D);

    aether::DegeneracyReason reason = aether::DegeneracyReason::NONE;
    bool ok = g_hilbert_memory->execute_orthogonal_deflation(g_mem_temp_out.data(), m_pack.data<float>(), m_target.data<float>(), reason);

    array m_out = array(g_mem_temp_out.data(), {static_cast<int>(D)}, float32);
    nb::dict d;
    d["result"]            = m_out;
    d["degenerate"]        = !ok;
    d["degeneracy_reason"] = static_cast<uint32_t>(reason);
    return d;
}

nb::dict hilbert_memory_style_transport_cpp(const array& h_truth, const array& u_style, float theta_s) {
    eval({h_truth, u_style});
    uint32_t D = h_truth.shape(-1);
    if (!g_hilbert_memory || g_hilbert_memory->dimension() != D) {
        g_hilbert_memory = std::make_unique<aether::HilbertMemoryCell>(D, 32);
    }
    if (g_mem_temp_out.size() < D) g_mem_temp_out.resize(D);

    aether::DegeneracyReason reason = aether::DegeneracyReason::NONE;
    bool ok = g_hilbert_memory->execute_style_transport(g_mem_temp_out.data(), h_truth.data<float>(), u_style.data<float>(), theta_s, reason);

    array m_out = array(g_mem_temp_out.data(), {static_cast<int>(D)}, float32);
    nb::dict d;
    d["result"]            = m_out;
    d["degenerate"]        = !ok;
    d["degeneracy_reason"] = static_cast<uint32_t>(reason);
    return d;
}


// Implementación Metal GPU pura para paridad
nb::dict hilbert_memory_pack_two_metal(const array& m_A, const array& m_B) {
    init_metal_hilbert_memory();
    eval({m_A, m_B});
    uint32_t D = static_cast<uint32_t>(m_A.size());
    size_t bytes = D * sizeof(float);
    ensure_metal_memory_buffers(bytes);

    @autoreleasepool {
        std::memcpy([g_buf_mem_mA contents], m_A.data<float>(), bytes);
        std::memcpy([g_buf_mem_mB contents], m_B.data<float>(), bytes);

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_metal_pso_pack];
        [enc setBuffer:g_buf_mem_mA offset:0 atIndex:0];
        [enc setBuffer:g_buf_mem_mB offset:0 atIndex:1];
        [enc setBuffer:g_buf_mem_out offset:0 atIndex:2];
        [enc setBuffer:g_buf_mem_degen offset:0 atIndex:3];
        [enc setBytes:&D length:sizeof(uint32_t) atIndex:4];

        NSUInteger tg_size = std::min(static_cast<uint32_t>(256), D);
        [enc setThreadgroupMemoryLength:tg_size * sizeof(float) atIndex:0];
        [enc dispatchThreads:MTLSizeMake(tg_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        uint32_t degen = *(uint32_t*)[g_buf_mem_degen contents];
        array m_out = array((float*)[g_buf_mem_out contents], {static_cast<int>(D)}, float32);

        nb::dict d;
        d["result"]            = m_out;
        d["degenerate"]        = (degen != 0);
        d["degeneracy_reason"] = degen;
        return d;
    }
}

nb::dict hilbert_memory_deflate_metal(const array& m_pack, const array& m_target) {
    init_metal_hilbert_memory();
    eval({m_pack, m_target});
    uint32_t D = static_cast<uint32_t>(m_pack.size());
    size_t bytes = D * sizeof(float);
    ensure_metal_memory_buffers(bytes);

    @autoreleasepool {
        std::memcpy([g_buf_mem_mA contents], m_pack.data<float>(), bytes);
        std::memcpy([g_buf_mem_mB contents], m_target.data<float>(), bytes);

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_metal_pso_deflate];
        [enc setBuffer:g_buf_mem_mA offset:0 atIndex:0];
        [enc setBuffer:g_buf_mem_mB offset:0 atIndex:1];
        [enc setBuffer:g_buf_mem_out offset:0 atIndex:2];
        [enc setBuffer:g_buf_mem_degen offset:0 atIndex:3];
        [enc setBytes:&D length:sizeof(uint32_t) atIndex:4];

        NSUInteger tg_size = std::min(static_cast<uint32_t>(256), D);
        [enc setThreadgroupMemoryLength:tg_size * sizeof(float) atIndex:0];
        [enc dispatchThreads:MTLSizeMake(tg_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        uint32_t degen = *(uint32_t*)[g_buf_mem_degen contents];
        array m_out = array((float*)[g_buf_mem_out contents], {static_cast<int>(D)}, float32);

        nb::dict d;
        d["result"]            = m_out;
        d["degenerate"]        = (degen != 0);
        d["degeneracy_reason"] = degen;
        return d;
    }
}

nb::dict hilbert_memory_style_transport_metal(const array& h_truth, const array& u_style, float theta_s) {
    init_metal_hilbert_memory();
    eval({h_truth, u_style});
    uint32_t D = static_cast<uint32_t>(h_truth.size());
    size_t bytes = D * sizeof(float);
    ensure_metal_memory_buffers(bytes);

    @autoreleasepool {
        std::memcpy([g_buf_mem_mA contents], h_truth.data<float>(), bytes);
        std::memcpy([g_buf_mem_mB contents], u_style.data<float>(), bytes);

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_metal_pso_style];
        [enc setBuffer:g_buf_mem_mA offset:0 atIndex:0];
        [enc setBuffer:g_buf_mem_mB offset:0 atIndex:1];
        [enc setBuffer:g_buf_mem_out offset:0 atIndex:2];
        [enc setBuffer:g_buf_mem_degen offset:0 atIndex:3];
        [enc setBytes:&theta_s length:sizeof(float) atIndex:4];
        [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];

        NSUInteger tg_size = std::min(static_cast<uint32_t>(256), D);
        [enc setThreadgroupMemoryLength:tg_size * sizeof(float) atIndex:0];
        [enc dispatchThreads:MTLSizeMake(tg_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        uint32_t degen = *(uint32_t*)[g_buf_mem_degen contents];
        array m_out = array((float*)[g_buf_mem_out contents], {static_cast<int>(D)}, float32);

        nb::dict d;
        d["result"]            = m_out;
        d["degenerate"]        = (degen != 0);
        d["degeneracy_reason"] = degen;
        return d;
    }
}

void hilbert_memory_reset_cpp() {
    if (g_hilbert_memory) g_hilbert_memory->reset();
}

// ─── 8. CONSULTAS DE MEMORIA INTER-CELULAR ──────────────────────────────────
uint32_t hilbert_memory_slot_count_cpp() {
    return g_hilbert_memory ? g_hilbert_memory->count() : 0;
}

nb::dict hilbert_memory_get_slot_cpp(uint32_t slot_idx) {
    nb::dict d;
    if (!g_hilbert_memory || slot_idx >= g_hilbert_memory->count()) {
        d["valid"] = false;
        return d;
    }
    uint32_t D = g_hilbert_memory->dimension();
    const float* ptr = g_hilbert_memory->get_slot_ptr(slot_idx);
    array slot_tensor = array(ptr, {static_cast<int>(D)}, float32);
    d["valid"] = true;
    d["tensor"] = slot_tensor;
    d["dimension"] = D;
    return d;
}

nb::dict hilbert_memory_query_resonance_cpp(const array& u_query) {
    eval({u_query});
    nb::dict d;
    if (!g_hilbert_memory) {
        d["count"] = static_cast<uint32_t>(0);
        return d;
    }
    uint32_t count = g_hilbert_memory->count();
    d["count"] = count;
    std::vector<float> resonances(count);
    for (uint32_t i = 0; i < count; ++i) {
        resonances[i] = g_hilbert_memory->query_slot_resonance(i, u_query.data<float>());
    }
    d["resonances"] = array(resonances.data(), {static_cast<int>(count)}, float32);
    return d;
}

// ─── 8. PUENTE C++: ENRUTADOR ASOCIATIVO Y DETECTOR DE CRESTA CINEMÁTICA (HITO 2.2) ────
static std::unique_ptr<aether::FactBandRouter> g_fact_router = nullptr;

nb::dict fact_band_detect_peak_cpp(const std::vector<array>& layer_arrays) {
    uint32_t num_layers = layer_arrays.size();
    if (num_layers == 0) throw std::invalid_argument("Vector de capas vacio");
    uint32_t D = layer_arrays[0].shape(-1);

    // Forzar evaluación de punteros contiguos MLX
    std::vector<const float*> raw_ptrs(num_layers);
    for (uint32_t l = 0; l < num_layers; ++l) {
        mlx::core::eval({layer_arrays[l]});
        raw_ptrs[l] = layer_arrays[l].data<float>();
    }

    std::vector<float> kappas;
    std::vector<aether::KinematicDegeneracy> degen;
    uint32_t peak_l = aether::FactBandRouter::detect_candidate_band_peak(
        raw_ptrs, num_layers, D, kappas, &degen
    );

    array kappas_arr = array(kappas.data(), {static_cast<int>(num_layers)}, float32);
    nb::dict d;
    d["peak_layer"]     = peak_l;
    d["relative_depth"] = static_cast<float>(peak_l) / static_cast<float>(num_layers);
    d["kappas"]         = kappas_arr;
    return d;
}

nb::dict fact_band_route_layer_cpp(const array& h_layer, float threshold = 0.45f, float beta = 16.0f) {
    uint32_t D = h_layer.shape(-1);
    if (!g_fact_router || g_fact_router->dimension() != D) {
        g_fact_router = std::make_unique<aether::FactBandRouter>(D, threshold, beta);
    }
    g_fact_router->set_threshold(threshold);

    if (!g_hilbert_memory) {
        throw std::runtime_error("HilbertMemoryCell no inicializada");
    }

    mlx::core::eval({h_layer});
    aether::RouterDecision dec = g_fact_router->evaluate_layer_routing(
        h_layer.data<float>(), *g_hilbert_memory
    );

    nb::dict d;
    d["selected_slot"]      = dec.selected_slot;
    d["max_resonance_r"]    = dec.max_resonance_r;
    d["second_resonance_r"] = dec.second_resonance_r;
    d["resonance_margin"]   = dec.resonance_margin;
    d["rectified_gate_g"]   = dec.rectified_gate_g;
    d["is_active"]          = dec.is_active_injection;
    return d;
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

    // Hito 1.3: Unión Conformal Direct-Pointer
    m.def("dispatch_conformal_coupling", &dispatch_conformal_coupling_cpp,
          "Ejecuta paso de Acoplamiento Conformal Direct-Pointer (Hito 1.3-R1)",
          nb::arg("h_state"), nb::arg("u_attractor"), nb::arg("step"),
          nb::arg("tau_eff") = 0.15f, nb::arg("kappa_att") = 0.80f,
          nb::arg("beta_gate") = 12.0f, nb::arg("theta_gate") = 0.35f,
          nb::arg("mode") = 1, nb::arg("force_g") = -1.0f);
    m.def("junction_reset", &junction_reset_cpp, "Reinicia la union conformal");

    // Hito 2.1: Célula de Memoria Geométrica en Espacio de Hilbert (CPU y Metal)
    m.def("hilbert_memory_ingest", &hilbert_memory_ingest_cpp,
          "Ingesta persistente de subproducto en la Celula de Memoria (exige D_C1 == D_C2)",
          nb::arg("h_subprod"), nb::arg("timestamp"), nb::arg("energy"));
    m.def("hilbert_memory_pack_two", &hilbert_memory_pack_two_cpp,
          "Empaquetamiento AND-like de 2 hechos en superposicion (CPU)",
          nb::arg("m_A"), nb::arg("m_B"));
    m.def("hilbert_memory_deflate", &hilbert_memory_deflate_cpp,
          "Deflacion NOT-like (Match and Peel) ortogonal (CPU)",
          nb::arg("m_pack"), nb::arg("m_target"));
    m.def("hilbert_memory_style_transport", &hilbert_memory_style_transport_cpp,
          "Transporte paralelo de estilo sobre el plano tangente (CPU)",
          nb::arg("h_truth"), nb::arg("u_style"), nb::arg("theta_s"));

    m.def("hilbert_memory_pack_two_metal", &hilbert_memory_pack_two_metal,
          "Empaquetamiento AND-like de 2 hechos en superposicion (Metal GPU)",
          nb::arg("m_A"), nb::arg("m_B"));
    m.def("hilbert_memory_deflate_metal", &hilbert_memory_deflate_metal,
          "Deflacion NOT-like (Match and Peel) ortogonal (Metal GPU)",
          nb::arg("m_pack"), nb::arg("m_target"));
    m.def("hilbert_memory_style_transport_metal", &hilbert_memory_style_transport_metal,
          "Transporte paralelo de estilo sobre el plano tangente (Metal GPU)",
          nb::arg("h_truth"), nb::arg("u_style"), nb::arg("theta_s"));

    m.def("hilbert_memory_reset", &hilbert_memory_reset_cpp, "Reinicia la celula de memoria Hilbert");

    // Hito 2.1-D: Consultas de Memoria Inter-Celular (Gap Junction)
    m.def("hilbert_memory_slot_count", &hilbert_memory_slot_count_cpp, "Numero de slots ocupados en la memoria");
    m.def("hilbert_memory_get_slot", &hilbert_memory_get_slot_cpp, "Recupera tensor de un slot de memoria",
          nb::arg("slot_idx"));
    m.def("hilbert_memory_query_resonance", &hilbert_memory_query_resonance_cpp,
          "Consulta resonancia de un query contra todos los slots de memoria",
          nb::arg("u_query"));

    // Hito 2.2: Sustrato de Direccionamiento y Detector de Cresta Cinemática
    m.def("fact_band_detect_peak", &fact_band_detect_peak_cpp,
          "Detecta la cresta cinematica kappa(l) de la Candidate Fact Band",
          nb::arg("layer_arrays"));
    m.def("fact_band_route_layer", &fact_band_route_layer_cpp,
          "Enrutamiento asociativo 1xK en hot-path con compuerta rectificada (cero fuga) y margen",
          nb::arg("h_layer"), nb::arg("threshold") = 0.45f, nb::arg("beta") = 16.0f);
}



