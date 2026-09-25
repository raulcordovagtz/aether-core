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
#include "../include/hilbert_memory_cell.h"
#include "../include/fact_band_router.h"
#include "../include/tetrapolar_predictor_cell.h"

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

// ─── 4. ESTADO COMPARTIDO METAL & MEMORIA GEOMÉTRICA (HITO 2.1 & 2.2) ────────
static id<MTLDevice> g_metal_device = nil;
static id<MTLCommandQueue> g_metal_queue = nil;
static std::unique_ptr<aether::HilbertMemoryCell> g_hilbert_memory = nullptr;


// ─── 5. PUENTE C++ & METAL: CÉLULA DE MEMORIA GEOMÉTRICA (HITO 2.1) ───────────

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

// ─── 8. PUENTE C++ & METAL: PREDICTOR GEODÉSICO TETRAPOLAR (C-022) ───────────
static id<MTLComputePipelineState> g_metal_pso_pred = nil;
static id<MTLBuffer> g_buf_pred_h = nil;
static id<MTLBuffer> g_buf_pred_v = nil;
static id<MTLBuffer> g_buf_pred_onto = nil;
static id<MTLBuffer> g_buf_pred_teleo = nil;
static id<MTLBuffer> g_buf_pred_anti = nil;
static id<MTLBuffer> g_buf_pred_eos = nil;
static id<MTLBuffer> g_buf_pred_out = nil;
static id<MTLBuffer> g_buf_pred_tel = nil;
static size_t g_buf_pred_capacity = 0;

static void init_metal_tetrapolar_predictor() {
    if (g_metal_pso_pred != nil) return;
    @autoreleasepool {
        if (!g_metal_device) {
            g_metal_device = MTLCreateSystemDefaultDevice();
        }
        if (!g_metal_queue) {
            g_metal_queue = [g_metal_device newCommandQueue];
        }

        NSError* err = nil;
        NSString* path = @"metal/tetrapolar_predictor_cell.metallib";
        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            path = @"/Users/crotalo/aether_engine/metal/tetrapolar_predictor_cell.metallib";
        }
        NSURL* libURL = [NSURL fileURLWithPath:path];
        id<MTLLibrary> lib = [g_metal_device newLibraryWithURL:libURL error:&err];
        if (!lib) {
            throw std::runtime_error("No se pudo cargar tetrapolar_predictor_cell.metallib: " +
                                     std::string(err ? [[err localizedDescription] UTF8String] : "unknown"));
        }

        id<MTLFunction> fn = [lib newFunctionWithName:@"dispatch_tetrapolar_predictor_step"];
        if (!fn) {
            throw std::runtime_error("Función dispatch_tetrapolar_predictor_step no encontrada en metallib");
        }

        g_metal_pso_pred = [g_metal_device newComputePipelineStateWithFunction:fn error:&err];
        if (!g_metal_pso_pred) {
            throw std::runtime_error("Error creando pipeline state Metal para predictor: " +
                                     std::string(err ? [[err localizedDescription] UTF8String] : "unknown"));
        }
    }
}

static void ensure_metal_predictor_buffers(size_t required_bytes) {
    if (g_buf_pred_h != nil && g_buf_pred_capacity >= required_bytes) return;
    size_t cap = std::max(required_bytes, static_cast<size_t>(8192 * sizeof(float)));
    g_buf_pred_h     = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    g_buf_pred_v     = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    g_buf_pred_onto  = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    g_buf_pred_teleo = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    g_buf_pred_anti  = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    g_buf_pred_eos   = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    g_buf_pred_out   = [g_metal_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
    if (!g_buf_pred_tel) {
        g_buf_pred_tel = [g_metal_device newBufferWithLength:sizeof(aether::PredictorTelemetry) options:MTLResourceStorageModeShared];
    }
    g_buf_pred_capacity = cap;
}

nb::dict tetrapolar_predictor_step_cpp(
    const array& h_in,
    const array& v_tangent,
    const array& u_onto,
    const array& u_teleo,
    const array& u_anti,
    const array& u_eos,
    float tau = 0.0f
) {
    eval({h_in, v_tangent, u_onto, u_teleo, u_anti, u_eos});
    uint32_t D = static_cast<uint32_t>(h_in.shape(-1));
    static std::vector<float> g_pred_h_star;
    if (g_pred_h_star.size() < D) g_pred_h_star.resize(D);

    aether::TetrapolarPoles poles{
        u_onto.data<float>(),
        u_teleo.data<float>(),
        u_anti.data<float>(),
        u_eos.data<float>()
    };

    aether::PredictorTelemetry tel{};
    aether::TetrapolarPredictorCell cell(D);
    cell.project_geodesic(g_pred_h_star.data(), h_in.data<float>(), v_tangent.data<float>(), tau, tel, poles);

    array h_star = array(g_pred_h_star.data(), {static_cast<int>(D)}, float32);

    nb::dict telemetry_dict;
    telemetry_dict["omega_angular_velocity"] = tel.omega_angular_velocity;
    telemetry_dict["curvature_kappa"]        = tel.curvature_kappa;
    telemetry_dict["grad_onto"]              = tel.grad_onto;
    telemetry_dict["grad_teleo"]             = tel.grad_teleo;
    telemetry_dict["grad_anti"]              = tel.grad_anti;
    telemetry_dict["grad_eos"]               = tel.grad_eos;
    telemetry_dict["teleology_alignment"]    = tel.teleology_alignment;

    nb::dict res;
    res["h_star"]    = h_star;
    res["telemetry"] = telemetry_dict;
    return res;
}

nb::dict tetrapolar_predictor_step_metal(
    const array& h_in,
    const array& v_tangent,
    const array& u_onto,
    const array& u_teleo,
    const array& u_anti,
    const array& u_eos,
    float tau = 0.0f
) {
    init_metal_tetrapolar_predictor();
    eval({h_in, v_tangent, u_onto, u_teleo, u_anti, u_eos});

    uint32_t D = static_cast<uint32_t>(h_in.shape(-1));
    size_t bytes_vec = D * sizeof(float);
    ensure_metal_predictor_buffers(bytes_vec);

    @autoreleasepool {
        std::memcpy([g_buf_pred_h contents], h_in.data<float>(), bytes_vec);
        std::memcpy([g_buf_pred_v contents], v_tangent.data<float>(), bytes_vec);
        std::memcpy([g_buf_pred_onto contents], u_onto.data<float>(), bytes_vec);
        std::memcpy([g_buf_pred_teleo contents], u_teleo.data<float>(), bytes_vec);
        std::memcpy([g_buf_pred_anti contents], u_anti.data<float>(), bytes_vec);
        std::memcpy([g_buf_pred_eos contents], u_eos.data<float>(), bytes_vec);

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_metal_pso_pred];
        [enc setBuffer:g_buf_pred_h offset:0 atIndex:0];
        [enc setBuffer:g_buf_pred_v offset:0 atIndex:1];
        [enc setBuffer:g_buf_pred_onto offset:0 atIndex:2];
        [enc setBuffer:g_buf_pred_teleo offset:0 atIndex:3];
        [enc setBuffer:g_buf_pred_anti offset:0 atIndex:4];
        [enc setBuffer:g_buf_pred_eos offset:0 atIndex:5];
        [enc setBuffer:g_buf_pred_out offset:0 atIndex:6];
        [enc setBuffer:g_buf_pred_tel offset:0 atIndex:7];
        [enc setBytes:&tau length:sizeof(float) atIndex:8];
        [enc setBytes:&D length:sizeof(uint32_t) atIndex:9];

        // 256 threads * 5 floats = 5120 bytes de memoria compartida
        [enc setThreadgroupMemoryLength:256 * 5 * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        float* p_out = (float*)[g_buf_pred_out contents];
        aether::PredictorTelemetry* p_tel = (aether::PredictorTelemetry*)[g_buf_pred_tel contents];

        array h_star(p_out, {static_cast<int>(D)}, float32);

        nb::dict telemetry_dict;
        telemetry_dict["omega_angular_velocity"] = p_tel->omega_angular_velocity;
        telemetry_dict["curvature_kappa"]        = p_tel->curvature_kappa;
        telemetry_dict["grad_onto"]              = p_tel->grad_onto;
        telemetry_dict["grad_teleo"]             = p_tel->grad_teleo;
        telemetry_dict["grad_anti"]              = p_tel->grad_anti;
        telemetry_dict["grad_eos"]               = p_tel->grad_eos;
        telemetry_dict["teleology_alignment"]    = p_tel->teleology_alignment;

        nb::dict res;
        res["h_star"]    = h_star;
        res["telemetry"] = telemetry_dict;
        return res;
    }
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

    // C-022: Predictor Geodésico Tetrapolar (CPU y Metal GPU)
    m.def("tetrapolar_predictor_step", &tetrapolar_predictor_step_cpp,
          "Extrapolación geodésica analítica en S^{D-1} y 4 líneas derivativas (CPU)",
          nb::arg("h_in"), nb::arg("v_tangent"),
          nb::arg("u_onto"), nb::arg("u_teleo"), nb::arg("u_anti"), nb::arg("u_eos"),
          nb::arg("tau") = 0.0f);
    m.def("tetrapolar_predictor_step_metal", &tetrapolar_predictor_step_metal,
          "Extrapolación geodésica analítica en S^{D-1} y 4 líneas derivativas (Metal GPU)",
          nb::arg("h_in"), nb::arg("v_tangent"),
          nb::arg("u_onto"), nb::arg("u_teleo"), nb::arg("u_anti"), nb::arg("u_eos"),
          nb::arg("tau") = 0.0f);
}



