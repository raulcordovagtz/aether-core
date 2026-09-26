// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: FACT BAND ROUTER & CANDIDATE PEAK DETECTOR (C-023 SSOT)
// Detección Cinemática de Cresta κ(l) en el Campo Tendiente (l >= 0.60 * N)
// ═════════════════════════════════════════════════════════════════════════════
#pragma once

#include "hilbert_memory_cell.h"
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace aether {

enum class InterventionTreatment : uint32_t {
    ConcentratedPeak    = 0,
    CurvatureWeighted   = 1
};

enum class KinematicDegeneracy : uint32_t {
    NONE                  = 0,
    ZERO_VELOCITY         = 1,
    NUMERIC_CLAMP_APPLIED = 2
};

struct alignas(16) RouterDecision {
    uint32_t selected_slot;
    float    max_resonance_r;
    float    second_resonance_r;
    float    resonance_margin;
    float    rectified_gate_g;
    float    layer_curvature_kappa;
    bool     is_active_injection;
};

class FactBandRouter {
public:
    FactBandRouter(
        uint32_t dimension = 2048,
        float resonance_threshold = 0.45f,
        float beta_sensitivity = 16.0f,
        float total_angular_bound = 0.15f
    ) : D_(dimension),
        theta_assoc_(resonance_threshold),
        beta_(beta_sensitivity),
        theta_bound_(total_angular_bound),
        treatment_(InterventionTreatment::ConcentratedPeak),
        memory_read_only_(true)
    {}

    float compute_rectified_gate(float r_max) const {
        if (r_max < theta_assoc_) {
            return 0.0f;
        }
        return 1.0f / (1.0f + std::exp(-beta_ * (r_max - theta_assoc_)));
    }

    // ─── DETECTOR DE CRESTA CINEMÁTICA EN CAMPO TENDIENTE (l >= 0.60 * N) ───
    static uint32_t detect_candidate_band_peak(
        const std::vector<const float*>& layer_states,
        uint32_t num_layers,
        uint32_t D,
        std::vector<float>& out_kappas,
        std::vector<KinematicDegeneracy>* out_degeneracies = nullptr
    ) {
        out_kappas.assign(num_layers, 0.0f);
        if (out_degeneracies) {
            out_degeneracies->assign(num_layers, KinematicDegeneracy::NONE);
        }
        if (num_layers < 4) return num_layers / 2;

        float max_kappa = -1.0f;
        uint32_t peak_l = static_cast<uint32_t>(num_layers * 0.79f);

        // Límite analítico: la Fact Band reside en el cono profundo post-sintaxis
        uint32_t l_start_tendiente = static_cast<uint32_t>(num_layers * 0.60f);

        std::vector<float> v_curr(D, 0.0f);
        std::vector<float> v_prev(D, 0.0f);

        for (uint32_t l = 1; l < num_layers; ++l) {
            float sq_v = 0.0f;
            for (uint32_t i = 0; i < D; ++i) {
                v_curr[i] = layer_states[l][i] - layer_states[l - 1][i];
                sq_v += v_curr[i] * v_curr[i];
            }

            if (l >= 2) {
                if (sq_v < 1e-8f) {
                    out_kappas[l] = 0.0f;
                    if (out_degeneracies) {
                        (*out_degeneracies)[l] = KinematicDegeneracy::ZERO_VELOCITY;
                    }
                    v_prev = v_curr;
                    continue;
                }

                float sq_a = 0.0f;
                float dot_va = 0.0f;
                for (uint32_t i = 0; i < D; ++i) {
                    float a = v_curr[i] - v_prev[i];
                    sq_a += a * a;
                    dot_va += v_curr[i] * a;
                }

                float bivector_sq = (sq_v * sq_a) - (dot_va * dot_va);
                KinematicDegeneracy degen = KinematicDegeneracy::NONE;
                if (bivector_sq < 0.0f) {
                    bivector_sq = 0.0f;
                    degen = KinematicDegeneracy::NUMERIC_CLAMP_APPLIED;
                }

                float kappa = std::sqrt(bivector_sq) / (std::pow(sq_v, 1.5f) + 1e-6f);
                out_kappas[l] = kappa;
                if (out_degeneracies) {
                    (*out_degeneracies)[l] = degen;
                }

                // La cresta máxima solo se busca dentro del campo tendiente de decisión
                if (l >= l_start_tendiente && kappa > max_kappa) {
                    max_kappa = kappa;
                    peak_l = l;
                }
            }
            v_prev = v_curr;
        }
        return peak_l;
    }

    RouterDecision evaluate_layer_routing(
        const float* h_layer,
        const HilbertMemoryCell& memory_cell
    ) const {
        RouterDecision dec{};
        uint32_t K = memory_cell.count();
        if (K == 0) {
            dec.selected_slot = 0;
            dec.max_resonance_r = -1.0f;
            dec.second_resonance_r = -1.0f;
            dec.resonance_margin = 0.0f;
            dec.rectified_gate_g = 0.0f;
            dec.is_active_injection = false;
            return dec;
        }

        float max_r = -2.0f;
        float second_r = -2.0f;
        uint32_t best_slot = 0;

        for (uint32_t k = 0; k < K; ++k) {
            float r_k = memory_cell.query_slot_resonance(k, h_layer);
            if (r_k > max_r) {
                second_r = max_r;
                max_r = r_k;
                best_slot = k;
            } else if (r_k > second_r) {
                second_r = r_k;
            }
        }

        if (K == 1) {
            second_r = -1.0f;
        }

        dec.selected_slot = best_slot;
        dec.max_resonance_r = max_r;
        dec.second_resonance_r = second_r;
        dec.resonance_margin = (K > 1) ? (max_r - second_r) : 1.0f;
        dec.rectified_gate_g = compute_rectified_gate(max_r);
        dec.is_active_injection = (dec.rectified_gate_g > 0.0f);
        return dec;
    }

    void set_treatment(InterventionTreatment t) { treatment_ = t; }
    InterventionTreatment get_treatment() const { return treatment_; }

    void set_read_only(bool ro) { memory_read_only_ = ro; }
    bool is_read_only() const { return memory_read_only_; }

    void set_threshold(float th) { theta_assoc_ = th; }
    float get_threshold() const { return theta_assoc_; }

    float get_angular_bound() const { return theta_bound_; }
    uint32_t dimension() const { return D_; }

private:
    uint32_t D_;
    float theta_assoc_;
    float beta_;
    float theta_bound_;
    InterventionTreatment treatment_;
    bool memory_read_only_;
};

} // namespace aether
