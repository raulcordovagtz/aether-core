// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: FACT BAND ROUTER & CANDIDATE PEAK DETECTOR (HITO 2.2-R1)
// Detección Cinemática de Cresta κ(l) y Enrutamiento Asociativo en Silicio UMA
// ═════════════════════════════════════════════════════════════════════════════
#pragma once

#include "hilbert_memory_cell.h"
#include "conformal_coupling_junction.h"
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace aether {

// Tratamientos experimentales de modulación en la Candidate Fact Band
enum class InterventionTreatment : uint32_t {
    ConcentratedPeak    = 0, // Tratamiento A: Intervención única en l* = argmax κ(l)
    CurvatureWeighted   = 1  // Tratamiento B: Intervención distribuida ponderada por κ(l)
};

// Causas de degeneración cinemática en el cálculo de curvatura
enum class KinematicDegeneracy : uint32_t {
    NONE                  = 0,
    ZERO_VELOCITY         = 1, // ||v|| ≈ 0: Flujo colapsado o idéntico entre capas
    NUMERIC_CLAMP_APPLIED = 2  // Redondeo negativo en ||v||^2 ||a||^2 - (v·a)^2 fijado a 0
};

struct alignas(16) RouterDecision {
    uint32_t selected_slot;        // k* = argmax <h, m_k>
    float    max_resonance_r;      // r_max
    float    second_resonance_r;   // r_second (segundo mejor)
    float    resonance_margin;     // margin = r_max - r_second
    float    rectified_gate_g;     // g(r_max) ∈ [0, 1] con cero absoluto estricto
    float    layer_curvature_kappa;// κ(l) de la capa evaluada
    bool     is_active_injection;  // g > 0.0f
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

    // ─── 1. COMPUERTA RECTIFICADA ESTRICTA (CERO FUGA) ───────────────────────
    // g(r) = 0 si r < theta;  g(r) = sigma(beta * (r - theta)) si r >= theta
    float compute_rectified_gate(float r_max) const {
        if (r_max < theta_assoc_) {
            return 0.0f; // Cero absoluto en silicio
        }
        return 1.0f / (1.0f + std::exp(-beta_ * (r_max - theta_assoc_)));
    }

    // ─── 2. DETECTOR DE CRESTA CINEMÁTICA κ(l) (MODO A: AGNÓSTICO PURO) ──────
    // Analiza la secuencia de estados residuales de prefill a lo largo de las N capas.
    // Retorna l* = argmax κ(l) y la tabla completa de curvaturas.
    // Protegido contra ||v|| ≈ 0 y clamp numérico del radicando de bivector.
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
        uint32_t peak_l = num_layers / 2;

        std::vector<float> v_curr(D, 0.0f);
        std::vector<float> v_prev(D, 0.0f);

        for (uint32_t l = 1; l < num_layers; ++l) {
            float sq_v = 0.0f;
            for (uint32_t i = 0; i < D; ++i) {
                v_curr[i] = layer_states[l][i] - layer_states[l - 1][i];
                sq_v += v_curr[i] * v_curr[i];
            }

            if (l >= 2) {
                // Si la velocidad es virtualmente nula, flujo estacionario => kappa = 0
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

                // Radicando del bivector de Lagrange: ||v||^2 ||a||^2 - (v·a)^2 >= 0
                // Clamp explícito contra errores de cancelación numérica en float32
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

                if (kappa > max_kappa) {
                    max_kappa = kappa;
                    peak_l = l;
                }
            }
            v_prev = v_curr;
        }
        return peak_l;
    }

    // ─── 3. ENRUTADOR ASOCIATIVO EN HOT-PATH (CERO ARCCOS, CON MARGEN) ────────
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
