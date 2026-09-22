// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: PERMEABILITY GATE (HITO 1.2 — DUAL-MODE)
// ═════════════════════════════════════════════════════════════════════════════
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace aether {

enum class GateInterventionMode : uint32_t {
    PassiveObserve = 0,  // Modo LAB 09: Observa, mide y registra sin alterar h
    ActiveCoupled  = 1   // Modo Inferencia: Inyecta h* si g_k supera el umbral
};

struct GateState {
    float dirichlet_q;
    float permeability_g;
    bool is_open;
    GateInterventionMode mode;
};

class PermeabilityGate {
public:
    PermeabilityGate(
        float beta = 12.0f,
        float theta = 0.50f,
        GateInterventionMode mode = GateInterventionMode::PassiveObserve
    ) : beta_(beta), theta_(theta), mode_(mode) {}

    GateState evaluate(float dirichlet_q) const {
        float g = 1.0f / (1.0f + std::exp(-beta_ * (dirichlet_q - theta_)));
        bool open = (g >= 0.50f);
        return GateState{dirichlet_q, g, open, mode_};
    }

    void set_mode(GateInterventionMode mode) { mode_ = mode; }
    GateInterventionMode get_mode() const { return mode_; }

    void set_parameters(float beta, float theta) {
        beta_ = beta;
        theta_ = theta;
    }

    void apply_boundary_filter(float* h_out, const float* h_in, const float* h_star, float g, uint32_t D) const {
        if (mode_ == GateInterventionMode::PassiveObserve) {
            std::memcpy(h_out, h_in, D * sizeof(float));
            return;
        }

        float sq_mix = 0.0f;
        for (uint32_t i = 0; i < D; ++i) {
            float mixed = (1.0f - g) * h_in[i] + g * h_star[i];
            h_out[i] = mixed;
            sq_mix += mixed * mixed;
        }
        float inv_norm = 1.0f / std::sqrt(sq_mix + 1e-12f);
        for (uint32_t i = 0; i < D; ++i) {
            h_out[i] *= inv_norm;
        }
    }

private:
    float beta_;
    float theta_;
    GateInterventionMode mode_;
};

} // namespace aether
