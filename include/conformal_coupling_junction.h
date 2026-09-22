// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: CONFORMAL COUPLING JUNCTION (HITO 1.3-R1)
// Enlace Inter-Modular: Puntero Directo UMA ↔ Búfer Markoviano ↔ Célula
// ═════════════════════════════════════════════════════════════════════════════
#pragma once

#include "geodesic_trajectory_cell.h"
#include "intracycle_state_buffer.h"
#include "permeability_gate.h"
#include <memory>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace aether {

struct CouplingMetrics {
    float correlation_r;          // Certeza con el atractor contextual [-1, 1]
    float curvature_kappa;        // Curvatura cinemática de Lagrange en R^D
    float dirichlet_tension_q;    // Tensión cinemática q_k
    float permeability_g;         // Coeficiente continuo g_k ∈ [0, 1]
    float angular_displacement;   // θ recorrido
    bool  gate_open;              // g_k >= 0.50 (umbral macro de apertura)
    bool  cell_evaluated;         // g_k >= 1e-4 y modo activo (ejecución vs bypass)
    bool  intervention_applied;   // g_k > 0 y modo activo (modificación aplicada)
    uint32_t active_regime;       // 0=Laminar, 1=Transición, 2=Turbulento
};

class ConformalCouplingJunction {
public:
    ConformalCouplingJunction(
        uint32_t dimension = 2048,
        float tau_effective = 0.15f,
        float kappa_attractor = 0.80f,
        float beta_gate = 12.0f,
        float theta_gate = 0.35f
    ) : D_(dimension),
        tau_eff_(tau_effective),
        kappa_att_(kappa_attractor),
        buffer_(std::make_unique<IntracycleStateBuffer>(dimension)),
        gate_(std::make_unique<PermeabilityGate>(beta_gate, theta_gate, GateInterventionMode::ActiveCoupled))
    {
        h_projected_.assign(D_, 0.0f);
        h_deflated_.assign(D_, 0.0f);
        h_out_.assign(D_, 0.0f);
    }

    void reset() {
        buffer_->reset();
        std::fill(h_projected_.begin(), h_projected_.end(), 0.0f);
        std::fill(h_deflated_.begin(), h_deflated_.end(), 0.0f);
        std::fill(h_out_.begin(), h_out_.end(), 0.0f);
    }

    void set_mode(GateInterventionMode mode) {
        gate_->set_mode(mode);
    }

    GateInterventionMode get_mode() const {
        return gate_->get_mode();
    }

    void set_hyperparameters(float tau, float kappa, float beta, float theta) {
        tau_eff_   = tau;
        kappa_att_ = kappa;
        gate_->set_parameters(beta, theta);
    }

    // Paso de acoplamiento consumiendo puntero contiguo UMA (CERO ALOCACIONES)
    CouplingMetrics couple_step(
        const float* h_in,            // [D] Puntero directo de entrada
        const float* u_attractor,     // [D] Atractor contextual (prompt/historial)
        uint32_t step,
        float force_g = -1.0f         // Permite forzar g=0.0f en condición Active-0
    ) {
        // 1. Ingesta directa en el Búfer Markoviano (Hito 1.2)
        KinematicState k = buffer_->push_state_zero_copy(h_in, step);
        GateState g = gate_->evaluate(k.dirichlet_tension_q);

        float effective_g = (force_g >= 0.0f) ? force_g : g.permeability_g;

        CouplingMetrics metrics{};
        metrics.dirichlet_tension_q  = k.dirichlet_tension_q;
        metrics.permeability_g       = effective_g;
        metrics.gate_open            = (effective_g >= 0.50f);
        metrics.cell_evaluated       = (effective_g >= 1e-4f) && 
                                       (gate_->get_mode() == GateInterventionMode::ActiveCoupled);
        metrics.intervention_applied = (effective_g > 0.0f) &&
                                       (gate_->get_mode() == GateInterventionMode::ActiveCoupled);

        // Si la compuerta está en modo pasivo o por debajo de 1e-4: Identidad exacta
        if (!metrics.cell_evaluated) {
            std::memcpy(h_out_.data(), h_in, D_ * sizeof(float));
            std::memcpy(h_projected_.data(), h_in, D_ * sizeof(float));
            std::fill(h_deflated_.begin(), h_deflated_.end(), 0.0f);
            metrics.correlation_r = 1.0f;
            metrics.curvature_kappa = 0.0f;
            metrics.angular_displacement = 0.0f;
            metrics.active_regime = 0;
            return metrics;
        }

        // 2. Extrapolación analítica de la Célula Proyectiva (Hito 1.1)
        // a_t cinemático se utiliza para kappa_kin (curvatura de Lagrange)
        // La trayectoria proyectada h* se deriva de h_t, v_t y u_attractor
        const float* v_ptr = buffer_->current_v();

        float inv_norm_h = 1.0f / (k.norm_h + 1e-12f);
        float coeff_hv = k.dot_hv * inv_norm_h * inv_norm_h;
        float sq_v_perp = std::max(0.0f, k.sq_v - (k.dot_hv * k.dot_hv * inv_norm_h * inv_norm_h));
        float norm_v_perp = std::sqrt(sq_v_perp + 1e-12f);
        float inv_norm_vp = 1.0f / (norm_v_perp + 1e-12f);

        // Curvatura de Lagrange en R^D basada en v_t y a_t cinemáticos
        float bivector_sq = std::max(0.0f, (k.sq_v * k.sq_a) - (k.dot_va * k.dot_va));
        float kappa_kin = std::sqrt(bivector_sq) / (std::pow(std::max(k.sq_v, 0.0f), 1.5f) + 1e-12f);

        float omega = norm_v_perp * inv_norm_h;
        float theta = omega * tau_eff_;
        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);

        float dot_hu = 0.0f;
        for (uint32_t i = 0; i < D_; ++i) dot_hu += h_in[i] * u_attractor[i];

        float sq_proj = 0.0f;
        float dot_proj_u = 0.0f;

        // Trayectoria geodésica sobre la esfera impulsada por v_perp y atractor contextual
        for (uint32_t i = 0; i < D_; ++i) {
            float v_p = v_ptr[i] - coeff_hv * h_in[i];
            float v_hat = v_p * inv_norm_vp;
            float a_att = kappa_att_ * (u_attractor[i] - (dot_hu * inv_norm_h * inv_norm_h) * h_in[i]);

            float h_star = (cos_t * h_in[i]) + (sin_t * v_hat) + (0.5f * tau_eff_ * tau_eff_ * a_att);
            h_projected_[i] = h_star;
            sq_proj += h_star * h_star;
            dot_proj_u += h_star * u_attractor[i];
        }

        float inv_norm_star = 1.0f / std::sqrt(sq_proj + 1e-12f);
        float r_val = dot_proj_u * inv_norm_star;

        for (uint32_t i = 0; i < D_; ++i) {
            h_projected_[i] *= inv_norm_star;
        }

        // 3. Subproducto Ortogonal Exacto (<h_projected, h_deflated> = 0)
        float dot_h_hstar = 0.0f;
        for (uint32_t i = 0; i < D_; ++i) dot_h_hstar += h_in[i] * h_projected_[i];
        for (uint32_t i = 0; i < D_; ++i) {
            h_deflated_[i] = h_in[i] - dot_h_hstar * h_projected_[i];
        }

        // 4. Retracción normalizada sobre la esfera: h_out = Normalize((1 - g) h_in + g h_projected)
        float sq_out = 0.0f;
        for (uint32_t i = 0; i < D_; ++i) {
            float mixed = (1.0f - effective_g) * h_in[i] + effective_g * h_projected_[i];
            h_out_[i] = mixed;
            sq_out += mixed * mixed;
        }
        float inv_norm_out = 1.0f / std::sqrt(sq_out + 1e-12f);
        for (uint32_t i = 0; i < D_; ++i) {
            h_out_[i] *= inv_norm_out;
        }

        metrics.correlation_r        = r_val;
        metrics.curvature_kappa      = kappa_kin;
        metrics.angular_displacement = theta;
        metrics.active_regime        = (r_val >= 0.95f) ? 0 : ((r_val >= 0.80f) ? 1 : 2);
        return metrics;
    }

    const float* get_steered_output() const { return h_out_.data(); }
    const float* get_projected_state() const { return h_projected_.data(); }
    const float* get_orthogonal_subproduct() const { return h_deflated_.data(); }

private:
    uint32_t D_;
    float tau_eff_;
    float kappa_att_;
    std::unique_ptr<IntracycleStateBuffer> buffer_;
    std::unique_ptr<PermeabilityGate> gate_;
    std::vector<float> h_projected_;
    std::vector<float> h_deflated_;
    std::vector<float> h_out_;
};

} // namespace aether
