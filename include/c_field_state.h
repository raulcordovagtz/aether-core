#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <iomanip>
#include <iostream>

namespace c_field {

constexpr uint32_t FIELD_DIM = 2048;

struct CFFieldInvariants {
    float kinetic_energy = 0.0f;
    float potential_energy = 0.0f;
    float semantic_density_rho = 0.0f;
    float curvature_scalar = 0.0f;
    float phase_coherence = 1.0f;
};

struct CFFieldParameters {
    bool absorption_enabled = false;
    bool homeostasis_enabled = false;
    bool phase_transition_enabled = false;

    float lambda_absorb = 0.0f;
    float lambda_homeo = 0.0f;
    float lambda_collapse = 0.0f;
    float dt = 1.0f;
};

class CFFieldState {
public:
    CFFieldState() {
        psi.assign(FIELD_DIM, 0.0f);
        psi_dot.assign(FIELD_DIM, 0.0f);
        f_base.assign(FIELD_DIM, 0.0f);
        f_absorb.assign(FIELD_DIM, 0.0f);
        f_homeo.assign(FIELD_DIM, 0.0f);
        f_collapse.assign(FIELD_DIM, 0.0f);
    }

    void initialize_from_substrate(const float* substrate_z) {
        double sq = 0.0;
        for (uint32_t i = 0; i < FIELD_DIM; ++i) {
            psi[i] = substrate_z[i];
            f_base[i] = substrate_z[i];
            psi_dot[i] = 0.0f;
            sq += double(substrate_z[i]) * double(substrate_z[i]);
        }
        tau = 0;
        invariants.semantic_density_rho = std::sqrt(sq);
        invariants.kinetic_energy = 0.0f;
    }

    // Integrador Geodésico Conformal con Conservación de Energía Noether
    void step_derivation(const float* substrate_z_next, const CFFieldParameters& params) {
        tau++;

        double base_sq = 0.0;
        for (uint32_t i = 0; i < FIELD_DIM; ++i) {
            f_base[i] = substrate_z_next[i];
            base_sq += double(f_base[i]) * double(f_base[i]);
        }
        float base_norm = std::sqrt(base_sq);

        if (!params.absorption_enabled && !params.homeostasis_enabled && !params.phase_transition_enabled) {
            // Modo Neutral: Identidad Conformal Absoluta
            for (uint32_t i = 0; i < FIELD_DIM; ++i) {
                psi[i] = f_base[i];
                psi_dot[i] = 0.0f;
            }
            invariants.semantic_density_rho = base_norm;
            invariants.kinetic_energy = 0.0f;
            return;
        }

        // Modo Físico Activo: Composición de Fuerzas en Espacio Tangente
        std::vector<float> total_force(FIELD_DIM);
        double total_sq = 0.0;

        for (uint32_t i = 0; i < FIELD_DIM; ++i) {
            float f_pert = 0.0f;
            if (params.absorption_enabled) {
                f_pert += params.lambda_absorb * f_absorb[i];
            }
            if (params.homeostasis_enabled) {
                f_pert += params.lambda_homeo * f_homeo[i];
            }
            if (params.phase_transition_enabled) {
                f_pert += params.lambda_collapse * f_collapse[i];
            }

            total_force[i] = f_base[i] + f_pert;
            total_sq += double(total_force[i]) * double(total_force[i]);
        }

        float total_norm = std::sqrt(total_sq);
        float conformal_scale = (total_norm > 1e-6f) ? (base_norm / total_norm) : 1.0f;

        double vel_sq = 0.0;
        for (uint32_t i = 0; i < FIELD_DIM; ++i) {
            float psi_next = total_force[i] * conformal_scale; // Proyección Conformal
            psi_dot[i] = psi_next - psi[i];
            psi[i] = psi_next;
            vel_sq += double(psi_dot[i]) * double(psi_dot[i]);
        }

        invariants.semantic_density_rho = base_norm;
        invariants.kinetic_energy = 0.5f * float(vel_sq);
    }

    const float* data() const { return psi.data(); }
    float* data() { return psi.data(); }

    uint32_t tau = 0;
    std::vector<float> psi;
    std::vector<float> psi_dot;
    std::vector<float> f_base;
    std::vector<float> f_absorb;
    std::vector<float> f_homeo;
    std::vector<float> f_collapse;

    CFFieldInvariants invariants;
};

} // namespace c_field
