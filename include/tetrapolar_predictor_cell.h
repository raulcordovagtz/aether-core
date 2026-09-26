// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: TETRAPOLAR PREDICTOR CELL (CÉLULA 1 - REVAMPED)
// Núcleo Geodésico Continuo en S^{D-1}, Membrana UMA y Líneas Derivativas
// CERO tiros parabólicos | CERO buffers temporales ciegos | CERO promedios
// ═════════════════════════════════════════════════════════════════════════════
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace aether {

// Estructura de las cuatro primitivas del campo de navegación
struct TetrapolarPoles {
    const float* u_onto;   // [D] Premisas / Anclaje factual inicial
    const float* u_teleo;  // [D] Intención deductiva / Meta de llegada
    const float* u_anti;   // [D] Subespacio de inconsistencia o contradicción
    const float* u_eos;    // [D] Vector de colapso terminal (End of Sequence)
};

// Telemetría analítica calculada por el Núcleo Celular
struct alignas(16) PredictorTelemetry {
    float omega_angular_velocity; // Velocidad angular en la variedad rad/paso
    float curvature_kappa;        // Curvatura de Lagrange instantánea
    float grad_onto;              // Componente tangencial hacia la ontología
    float grad_teleo;             // Componente tangencial hacia la teleología
    float grad_anti;              // Componente tangencial hacia la antítesis
    float grad_eos;               // Componente tangencial hacia el pozo EOS
    float teleology_alignment;    // Proyección estática <h*, u_teleo>
};

class TetrapolarPredictorCell {
public:
    explicit TetrapolarPredictorCell(uint32_t dimension = 2048) : D_(dimension) {
        if (D_ == 0) throw std::invalid_argument("La dimensión debe ser mayor a 0");
    }

    // ─── NÚCLEO: EXTRAPOLACIÓN GEODÉSICA PURA EN S^{D-1} ────────────────────
    // h*(tau) = cos(theta) * h_unit + sin(theta) * v_hat
    // Conserva analíticamente la norma unitaria en cualquier horizonte tau
    void project_geodesic(
        float* h_star_out,
        const float* h_in,
        const float* v_tangent,
        float tau,
        PredictorTelemetry& telemetry,
        const TetrapolarPoles& poles
    ) const {
        float sq_h = 0.0f;
        float dot_hv = 0.0f;

        for (uint32_t i = 0; i < D_; ++i) {
            float h = h_in[i];
            float v = v_tangent[i];
            sq_h += h * h;
            dot_hv += h * v;
        }

        float norm_h = std::sqrt(sq_h + 1e-12f);
        float inv_norm_h = 1.0f / norm_h;

        // Descomposición tangencial estricta: v_perp = v - (h·v / ||h||^2) * h
        float coeff_hv = dot_hv * inv_norm_h * inv_norm_h;
        float sq_v_perp = 0.0f;
        
        for (uint32_t i = 0; i < D_; ++i) {
            float vp = v_tangent[i] - coeff_hv * h_in[i];
            sq_v_perp += vp * vp;
        }

        float norm_v_perp = std::sqrt(sq_v_perp + 1e-12f);
        float inv_norm_vp = 1.0f / norm_v_perp;

        // Parámetros geodésicos en S^{D-1}
        float omega = norm_v_perp * inv_norm_h;
        float theta = omega * tau;
        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);

        // Curvatura de flujo local
        telemetry.omega_angular_velocity = omega;
        telemetry.curvature_kappa = (norm_v_perp > 1e-6f) ? (omega / norm_h) : 0.0f;

        // Evaluación de las 4 líneas derivativas primitivas contra v_hat:
        // grad_pole = <v_hat, U_pole>
        float g_onto = 0.0f, g_teleo = 0.0f, g_anti = 0.0f, g_eos = 0.0f;
        float dot_teleo_proj = 0.0f;

        for (uint32_t i = 0; i < D_; ++i) {
            float h_u = h_in[i] * inv_norm_h;
            float vp = v_tangent[i] - coeff_hv * h_in[i];
            float v_hat = (norm_v_perp > 1e-8f) ? (vp * inv_norm_vp) : 0.0f;

            // Extrapolación geodésica analítica (Cero tiros parabólicos)
            float star_val = cos_t * h_u + sin_t * v_hat;
            h_star_out[i] = star_val;

            // Proyecciones derivativas
            if (poles.u_onto)  g_onto  += v_hat * poles.u_onto[i];
            if (poles.u_teleo) g_teleo += v_hat * poles.u_teleo[i];
            if (poles.u_anti)  g_anti  += v_hat * poles.u_anti[i];
            if (poles.u_eos)   g_eos   += v_hat * poles.u_eos[i];

            if (poles.u_teleo) dot_teleo_proj += star_val * poles.u_teleo[i];
        }

        // Blindaje a dominio no negativo R+
        telemetry.grad_onto  = std::max(0.0f, g_onto);
        telemetry.grad_teleo = std::max(0.0f, g_teleo);
        telemetry.grad_anti  = std::max(0.0f, g_anti);
        telemetry.grad_eos   = std::max(0.0f, g_eos);
        telemetry.teleology_alignment = std::max(0.0f, dot_teleo_proj);
    }

    uint32_t dimension() const { return D_; }

private:
    uint32_t D_;
};

} // namespace aether
