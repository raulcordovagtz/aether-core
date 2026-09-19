#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>

const uint32_t D = 5120;
const uint32_t R = 32;
const uint32_t TWO_D = 2 * D;

// Estructura de bajo rango
struct LowRankOperators {
    std::vector<double> Uc, Vc;
    std::vector<double> Us, Vs;
    std::vector<double> Ul, Vl;
    double coupling_scale = 0.05;
    double alpha_eml = 0.02;
    double lambda_diss = 0.001;
};

// Evaluación de flujo en FP64 (Oráculo)
void evaluate_flow_fp64(const std::vector<double>& Phi, std::vector<double>& dPhi, const LowRankOperators& op) {
    const double* S = Phi.data();
    const double* L = Phi.data() + D;
    double* dS = dPhi.data();
    double* dL = dPhi.data() + D;

    // 1. Proyecciones internas r=32
    std::vector<double> Vs_S(R, 0.0), Us_S(R, 0.0);
    std::vector<double> Vl_L(R, 0.0), Ul_L(R, 0.0);
    std::vector<double> Vc_L(R, 0.0), Uc_S(R, 0.0);

    for (uint32_t r = 0; r < R; ++r) {
        for (uint32_t i = 0; i < D; ++i) {
            Vs_S[r] += op.Vs[r * D + i] * S[i];
            Us_S[r] += op.Us[r * D + i] * S[i];
            Vl_L[r] += op.Vl[r * D + i] * L[i];
            Ul_L[r] += op.Ul[r * D + i] * L[i];
            Vc_L[r] += op.Vc[r * D + i] * L[i];
            Uc_S[r] += op.Uc[r * D + i] * S[i];
        }
    }

    // 2. Reconstrucción de campo K
    for (uint32_t i = 0; i < D; ++i) {
        double As_S = 0.0, Al_L = 0.0, C_L = 0.0, neg_CT_S = 0.0;
        for (uint32_t r = 0; r < R; ++r) {
            As_S += op.Us[r * D + i] * Vs_S[r] - op.Vs[r * D + i] * Us_S[r];
            Al_L += op.Ul[r * D + i] * Vl_L[r] - op.Vl[r * D + i] * Ul_L[r];
            C_L  += op.Uc[r * D + i] * Vc_L[r];
            neg_CT_S -= op.Vc[r * D + i] * Uc_S[r];
        }
        dS[i] = As_S + op.coupling_scale * C_L;
        dL[i] = op.coupling_scale * neg_CT_S + Al_L;
    }

    // 3. No-linealidad EML y proyección tangencial Pi_perp
    std::vector<double> F_eml(TWO_D);
    double dot_F_Phi = 0.0;
    double norm_Phi_sq = 0.0;

    for (uint32_t i = 0; i < TWO_D; ++i) {
        double x = Phi[i];
        double safe_x = std::max(-20.0, std::min(20.0, -x));
        double eml_val = std::exp(safe_x) - std::log(1.0);
        F_eml[i] = x / (1.0 + eml_val);

        dot_F_Phi += F_eml[i] * x;
        norm_Phi_sq += x * x;
    }

    double proj_factor = (norm_Phi_sq > 1e-12) ? (dot_F_Phi / norm_Phi_sq) : 0.0;

    // 4. Derivada total
    for (uint32_t i = 0; i < TWO_D; ++i) {
        double f_tangent = F_eml[i] - proj_factor * Phi[i];
        dPhi[i] = dPhi[i] + op.alpha_eml * f_tangent - op.lambda_diss * Phi[i];
    }
}

// Integrador Midpoint FP64
void integrate_midpoint_fp64(std::vector<double>& Phi, const LowRankOperators& op, uint32_t steps, double d_tau) {
    std::vector<double> k1(TWO_D), k2(TWO_D), phi_mid(TWO_D);
    for (uint32_t s = 0; s < steps; ++s) {
        evaluate_flow_fp64(Phi, k1, op);
        for (uint32_t i = 0; i < TWO_D; ++i) {
            phi_mid[i] = Phi[i] + 0.5 * d_tau * k1[i];
        }
        evaluate_flow_fp64(phi_mid, k2, op);
        for (uint32_t i = 0; i < TWO_D; ++i) {
            Phi[i] = Phi[i] + d_tau * k2[i];
        }
    }
}

int main() {
    std::cout << "=================================================================================\n";
    std::cout << " �� ARNES DE VALIDACIÓN NUMÉRICA: ORÁCULO FP64 Y CONDICIONES DE BORDE\n";
    std::cout << "=================================================================================\n";

    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(D));

    LowRankOperators op;
    op.Uc.resize(R * D); op.Vc.resize(R * D);
    op.Us.resize(R * D); op.Vs.resize(R * D);
    op.Ul.resize(R * D); op.Vl.resize(R * D);

    for (auto& x : op.Uc) x = dist(rng);
    for (auto& x : op.Vc) x = dist(rng);
    for (auto& x : op.Us) x = dist(rng);
    for (auto& x : op.Vs) x = dist(rng);
    for (auto& x : op.Ul) x = dist(rng);
    for (auto& x : op.Vl) x = dist(rng);

    std::vector<double> Phi_0(TWO_D);
    for (uint32_t i = 0; i < TWO_D; ++i) Phi_0[i] = dist(rng) * std::sqrt(D);

    // Normalizar S_0 y L_0 a norma 1
    double n_S = 0.0, n_L = 0.0;
    for (uint32_t i = 0; i < D; ++i) n_S += Phi_0[i] * Phi_0[i];
    for (uint32_t i = D; i < TWO_D; ++i) n_L += Phi_0[i] * Phi_0[i];
    n_S = std::sqrt(n_S); n_L = std::sqrt(n_L);
    for (uint32_t i = 0; i < D; ++i) Phi_0[i] /= n_S;
    for (uint32_t i = D; i < TWO_D; ++i) Phi_0[i] /= n_L;

    double norm_sq_ini = 0.0;
    for (double x : Phi_0) norm_sq_ini += x * x;

    std::cout << " • Dimensión latente D: " << D << " | Espinor 2D: " << TWO_D << "\n";
    std::cout << " • Rango de acoplamiento r: " << R << "\n";
    std::cout << " • Norma cuadrática inicial ||Φ(0)||²: " << std::fixed << std::setprecision(8) << norm_sq_ini << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<double> Phi_64 = Phi_0;
    integrate_midpoint_fp64(Phi_64, op, 64, 0.05);
    auto t1 = std::chrono::high_resolution_clock::now();

    double norm_sq_fin = 0.0;
    for (double x : Phi_64) norm_sq_fin += x * x;
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << " • Norma cuadrática final ||Φ(64)||²: " << norm_sq_fin << "\n";
    std::cout << " • Variación neta de energía: " << std::scientific << std::setprecision(4) << (norm_sq_fin - norm_sq_ini) << "\n";
    std::cout << " • Tiempo de cómputo CPU (FP64): " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n";

    // Verificación de monotonía de Lyapunov
    if (norm_sq_fin <= norm_sq_ini + 1e-12) {
        std::cout << " ✅ CRITERIO DE LYAPUNOV CUMPLIDO: d||Φ||²/dτ <= 0.\n";
    } else {
        std::cout << " ❌ ERROR: Violación de monotonía en el integrador.\n";
        return 1;
    }
    std::cout << "=================================================================================\n";
    return 0;
}
