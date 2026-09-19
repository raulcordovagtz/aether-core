#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace c_field {

struct IntentFieldPrimitives {
    float epistemic_rigidity;       // E_Q: Escalar de masa / rigidez cognitiva
    std::vector<float> u_ontology;   // u_O: Centroide de permanencia ontológica
    std::vector<float> u_teleology;  // u_T: Atractor de cuenca resolutiva
    std::vector<float> u_deontic;    // u_D: Vector de restricción de frontera
    uint32_t D;

    IntentFieldPrimitives(uint32_t dim) : D(dim), epistemic_rigidity(1.0f) {
        u_ontology.assign(dim, 0.0f);
        u_teleology.assign(dim, 0.0f);
        u_deontic.assign(dim, 0.0f);
    }

    // Extrae analíticamente las 4 primitivas a partir de la historia del prefill
    void extract_from_prefill(const std::vector<std::vector<float>>& prompt_states) {
        size_t N = prompt_states.size();
        if (N == 0) return;

        // 1. TELEOLOGÍA: Estado del último token proyectado a S^{D-1}
        const auto& last_state = prompt_states.back();
        double sq_t = 0.0;
        for (uint32_t i = 0; i < D; ++i) sq_t += double(last_state[i]) * double(last_state[i]);
        float inv_t = 1.0f / float(std::sqrt(sq_t) + 1e-6f);
        for (uint32_t i = 0; i < D; ++i) u_teleology[i] = last_state[i] * inv_t;

        // 2. ONTOLOGÍA: Centroide normalizado de todos los tokens de contexto
        std::vector<double> accum_o(D, 0.0);
        for (size_t t = 0; t < N; ++t) {
            for (uint32_t i = 0; i < D; ++i) {
                accum_o[i] += double(prompt_states[t][i]);
            }
        }
        double sq_o = 0.0;
        for (uint32_t i = 0; i < D; ++i) {
            double mean_val = accum_o[i] / double(N);
            sq_o += mean_val * mean_val;
        }
        float inv_o = 1.0f / float(std::sqrt(sq_o) + 1e-6f);
        for (uint32_t i = 0; i < D; ++i) u_ontology[i] = float(accum_o[i] / double(N)) * inv_o;

        // 3. DEÓNTICA: Componente ortogonal u_D = normalize(u_T - <u_T, u_O> * u_O)
        double dot_to = 0.0;
        for (uint32_t i = 0; i < D; ++i) dot_to += double(u_teleology[i]) * double(u_ontology[i]);
        double sq_d = 0.0;
        for (uint32_t i = 0; i < D; ++i) {
            float diff = u_teleology[i] - float(dot_to) * u_ontology[i];
            u_deontic[i] = diff;
            sq_d += double(diff) * double(diff);
        }
        float inv_d = 1.0f / float(std::sqrt(sq_d) + 1e-6f);
        for (uint32_t i = 0; i < D; ++i) u_deontic[i] *= inv_d;

        // 4. EPISTEMIA: Rigidez deducida de la coherencia angular global
        // Mayor alineación entre ontología y teleología => problema más técnico y convergente
        float alignment = std::abs(float(dot_to));
        epistemic_rigidity = 0.5f + 1.5f * alignment;

        std::cout << "\n ┌─────────────────────────────────────────────────────────────┐\n";
        std::cout << " │ 🧭 CAMPO DE INTENCIÓN DESCOMPUESTO EN 4 PRIMITIVAS:        │\n";
        std::cout << " ├─────────────────────────────────────────────────────────────┤\n";
        std::cout << " │ • Epistemia (Rigidez Cognitiva) : E_Q = " << std::fixed << std::setprecision(4) << epistemic_rigidity << "           │\n";
        std::cout << " │ • Ontología (Permanencia Símbolo): ||u_O|| = 1.0000        │\n";
        std::cout << " │ • Teleología (Atractor Cuenca)  : ||u_T|| = 1.0000        │\n";
        std::cout << " │ • Deóntica  (Frontera Ortogonal): ||u_D|| = 1.0000        │\n";
        std::cout << " │ • Proximidad Onto-Teleológica   : <u_T, u_O> = " << std::setprecision(4) << dot_to << "       │\n";
        std::cout << " └─────────────────────────────────────────────────────────────┘\n\n";
    }
};

} // namespace c_field
