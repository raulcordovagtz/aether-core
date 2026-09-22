// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: INTRACYCLE STATE BUFFER (HITO 1.2 — ZERO-ALLOCATION)
// Búfer Circular Markoviano en Memoria Unificada UMA
// ═════════════════════════════════════════════════════════════════════════════
#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace aether {

struct KinematicState {
    float norm_h;               // ||h_t||
    float sq_v;                 // ||v_t||^2
    float sq_a;                 // ||a_t||^2
    float dot_hv;               // <h_t, v_t> (Tangencia)
    float dot_va;               // <v_t, a_t>
    float dirichlet_tension_q;  // q_k = ||v_perp||^2 / ||h||^2
    uint32_t token_step;        // Índice del paso temporal macro
};

class IntracycleStateBuffer {
public:
    IntracycleStateBuffer(uint32_t dimension = 2048) 
        : D_(dimension), capacity_(3), count_(0), head_(0) {
        storage_.assign(capacity_ * D_, 0.0f);
        v_current_.assign(D_, 0.0f);
        a_current_.assign(D_, 0.0f);
    }

    void reset() {
        count_ = 0;
        head_ = 0;
        std::fill(storage_.begin(), storage_.end(), 0.0f);
        std::fill(v_current_.begin(), v_current_.end(), 0.0f);
        std::fill(a_current_.begin(), a_current_.end(), 0.0f);
    }

    // Ingestión de h_ptr directo desde MLX (CERO alocaciones dinámicas)
    KinematicState push_state_zero_copy(const float* h_ptr, uint32_t step) {
        head_ = (head_ + 1) % capacity_;
        float* dest = storage_.data() + (head_ * D_);
        std::memcpy(dest, h_ptr, D_ * sizeof(float));

        if (count_ < capacity_) {
            count_++;
        }

        const float* h_t   = get_slot(0);
        const float* h_tm1 = (count_ >= 2) ? get_slot(1) : nullptr;
        const float* h_tm2 = (count_ >= 3) ? get_slot(2) : nullptr;

        KinematicState k{};
        k.token_step = step;

        float sq_h = 0.0f, sq_v = 0.0f, sq_a = 0.0f;
        float dot_hv = 0.0f, dot_va = 0.0f;

        // 1. Velocidad discreta: v_t = h_t - h_{t-1}
        if (h_tm1) {
            for (uint32_t i = 0; i < D_; ++i) {
                float v = h_t[i] - h_tm1[i];
                v_current_[i] = v;
                sq_v += v * v;
                sq_h += h_t[i] * h_t[i];
                dot_hv += h_t[i] * v;
            }
        } else {
            for (uint32_t i = 0; i < D_; ++i) {
                sq_h += h_t[i] * h_t[i];
                v_current_[i] = 0.0f;
            }
        }

        // 2. Aceleración discreta: a_t = v_t - v_{t-1} = h_t - 2h_{t-1} + h_{t-2}
        if (h_tm2 && h_tm1) {
            for (uint32_t i = 0; i < D_; ++i) {
                float v_prev = h_tm1[i] - h_tm2[i];
                float a = v_current_[i] - v_prev;
                a_current_[i] = a;
                sq_a += a * a;
                dot_va += v_current_[i] * a;
            }
        } else {
            std::fill(a_current_.begin(), a_current_.end(), 0.0f);
        }

        k.norm_h = std::sqrt(sq_h + 1e-12f);
        k.sq_v   = sq_v;
        k.sq_a   = sq_a;
        k.dot_hv = dot_hv;
        k.dot_va = dot_va;

        float sq_v_perp = std::max(0.0f, sq_v - (dot_hv * dot_hv / (sq_h + 1e-12f)));
        k.dirichlet_tension_q = sq_v_perp / (sq_h + 1e-12f);

        return k;
    }

    const float* current_h() const { return get_slot(0); }
    const float* current_v() const { return v_current_.data(); }
    const float* current_a() const { return a_current_.data(); }

    uint32_t dimension() const { return D_; }
    uint32_t count() const { return count_; }

private:
    uint32_t D_;
    uint32_t capacity_;
    uint32_t count_;
    uint32_t head_;
    std::vector<float> storage_;
    std::vector<float> v_current_;
    std::vector<float> a_current_;

    const float* get_slot(uint32_t back_index) const {
        int32_t idx = static_cast<int32_t>(head_) - static_cast<int32_t>(back_index);
        while (idx < 0) idx += capacity_;
        return storage_.data() + (idx * D_);
    }
};

} // namespace aether
