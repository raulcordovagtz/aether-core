// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: HILBERT MEMORY CELL (HITO 2.1-A & 2.1-B)
// Órgano de Persistencia y Composición Geométrica sobre S^{D-1}
// SSOT: spec/C13_boolean_attention_algebra.yaml
// ═════════════════════════════════════════════════════════════════════════════
#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace aether {

enum class DegeneracyReason : uint32_t {
    NONE = 0,
    ZERO_SUPERPOSITION = 1,  // Antiparalelismo en empaquetado: sum(m_k) ≈ 0
    COLLINEAR_TARGET   = 2,  // Deflación colineal: m_pack || target (+A o -A)
    ZERO_TANGENT       = 3,  // Estilo colineal a la verdad: v_style ≈ 0
    INVALID_NORM       = 4   // Vector de entrada nulo o no normalizado
};

struct alignas(16) MemorySlotInfo {
    uint32_t slot_id;
    uint32_t timestamp;
    float semantic_energy;
    bool is_active;
    DegeneracyReason degeneracy_reason;
};

class HilbertMemoryCell {
public:
    HilbertMemoryCell(uint32_t dimension = 5120, uint32_t max_slots = 32)
        : D_(dimension), max_slots_(max_slots), count_(0), head_(0)
    {
        ring_storage_.assign(max_slots_ * D_, 0.0f);
        slots_.resize(max_slots_);
        for (uint32_t i = 0; i < max_slots_; ++i) {
            slots_[i] = {i, 0, 0.0f, false, DegeneracyReason::NONE};
        }
    }

    void reset() {
        count_ = 0;
        head_ = 0;
        std::fill(ring_storage_.begin(), ring_storage_.end(), 0.0f);
        for (auto& s : slots_) {
            s.is_active = false;
            s.degeneracy_reason = DegeneracyReason::NONE;
        }
    }

    // Ingestión en anillo: exige explícitamente que D_input == D_
    uint32_t ingest_residual_subproduct(const float* h_ortho_ptr, uint32_t dim_input, uint32_t timestamp, float energy) {
        if (dim_input != D_) {
            throw std::invalid_argument("Dimension mismatch: D_C1 != D_C2 en la interfaz de memoria");
        }
        uint32_t slot_idx = head_;
        head_ = (head_ + 1) % max_slots_;
        if (count_ < max_slots_) count_++;

        float* dest = ring_storage_.data() + (slot_idx * D_);
        std::memcpy(dest, h_ortho_ptr, D_ * sizeof(float));

        slots_[slot_idx].timestamp = timestamp;
        slots_[slot_idx].semantic_energy = energy;
        slots_[slot_idx].is_active = true;
        slots_[slot_idx].degeneracy_reason = DegeneracyReason::NONE;
        return slot_idx;
    }

    // ─── 1. EMPAQUETAMIENTO EN SUPERPOSICIÓN N-ARIA ──────────────────────────
    // m_pack = Normalize(sum(m_i))
    bool execute_superposition_packing(
        float* out_ptr,
        const float* const* facts,
        uint32_t num_facts,
        DegeneracyReason& reason
    ) const {
        reason = DegeneracyReason::NONE;
        if (num_facts == 0) {
            reason = DegeneracyReason::INVALID_NORM;
            return false;
        }
        float sq_sum = 0.0f;
        for (uint32_t i = 0; i < D_; ++i) {
            float sum_val = 0.0f;
            for (uint32_t k = 0; k < num_facts; ++k) {
                sum_val += facts[k][i];
            }
            out_ptr[i] = sum_val;
            sq_sum += sum_val * sum_val;
        }
        // Detección de degeneración antiparalela: A + B ≈ 0
        if (sq_sum < 1e-10f) {
            std::fill(out_ptr, out_ptr + D_, 0.0f);
            reason = DegeneracyReason::ZERO_SUPERPOSITION;
            return false;
        }
        float inv_norm = 1.0f / std::sqrt(sq_sum);
        for (uint32_t i = 0; i < D_; ++i) out_ptr[i] *= inv_norm;
        return true;
    }

    // ─── 2. DEFLACIÓN ORTOGONAL (MATCH & PEEL) ───────────────────────────────
    // m_peeled = Normalize(m_pack - <m_pack, m_target> m_target)
    // Garantiza ortogonalidad física: |<m_peeled, m_target>| < epsilon
    bool execute_orthogonal_deflation(
        float* out_ptr,
        const float* m_pack,
        const float* m_target,
        DegeneracyReason& reason
    ) const {
        reason = DegeneracyReason::NONE;
        float dot_val = 0.0f;
        for (uint32_t i = 0; i < D_; ++i) dot_val += m_pack[i] * m_target[i];

        float sq_rem = 0.0f;
        for (uint32_t i = 0; i < D_; ++i) {
            float val = m_pack[i] - dot_val * m_target[i];
            out_ptr[i] = val;
            sq_rem += val * val;
        }
        // Detección de degeneración colineal en ambas orientaciones: (+A o -A)
        if (sq_rem < 1e-10f) {
            std::fill(out_ptr, out_ptr + D_, 0.0f);
            reason = DegeneracyReason::COLLINEAR_TARGET;
            return false;
        }
        float inv_norm = 1.0f / std::sqrt(sq_rem);
        for (uint32_t i = 0; i < D_; ++i) out_ptr[i] *= inv_norm;
        return true;
    }

    // ─── 3. TRANSPORTE PARALELO DE ESTILO EN EL ESPACIO TANGENTE ─────────────
    // Rota hacia el estilo manteniendo la componente fáctica bajo cos(theta_s)
    bool execute_style_transport(
        float* out_ptr,
        const float* h_truth,
        const float* u_style,
        float theta_s,
        DegeneracyReason& reason
    ) const {
        reason = DegeneracyReason::NONE;
        float dot_ts = 0.0f;
        for (uint32_t i = 0; i < D_; ++i) dot_ts += h_truth[i] * u_style[i];

        float sq_v = 0.0f;
        for (uint32_t i = 0; i < D_; ++i) {
            float v = u_style[i] - dot_ts * h_truth[i];
            out_ptr[i] = v;
            sq_v += v * v;
        }
        // Detección de degeneración: el estilo ya es colineal a la verdad (+ o -)
        if (sq_v < 1e-10f) {
            std::memcpy(out_ptr, h_truth, D_ * sizeof(float));
            reason = DegeneracyReason::ZERO_TANGENT;
            return false;
        }
        float inv_norm_v = 1.0f / std::sqrt(sq_v);
        float cos_s = std::cos(theta_s);
        float sin_s = std::sin(theta_s);

        for (uint32_t i = 0; i < D_; ++i) {
            out_ptr[i] = cos_s * h_truth[i] + sin_s * (out_ptr[i] * inv_norm_v);
        }
        return true;
    }

    // Consulta de resonancia: r = <u_query, m_slot>
    float query_slot_resonance(uint32_t slot_idx, const float* u_query) const {
        if (slot_idx >= max_slots_ || !slots_[slot_idx].is_active) return -2.0f;
        const float* m_slot = ring_storage_.data() + (slot_idx * D_);
        float dot_q = 0.0f;
        for (uint32_t i = 0; i < D_; ++i) dot_q += u_query[i] * m_slot[i];
        return dot_q;
    }

    uint32_t dimension() const { return D_; }
    uint32_t count() const { return count_; }
    const float* get_slot_ptr(uint32_t slot_idx) const { return ring_storage_.data() + (slot_idx * D_); }

private:
    uint32_t D_;
    uint32_t max_slots_;
    uint32_t count_;
    uint32_t head_;
    std::vector<float> ring_storage_;
    std::vector<MemorySlotInfo> slots_;
};

} // namespace aether
