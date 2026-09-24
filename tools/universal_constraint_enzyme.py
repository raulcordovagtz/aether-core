#!/usr/bin/env python3
import math
import numpy as np

def eml(x: float, y: float) -> float:
    safe_y = max(float(y), 1e-12)
    safe_x = min(float(x), 50.0)
    return math.exp(safe_x) - math.log(safe_y)

def eml_ln(x: float) -> float:
    return eml(1.0, eml(eml(1.0, max(float(x), 1e-12)), 1.0))

def eml_exp(x: float) -> float:
    return eml(float(x), 1.0)

def eml_sub(x: float, y: float) -> float:
    return eml(eml_ln(x), math.exp(y))

def eml_add(x: float, y: float) -> float:
    return eml_ln(eml_exp(x) * eml_exp(y))

def eml_mul(x: float, y: float) -> float:
    return eml_exp(eml_add(eml_ln(x), eml_ln(y)))

class OpType:
    NONE = 0
    ADD  = 1
    MUL  = 2

class Channel:
    ACTIVE   = 0
    EQ       = 1
    GT       = 2
    LT       = 3
    SCALAR   = 4
    MULT     = 5
    OFFSET   = 6
    HAS_VAL  = 7
    NUM_CHANNELS = 8

class UniversalTensorEnzyme:
    def __init__(self, max_entities: int = 16, latent_dim: int = 2048):
        self.N = max_entities
        self.D = latent_dim
        self.K = Channel.NUM_CHANNELS
        self.S = np.zeros((self.N, self.N, self.K), dtype=np.float32)
        self.entity_names = {}
        self.name_to_idx = {}

    def reset(self):
        self.S.fill(0.0)
        self.entity_names.clear()
        self.name_to_idx.clear()

    def register_entity(self, name: str) -> int:
        if name in self.name_to_idx:
            return self.name_to_idx[name]
        idx = len(self.name_to_idx)
        if idx >= self.N:
            raise ValueError(f"Capacidad excedida: {idx} >= {self.N}")
        self.name_to_idx[name] = idx
        self.entity_names[idx] = name
        self.S[idx, idx, Channel.ACTIVE] = 1.0
        self.S[idx, idx, Channel.EQ] = 1.0
        return idx

    def set_greater_than(self, entity_a: str, entity_b: str):
        i = self.register_entity(entity_a)
        j = self.register_entity(entity_b)
        self.S[i, j, Channel.GT] = 1.0
        self.S[j, i, Channel.LT] = 1.0

    def set_scalar_value(self, entity: str, value: float):
        i = self.register_entity(entity)
        self.S[i, i, Channel.SCALAR] = float(value)
        self.S[i, i, Channel.HAS_VAL] = 1.0

    def set_linear_relation(self, entity_target: str, entity_source: str, mult: float = 1.0, offset: float = 0.0):
        i = self.register_entity(entity_target)
        j = self.register_entity(entity_source)
        self.S[i, j, Channel.MULT] = float(mult)
        self.S[i, j, Channel.OFFSET] = float(offset)

    def propagate_fixed_point(self, max_iters: int = 16, tol: float = 1e-6) -> int:
        total_steps = 0
        for iteration in range(max_iters):
            S_prev = self.S.copy()
            for k in range(self.N):
                if self.S[k, k, Channel.ACTIVE] < 0.5:
                    continue
                for i in range(self.N):
                    if self.S[i, i, Channel.ACTIVE] < 0.5:
                        continue
                    if self.S[i, k, Channel.GT] > 0.5:
                        for j in range(self.N):
                            if self.S[j, j, Channel.ACTIVE] < 0.5:
                                continue
                            if self.S[k, j, Channel.GT] > 0.5:
                                self.S[i, j, Channel.GT] = 1.0
                                self.S[j, i, Channel.LT] = 1.0

            for i in range(self.N):
                if self.S[i, i, Channel.ACTIVE] > 0.5 and self.S[i, i, Channel.HAS_VAL] < 0.5:
                    for j in range(self.N):
                        if i != j and self.S[j, j, Channel.HAS_VAL] > 0.5:
                            mult = self.S[i, j, Channel.MULT]
                            offset = self.S[i, j, Channel.OFFSET]
                            if abs(mult) > 1e-6 or abs(offset) > 1e-6:
                                val_j = self.S[j, j, Channel.SCALAR]
                                mult_res = eml_mul(mult, val_j) if abs(mult) > 1e-6 else 0.0
                                val_i = eml_add(mult_res, offset) if abs(offset) > 1e-6 else mult_res
                                self.S[i, i, Channel.SCALAR] = float(val_i)
                                self.S[i, i, Channel.HAS_VAL] = 1.0
                                break

            delta = np.max(np.abs(self.S - S_prev))
            total_steps += 1
            if delta < tol:
                return total_steps
        return total_steps

    def compute_residual_inoculation(self, initial_tensor: np.ndarray) -> dict:
        np.random.seed(1337)
        W_proj = np.random.randn(self.D, self.N).astype(np.float32) / np.sqrt(self.D)
        delta_matrix = self.S[:, :, Channel.SCALAR] - initial_tensor[:, :, Channel.SCALAR]
        delta_R = np.zeros(self.D, dtype=np.float32)
        for i in range(self.N):
            if self.S[i, i, Channel.HAS_VAL] > 0.5:
                delta_R += delta_matrix[i, i] * W_proj[:, i] * 0.05
        norm_delta = np.linalg.norm(delta_R)
        return {
            "delta_R": delta_R,
            "norm_delta": float(norm_delta),
            "is_effective": norm_delta > 1e-6
        }
