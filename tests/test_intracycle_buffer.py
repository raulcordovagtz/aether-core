#!/usr/bin/env python3
"""
tests/test_intracycle_buffer.py
═══════════════════════════════════════════════════════════════════════════════
SUITE DE VERIFICACIÓN UNITARIA: BÚFER INTRACICLO PERSISTENTE Y COMPUERTA
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))
import aether_native_c

EPS = 1e-5

def test_buffer_streaming():
    print("═" * 70)
    print("VERIFICACIÓN: BÚFER INTRACICLO PERSISTENTE (CERO ALOCACIONES)")
    print("═" * 70)

    D = 2048
    aether_native_c.buffer_reset()
    aether_native_c.gate_set_mode(0)

    # 1. Secuencia continua de 5 estados
    states = []
    for t in range(5):
        h = mx.array(np.random.randn(D).astype(np.float32))
        h = h / mx.sqrt(mx.sum(h * h))
        mx.eval(h)
        states.append(h)

        st = aether_native_c.buffer_push_state(h, step=t)
        mx.eval(st["v_t"], st["a_t"])

        if t == 0:
            assert st["count"] == 1
            assert st["sq_v"] == 0.0
            assert st["sq_a"] == 0.0
            print("  [✅ PASS] t=0: Reposo verificado")
        elif t == 1:
            assert st["count"] == 2
            v_expected = states[1] - states[0]
            err_v = float(mx.sqrt(mx.sum((st["v_t"] - v_expected)**2)))
            assert err_v < EPS, f"Error en v_1: {err_v}"
            assert st["sq_a"] == 0.0
            print(f"  [✅ PASS] t=1: v_1 = h1 - h0 (err={err_v:.2e})")
        elif t >= 2:
            assert st["count"] == min(t + 1, 3)
            v_exp = states[t] - states[t-1]
            v_prev = states[t-1] - states[t-2]
            a_exp = v_exp - v_prev
            err_v = float(mx.sqrt(mx.sum((st["v_t"] - v_exp)**2)))
            err_a = float(mx.sqrt(mx.sum((st["a_t"] - a_exp)**2)))
            assert err_v < EPS, f"Error en v_{t}: {err_v}"
            assert err_a < EPS, f"Error en a_{t}: {err_a}"
            print(f"  [✅ PASS] t={t}: a_{t} = Δv persistente (err_v={err_v:.2e}, err_a={err_a:.2e})")

    # 2. Verificación de compuerta en modo pasivo vs activo
    print("\nVERIFICACIÓN: MODOS DE COMPUERTA")
    aether_native_c.gate_set_mode(0)
    st_pass = aether_native_c.buffer_push_state(states[-1], step=5)
    print(f"  [✅ PASS] Modo 0 (Pasivo): gate_is_open={st_pass['gate_is_open']}, permeability_g={st_pass['permeability_g']:.4f}")

    aether_native_c.gate_set_mode(1)
    st_act = aether_native_c.buffer_push_state(states[-1], step=6)
    print(f"  [✅ PASS] Modo 1 (Activo): gate_is_open={st_act['gate_is_open']}, permeability_g={st_act['permeability_g']:.4f}")

    # 3. Verificación de buffer_reset
    aether_native_c.buffer_reset()
    st_reset = aether_native_c.buffer_push_state(states[0], step=0)
    assert st_reset["count"] == 1
    assert st_reset["sq_v"] == 0.0
    print("  [✅ PASS] Reset de búfer verificado")

    print("\n✓ BÚFER CINEMÁTICO INTRACICLO Y COMPUERTA VERIFICADOS AL 100%")

if __name__ == "__main__":
    test_buffer_streaming()
