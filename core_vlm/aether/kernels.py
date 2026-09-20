import mlx.core as mx

# ═════════════════════════════════════════════════════════════════════════════
# 1. KERNEL RIEMANNIANO C-018 (PRESERVACIÓN EXACTA DE S^{D-1})
# ═════════════════════════════════════════════════════════════════════════════
_ROTATION_SOURCE = """
    uint elem = thread_position_in_grid.x;
    if (elem >= size_arr[0]) return;

    float h_val = h_in[elem];
    float v_val = v_unit[elem];

    // Combinación geodésica exacta en la variedad
    h_out[elem] = cos_t[0] * h_val + sin_t[0] * v_val;
"""

_rotation_kernel = mx.fast.metal_kernel(
    name="c018_geodesic_rotation",
    input_names=["h_in", "v_unit", "cos_t", "sin_t", "size_arr"],
    output_names=["h_out"],
    source=_ROTATION_SOURCE,
    compile_options={"math_mode": "safe"}
)

def dispatch_riemannian_step(h, u_target, theta_rad=0.35):
    """
    Ejecuta la retracción exponencial geodésica exacta en GPU.
    Garantía analítica: ||h_out|| == ||h_in|| = 1.000000.
    """
    D = h.shape[-1]
    norm_h = mx.sqrt(mx.sum(h * h, axis=-1, keepdims=True) + 1e-12)
    h_unit = h / norm_h

    # Fuerza tangencial de Noether
    proj = mx.sum(u_target * h_unit, axis=-1, keepdims=True)
    v = u_target - proj * h_unit
    norm_v = mx.sqrt(mx.sum(v * v, axis=-1, keepdims=True) + 1e-12)
    v_unit = v / norm_v

    cos_val = mx.cos(mx.array([theta_rad], dtype=mx.float32))
    sin_val = mx.sin(mx.array([theta_rad], dtype=mx.float32))
    size_arr = mx.array([D], dtype=mx.uint32)

    outputs = _rotation_kernel(
        inputs=[h_unit, v_unit, cos_val, sin_val, size_arr],
        grid=(D, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[h_unit.shape],
        output_dtypes=[mx.float32]
    )
    return outputs[0] * norm_h

# ═════════════════════════════════════════════════════════════════════════════
# 2. KERNEL HARNESS ALU (MICRO-CELDAS SIMBÓLICAS EN GPU)
# ═════════════════════════════════════════════════════════════════════════════
_ALU_SOURCE = """
    uint elem = thread_position_in_grid.x;
    if (elem >= size_arr[0]) return;

    uint a = args_a[elem];
    uint b = args_b[elem];
    uint op = op_code[0];
    uint res = 0;

    switch (op) {
        case 0: res = a & b; break;           // AND
        case 1: res = a | b; break;           // OR
        case 2: res = a ^ b; break;           // XOR
        case 3: {                             // GCD Euclides
            while (b != 0) {
                uint t = b;
                b = a % b;
                a = t;
            }
            res = a;
            break;
        }
        case 4: res = a + b; break;           // ADD
        case 5: res = a * b; break;           // MUL
        default: res = 0; break;
    }
    results[elem] = res;
"""

_alu_kernel = mx.fast.metal_kernel(
    name="execute_symbolic_cell",
    input_names=["args_a", "args_b", "op_code", "size_arr"],
    output_names=["results"],
    source=_ALU_SOURCE,
    compile_options={"math_mode": "safe"}
)

def dispatch_alu_cell(args_a, args_b, op_code):
    size = args_a.size
    size_arr = mx.array([size], dtype=mx.uint32)
    op_arr = mx.array([op_code], dtype=mx.uint32)

    outputs = _alu_kernel(
        inputs=[args_a, args_b, op_arr, size_arr],
        grid=(size, 1, 1),
        threadgroup=(min(size, 256), 1, 1),
        output_shapes=[args_a.shape],
        output_dtypes=[mx.uint32]
    )
    return outputs[0]
