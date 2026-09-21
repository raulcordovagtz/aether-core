import mlx.core as mx

# ═════════════════════════════════════════════════════════════════════════════
# 1. KERNEL DE ROTACIÓN GEODÉSICA EN S^{D-1} (CERTIFICADO NORMA 1.000000)
# ═════════════════════════════════════════════════════════════════════════════
_ROTATION_SOURCE = """
    uint elem = thread_position_in_grid.x;
    if (elem >= size_arr[0]) return;
    float h_val = h_in[elem];
    float v_val = v_unit[elem];
    h_out[elem] = cos_t[0] * h_val + sin_t[0] * v_val;
"""

_rotation_kernel = mx.fast.metal_kernel(
    name="c018_geodesic_rotation",
    input_names=["h_in", "v_unit", "cos_t", "sin_t", "size_arr"],
    output_names=["h_out"],
    source=_ROTATION_SOURCE,
    compile_options={"math_mode": "safe"}
)

def _apply_geodesic_rotation(h_unit, v_vec, theta_rad):
    D = h_unit.shape[-1]
    # Proyección tangencial de Noether
    proj = mx.sum(v_vec * h_unit, axis=-1, keepdims=True)
    v_tangent = v_vec - proj * h_unit
    norm_v = mx.sqrt(mx.sum(v_tangent * v_tangent, axis=-1, keepdims=True) + 1e-12)
    v_unit = v_tangent / norm_v

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
    return outputs[0]

# ═════════════════════════════════════════════════════════════════════════════
# 2. INTEGRADOR PARALELO DEL BIESPINOR [S, L] (C-007 A C-018)
# ═════════════════════════════════════════════════════════════════════════════
def dispatch_spinor_parallel_step(
    S, L, u_vis, u_txt, u_exact,
    Uc, Vc, dt=0.05, kappa=0.01397, M_diss=0.25, nu_diff=0.05, beta_alu=2.5, gate_alu=0.0
):
    norm_s = mx.sqrt(mx.sum(S * S) + 1e-12)
    norm_l = mx.sqrt(mx.sum(L * L) + 1e-12)
    s_hat = S / norm_s
    l_hat = L / norm_l

    # Proyecciones de coherencia
    cos_v = mx.sum(s_hat * u_vis)
    cos_t = mx.sum(l_hat * u_txt)
    cos_vl = mx.sum(s_hat * l_hat)

    # 1. Operador simpléctico de Poisson J
    # C_VL = Uc @ (Vc^T @ L)  y  -C_VL^T = -Vc @ (Uc^T @ S)
    Vc_dot_L = mx.sum(Vc * l_hat)
    Uc_dot_S = mx.sum(Uc * s_hat)
    c_vl_s = Uc * Vc_dot_L
    neg_ct_l = - Vc * Uc_dot_S

    # 2. Gradientes Puerto-Hamiltonianos
    grad_s = - (1.0 - cos_v) * (u_vis - cos_v * s_hat) - 0.5 * (1.0 - cos_vl) * (l_hat - cos_vl * s_hat)
    grad_l = - (1.0 - cos_t) * (u_txt - cos_t * l_hat) - 0.5 * (1.0 - cos_vl) * (s_hat - cos_vl * l_hat)

    # 3. Difusión Laplaciana de Beltrami L_G >= 0
    lap_s = s_hat - cos_v * u_vis
    lap_l = l_hat - cos_t * u_txt

    # 4. Velocidades tangenciales
    v_s = kappa * c_vl_s - M_diss * grad_s - nu_diff * lap_s
    v_l = kappa * neg_ct_l - M_diss * grad_l - nu_diff * lap_l

    # Acoplamiento del Harness en el canal de lenguaje
    if gate_alu > 0.01:
        cos_sol = mx.sum(l_hat * u_exact)
        b_k = u_exact - cos_sol * l_hat
        v_l = v_l + gate_alu * beta_alu * b_k

    # 5. Ángulos geodésicos globales
    mag_v_s = mx.sqrt(mx.sum(v_s * v_s) + 1e-12)
    mag_v_l = mx.sqrt(mx.sum(v_l * v_l) + 1e-12)
    theta_s = float(dt * mag_v_s)
    theta_l = float(dt * mag_v_l)

    # 6. Retracción geodésica exacta en registros Metal
    S_next = _apply_geodesic_rotation(s_hat, v_s, theta_s) * norm_s
    L_next = _apply_geodesic_rotation(l_hat, v_l, theta_l) * norm_l

    return S_next, L_next

# ═════════════════════════════════════════════════════════════════════════════
# 3. RETRACCIÓN RIEMANNIANA UNIDIMENSIONAL (C-018)
# ═════════════════════════════════════════════════════════════════════════════
def dispatch_riemannian_step(h, u_target, theta_rad=0.35):
    norm_h = mx.sqrt(mx.sum(h * h, axis=-1, keepdims=True) + 1e-12)
    h_unit = h / norm_h
    h_steered = _apply_geodesic_rotation(h_unit, u_target, theta_rad)
    return h_steered * norm_h

# ═════════════════════════════════════════════════════════════════════════════
# 4. HARNESS ALU DETERMINISTA
# ═════════════════════════════════════════════════════════════════════════════
_ALU_SOURCE = """
    uint elem = thread_position_in_grid.x;
    if (elem >= size_arr[0]) return;
    uint a = args_a[elem];
    uint b = args_b[elem];
    uint op = op_code[0];
    uint res = 0;
    switch (op) {
        case 0: res = a & b; break;
        case 1: res = a | b; break;
        case 2: res = a ^ b; break;
        case 3: {
            while (b != 0) { uint t = b; b = a % b; a = t; }
            res = a; break;
        }
        case 4: res = a + b; break;
        case 5: res = a * b; break;
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

# ═════════════════════════════════════════════════════════════════════════════
# 5. KERNEL METAL C++: CONDENSACIÓN DE VAPOR DE FLUIDOS EN LOGITS (C-021)
# ═════════════════════════════════════════════════════════════════════════════
_VAPOR_CONDENSATION_SOURCE = """
    uint tid = thread_position_in_grid.x;
    if (tid >= vocab_size[0]) return;

    float z_val   = z_impact[tid];
    float z_star  = z_L_star[tid];
    float z_mean  = mean_val[0];
    float nu_val  = nu[0];
    float kappa_n = kappa[0];

    // 1. Amortiguamiento Viscoso Laminar: (1 - ν)·z + ν·mean(z)
    float z_visc = (1.0f - nu_val) * z_val + nu_val * z_mean;

    // 2. Condensación por Balance de Energía (Nucleación hacia L*)
    float z_cond = z_visc + kappa_n * z_star;

    z_out[tid] = z_cond;
"""

_condensation_kernel = mx.fast.metal_kernel(
    name="c021_vapor_condensation_step",
    input_names=["z_impact", "z_L_star", "mean_val", "nu", "kappa", "vocab_size"],
    output_names=["z_out"],
    source=_VAPOR_CONDENSATION_SOURCE,
    compile_options={"math_mode": "safe"}
)

def dispatch_vapor_condensation(z_impact, z_L_star, nu=0.12, kappa_n=0.15):
    """
    Ejecuta la ecuación de condensación de fluidos en GPU Metal pura.
    Cero bucles de Python, cero cálculo intermedio en CPU.
    """
    V = z_impact.shape[-1]
    mean_val = mx.mean(z_impact, axis=-1, keepdims=True)
    
    V_arr = mx.array([V], dtype=mx.uint32)
    nu_arr = mx.array([nu], dtype=mx.float32)
    kappa_arr = mx.array([kappa_n], dtype=mx.float32)

    outputs = _condensation_kernel(
        inputs=[z_impact, z_L_star, mean_val, nu_arr, kappa_arr, V_arr],
        grid=(V, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[z_impact.shape],
        output_dtypes=[mx.float32]
    )
    return outputs[0]
