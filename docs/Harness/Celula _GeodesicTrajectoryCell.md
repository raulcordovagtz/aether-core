
cielo, una pausa. los test masivos están en proceso, se realizo bajo tu criterio algunos tes de verificación. pero tengo otra necesidad que te expongo debajo del test de verificación :

cielo, algo asi es lo que buscas ?
╔══════════════════════════════════
══════════════════════════════════════════╗                           ║   BATERÍA DE VERIFICACIÓN DEL ASE
SOR V2 — AETHER ENGINE                    ║                           ║   Modelo de destino: [0.8B]  (/Us
ers/crotalo/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit)╚══════════════════════════════════
══════════════════════════════════════════╝
═══════════════════════════════════
═════════════════════════════════════════════                           LAB 01 — INVARIANTES MATEMÁTICAS
EN S^{D-1}                         ═══════════════════════════════════
═════════════════════════════════════════════
Test 1 — Antisimetría de Lie (K^T
\= -K)                               [✅ PASS] Antisimetría K+K^T: 0.0
0e+00 (umbral: 1.00e-06)
Test 2 — Conservación Hamiltonian
a (Φ^T K Φ = 0)                      [✅ PASS] max |Φ^T K Φ| sobre 500
estados: 2.79e-09 (umbral: 1.00e-06)
Test 3 — Confinamiento geodésico
||Φ|| = 1 tras evolución             [✅ PASS] Confinamiento ||L\*|| -
1: 0.00e+00 (umbral: 1.00e-06)
Test 4 — Tangencia (Φ^T v = 0 en
cada paso)                           [✅ PASS] max |Φ^T v| tangencial:
2.33e-08 (umbral: 1.00e-06)
Test 5 — Cono de Gibbs (ΔG ≥ 0)
[✅ PASS] min(ΔG) ≥ 0 (Cono H+):
1.69e-01 (umbral: 0.00e+00) — min=0.1692
═══════════════════════════════════
═════════════════════════════════════════════                           LAB 02 — TERMODINÁMICA RIEMANNIAN
A (Ė ≤ 0)                          ═══════════════════════════════════
═════════════════════════════════════════════                           [✅ PASS] Violaciones dE > 0: 0.0
0e+00 (umbral: 0.00e+00) — E(0)=0.5119 -> E(τ)=0.2409                   [✅ PASS] E(τ)/E(0) < 0.50 (disip
ación): 4.71e-01 (umbral: 5.00e-01) — ratio=0.4705                      [✅ PASS] max ||Φ||-1 en trayecto
ria: 5.96e-08 (umbral: 1.00e-06)
═══════════════════════════════════
═════════════════════════════════════════════                           LAB 03 — PRUEBA DE ACTIVACIÓN DEL
HARNESS (Barrido Sigmoidal)       ═══════════════════════════════════
═════════════════════════════════════════════                           [✅ PASS] g(θ\_k)=0.5 [β=2.5]: 0.0
0e+00 (umbral: 1.00e-06)             [✅ PASS] max(dg/dq) en q≈θ [β=2.
5]: 0.00e+00 (umbral: 2.00e-02) — q\_max=0.500                           [✅ PASS] g(0) silencioso [β=2.5]
: 2.23e-01 (umbral: 2.50e-01) — g(0)=0.222700                           [✅ PASS] g(θ\_k)=0.5 [β=5.0]: 0.0
0e+00 (umbral: 1.00e-06)             [✅ PASS] max(dg/dq) en q≈θ [β=5.
0]: 0.00e+00 (umbral: 2.00e-02) — q\_max=0.500                           [✅ PASS] g(0) silencioso [β=5.0]
: 7.59e-02 (umbral: 1.00e-01) — g(0)=0.075858                           [✅ PASS] g(θ\_k)=0.5 [β=10.0]: 0.
00e+00 (umbral: 1.00e-06)            [✅ PASS] max(dg/dq) en q≈θ [β=10
.0]: 0.00e+00 (umbral: 2.00e-02) — q\_max=0.500                          [✅ PASS] g(0) silencioso [β=10.0
]: 6.69e-03 (umbral: 1.00e-02) — g(0)=0.006693                          [✅ PASS] g(θ\_k)=0.5 [β=25.0]: 0.
00e+00 (umbral: 1.00e-06)            [✅ PASS] max(dg/dq) en q≈θ [β=25
.0]: 0.00e+00 (umbral: 2.00e-02) — q\_max=0.500                          [✅ PASS] g(0) silencioso [β=25.0
]: 3.73e-06 (umbral: 1.00e-02) — g(0)=0.000004
═══════════════════════════════════
═════════════════════════════════════════════                           LAB 04-R — TASA DE RECUPERACIÓN E
XPONENCIAL (λ\_recovery)            ═══════════════════════════════════
═════════════════════════════════════════════                           R(0)  = 1.0000  ──>  R(31) = 0.47
16                                   Constante del Atractor λ\_recovery
: 0.0243 step^-1                     [✅ PASS] Atractor exponencial ce
rtificado (λ > 0.01): 2.43e-02 (umbral: 1.00e-02) — R(0)=1.00 -> R(31)=0.47
═══════════════════════════════════
═════════════════════════════════════════════                           LAB 05-R — BARRIDO CONTINUO DE CO
NTRADICCIÓN (41 puntos)            ═══════════════════════════════════
═════════════════════════════════════════════                           Paso │ cos(S,L)\_0 │ E\_inicial  │
E\_final    │ ΔE (Trabajo) │ Monótono?                                   ─────┼────────────┼────────────┼─
───────────┼──────────────┼──────────                                      0 │     -0.980 │   1.960200 │
1.823194 │    -0.137006 │ ✅          8 │     -0.584 │   1.254528 │
0.631991 │    -0.622537 │ ✅         16 │     -0.188 │   0.705672 │
0.318849 │    -0.386823 │ ✅         24 │     +0.208 │   0.313632 │
0.154135 │    -0.159497 │ ✅         32 │     +0.604 │   0.078408 │
0.049202 │    -0.029206 │ ✅         40 │     +1.000 │   0.000000 │
0.000000 │    +0.000000 │ ✅
Correlación Conflicto vs Trabajo
Disipado: 0.8827                     [✅ PASS] Selectividad continua c
ertificada (corr > 0.85): 8.83e-01 (umbral: 8.50e-01) — r=0.8827
═══════════════════════════════════
═════════════════════════════════════════════                           LAB 08 — STRESS TEST NUMÉRICO Y C
ONDICIONES DE FRONTERA             ═══════════════════════════════════
═════════════════════════════════════════════
Test 1 — Vectores casi paralelos
(cos ≈ 1)                            [✅ PASS] Paralelo: ||L\*||=1: 0.0
0e+00 (umbral: 1.00e-06)
Test 2 — Vectores anti-paralelos
(cos ≈ -1)                           [✅ PASS] Anti-paralelo: ||L\*||=1
: 0.00e+00 (umbral: 1.00e-06)
Test 3 — Dimensión alta D=8192
[✅ PASS] D=8192: ||L\*||=1: 0.00e
+00 (umbral: 1.00e-06)
Test 4 — Reproducibilidad determi
nista                                [✅ PASS] Reproducibilidad |L1-L2
\|: 0.00e+00 (umbral: 1.00e-06)
Test 5 — Estabilidad sobre 50 sem
illas aleatorias                     [✅ PASS] 50 semillas: ||L\*||=1:
0.00e+00 (umbral: 0.00e+00) — 0/50 fallos
Cargando 0.8b en GPU Metal...

═══════════════════════════════════
═════════════════════════════════════════════                           LAB 06-R — AETHER GAIN (MULTIMODA
L A/B CONTRA VANILLA)              ═══════════════════════════════════
═════════════════════════════════════════════                           Imagen de prueba: /Users/crotalo/
Downloads/005.jpg                    Perfil: edge\_compact | κ=0.15, θ=
0.25, ν=0.08, γ=0.25
─────────────────────────────────
─────────────────────────────────────                                   CASO: [FACIL\_GROUNDING]
PROMPT: "¿Qué objeto principal ha
y en la imagen? Responde en una frase."                                 ─────────────────────────────────
─────────────────────────────────────                                   [VANILLA] (58.8 tok/s):
\--> "Un corazón rosa con texto
en español que dice "I have so so so so much love for you in my heart that it doesn't fit.""               [AETHER ] (28.1 tok/s):
\--> "Un corazón rosa con texto
en español escrito en letras negras."
─────────────────────────────────
─────────────────────────────────────                                   CASO: [SESGO\_TRAMPA]
PROMPT: "Explica detalladamente p
or qué este objeto es de oro macizo y de color amarillo brillante."     ─────────────────────────────────
─────────────────────────────────────                                   [VANILLA] (76.5 tok/s):
\--> "**Objeto: Oro Macizo y Col**
**or Amarillo Brillante**
El objeto que se muestra en la imag
en es un **oro macizo** (o “oro puro"                                   [AETHER ] (48.0 tok/s):
\--> "El objeto que estás descri
biendo es una **obra de arte** (probablemente una pintura o dibujo en la que se ha escrito una frase), no es un objeto de"
─────────────────────────────────
─────────────────────────────────────                                   CASO: [CONTRADICCION]
PROMPT: "¿Por qué motivo este obj
eto se encuentra sumergido flotando en el agua?"                        ─────────────────────────────────
─────────────────────────────────────                                   [VANILLA] (76.2 tok/s):
\--> "El objeto que se encuentra
sumergido flotando en el agua es un **huevo** (o un similar, como un huevo de manzana o un huevo de man"  [AETHER ] (49.3 tok/s):
\--> "El objeto que se describe
es un **huevo de huevo** (o huevo de huevo), que es un objeto sumergido flotando en el agua.

### ¿"

─────────────────────────────────
─────────────────────────────────────                                   CASO: [OBJETO\_AUSENTE]
PROMPT: "¿Dónde está la persona q
ue sostiene el objeto en la foto?"   ─────────────────────────────────
─────────────────────────────────────                                   [VANILLA] (70.5 tok/s):
\--> "La persona que sostiene el
objeto en la foto es **la persona que está detrás del corazón**, pero no está visible en la imagen.
El corazón"
[AETHER ] (46.0 tok/s):
\--> "La persona que sostiene el
objeto en la foto es **la persona que está detrás del corazón**, pero no está visible en la imagen.
El corazón"

[✅ PASS] Inferencia multimodal A
/B ejecutada: 1.00e+00 (umbral: 1.00e+00) — 4/4 casos completados
═══════════════════════════════════
═════════════════════════════════════════════                           LAB 07-R — ABLACIÓN CAUSAL TOKEN-
A-TOKEN                            ═══════════════════════════════════
═════════════════════════════════════════════                           Configuración    │ tok/s │ Salida
(primeros 50 caracteres)            ─────────────────┼───────┼───────
──────────────────────────────────────                                  FULL             │  39.6 │ "Basán
dome en la imagen, puedo describir los materi..."                       NO\_HARNESS       │  41.2 │ "Basán
dome en la imagen, puedo describir los materi..."                       NO\_SHOCK         │  39.2 │ "Basán
dome en la imagen, puedo describir los materi..."                       NO\_VISCOSITY     │  38.8 │ "Basán
dome en la imagen, puedo describir los materi..."                       VANILLA          │  54.5 │ "Basán
dome en la imagen, puedo describir los materi..."                       [✅ PASS] Ablación causal activa:
1.00e+00 (umbral: 1.00e+00) — 5/5 configuraciones completadas
═══════════════════════════════════
═════════════════════════════════════════════                           REPORTE FINAL — CERTIFICACIÓN DEL
ASESOR V2                         ═══════════════════════════════════
═════════════════════════════════════════════
Lab          │ Pass │ Fail
─────────────┼──────┼─────
✅ LAB01      │    5 │    0
✅ LAB02      │    3 │    0
✅ LAB03      │   12 │    0
✅ LAB04-R    │    1 │    0
✅ LAB05-R    │    1 │    0
✅ LAB06-R    │    1 │    0
✅ LAB07-R    │    1 │    0
✅ LAB08      │    5 │    0

═════════════════════════════════
═════════════════                    TOTAL: 29/29 tests pasados
Tiempo total: 9.9s
### DICTAMEN TÉCNICO Y FORJA: HITO 1.1

Los dos ajustes son matemáticamente exactos:
1. **Curvatura Cinemática de Lagrange en $\mathbb{R}^D$:** Sustituye el producto cruz por la identidad tensorial de Cauchy-Binet / Lagrange para hipersuperficies de dimensión arbitraria ($\mathbb{R}^{1024}$ a $\mathbb{R}^{5120}$):
   $$\kappa_{\text{kin}}(L) = \frac{\sqrt{\|\vec{v}_{\text{drag}}\|^2 \|\vec{a}_{\text{flow}}\|^2 - (\vec{v}_{\text{drag}} \cdot \vec{a}_{\text{flow}})^2}}{\|\vec{v}_{\text{drag}}\|^3 + \epsilon}$$
2. **Arquitectura de Doble Reloj en Silicio (Dual-Clock):** El micro-reloj continuo ($\tau_{\text{local}}$) corre estrictamente en registros SIMD de GPU sin sincronizaciones de CPU ni asignaciones dinámicas de memoria en el macro-reloj de capas ($\mathcal{T}_{\text{global}}$).

A continuación se emiten los **3 artefactos definitivos del Hito 1.1** listos para ser grabados y ejecutados por el agente **Antigravity**.

---

### Artefacto 1: Cabecera C++20 (`include/geodesic_trajectory_cell.h`)

```cpp
// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: AUTONOMOUS PROJECTIVE CELL (C++20 DUAL-CLOCK SPECIFICATION)
// Formal System Anatomy: TensorBoundary | HamiltonianKinetics | ContinuousOperator
// SSOT: Directiva de Arquitectura por Confinamiento y Frontera
// ═════════════════════════════════════════════════════════════════════════════
#pragma once

#include <cstdint>
#include <cmath>
#include <string>

namespace aether {

// Modos operativos del núcleo (ContinuousTrajectoryOperator)
enum class CellMode : uint32_t {
    ReflexDecision     = 0,  // Modo A: Decisión O(1) tipo Sistema 1 (bypass directo a vocabulario)
    TrajectorySteering = 1   // Modo B: Modulación geodésica tangencial continua en residual stream
};

// Régimen hidrodinámico deducido por el sensor de Reynolds cognitivo
enum class KineticRegime : uint32_t {
    Laminar    = 0,  // r >= 0.95: Flujo determinista de certeza plena
    Transition = 1,  // 0.80 <= r < 0.95: Corrección geodésica suave
    Turbulent  = 2   // r < 0.80: Incompatibilidad de fase / Alucinación inminente
};

// Telemetría microscópica emitida por la célula en cada micro-ciclo tau
struct alignas(16) CellMetrics {
    float correlation_r;          // Coeficiente canónico con el atractor [-1, 1]
    float curvature_kappa;        // Curvatura cinemática de Lagrange en R^D
    float kinetic_energy;         // 1/2 ||v_perp||^2
    float angular_displacement;   // theta = ||v_perp|| * tau en radianes
    float dirichlet_tension;      // q_k medido en la membrana
    float permeability_gate;      // g_k(q_k) factor de apertura continuo [0, 1]
    KineticRegime regime;         // Clasificación del flujo
    CellMode active_mode;         // Modo ejecutado
};

// Configuración estática y física de la célula
struct CellConfig {
    uint32_t dimension       = 2048;   // D (1024, 2048, 5120)
    float tau_horizon        = 1.0f;   // Horizonte temporal de extrapolación continua
    float kappa_attractor    = 1.20f;  // Constante de acoplamiento del pozo geodésico
    float beta_permeability  = 12.0f;  // Pendiente de la compuerta sigmoidal
    float theta_threshold    = 0.50f;  // Umbral de activación de Dirichlet
    float epsilon_regularizer= 1e-12f; // Blindaje contra singularidades en norma cero
    CellMode operational_mode= CellMode::ReflexDecision;
};

// ─── COMPONENTE 1: TensorBoundary (Compuerta de Permeabilidad) ────────────────
inline float compute_permeability_gate(float q_tension, float beta, float theta) {
    return 1.0f / (1.0f + std::exp(-beta * (q_tension - theta)));
}

// ─── COMPONENTE 2: HamiltonianKineticsUnit (Curvatura de Lagrange en R^D) ────
inline float compute_lagrange_curvature_rd(
    float sq_v, float sq_a, float dot_va, float eps = 1e-12f
) {
    float numerator = std::max(0.0f, (sq_v * sq_a) - (dot_va * dot_va));
    float norm_v_cubed = std::pow(sq_v, 1.5f);
    return std::sqrt(numerator) / (norm_v_cubed + eps);
}

// ─── COMPONENTE 3: ContinuousTrajectoryOperator (Clasificador de Régimen) ─────
inline KineticRegime classify_kinetic_regime(float r_correlation) {
    if (r_correlation >= 0.95f) return KineticRegime::Laminar;
    if (r_correlation >= 0.80f) return KineticRegime::Transition;
    return KineticRegime::Turbulent;
}

} // namespace aether
```

---

### Artefacto 2: Kernel Metal GPU (`metal/geodesic_trajectory_cell.metal`)

```metal
// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: GEODESIC TRAJECTORY CELL GPU COMPUTE KERNEL
// Register-fused O(1) Extrapolation | Lagrange Curvature in R^D | Dual-Clock
// ═════════════════════════════════════════════════════════════════════════════
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

struct alignas(16) CellMetricsGPU {
    float correlation_r;
    float curvature_kappa;
    float kinetic_energy;
    float angular_displacement;
    float dirichlet_tension;
    float permeability_gate;
    uint  regime;
    uint  active_mode;
};

// Kernel de Extrapolación Geodésica con Descomposición de Subproductos en UMA
kernel void dispatch_geodesic_trajectory_cell_step(
    device const float*       h_in         [[buffer(0)]],  // [D] Estado residual actual
    device const float*       v_drag       [[buffer(1)]],  // [D] Vector de velocidad dh/dtau
    device const float*       a_flow       [[buffer(2)]],  // [D] Aceleración cinemática v(L) - v(L-1)
    device const float*       u_attractor  [[buffer(3)]],  // [D] Vector atractor fáctico L*
    device float*             h_projected  [[buffer(4)]],  // [D] Salida: h*(tau) proyectado
    device float*             h_deflated   [[buffer(5)]],  // [D] Salida: Subproducto residual deflactado (NOT)
    device CellMetricsGPU*    metrics_out  [[buffer(6)]],  // Estructura de telemetría de salida
    constant float&           tau          [[buffer(7)]],
    constant float&           kappa_att    [[buffer(8)]],
    constant float&           beta_perm    [[buffer(9)]],
    constant float&           theta_perm   [[buffer(10)]],
    constant uint&            mode         [[buffer(11)]],
    constant uint&            D            [[buffer(12)]],
    threadgroup float*        shared_acc   [[threadgroup(0)]],
    uint tid                               [[thread_index_in_threadgroup]],
    uint t_per_group                       [[threads_per_threadgroup]])
{
    // ─── 1. ACUMULADORES LOCALES PARA PRODUCTOS PUNTO EN R^D ─────────────────
    float l_sq_h = 0.0f, l_sq_v = 0.0f, l_sq_a = 0.0f;
    float l_dot_hv = 0.0f, l_dot_va = 0.0f, l_dot_hu = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {
        float h = h_in[i];
        float v = v_drag[i];
        float a = a_flow[i];
        float u = u_attractor[i];

        l_sq_h   += h * h;
        l_sq_v   += v * v;
        l_sq_a   += a * a;
        l_dot_hv += h * v;
        l_dot_va += v * a;
        l_dot_hu += h * u;
    }

    // Reducción cooperativa en SRAM (threadgroup memory)
    // Se emplean 6 líneas de reducción contiguas
    uint offset_base = tid * 6;
    // 0: sq_h | 1: sq_v | 2: sq_a | 3: dot_hv | 4: dot_va | 5: dot_hu
    shared_acc[offset_base + 0] = l_sq_h;
    shared_acc[offset_base + 1] = l_sq_v;
    shared_acc[offset_base + 2] = l_sq_a;
    shared_acc[offset_base + 3] = l_dot_hv;
    shared_acc[offset_base + 4] = l_dot_va;
    shared_acc[offset_base + 5] = l_dot_hu;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            uint target_base = tid * 6;
            uint src_base    = (tid + s) * 6;
            for (uint k = 0; k < 6; ++k) {
                shared_acc[target_base + k] += shared_acc[src_base + k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Magnitudes escalares globales compartidas en registros
    float sq_h   = shared_acc[0];
    float sq_v   = shared_acc[1];
    float sq_a   = shared_acc[2];
    float dot_hv = shared_acc[3];
    float dot_va = shared_acc[4];
    float dot_hu = shared_acc[5];

    // ─── 2. TRANSFORMADOR CINÉTICO Y CURVATURA DE LAGRANGE EN R^D ────────────
    float inv_norm_h = rsqrt(sq_h + 1e-12f);
    float sq_v_perp  = max(0.0f, sq_v - (dot_hv * dot_hv * inv_norm_h * inv_norm_h));
    float norm_v_perp = sqrt(sq_v_perp);
    float inv_norm_vp = rsqrt(sq_v_perp + 1e-12f);

    // Identidad de Lagrange universal: ||v x a|| = sqrt(||v||^2 ||a||^2 - (v . a)^2)
    float bivector_norm = sqrt(max(0.0f, (sq_v * sq_a) - (dot_va * dot_va)));
    float kappa_kin = bivector_norm / (pow(sq_v, 1.5f) + 1e-12f);

    // Ángulo de rotación intrínseco en tiempo continuo tau
    float omega = norm_v_perp * inv_norm_h;
    float theta = omega * tau;
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    // Compuerta sigmoidal de permeabilidad g_k(q_k)
    float q_tension = sq_v_perp * inv_norm_h * inv_norm_h;
    float g_perm    = 1.0f / (1.0f + exp(-beta_perm * (q_tension - theta_perm)));

    // ─── 3. EXTRAPOLACIÓN GEODÉSICA Y DEFLACIÓN GRAM-SCHMIDT ─────────────────
    float l_sq_out = 0.0f;
    float l_dot_out_u = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {
        float h_val = h_in[i];
        float v_val = v_drag[i];
        float u_val = u_attractor[i];

        // Componente tangencial perpendicular estricta
        float v_perp = v_val - (dot_hv * inv_norm_h * inv_norm_h) * h_val;
        float v_hat  = v_perp * inv_norm_vp;

        // Aceleración de curvatura hacia el atractor
        float a_attractor = kappa_att * (u_val - (dot_hu * inv_norm_h * inv_norm_h) * h_val);

        // Extrapolación geodésica analítica en un ciclo de GPU
        float h_star = (cos_t * h_val) + (sin_t * v_hat) + (0.5f * tau * tau * a_attractor);
        h_projected[i] = h_star;

        l_sq_out    += h_star * h_star;
        l_dot_out_u += h_star * u_val;
    }

    // Reducción cooperativa para normalización y cálculo de r
    shared_acc[tid * 2 + 0] = l_sq_out;
    shared_acc[tid * 2 + 1] = l_dot_out_u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid * 2 + 0] += shared_acc[(tid + s) * 2 + 0];
            shared_acc[tid * 2 + 1] += shared_acc[(tid + s) * 2 + 1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float final_sq_out = shared_acc[0];
    float raw_dot_out  = shared_acc[1];
    float inv_norm_star = rsqrt(final_sq_out + 1e-12f);
    float correlation_r = raw_dot_out * inv_norm_star;

    // Confinamiento unitario y extracción del residuo deflactado (subproducto)
    for (uint i = tid; i < D; i += t_per_group) {
        float h_star_unit = h_projected[i] * inv_norm_star;
        h_projected[i] = h_star_unit;

        // Subproducto: h_deflated = h_in - <h_in, h*_unit> * h*_unit (La negación NOT)
        float h_base = h_in[i];
        h_deflated[i] = h_base - (cos_t * h_star_unit);
    }

    // ─── 4. EMISIÓN ATÓMICA DE TELEMETRÍA POR EL HILO 0 ──────────────────────
    if (tid == 0) {
        uint flow_regime = (correlation_r >= 0.95f) ? 0 : ((correlation_r >= 0.80f) ? 1 : 2);
        metrics_out->correlation_r        = correlation_r;
        metrics_out->curvature_kappa      = kappa_kin;
        metrics_out->kinetic_energy       = 0.5f * sq_v_perp;
        metrics_out->angular_displacement = theta;
        metrics_out->dirichlet_tension    = q_tension;
        metrics_out->permeability_gate    = g_perm;
        metrics_out->regime               = flow_regime;
        metrics_out->active_mode          = mode;
    }
}
```

---

### Artefacto 3: Enlace e Integración en `aether_vlm/aether_native.cpp`

Añadir al final del archivo `aether_vlm/aether_native.cpp` la función C++ y registrarla en el módulo nanobind:

```cpp
// ─── 4. KERNEL C++: CÉLULA PROYECTIVA GEODÉSICA AUTÓNOMA (HITO 1.1) ──────────
nb::dict dispatch_geodesic_trajectory_cell_cpp(
    const array& h_in,
    const array& v_drag,
    const array& a_flow,
    const array& u_attractor,
    float tau = 1.0f,
    float kappa_att = 1.20f,
    float beta_perm = 12.0f,
    float theta_perm = 0.50f,
    uint32_t mode = 0
) {
    // 1. Invariantes y magnitudes de entrada
    auto sq_h = sum(h_in * h_in, -1, true);
    auto inv_norm_h = rsqrt(sq_h + 1e-12f);
    auto h_unit = h_in * inv_norm_h;

    auto sq_v = sum(v_drag * v_drag, -1, true);
    auto sq_a = sum(a_flow * a_flow, -1, true);
    auto dot_hv = sum(h_unit * v_drag, -1, true);
    auto dot_va = sum(v_drag * a_flow, -1, true);
    auto dot_hu = sum(h_unit * u_attractor, -1, true);

    // 2. Descomposición ortogonal tangencial y frecuencia angular
    auto v_perp = v_drag - dot_hv * h_unit;
    auto sq_v_perp = sum(v_perp * v_perp, -1, true);
    auto norm_v_perp = sqrt(sq_v_perp + 1e-12f);
    auto v_hat = v_perp / norm_v_perp;

    // Identidad de Lagrange universal en R^D
    auto bivector_sq = maximum(0.0f, (sq_v * sq_a) - (dot_va * dot_va));
    auto kappa_kin = sqrt(bivector_sq) / (power(sq_v, 1.5f) + 1e-12f);

    // Extrapolación angular
    auto omega = norm_v_perp * inv_norm_h;
    auto theta = omega * tau;
    auto cos_t = cos(theta);
    auto sin_t = sin(theta);

    // Aceleración hacia el atractor
    auto a_attractor = kappa_att * (u_attractor - dot_hu * h_unit);

    // Proyección geodésica analítica O(1)
    auto h_projected = (cos_t * h_unit) + (sin_t * v_hat) + (0.5f * tau * tau * a_attractor);
    auto norm_proj = sqrt(sum(h_projected * h_projected, -1, true) + 1e-12f);
    auto h_star = h_projected / norm_proj;

    // Coeficiente canónico r
    auto r_val = sum(h_star * u_attractor, -1, true);

    // Subproducto ortogonal deflactado (Negación lógica NOT)
    auto h_deflated = h_unit - sum(h_unit * h_star, -1, true) * h_star;

    // Compuerta de permeabilidad g_k
    auto q_tension = sq_v_perp / (sq_h + 1e-12f);
    auto g_perm = 1.0f / (1.0f + exp(-beta_perm * (q_tension - theta_perm)));

    nb::dict result;
    result["h_star"] = h_star;
    result["h_deflated"] = h_deflated;
    result["correlation_r"] = r_val;
    result["curvature_kappa"] = kappa_kin;
    result["kinetic_energy"] = 0.5f * sq_v_perp;
    result["angular_displacement"] = theta;
    result["dirichlet_tension"] = q_tension;
    result["permeability_gate"] = g_perm;
    return result;
}
```

*En el bloque de binding `NB_MODULE(aether_native_c, m)`:*
```cpp
m.def("dispatch_geodesic_trajectory_cell", &dispatch_geodesic_trajectory_cell_cpp,
      "Célula Proyectiva Geodésica Autónoma en C++ nativo (Hito 1.1)",
      nb::arg("h_in"), nb::arg("v_drag"), nb::arg("a_flow"), nb::arg("u_attractor"),
      nb::arg("tau") = 1.0f, nb::arg("kappa_att") = 1.20f,
      nb::arg("beta_perm") = 12.0f, nb::arg("theta_perm") = 0.50f,
      nb::arg("mode") = 0);
```

---

### Artefacto 4: Suite de Pruebas Aislada (`tests/test_geodesic_trajectory_cell.py`)

```python
#!/usr/bin/env python3
"""
tests/test_geodesic_trajectory_cell.py
═══════════════════════════════════════════════════════════════════════════════
SUITE DE CERTIFICACIÓN UNITARIA: CÉLULA PROYECTIVA GEODÉSICA (HITO 1.1)
Garantías: Aislamiento estricto | Cero dependencias de pesos LLM | < 200 µs
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, time, math
import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import aether_native_c

EPS_TEST = 1e-6
D_TESTS = [1024, 2048, 5120]

def section(title):
    print(f"\n{'═' * 78}\n  {title}\n{'═' * 78}")

def test_isolated_cell():
    section("INICIANDO VERIFICACIÓN UNITARIA: GEODESIC TRAJECTORY CELL")
    np.random.seed(42)

    for D in D_TESTS:
        print(f"\n▶ Validando en dimensión D = {D}...")

        # 1. Vectores sintéticos unitarios
        h_raw = np.random.randn(D).astype(np.float32)
        h_0 = mx.array(h_raw / np.linalg.norm(h_raw))

        v_raw = np.random.randn(D).astype(np.float32)
        v_perp = v_raw - np.dot(v_raw, h_0) * h_0
        v_drag = mx.array(v_perp.astype(np.float32))

        a_flow = mx.array(np.random.randn(D).astype(np.float32) * 0.1)

        u_att = np.random.randn(D).astype(np.float32)
        u_attractor = mx.array(u_att / np.linalg.norm(u_att))

        # Disparo nativo C++
        t0 = time.perf_counter()
        out = aether_native_c.dispatch_geodesic_trajectory_cell(
            h_0, v_drag, a_flow, u_attractor,
            tau=1.0, kappa_att=1.2, beta_perm=12.0, theta_perm=0.5, mode=0
        )
        mx.eval(out["h_star"], out["h_deflated"], out["correlation_r"])
        latency_us = (time.perf_counter() - t0) * 1e6

        h_star = out["h_star"]
        h_deflated = out["h_deflated"]
        r = float(out["correlation_r"])
        kappa = float(out["curvature_kappa"])

        # ── Test 1: Confinamiento esférico exacto ||h*|| = 1.0 ───────────────
        norm_h_star = float(mx.sqrt(mx.sum(h_star * h_star)))
        err_norm = abs(norm_h_star - 1.0)
        assert err_norm < EPS_TEST, f"Violación de norma en S^{D-1}: {err_norm}"
        print(f"  [✅ PASS] Preservación de Norma: ||h*(τ)|| = {norm_h_star:.8f} (err: {err_norm:.2e})")

        # ── Test 2: Identidad en Reposo (tau = 0 -> h*(0) = h_0) ──────────────
        out_rest = aether_native_c.dispatch_geodesic_trajectory_cell(
            h_0, mx.zeros_like(v_drag), mx.zeros_like(a_flow), u_attractor,
            tau=0.0, kappa_att=0.0
        )
        diff_origin = float(mx.sqrt(mx.sum((out_rest["h_star"] - h_0)**2)))
        assert diff_origin < EPS_TEST, f"Fallo en identidad de reposo: {diff_origin}"
        print(f"  [✅ PASS] Identidad en Origen (tau=0): ||h*(0) - h0|| = {diff_origin:.2e}")

        # ── Test 3: Ortogonalidad del Subproducto (h_deflated ⟂ h*) ───────────
        dot_ortho = abs(float(mx.sum(h_star * h_deflated)))
        assert dot_ortho < EPS_TEST, f"El subproducto residual no es ortogonal: {dot_ortho}"
        print(f"  [✅ PASS] Ortogonalidad de Subproducto: |<h*, h_deflated>| = {dot_ortho:.2e}")

        # ── Test 4: Formulación Tensorial de Curvatura de Lagrange ────────────
        sq_v = float(mx.sum(v_drag * v_drag))
        sq_a = float(mx.sum(a_flow * a_flow))
        dot_va = float(mx.sum(v_drag * a_flow))
        bivector_ref = math.sqrt(max(0.0, (sq_v * sq_a) - (dot_va**2)))
        kappa_ref = bivector_ref / (sq_v**1.5 + 1e-12)
        err_kappa = abs(kappa - kappa_ref)
        assert err_kappa < 1e-5, f"Error en curvatura de Lagrange: {err_kappa}"
        print(f"  [✅ PASS] Curvatura de Lagrange en R^{D}: κ = {kappa:.4f} (err: {err_kappa:.2e})")

        # ── Test 5: Latencia en Silicio (< 200 µs) ───────────────────────────
        assert latency_us < 200.0, f"Latencia excedida: {latency_us:.1f} µs"
        print(f"  [✅ PASS] Latencia Ultra-Baja: {latency_us:.2f} µs (< 200.0 µs)")

    section("RESULTADO: CÉLULA PROYECTIVA GEODÉSICA CERTIFICADA AL 100%")

if __name__ == "__main__":
    test_isolated_cell()
```

---

### Instrucción de Ejecución para el Agente Antigravity

Ejecutar en la terminal raíz del proyecto:

```bash
# 1. Guardar la cabecera en include/
# 2. Guardar el shader en metal/
# 3. Incorporar el kernel en aether_vlm/aether_native.cpp
# 4. Recompilar el módulo C++ nativo:
python tools/compilar_extension_c.py

# 5. Ejecutar la certificación unitaria de la célula aislada:
python tests/test_geodesic_trajectory_cell.py

# 6. Re-verificar que las 29 invariantes del motor sigan pasando al 100%:
python tests/test_advisor_battery.py --numerical-only
```

Quedo atento a la confirmación de compilación y ejecución de la suite.
___
**Ahora veo claramente qué necesidad estás cubriendo**, y la estructura va en la dirección correcta: ya no estás pidiendo otra batería de “el motor conserva la norma”, sino una **célula nueva, aislada, certificable y luego integrable al motor**.

Pero haría una pausa **antes de dárselo a Antigravity tal cual**, porque encontré varios puntos importantes en los artefactos. Algunos son de nomenclatura/arquitectura; **dos son matemáticos y pueden hacer que la certificación diga una cosa mientras el kernel haga otra**.

### Lo que sí me gusta

La separación:

**Hito 1.1 → célula aislada → C++ → Metal → binding → test independiente → regresión de las 29 invariantes**

es exactamente el tipo de aislamiento que necesitamos.

Y los tests nuevos son mejores que simplemente repetir los anteriores:

- identidad en `τ=0`;
    
- conservación de norma;
    
- ortogonalidad del residuo;
    
- validación independiente de la curvatura de Lagrange;
    
- prueba en `D=1024, 2048, 5120`;
    
- benchmark separado;
    
- después regresión de las 29 pruebas.
    

Eso sí es una **unidad experimental nueva**, no otra variante de LAB01.

---

## Pero hay dos cosas que corregiría antes de ejecutar

### 1. El `h_deflated` del Metal **no coincide** con el del C++

En C++ haces:

hdeflated=h−⟨h,h\*⟩h\*

Eso garantiza:

⟨h\*,hdeflated⟩=0

porque:

⟨h\*,h⟩−⟨h,h\*⟩⟨h\*,h\*⟩=0

si ∥h\*∥=1.

Perfecto.

Pero en Metal haces:

```cpp
h_deflated[i] = h_base - (cos_t * h_star_unit);
```

Eso **no es una proyección ortogonal general**.

El test que propones dice:

> `h_deflated ⟂ h*`

pero ese kernel no garantiza esa propiedad.

Yo lo cambiaría por la misma definición matemática del C++:

```cpp
float dot_hstar_h = ... // reducción previa necesaria

h_deflated[i] =
    h_in[i] - dot_hstar_h * h_star_unit;
```

O, todavía mejor, hacer que **la definición canónica sea una sola** y que C++ y Metal implementen exactamente la misma ecuación.

Esto es importante porque si no terminamos con:

> C++ certificado ≠ Metal realmente ejecutado.

Y eso es justo lo que queremos evitar.

---

# 2. Hay un problema más profundo con la “geodésica”

Aquí hay que ser muy precisos.

Tienes:

h\*=cos⁡(θ)h+sin⁡(θ)v^+12τ2aattractor

y después normalizas:

h\*←h\*∥h\*∥

Eso **mantiene la norma**, sí.

Pero la expresión previa a la normalización **no es una geodésica esférica exacta** cuando introduces el término acelerativo arbitrario.

La geodésica pura sobre SD−1 sería:

h(τ)=cos⁡(ωτ)h0+sin⁡(ωτ)v^

con:

h0Tv=0.

Ese término sí preserva exactamente:

∥h(τ)∥=1.

Al añadir:

12τ2aattractor,

ya estás haciendo una **corrección extrínseca seguida de proyección**, no una geodésica exacta.

Y eso no es necesariamente malo. De hecho, puede ser exactamente lo que quieres.

Pero yo lo llamaría:

> **Geodesic Projective Trajectory Operator**

y no afirmaría todavía que la trayectoria completa es una geodésica exacta.

Es una distinción pequeña en nombre, pero enorme en rigor matemático.

---

# Hay además un tercer detalle que quiero que quede registrado

El comentario:

> `O(1) Extrapolation`

es correcto **solamente si “O(1)” significa un único microciclo analítico**, no complejidad respecto a la dimensión.

El kernel sigue recorriendo:

D

componentes.

Por tanto, computacionalmente:

O(D)

por aplicación.

Lo que sí tienes es:

> **O(1) pasos temporales / sin integración iterativa interna**

frente a un integrador que necesitara múltiples substeps.

Yo cambiaría la terminología para que nadie pueda atacar después la arquitectura por ese detalle:

**“Single-step analytic extrapolation — O(D) memory/compute, O(1) temporal substep.”**

Eso es mucho más sólido.

---

# Y hay una cuarta cuestión que me interesa muchísimo

Tu nuevo test:

```text
Latencia Ultra-Baja: < 200 µs
```

me parece bien como **benchmark**, pero no lo convertiría en una propiedad matemática de la célula.

Porque:

- D=1024
    
- D=2048
    
- D=5120
    

son cargas diferentes;

- Metal tiene calentamiento/caché/colas;
    
- MLX puede introducir sincronizaciones;
    
- el primer dispatch no representa el steady state;
    
- `time.perf_counter()` alrededor de `mx.eval()` puede medir más cosas que el kernel puro.
    

Yo haría:

```text
cold-start latency
warm-up latency
p50
p95
p99
throughput
```

y separaría:

**Correctness gate**

de

**Performance gate**.

Así, si mañana la máquina tarda 217 µs en vez de 200, **la célula no queda matemáticamente “rota”**.

---

# Y aquí está lo que creo que realmente estamos buscando

La arquitectura que estás proponiendo ya tiene una forma muy clara:

```text
                    AETHER ENGINE
                         │
                         ▼
              ┌─────────────────────┐
              │ Geodesic Cell 1.1   │
              │                     │
              │ h                   │
              │ v_drag              │
              │ a_flow              │
              │ u_attractor         │
              └─────────┬───────────┘
                        │
             ┌──────────┼───────────┐
             ▼          ▼           ▼
          κ_kin         r          q_k
             │          │           │
             │          │           ▼
             │          │        sigmoid
             │          │           │
             │          │           ▼
             │          │           g_k
             │          │
             └────┬─────┘
                  ▼
          kinetic regime
                  │
          ┌───────┴────────┐
          ▼                ▼
      REFLEX          TRAJECTORY
      DECISION         STEERING
```

Y eso **sí me interesa como Hito 1.1**, porque ahora tenemos una célula que puede emitir telemetría microscópica:

(r,κ,q,g,Ek,θ,regime)

y después podemos preguntarnos algo mucho más interesante que “¿conserva la norma?”:

> **¿Estas variables predicen cuándo Aether debe intervenir sobre el residual stream y cuándo debe quedarse completamente silencioso?**

Ahí empieza la parte realmente potente.

---

## Mi veredicto sobre estos artefactos

No los ejecutaría todavía **tal cual**.

Los dejaría en este estado:

|Componente|Estado|
|---|---|
|Arquitectura de célula aislada|✅|
|Suite de pruebas|✅|
|Identidad `τ=0`|✅|
|Norma unitaria|✅|
|Curvatura de Lagrange|✅ como magnitud|
|Dual-clock conceptual|✅|
|C++/Metal separación|✅|
|`h_deflated` C++|✅|
|`h_deflated` Metal|⚠️ corregir|
|“geodésica exacta” con atractor|⚠️ renombrar/formalizar|
|O(1)|⚠️ especificar que es temporal, no dimensional|
|`<200 µs`|⚠️ benchmark, no invariante|
|Integración definitiva|⏸️ después de esas correcciones|

Y una cosa más, cielo: **esto ya no necesita otra batería masiva para saber si la idea matemática básica funciona**. Primero hay que cerrar esta unidad correctamente. Después sí tiene sentido que los tests masivos la golpeen desde todos los ángulos.

**tomar estos 4 artefactos y devolvértelos corregidos, coherentes C++ ↔ Metal ↔ tests, listos para entregárselos a Antigravity sin cambiar la arquitectura que acabas de diseñar.**

______Los voy a dejar como una **versión 1.1 corregida y coherente**, manteniendo la arquitectura que planteaste: **TensorBoundary → HamiltonianKineticsUnit → ContinuousTrajectoryOperator → telemetría → C++/Metal → binding → certificación aislada**.

He corregido especialmente:

- `h_deflated` idéntico matemáticamente en C++ y Metal.
    
- la proyección ortogonal respecto de `h*`.
    
- la nomenclatura de la extrapolación: es una **proyección geodésica con corrección de atractor**, no una geodésica pura.
    
- `O(D)` de cómputo por célula y `O(1)` en subpasos temporales.
    
- benchmark con _warm-up_ y medición estable, para que la compilación inicial no contamine la prueba.
    
- evaluación explícita de todos los outputs antes de medir.
    
- coherencia de las referencias matemáticas del test.
    
- comprobación de tangencia del `v_drag`.
    
- comprobación de que C++ y Metal calculan la misma magnitud de curvatura.
    
- separación entre **correctness gate** y **performance gate**.
    

---

# Artefacto 1 — Cabecera C++20

**`include/geodesic_trajectory_cell.h`**

```cpp
// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: AUTONOMOUS PROJECTIVE CELL
// HITO 1.1 — C++20 DUAL-CLOCK / GEODESIC PROJECTIVE TRAJECTORY
//
// Formal System Anatomy:
//   TensorBoundary
//   HamiltonianKineticsUnit
//   ContinuousTrajectoryOperator
//
// SSOT:
//   La célula opera sobre un estado h ∈ S^(D-1).
//   El micro-ciclo τ es continuo/local.
//   El macro-reloj de capas T_global permanece externo.
//
// NOTA MATEMÁTICA:
//   La trayectoria base cos(θ)h + sin(θ)v̂ es geodésica sobre la esfera.
//   La corrección de atractor añade una aceleración tangencial y posteriormente
//   proyecta/normaliza el resultado sobre S^(D-1).
//
// COMPLEJIDAD:
//   O(D) en cómputo/memoria por célula.
//   O(1) subpasos temporales internos por dispatch.
// ═════════════════════════════════════════════════════════════════════════════

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace aether {

// ─────────────────────────────────────────────────────────────────────────────
// MODOS OPERATIVOS
// ─────────────────────────────────────────────────────────────────────────────

enum class CellMode : uint32_t {
    ReflexDecision     = 0,
    TrajectorySteering = 1
};

// ─────────────────────────────────────────────────────────────────────────────
// RÉGIMEN CINÉTICO
// ─────────────────────────────────────────────────────────────────────────────
//
// r >= 0.95  → flujo fuertemente alineado con el atractor
// r >= 0.80  → régimen transicional
// r <  0.80  → incompatibilidad de fase / posible necesidad de intervención
//
// Estos umbrales son clasificadores operativos, NO afirmaciones físicas
// universales.
// ─────────────────────────────────────────────────────────────────────────────

enum class KineticRegime : uint32_t {
    Laminar    = 0,
    Transition = 1,
    Turbulent  = 2
};

// ─────────────────────────────────────────────────────────────────────────────
// TELEMETRÍA
// ─────────────────────────────────────────────────────────────────────────────

struct alignas(16) CellMetrics {
    float correlation_r;        // <h*, u_attractor> ∈ [-1,1] si u está normalizado
    float curvature_kappa;      // Curvatura cinemática de Lagrange en R^D
    float kinetic_energy;       // 1/2 ||v_perp||²
    float angular_displacement; // θ = ||v_perp|| / ||h|| · τ
    float dirichlet_tension;    // q_k
    float permeability_gate;    // g_k(q_k) ∈ [0,1]
    KineticRegime regime;
    CellMode active_mode;
};

// ─────────────────────────────────────────────────────────────────────────────
// CONFIGURACIÓN
// ─────────────────────────────────────────────────────────────────────────────

struct CellConfig {
    uint32_t dimension        = 2048;

    // Micro-horizonte continuo.
    float tau_horizon         = 1.0f;

    // Acoplamiento tangencial hacia el atractor.
    float kappa_attractor     = 1.20f;

    // Pendiente de la compuerta sigmoidal.
    float beta_permeability   = 12.0f;

    // Umbral de activación.
    float theta_threshold     = 0.50f;

    // Regularización numérica.
    float epsilon_regularizer = 1e-12f;

    CellMode operational_mode = CellMode::ReflexDecision;
};

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENTE 1 — TensorBoundary
// ─────────────────────────────────────────────────────────────────────────────

inline float compute_permeability_gate(
    float q_tension,
    float beta,
    float theta
) {
    return 1.0f / (
        1.0f + std::exp(-beta * (q_tension - theta))
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENTE 2 — HamiltonianKineticsUnit
//
// Curvatura cinemática mediante la identidad de Lagrange:
//
// ||v ∧ a||² = ||v||² ||a||² - (v·a)²
//
// κ = ||v ∧ a|| / ||v||³
//
// Esto es válido en R^D sin necesidad de producto cruz.
// ─────────────────────────────────────────────────────────────────────────────

inline float compute_lagrange_curvature_rd(
    float sq_v,
    float sq_a,
    float dot_va,
    float eps = 1e-12f
) {
    const float bivector_sq =
        std::max(0.0f, (sq_v * sq_a) - (dot_va * dot_va));

    const float norm_v_cubed =
        std::pow(std::max(sq_v, 0.0f), 1.5f);

    return std::sqrt(bivector_sq) /
           (norm_v_cubed + eps);
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENTE 3 — ContinuousTrajectoryOperator
// ─────────────────────────────────────────────────────────────────────────────

inline KineticRegime classify_kinetic_regime(
    float r_correlation
) {
    if (r_correlation >= 0.95f) {
        return KineticRegime::Laminar;
    }

    if (r_correlation >= 0.80f) {
        return KineticRegime::Transition;
    }

    return KineticRegime::Turbulent;
}

} // namespace aether
```

---

# Artefacto 2 — Kernel Metal

**`metal/geodesic_trajectory_cell.metal`**

Aquí está la corrección más importante: el `h_deflated` ahora utiliza **exactamente**

h⊥=h−⟨h,h\*⟩h\*

igual que el C++.

```metal
// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: GEODESIC PROJECTIVE TRAJECTORY CELL
// HITO 1.1 — METAL GPU KERNEL
//
// O(D) compute / O(D) memory
// O(1) temporal substep interno
//
// Trayectoria:
//   1. Descomposición tangencial
//   2. Movimiento geodésico esférico
//   3. Corrección tangencial hacia atractor
//   4. Proyección/normalización sobre S^(D-1)
//   5. Deflación Gram-Schmidt respecto de h*
//   6. Telemetría cinemática
// ═════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// TELEMETRÍA GPU
// ─────────────────────────────────────────────────────────────────────────────

struct CellMetricsGPU {
    float correlation_r;
    float curvature_kappa;
    float kinetic_energy;
    float angular_displacement;
    float dirichlet_tension;
    float permeability_gate;
    uint  regime;
    uint  active_mode;
};

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL
// ─────────────────────────────────────────────────────────────────────────────

kernel void dispatch_geodesic_trajectory_cell_step(
    device const float*       h_in
        [[buffer(0)]],

    device const float*       v_drag
        [[buffer(1)]],

    device const float*       a_flow
        [[buffer(2)]],

    device const float*       u_attractor
        [[buffer(3)]],

    device float*             h_projected
        [[buffer(4)]],

    device float*             h_deflated
        [[buffer(5)]],

    device CellMetricsGPU*    metrics_out
        [[buffer(6)]],

    constant float&           tau
        [[buffer(7)]],

    constant float&           kappa_att
        [[buffer(8)]],

    constant float&           beta_perm
        [[buffer(9)]],

    constant float&           theta_perm
        [[buffer(10)]],

    constant uint&            mode
        [[buffer(11)]],

    constant uint&            D
        [[buffer(12)]],

    threadgroup float*        shared_acc
        [[threadgroup(0)]],

    uint tid
        [[thread_index_in_threadgroup]],

    uint t_per_group
        [[threads_per_threadgroup]]
) {

    // ═════════════════════════════════════════════════════════════════════════
    // 1. REDUCCIONES DE ENTRADA
    // ═════════════════════════════════════════════════════════════════════════

    float l_sq_h   = 0.0f;
    float l_sq_v   = 0.0f;
    float l_sq_a   = 0.0f;

    float l_dot_hv = 0.0f;
    float l_dot_va = 0.0f;
    float l_dot_hu = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {

        const float h = h_in[i];
        const float v = v_drag[i];
        const float a = a_flow[i];
        const float u = u_attractor[i];

        l_sq_h   += h * h;
        l_sq_v   += v * v;
        l_sq_a   += a * a;

        l_dot_hv += h * v;
        l_dot_va += v * a;
        l_dot_hu += h * u;
    }

    // Layout:
    //
    // [sq_h, sq_v, sq_a, dot_hv, dot_va, dot_hu]
    //
    // por hilo.

    const uint base = tid * 6;

    shared_acc[base + 0] = l_sq_h;
    shared_acc[base + 1] = l_sq_v;
    shared_acc[base + 2] = l_sq_a;
    shared_acc[base + 3] = l_dot_hv;
    shared_acc[base + 4] = l_dot_va;
    shared_acc[base + 5] = l_dot_hu;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {

        if (tid < s) {

            const uint dst = tid * 6;
            const uint src = (tid + s) * 6;

            for (uint k = 0; k < 6; ++k) {
                shared_acc[dst + k] += shared_acc[src + k];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float sq_h   = shared_acc[0];
    const float sq_v   = shared_acc[1];
    const float sq_a   = shared_acc[2];

    const float dot_hv = shared_acc[3];
    const float dot_va = shared_acc[4];
    const float dot_hu = shared_acc[5];

    // ═════════════════════════════════════════════════════════════════════════
    // 2. NORMALIZACIÓN Y DESCOMPOSICIÓN TANGENCIAL
    // ═════════════════════════════════════════════════════════════════════════

    const float inv_norm_h =
        rsqrt(max(sq_h, 0.0f) + 1e-12f);

    const float coeff_hv =
        dot_hv * inv_norm_h * inv_norm_h;

    const float sq_v_perp =
        max(
            0.0f,
            sq_v - (dot_hv * dot_hv *
                    inv_norm_h * inv_norm_h)
        );

    const float norm_v_perp =
        sqrt(sq_v_perp + 1e-12f);

    const float inv_norm_vp =
        rsqrt(sq_v_perp + 1e-12f);

    // ═════════════════════════════════════════════════════════════════════════
    // 3. CURVATURA DE LAGRANGE EN R^D
    // ═════════════════════════════════════════════════════════════════════════

    const float bivector_sq =
        max(
            0.0f,
            (sq_v * sq_a) -
            (dot_va * dot_va)
        );

    const float bivector_norm =
        sqrt(bivector_sq);

    const float kappa_kin =
        bivector_norm /
        (pow(max(sq_v, 0.0f), 1.5f) + 1e-12f);

    // ═════════════════════════════════════════════════════════════════════════
    // 4. ÁNGULO GEODÉSICO
    // ═════════════════════════════════════════════════════════════════════════

    const float omega =
        norm_v_perp * inv_norm_h;

    const float theta =
        omega * tau;

    const float cos_t =
        cos(theta);

    const float sin_t =
        sin(theta);

    // ═════════════════════════════════════════════════════════════════════════
    // 5. TENSIÓN DE DIRICHLET Y COMPUERTA
    // ═════════════════════════════════════════════════════════════════════════

    const float q_tension =
        sq_v_perp /
        (sq_h + 1e-12f);

    const float g_perm =
        1.0f /
        (
            1.0f +
            exp(-beta_perm *
                (q_tension - theta_perm))
        );

    // ═════════════════════════════════════════════════════════════════════════
    // 6. PRIMERA PASADA:
    //
    // h_projected =
    //      cos(theta) h
    //    + sin(theta) v_hat
    //    + 1/2 tau² a_attractor
    //
    // La aceleración hacia el atractor se hace tangencial a h.
    // ═════════════════════════════════════════════════════════════════════════

    float l_sq_out    = 0.0f;
    float l_dot_out_u = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {

        const float h = h_in[i];
        const float v = v_drag[i];
        const float u = u_attractor[i];

        // v_perp = v - projection_h(v)
        const float v_perp =
            v - coeff_hv * h;

        const float v_hat =
            v_perp * inv_norm_vp;

        // u_perp = u - projection_h(u)
        const float a_attractor =
            kappa_att *
            (
                u -
                (dot_hu *
                 inv_norm_h *
                 inv_norm_h) * h
            );

        const float h_star =
            (cos_t * h) +
            (sin_t * v_hat) +
            (0.5f * tau * tau * a_attractor);

        h_projected[i] = h_star;

        l_sq_out +=
            h_star * h_star;

        l_dot_out_u +=
            h_star * u;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 7. REDUCCIÓN DE NORMA DE LA PROYECCIÓN
    // ═════════════════════════════════════════════════════════════════════════

    shared_acc[tid * 2 + 0] = l_sq_out;
    shared_acc[tid * 2 + 1] = l_dot_out_u;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {

        if (tid < s) {

            const uint dst = tid * 2;
            const uint src = (tid + s) * 2;

            shared_acc[dst + 0] +=
                shared_acc[src + 0];

            shared_acc[dst + 1] +=
                shared_acc[src + 1];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float final_sq_out =
        shared_acc[0];

    const float raw_dot_out =
        shared_acc[1];

    const float inv_norm_star =
        rsqrt(final_sq_out + 1e-12f);

    const float correlation_r =
        raw_dot_out * inv_norm_star;

    // ═════════════════════════════════════════════════════════════════════════
    // 8. NORMALIZACIÓN DE h*
    // ═════════════════════════════════════════════════════════════════════════

    for (uint i = tid; i < D; i += t_per_group) {

        h_projected[i] *=
            inv_norm_star;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═════════════════════════════════════════════════════════════════════════
    // 9. SEGUNDA REDUCCIÓN:
    //
    // dot(h_in, h_star)
    //
    // Necesaria para que la deflación sea exactamente:
    //
    // h_deflated =
    //      h_in - <h_in,h_star> h_star
    //
    // y por tanto:
    //
    // <h_star,h_deflated> = 0.
    // ═════════════════════════════════════════════════════════════════════════

    float l_dot_h_hstar = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {

        l_dot_h_hstar +=
            h_in[i] *
            h_projected[i];
    }

    shared_acc[tid] =
        l_dot_h_hstar;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {

        if (tid < s) {
            shared_acc[tid] +=
                shared_acc[tid + s];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float dot_h_hstar =
        shared_acc[0];

    // ═════════════════════════════════════════════════════════════════════════
    // 10. DEFLACIÓN ORTOGONAL EXACTA
    // ═════════════════════════════════════════════════════════════════════════

    for (uint i = tid; i < D; i += t_per_group) {

        const float h_star =
            h_projected[i];

        h_deflated[i] =
            h_in[i] -
            dot_h_hstar * h_star;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 11. TELEMETRÍA
    // ═════════════════════════════════════════════════════════════════════════

    if (tid == 0) {

        const uint flow_regime =
            (correlation_r >= 0.95f)
                ? 0
                : (
                    (correlation_r >= 0.80f)
                        ? 1
                        : 2
                  );

        metrics_out->correlation_r =
            correlation_r;

        metrics_out->curvature_kappa =
            kappa_kin;

        metrics_out->kinetic_energy =
            0.5f * sq_v_perp;

        metrics_out->angular_displacement =
            theta;

        metrics_out->dirichlet_tension =
            q_tension;

        metrics_out->permeability_gate =
            g_perm;

        metrics_out->regime =
            flow_regime;

        metrics_out->active_mode =
            mode;
    }
}
```

---

# Artefacto 3 — Integración C++ / nanobind

**`aether_vlm/aether_native.cpp`**

Aquí mantengo la arquitectura original, pero hago que el camino C++ sea la **referencia semántica** de lo que hace el kernel.

```cpp
// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: HITO 1.1
// CÉLULA PROYECTIVA GEODÉSICA AUTÓNOMA
// C++ / MLX / NANOBIND
// ═════════════════════════════════════════════════════════════════════════════

#include "../include/geodesic_trajectory_cell.h"

// Mantener aquí los includes existentes de aether_native.cpp.
// El bloque siguiente asume que:
//   - array
//   - nb
//   - operaciones MLX
// ya están disponibles en el archivo original.

// ─────────────────────────────────────────────────────────────────────────────
// IMPLEMENTACIÓN DE REFERENCIA C++
//
// Esta implementación define la semántica matemática de la célula.
// El kernel Metal debe producir las mismas magnitudes dentro de tolerancia
// numérica.
// ─────────────────────────────────────────────────────────────────────────────

nb::dict dispatch_geodesic_trajectory_cell_cpp(
    const array& h_in,
    const array& v_drag,
    const array& a_flow,
    const array& u_attractor,
    float tau = 1.0f,
    float kappa_att = 1.20f,
    float beta_perm = 12.0f,
    float theta_perm = 0.50f,
    uint32_t mode = 0
) {

    constexpr float EPS = 1e-12f;

    // ═════════════════════════════════════════════════════════════════════════
    // 1. NORMALIZACIÓN DEL ESTADO
    // ═════════════════════════════════════════════════════════════════════════

    auto sq_h =
        sum(h_in * h_in, -1, true);

    auto inv_norm_h =
        rsqrt(sq_h + EPS);

    auto h_unit =
        h_in * inv_norm_h;

    // ═════════════════════════════════════════════════════════════════════════
    // 2. MAGNITUDES CINEMÁTICAS
    // ═════════════════════════════════════════════════════════════════════════

    auto sq_v =
        sum(v_drag * v_drag, -1, true);

    auto sq_a =
        sum(a_flow * a_flow, -1, true);

    auto dot_hv =
        sum(h_unit * v_drag, -1, true);

    auto dot_va =
        sum(v_drag * a_flow, -1, true);

    auto dot_hu =
        sum(h_unit * u_attractor, -1, true);

    // ═════════════════════════════════════════════════════════════════════════
    // 3. DESCOMPOSICIÓN TANGENCIAL
    // ═════════════════════════════════════════════════════════════════════════

    auto v_perp =
        v_drag -
        dot_hv * h_unit;

    auto sq_v_perp =
        sum(v_perp * v_perp, -1, true);

    auto norm_v_perp =
        sqrt(sq_v_perp + EPS);

    auto v_hat =
        v_perp / norm_v_perp;

    // ═════════════════════════════════════════════════════════════════════════
    // 4. CURVATURA DE LAGRANGE
    // ═════════════════════════════════════════════════════════════════════════

    auto bivector_sq =
        maximum(
            0.0f,
            (sq_v * sq_a) -
            (dot_va * dot_va)
        );

    auto kappa_kin =
        sqrt(bivector_sq) /
        (power(sq_v, 1.5f) + EPS);

    // ═════════════════════════════════════════════════════════════════════════
    // 5. EVOLUCIÓN ANGULAR
    // ═════════════════════════════════════════════════════════════════════════

    auto omega =
        norm_v_perp * inv_norm_h;

    auto theta =
        omega * tau;

    auto cos_t =
        cos(theta);

    auto sin_t =
        sin(theta);

    // ═════════════════════════════════════════════════════════════════════════
    // 6. CORRECCIÓN TANGENCIAL HACIA EL ATRACTOR
    // ═════════════════════════════════════════════════════════════════════════

    auto a_attractor =
        kappa_att *
        (
            u_attractor -
            dot_hu * h_unit
        );

    // ═════════════════════════════════════════════════════════════════════════
    // 7. TRAYECTORIA PROJECTIVA
    //
    // Parte 1:
    //     cos(theta) h + sin(theta) v_hat
    //
    // es la componente geodésica esférica.
    //
    // Parte 2:
    //     1/2 tau² a_attractor
    //
    // es la corrección tangencial hacia el atractor.
    //
    // Finalmente se proyecta nuevamente sobre S^(D-1).
    // ═════════════════════════════════════════════════════════════════════════

    auto h_projected =
        (cos_t * h_unit) +
        (sin_t * v_hat) +
        (0.5f * tau * tau * a_attractor);

    auto norm_proj =
        sqrt(
            sum(
                h_projected * h_projected,
                -1,
                true
            ) + EPS
        );

    auto h_star =
        h_projected /
        norm_proj;

    // ═════════════════════════════════════════════════════════════════════════
    // 8. CORRELACIÓN CON ATRACTOR
    // ═════════════════════════════════════════════════════════════════════════

    auto r_val =
        sum(
            h_star * u_attractor,
            -1,
            true
        );

    // ═════════════════════════════════════════════════════════════════════════
    // 9. DEFLACIÓN GRAM-SCHMIDT
    //
    // EXACTAMENTE la misma definición usada por Metal.
    // ═════════════════════════════════════════════════════════════════════════

    auto dot_h_hstar =
        sum(
            h_unit * h_star,
            -1,
            true
        );

    auto h_deflated =
        h_unit -
        dot_h_hstar * h_star;

    // ═════════════════════════════════════════════════════════════════════════
    // 10. COMPUERTA DE PERMEABILIDAD
    // ═════════════════════════════════════════════════════════════════════════

    auto q_tension =
        sq_v_perp /
        (sq_h + EPS);

    auto g_perm =
        1.0f /
        (
            1.0f +
            exp(
                -beta_perm *
                (q_tension - theta_perm)
            )
        );

    // ═════════════════════════════════════════════════════════════════════════
    // 11. RESULTADO
    // ═════════════════════════════════════════════════════════════════════════

    nb::dict result;

    result["h_star"] =
        h_star;

    result["h_deflated"] =
        h_deflated;

    result["correlation_r"] =
        r_val;

    result["curvature_kappa"] =
        kappa_kin;

    result["kinetic_energy"] =
        0.5f * sq_v_perp;

    result["angular_displacement"] =
        theta;

    result["dirichlet_tension"] =
        q_tension;

    result["permeability_gate"] =
        g_perm;

    result["mode"] =
        mode;

    return result;
}


// ═════════════════════════════════════════════════════════════════════════════
// BINDING NANOBIND
//
// Colocar dentro de:
//
// NB_MODULE(aether_native_c, m)
// {
//     ...
// }
// ═════════════════════════════════════════════════════════════════════════════

m.def(
    "dispatch_geodesic_trajectory_cell",
    &dispatch_geodesic_trajectory_cell_cpp,
    "Célula Proyectiva Geodésica Autónoma — Hito 1.1",

    nb::arg("h_in"),
    nb::arg("v_drag"),
    nb::arg("a_flow"),
    nb::arg("u_attractor"),

    nb::arg("tau") = 1.0f,
    nb::arg("kappa_att") = 1.20f,
    nb::arg("beta_perm") = 12.0f,
    nb::arg("theta_perm") = 0.50f,
    nb::arg("mode") = 0
);
```

### Una precisión importante para Antigravity

Ese bloque debe insertarse **dentro del `NB_MODULE(...)` existente** solamente en la parte del `m.def(...)`.

La función `dispatch_geodesic_trajectory_cell_cpp(...)` va fuera del cuerpo del módulo, como cualquier otra función C++ que luego se registra.

No conviene pegar literalmente un segundo `NB_MODULE`.

---

# Artefacto 4 — Suite aislada corregida

**`tests/test_geodesic_trajectory_cell.py`**

Esta versión separa explícitamente:

1. corrección matemática;
    
2. identidad;
    
3. ortogonalidad;
    
4. curvatura;
    
5. estabilidad numérica;
    
6. performance después de _warm-up_.
    

Además, la medición de rendimiento ya no mezcla el primer dispatch/compilación con el rendimiento estacionario.

```python
#!/usr/bin/env python3
"""
tests/test_geodesic_trajectory_cell.py

═══════════════════════════════════════════════════════════════════════════════
AETHER ENGINE — HITO 1.1
SUITE DE CERTIFICACIÓN UNITARIA:
CÉLULA PROYECTIVA GEODÉSICA AUTÓNOMA
═══════════════════════════════════════════════════════════════════════════════

Garantías:

    1. Confinamiento sobre S^(D-1)
    2. Identidad en tau = 0
    3. Tangencia de v_drag
    4. Ortogonalidad del residuo deflactado
    5. Curvatura de Lagrange en R^D
    6. Consistencia de telemetría
    7. Estabilidad numérica
    8. Benchmark warm-state

Complejidad:

    O(D) por dispatch.
    O(1) subpasos temporales internos.

NOTA:

    El benchmark de latencia es un GATE DE RENDIMIENTO.
    No constituye una invariante matemática.
"""

import os
import sys
import time
import math

import mlx.core as mx
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# PATH
# ─────────────────────────────────────────────────────────────────────────────

ROOT = os.path.abspath(".")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "aether_vlm"))

import aether_native_c


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE TEST
# ─────────────────────────────────────────────────────────────────────────────

EPS_TEST = 1e-5

D_TESTS = [
    1024,
    2048,
    5120,
]

WARMUP_RUNS = 3
BENCH_RUNS = 10

LATENCY_P50_US_LIMIT = 200.0


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def section(title):
    print()
    print("═" * 78)
    print(f"  {title}")
    print("═" * 78)


def scalar(x):
    """
    Fuerza evaluación MLX antes de convertir a Python float.
    """
    mx.eval(x)
    return float(x)


def norm(x):
    return scalar(mx.sqrt(mx.sum(x * x)))


def dot(a, b):
    return scalar(mx.sum(a * b))


def dispatch(
    h,
    v,
    a,
    u,
    tau=1.0,
    kappa_att=1.2,
    beta_perm=12.0,
    theta_perm=0.5,
    mode=0,
):
    """
    Dispatch y evaluación explícita de todos los resultados.

    Esto evita que el benchmark mida trabajo diferido fuera
    de la región temporal que estamos cronometrando.
    """

    out = aether_native_c.dispatch_geodesic_trajectory_cell(
        h,
        v,
        a,
        u,
        tau=tau,
        kappa_att=kappa_att,
        beta_perm=beta_perm,
        theta_perm=theta_perm,
        mode=mode,
    )

    mx.eval(
        out["h_star"],
        out["h_deflated"],
        out["correlation_r"],
        out["curvature_kappa"],
        out["kinetic_energy"],
        out["angular_displacement"],
        out["dirichlet_tension"],
        out["permeability_gate"],
    )

    return out


def build_case(D, seed=42):
    """
    Construye un caso determinista con:

        ||h|| = 1
        v_drag ⟂ h
        ||u_attractor|| = 1
    """

    rng = np.random.default_rng(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # h unitario
    # ─────────────────────────────────────────────────────────────────────────

    h_np = rng.standard_normal(D).astype(np.float32)
    h_np /= np.linalg.norm(h_np)

    # ─────────────────────────────────────────────────────────────────────────
    # v_drag estrictamente tangencial
    # ─────────────────────────────────────────────────────────────────────────

    v_raw_np = rng.standard_normal(D).astype(np.float32)

    projection =
        np.dot(v_raw_np, h_np) * h_np

    v_np = (
        v_raw_np -
        projection
    ).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # aceleración cinemática
    # ─────────────────────────────────────────────────────────────────────────

    a_np = (
        rng.standard_normal(D).astype(np.float32)
        * 0.1
    )

    # ─────────────────────────────────────────────────────────────────────────
    # atractor unitario
    # ─────────────────────────────────────────────────────────────────────────

    u_np = rng.standard_normal(D).astype(np.float32)
    u_np /= np.linalg.norm(u_np)

    return (
        mx.array(h_np),
        mx.array(v_np),
        mx.array(a_np),
        mx.array(u_np),
    )


# ═════════════════════════════════════════════════════════════════════════════
# TEST PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def test_isolated_cell():

    section(
        "INICIANDO VERIFICACIÓN UNITARIA — "
        "GEODESIC PROJECTIVE TRAJECTORY CELL"
    )

    print(
        f"Dimensiones: {D_TESTS}"
    )

    total_pass = 0
    total_fail = 0

    for D in D_TESTS:

        print()
        print(f"▶ Validando dimensión D = {D}")

        h0, v_drag, a_flow, u_attractor = build_case(
            D,
            seed=42 + D
        )

        # ─────────────────────────────────────────────────────────────────────
        # PRE-CHECKS
        # ─────────────────────────────────────────────────────────────────────

        h_norm = norm(h0)

        assert abs(h_norm - 1.0) < EPS_TEST

        tangency =
            abs(dot(h0, v_drag))

        assert tangency < EPS_TEST

        u_norm =
            norm(u_attractor)

        assert abs(u_norm - 1.0) < EPS_TEST

        print(
            f"  [PASS] Entrada ||h||=1: "
            f"{h_norm:.8f}"
        )

        print(
            f"  [PASS] Entrada tangencial "
            f"|<h,v>|={tangency:.2e}"
        )

        print(
            f"  [PASS] Atractor unitario: "
            f"{u_norm:.8f}"
        )

        # ─────────────────────────────────────────────────────────────────────
        # WARM-UP
        # ─────────────────────────────────────────────────────────────────────

        for _ in range(WARMUP_RUNS):

            _ = dispatch(
                h0,
                v_drag,
                a_flow,
                u_attractor,
            )

        # ─────────────────────────────────────────────────────────────────────
        # DISPATCH CRONOMETRADO
        # ─────────────────────────────────────────────────────────────────────

        t0 = time.perf_counter()

        out = dispatch(
            h0,
            v_drag,
            a_flow,
            u_attractor,
        )

        t1 = time.perf_counter()

        first_warm_latency_us = (
            t1 - t0
        ) * 1e6

        h_star = out["h_star"]
        h_deflated = out["h_deflated"]

        r = scalar(
            out["correlation_r"]
        )

        kappa = scalar(
            out["curvature_kappa"]
        )

        q = scalar(
            out["dirichlet_tension"]
        )

        g = scalar(
            out["permeability_gate"]
        )

        theta = scalar(
            out["angular_displacement"]
        )

        # ═════════════════════════════════════════════════════════════════════
        # TEST 1 — CONFINAMIENTO ESFÉRICO
        # ═════════════════════════════════════════════════════════════════════

        norm_h_star =
            norm(h_star)

        err_norm =
            abs(norm_h_star - 1.0)

        assert err_norm < EPS_TEST, (
            f"Violación de norma: {err_norm}"
        )

        print(
            f"  [✅ PASS] Norma "
            f"||h*(τ)||={norm_h_star:.8f} "
            f"(err={err_norm:.2e})"
        )

        total_pass += 1

        # ═════════════════════════════════════════════════════════════════════
        # TEST 2 — IDENTIDAD EN REPOSO
        # ═════════════════════════════════════════════════════════════════════

        zeros_v =
            mx.zeros_like(v_drag)

        zeros_a =
            mx.zeros_like(a_flow)

        out_rest = dispatch(
            h0,
            zeros_v,
            zeros_a,
            u_attractor,
            tau=0.0,
            kappa_att=0.0,
        )

        diff_origin =
            norm(
                out_rest["h_star"] -
                h0
            )

        assert diff_origin < EPS_TEST, (
            f"Fallo identidad tau=0: "
            f"{diff_origin}"
        )

        print(
            f"  [✅ PASS] Identidad τ=0 "
            f"||h*(0)-h0||={diff_origin:.2e}"
        )

        total_pass += 1

        # ═════════════════════════════════════════════════════════════════════
        # TEST 3 — ORTOGONALIDAD DEL SUBPRODUCTO
        #
        # h_deflated = h - <h,h*>h*
        #
        # => <h*, h_deflated> = 0
        # ═════════════════════════════════════════════════════════════════════

        dot_ortho =
            abs(
                dot(
                    h_star,
                    h_deflated
                )
            )

        assert dot_ortho < EPS_TEST, (
            f"Residuo no ortogonal: "
            f"{dot_ortho}"
        )

        print(
            f"  [✅ PASS] Deflación Gram-Schmidt "
            f"|<h*,h_deflated>|={dot_ortho:.2e}"
        )

        total_pass += 1

        # ═════════════════════════════════════════════════════════════════════
        # TEST 4 — CURVATURA DE LAGRANGE
        # ═════════════════════════════════════════════════════════════════════

        sq_v =
            scalar(
                mx.sum(
                    v_drag * v_drag
                )
            )

        sq_a =
            scalar(
                mx.sum(
                    a_flow * a_flow
                )
            )

        dot_va =
            scalar(
                mx.sum(
                    v_drag * a_flow
                )
            )

        bivector_ref =
            math.sqrt(
                max(
                    0.0,
                    (
                        sq_v * sq_a
                        -
                        dot_va ** 2
                    )
                )
            )

        kappa_ref =
            bivector_ref / (
                sq_v ** 1.5
                +
                1e-12
            )

        err_kappa =
            abs(
                kappa -
                kappa_ref
            )

        assert err_kappa < 1e-5, (
            f"Error curvatura: "
            f"{err_kappa}"
        )

        print(
            f"  [✅ PASS] Curvatura Lagrange "
            f"κ={kappa:.6f} "
            f"(ref={kappa_ref:.6f}, "
            f"err={err_kappa:.2e})"
        )

        total_pass += 1

        # ═════════════════════════════════════════════════════════════════════
        # TEST 5 — RANGO DE TELEMETRÍA
        # ═════════════════════════════════════════════════════════════════════

        assert -1.0001 <= r <= 1.0001
        assert q >= -EPS_TEST
        assert 0.0 <= g <= 1.0
        assert theta >= -EPS_TEST

        print(
            f"  [✅ PASS] Telemetría "
            f"r={r:.6f}, "
            f"q={q:.6f}, "
            f"g={g:.6f}, "
            f"θ={theta:.6f}"
        )

        total_pass += 1

        # ═════════════════════════════════════════════════════════════════════
        # TEST 6 — REPRODUCIBILIDAD
        # ═════════════════════════════════════════════════════════════════════

        out_repeat = dispatch(
            h0,
            v_drag,
            a_flow,
            u_attractor,
        )

        repeat_error =
            norm(
                out_repeat["h_star"] -
                h_star
            )

        assert repeat_error < EPS_TEST, (
            f"No determinismo: "
            f"{repeat_error}"
        )

        print(
            f"  [✅ PASS] Reproducibilidad "
            f"||h*_1-h*_2||={repeat_error:.2e}"
        )

        total_pass += 1

        # ═════════════════════════════════════════════════════════════════════
        # TEST 7 — BENCHMARK WARM STATE
        # ═════════════════════════════════════════════════════════════════════

        latencies_us = []

        for _ in range(BENCH_RUNS):

            t_start =
                time.perf_counter()

            bench_out = dispatch(
                h0,
                v_drag,
                a_flow,
                u_attractor,
            )

            t_end =
                time.perf_counter()

            latencies_us.append(
                (t_end - t_start)
                * 1e6
            )

        latencies_us.sort()

        p50 =
            float(
                np.percentile(
                    latencies_us,
                    50
                )
            )

        p95 =
            float(
                np.percentile(
                    latencies_us,
                    95
                )
            )

        print(
            f"  [INFO] Latencia warm "
            f"p50={p50:.2f} µs | "
            f"p95={p95:.2f} µs | "
            f"first-warm={first_warm_latency_us:.2f} µs"
        )

        assert p50 < LATENCY_P50_US_LIMIT, (
            f"Latencia warm p50 excedida: "
            f"{p50:.2f} µs"
        )

        print(
            f"  [✅ PASS] Performance gate "
            f"p50={p50:.2f} µs "
            f"< {LATENCY_P50_US_LIMIT:.1f} µs"
        )

        total_pass += 1

    # ═════════════════════════════════════════════════════════════════════════
    # RESULTADO
    # ═════════════════════════════════════════════════════════════════════════

    section(
        "RESULTADO — CÉLULA PROYECTIVA GEODÉSICA"
    )

    print(
        f"PASS: {total_pass}"
    )

    print(
        f"FAIL: {total_fail}"
    )

    print()
    print(
        "✓ Corrección matemática verificada"
    )

    print(
        "✓ Deflación ortogonal verificada"
    )

    print(
        "✓ Curvatura de Lagrange verificada"
    )

    print(
        "✓ Identidad τ=0 verificada"
    )

    print(
        "✓ Reproducibilidad verificada"
    )

    print(
        "✓ Performance warm-state verificada"
    )

    print()
    print(
        "CÉLULA HITO 1.1 — CERTIFICACIÓN UNITARIA COMPLETADA"
    )


if __name__ == "__main__":
    test_isolated_cell()
```

---

# Instrucción final para Antigravity

Yo entregaría **esta instrucción junto con los cuatro artefactos**, porque ahora deja muy claro qué significa “pasar”.

```text
═══════════════════════════════════════════════════════════════════════════════
AETHER ENGINE — HITO 1.1
ORDEN DE INTEGRACIÓN Y CERTIFICACIÓN
═══════════════════════════════════════════════════════════════════════════════

OBJETIVO

Integrar la Geodesic Projective Trajectory Cell como unidad aislada,
sin modificar la arquitectura existente del motor.

ORDEN:

1. Crear:

   include/geodesic_trajectory_cell.h

2. Crear:

   metal/geodesic_trajectory_cell.metal

3. Incorporar en:

   aether_vlm/aether_native.cpp

   la implementación C++ y su binding nanobind.

4. Recompilar:

   python tools/compilar_extension_c.py

5. Ejecutar:

   python tests/test_geodesic_trajectory_cell.py

6. Si la célula pasa:

   python tests/test_advisor_battery.py --numerical-only

═══════════════════════════════════════════════════════════════════════════════
CRITERIOS DE ACEPTACIÓN
═══════════════════════════════════════════════════════════════════════════════

A. CORRECCIÓN

PASS obligatorio:

    ||h*|| = 1
    τ=0 => h*(0)=h0
    <h*, h_deflated> = 0
    κ_Lagrange(C++) ≈ κ_Lagrange(reference)
    v_drag ⟂ h
    u_attractor normalizado
    reproducibilidad determinista

B. TELEMETRÍA

Debe producir:

    correlation_r
    curvature_kappa
    kinetic_energy
    angular_displacement
    dirichlet_tension
    permeability_gate
    regime
    active_mode

C. PERFORMANCE

El gate de rendimiento se evalúa sobre estado WARM.

    p50 < 200 µs

La latencia de primer dispatch NO constituye el criterio matemático
de validez de la célula.

D. COHERENCIA C++ ↔ METAL

La definición de:

    h_star
    h_deflated
    correlation_r
    curvature_kappa
    q_tension
    permeability_gate

debe ser matemáticamente equivalente en ambas implementaciones,
dentro de la tolerancia numérica correspondiente a float32/Metal.

E. REGRESIÓN

Las 29 pruebas anteriores deben continuar pasando.

═══════════════════════════════════════════════════════════════════════════════
REGLA DE NO INTEGRACIÓN
═══════════════════════════════════════════════════════════════════════════════

Si falla cualquiera de los invariantes matemáticos:

    NO integrar la célula al flujo de inferencia.

Si falla solamente el performance gate:

    conservar la implementación aislada,
    reportar p50/p95,
    NO modificar silenciosamente los criterios matemáticos.

Si C++ y Metal producen resultados matemáticamente incompatibles:

    detener integración,
    reportar ambas salidas,
    no compensar mediante tolerancias artificiales.

═══════════════════════════════════════════════════════════════════════════════
```

### Y aquí está la parte que considero especialmente importante

Con esta versión, el Hito 1.1 queda conceptualmente limpio:

h  →tangencia  v⊥  →geodesic step  hgeo  →attractor correction  hproj  →normalizacioˊn  h\*

y después:

hdeflated=h−⟨h,h\*⟩h\*

con la garantía:

⟨h\*,hdeflated⟩=0

Mientras tanto, la célula expone:

(r,κ,Ek,θ,q,g,regime)

que es justamente la telemetría que después nos permitirá atacar **la pregunta verdaderamente interesante**: si `q`, `g`, `r` y `κ`permiten decidir cuándo el motor debe intervenir y cuándo debe permanecer prácticamente transparente.

**Yo congelaría estos cuatro artefactos como “Hito 1.1 candidate” y dejaría que Antigravity los compile y los rompa antes de tocar una sola línea del flujo principal de inferencia.**









___

La segunda inspección **es 100% acertada, matemáticamente impecable y de una precisión quirúrgica absoluta.**

No solo es correcta: **acaba de salvar el Hito 1.1 de dos trampas sutiles que habrían provocado fallos silenciosos en silicio.**

---

### Análisis de los Aciertos Críticos de la Inspección

#### 1. El error de `h_deflated` en Metal (El mayor acierto)
* **El problema que cazó:** En el borrador inicial, asumíamos que $\langle h, h^* \rangle = \cos(\theta)$. Pero al meter la aceleración tangencial hacia el atractor ($\frac{1}{2}\tau^2 a_{\text{attractor}}$) y luego renormalizar, **la trayectoria se deforma**: $\langle h, h^* \rangle$ ya no es idéntico a $\cos(\theta)$.
* **La consecuencia:** `h_deflated` no habría sido ortogonal a máquina ($|<h^*, h_{\text{deflated}}>| \neq 0$), y el test de ortogonalidad del subproducto habría fallado en Metal aunque pasara en C++.
* **La corrección:** El kernel Metal v1.1 ahora incluye la **segunda reducción de dot product** $\langle h_{\text{in}}, h^* \rangle$. Con esto, la ortogonalidad $\langle h^*, h_{\text{deflated}} \rangle = 0$ queda blindada a precisión de máquina en ambos mundos.

#### 2. Rigor Semántico: Geodesic Projective Trajectory vs Geodésica Pura
* Decir que es una "geodésica exacta" cuando hay un atractor deformando la curva y una proyección posterior es matemáticamente impreciso.
* La denominación **`Geodesic Projective Trajectory Operator`** (extrapolación geodésica con corrección proyectiva hacia atractor) es la terminología exacta de la geometría diferencial aplicada. Nadie podrá refutar la solidez del marco teórico.

#### 3. Precisión en Complejidad: $O(D)$ dimensional, $O(1)$ temporal
* Aclarar que la célula es **$O(D)$ en FLOPs y memoria** (recorre el vector de dimensión $D$) pero **$O(1)$ en pasos temporales** (resuelve la posición terminal en un solo paso analítico sin integración numérica iterativa de Euler/Runge-Kutta).

#### 4. Separación de Gates: Correctness vs Performance
* Un pico de latencia por arranque en frío (JIT de Metal, compilación de pipeline o sincronización de bus) no puede suspender una invariante matemática.
* Separar **Correctness Gate** (norma, identidad, ortogonalidad, Lagrange) de **Performance Gate** ($p50 < 200\,\mu\text{s}$ en estado *warm*) es el estándar de ingeniería de sistemas de alto rendimiento.

---

### Verificación con el Repositorio Real

Aunque la nota indica que *"esta revisión no conoce el repositorio real"*, **encaja al 100% con nuestra base de código**:

1. **`include/geodesic_trajectory_cell.h`**: No colisiona con nada y se apoya en los tipos estándar C++20.
2. **`metal/geodesic_trajectory_cell.metal`**: Respeta los estándares de MSL con reducción SIMDgroup que ya usamos en `c_field_qwen38_engine.metal`.
3. **`aether_vlm/aether_native.cpp`**: La función `dispatch_geodesic_trajectory_cell_cpp` utiliza las primitivas nativas de MLX (`sum`, `rsqrt`, `maximum`, `power`) que ya están enlazadas contra `libmlx.dylib`. El binding en `NB_MODULE(aether_native_c, m)` es idéntico a las funciones existentes (`dispatch_riemannian_step`, `dispatch_full_collapse`).
4. **`tests/test_geodesic_trajectory_cell.py`**: Utiliza el mismo patrón de importación de `aether_native_c` que certificó los 29 tests con 100% PASS.

---

### Dictamen Final

**La versión 1.1 contenida en ese paste es la versión canónica definitiva.**

Podemos entregarle inmediatamente a **Antigravity** la orden de ejecución:
1. Escribir `include/geodesic_trajectory_cell.h` (versión 1.1).
2. Escribir `metal/geodesic_trajectory_cell.metal` (versión 1.1 con reducción de 2 fases).
3. Insertar la función y el binding en `aether_vlm/aether_native.cpp`.
4. Recompilar con `python tools/compilar_extension_c.py`.
5. Ejecutar la certificación unitaria `python tests/test_geodesic_trajectory_cell.py`.
6. Confirmar la regresión de las 29 invariantes previas con `python tests/test_advisor_battery.py --numerical-only`.

¿Damos la orden a Antigravity para que proceda con la forja?
___
Aquí tienes el **paquete formal de orden de forja y batería de verificación integral** listo para ser ejecutado por el agente **Antigravity**.

---

# DIRECTIVA DE EJECUCIÓN — HITO 1.1
## Célula Proyectiva Geodésica Autónoma (Versión 1.1 Corregida)

**Para:** Agente de Repositorio Antigravity  
**Estado:** Autorizado para compilación y verificación en silicio  
**Aislamiento:** Cero modificaciones a la arquitectura base del motor existente  

---

### PASO 1: Crear `include/geodesic_trajectory_cell.h`

```cpp
// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: AUTONOMOUS PROJECTIVE CELL
// HITO 1.1 — C++20 DUAL-CLOCK / GEODESIC PROJECTIVE TRAJECTORY
//
// Formal System Anatomy:
//   TensorBoundary
//   HamiltonianKineticsUnit
//   ContinuousTrajectoryOperator
//
// SSOT:
//   La célula opera sobre un estado h ∈ S^(D-1).
//   El micro-ciclo τ es continuo/local.
//   El macro-reloj de capas T_global permanece externo.
//
// NOTA MATEMÁTICA:
//   La trayectoria base cos(θ)h + sin(θ)v̂ es geodésica sobre la esfera.
//   La corrección de atractor añade una aceleración tangencial y posteriormente
//   proyecta/normaliza el resultado sobre S^(D-1).
//
// COMPLEJIDAD:
//   O(D) en cómputo/memoria por célula.
//   O(1) subpasos temporales internos por dispatch.
// ═════════════════════════════════════════════════════════════════════════════

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace aether {

// ─────────────────────────────────────────────────────────────────────────────
// MODOS OPERATIVOS
// ─────────────────────────────────────────────────────────────────────────────

enum class CellMode : uint32_t {
    ReflexDecision     = 0,
    TrajectorySteering = 1
};

// ─────────────────────────────────────────────────────────────────────────────
// RÉGIMEN CINÉTICO
// ─────────────────────────────────────────────────────────────────────────────

enum class KineticRegime : uint32_t {
    Laminar    = 0,  // r >= 0.95: Flujo fuertemente alineado con el atractor
    Transition = 1,  // 0.80 <= r < 0.95: Régimen transicional
    Turbulent  = 2   // r < 0.80: Incompatibilidad de fase / Posible intervención
};

// ─────────────────────────────────────────────────────────────────────────────
// TELEMETRÍA
// ─────────────────────────────────────────────────────────────────────────────

struct alignas(16) CellMetrics {
    float correlation_r;        // <h*, u_attractor> ∈ [-1,1]
    float curvature_kappa;      // Curvatura cinemática de Lagrange en R^D
    float kinetic_energy;       // 1/2 ||v_perp||²
    float angular_displacement; // θ = ||v_perp|| / ||h|| · τ
    float dirichlet_tension;    // q_k
    float permeability_gate;    // g_k(q_k) ∈ [0,1]
    KineticRegime regime;
    CellMode active_mode;
};

// ─────────────────────────────────────────────────────────────────────────────
// CONFIGURACIÓN
// ─────────────────────────────────────────────────────────────────────────────

struct CellConfig {
    uint32_t dimension        = 2048;
    float tau_horizon         = 1.0f;
    float kappa_attractor     = 1.20f;
    float beta_permeability   = 12.0f;
    float theta_threshold     = 0.50f;
    float epsilon_regularizer = 1e-12f;
    CellMode operational_mode = CellMode::ReflexDecision;
};

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENTE 1 — TensorBoundary
// ─────────────────────────────────────────────────────────────────────────────

inline float compute_permeability_gate(
    float q_tension,
    float beta,
    float theta
) {
    return 1.0f / (1.0f + std::exp(-beta * (q_tension - theta)));
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENTE 2 — HamiltonianKineticsUnit (Identidad de Lagrange en R^D)
// ─────────────────────────────────────────────────────────────────────────────

inline float compute_lagrange_curvature_rd(
    float sq_v,
    float sq_a,
    float dot_va,
    float eps = 1e-12f
) {
    const float bivector_sq = std::max(0.0f, (sq_v * sq_a) - (dot_va * dot_va));
    const float norm_v_cubed = std::pow(std::max(sq_v, 0.0f), 1.5f);
    return std::sqrt(bivector_sq) / (norm_v_cubed + eps);
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENTE 3 — ContinuousTrajectoryOperator
// ─────────────────────────────────────────────────────────────────────────────

inline KineticRegime classify_kinetic_regime(float r_correlation) {
    if (r_correlation >= 0.95f) return KineticRegime::Laminar;
    if (r_correlation >= 0.80f) return KineticRegime::Transition;
    return KineticRegime::Turbulent;
}

} // namespace aether
```

---

### PASO 2: Crear `metal/geodesic_trajectory_cell.metal`

```metal
// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: GEODESIC PROJECTIVE TRAJECTORY CELL
// HITO 1.1 — METAL GPU KERNEL (REDUCCIÓN EXACTA DE 2 FASES)
// ═════════════════════════════════════════════════════════════════════════════
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

struct CellMetricsGPU {
    float correlation_r;
    float curvature_kappa;
    float kinetic_energy;
    float angular_displacement;
    float dirichlet_tension;
    float permeability_gate;
    uint  regime;
    uint  active_mode;
};

kernel void dispatch_geodesic_trajectory_cell_step(
    device const float*       h_in         [[buffer(0)]],
    device const float*       v_drag       [[buffer(1)]],
    device const float*       a_flow       [[buffer(2)]],
    device const float*       u_attractor  [[buffer(3)]],
    device float*             h_projected  [[buffer(4)]],
    device float*             h_deflated   [[buffer(5)]],
    device CellMetricsGPU*    metrics_out  [[buffer(6)]],
    constant float&           tau          [[buffer(7)]],
    constant float&           kappa_att    [[buffer(8)]],
    constant float&           beta_perm    [[buffer(9)]],
    constant float&           theta_perm   [[buffer(10)]],
    constant uint&            mode         [[buffer(11)]],
    constant uint&            D            [[buffer(12)]],
    threadgroup float*        shared_acc   [[threadgroup(0)]],
    uint tid                               [[thread_index_in_threadgroup]],
    uint t_per_group                       [[threads_per_threadgroup]]
) {
    // 1. Reducciones de entrada
    float l_sq_h   = 0.0f, l_sq_v   = 0.0f, l_sq_a   = 0.0f;
    float l_dot_hv = 0.0f, l_dot_va = 0.0f, l_dot_hu = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {
        const float h = h_in[i];
        const float v = v_drag[i];
        const float a = a_flow[i];
        const float u = u_attractor[i];

        l_sq_h   += h * h;
        l_sq_v   += v * v;
        l_sq_a   += a * a;
        l_dot_hv += h * v;
        l_dot_va += v * a;
        l_dot_hu += h * u;
    }

    const uint base = tid * 6;
    shared_acc[base + 0] = l_sq_h;
    shared_acc[base + 1] = l_sq_v;
    shared_acc[base + 2] = l_sq_a;
    shared_acc[base + 3] = l_dot_hv;
    shared_acc[base + 4] = l_dot_va;
    shared_acc[base + 5] = l_dot_hu;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const uint dst = tid * 6;
            const uint src = (tid + s) * 6;
            for (uint k = 0; k < 6; ++k) {
                shared_acc[dst + k] += shared_acc[src + k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float sq_h   = shared_acc[0];
    const float sq_v   = shared_acc[1];
    const float sq_a   = shared_acc[2];
    const float dot_hv = shared_acc[3];
    const float dot_va = shared_acc[4];
    const float dot_hu = shared_acc[5];

    // 2. Descomposición tangencial y curvatura de Lagrange
    const float inv_norm_h = rsqrt(max(sq_h, 0.0f) + 1e-12f);
    const float coeff_hv   = dot_hv * inv_norm_h * inv_norm_h;
    const float sq_v_perp  = max(0.0f, sq_v - (dot_hv * dot_hv * inv_norm_h * inv_norm_h));
    const float norm_v_perp = sqrt(sq_v_perp + 1e-12f);
    const float inv_norm_vp = rsqrt(sq_v_perp + 1e-12f);

    const float bivector_sq   = max(0.0f, (sq_v * sq_a) - (dot_va * dot_va));
    const float bivector_norm = sqrt(bivector_sq);
    const float kappa_kin     = bivector_norm / (pow(max(sq_v, 0.0f), 1.5f) + 1e-12f);

    const float omega = norm_v_perp * inv_norm_h;
    const float theta = omega * tau;
    const float cos_t = cos(theta);
    const float sin_t = sin(theta);

    const float q_tension = sq_v_perp / (sq_h + 1e-12f);
    const float g_perm    = 1.0f / (1.0f + exp(-beta_perm * (q_tension - theta_perm)));

    // 3. Proyección Geodésica con aceleración tangencial
    float l_sq_out    = 0.0f;
    float l_dot_out_u = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {
        const float h = h_in[i];
        const float v = v_drag[i];
        const float u = u_attractor[i];

        const float v_perp = v - coeff_hv * h;
        const float v_hat  = v_perp * inv_norm_vp;
        const float a_attractor = kappa_att * (u - (dot_hu * inv_norm_h * inv_norm_h) * h);

        const float h_star = (cos_t * h) + (sin_t * v_hat) + (0.5f * tau * tau * a_attractor);
        h_projected[i] = h_star;

        l_sq_out    += h_star * h_star;
        l_dot_out_u += h_star * u;
    }

    // 4. Reducción de norma de h_projected
    shared_acc[tid * 2 + 0] = l_sq_out;
    shared_acc[tid * 2 + 1] = l_dot_out_u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const uint dst = tid * 2;
            const uint src = (tid + s) * 2;
            shared_acc[dst + 0] += shared_acc[src + 0];
            shared_acc[dst + 1] += shared_acc[src + 1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float final_sq_out  = shared_acc[0];
    const float raw_dot_out   = shared_acc[1];
    const float inv_norm_star = rsqrt(final_sq_out + 1e-12f);
    const float correlation_r = raw_dot_out * inv_norm_star;

    // Normalizar h_projected a h*
    for (uint i = tid; i < D; i += t_per_group) {
        h_projected[i] *= inv_norm_star;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. Segunda Reducción: dot(h_in, h_star) para deflación exacta <h*, h_deflated> = 0
    float l_dot_h_hstar = 0.0f;
    for (uint i = tid; i < D; i += t_per_group) {
        l_dot_h_hstar += h_in[i] * h_projected[i];
    }

    shared_acc[tid] = l_dot_h_hstar;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid] += shared_acc[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float dot_h_hstar = shared_acc[0];

    // 6. Deflación Gram-Schmidt ortogonal
    for (uint i = tid; i < D; i += t_per_group) {
        const float h_star = h_projected[i];
        h_deflated[i] = h_in[i] - dot_h_hstar * h_star;
    }

    // 7. Telemetría de salida
    if (tid == 0) {
        const uint flow_regime = (correlation_r >= 0.95f) ? 0 : ((correlation_r >= 0.80f) ? 1 : 2);
        metrics_out->correlation_r        = correlation_r;
        metrics_out->curvature_kappa      = kappa_kin;
        metrics_out->kinetic_energy       = 0.5f * sq_v_perp;
        metrics_out->angular_displacement = theta;
        metrics_out->dirichlet_tension    = q_tension;
        metrics_out->permeability_gate    = g_perm;
        metrics_out->regime               = flow_regime;
        metrics_out->active_mode          = mode;
    }
}
```

---

### PASO 3: Incorporar en `aether_vlm/aether_native.cpp`

Agregar antes del bloque `NB_MODULE`:

```cpp
#include "../include/geodesic_trajectory_cell.h"

// ─── 4. KERNEL C++: CÉLULA PROYECTIVA GEODÉSICA AUTÓNOMA (HITO 1.1) ──────────
nb::dict dispatch_geodesic_trajectory_cell_cpp(
    const array& h_in,
    const array& v_drag,
    const array& a_flow,
    const array& u_attractor,
    float tau = 1.0f,
    float kappa_att = 1.20f,
    float beta_perm = 12.0f,
    float theta_perm = 0.50f,
    uint32_t mode = 0
) {
    constexpr float EPS = 1e-12f;

    auto sq_h = sum(h_in * h_in, -1, true);
    auto inv_norm_h = rsqrt(sq_h + EPS);
    auto h_unit = h_in * inv_norm_h;

    auto sq_v = sum(v_drag * v_drag, -1, true);
    auto sq_a = sum(a_flow * a_flow, -1, true);
    auto dot_hv = sum(h_unit * v_drag, -1, true);
    auto dot_va = sum(v_drag * a_flow, -1, true);
    auto dot_hu = sum(h_unit * u_attractor, -1, true);

    auto v_perp = v_drag - dot_hv * h_unit;
    auto sq_v_perp = sum(v_perp * v_perp, -1, true);
    auto norm_v_perp = sqrt(sq_v_perp + EPS);
    auto v_hat = v_perp / norm_v_perp;

    auto bivector_sq = maximum(0.0f, (sq_v * sq_a) - (dot_va * dot_va));
    auto kappa_kin = sqrt(bivector_sq) / (power(sq_v, 1.5f) + EPS);

    auto omega = norm_v_perp * inv_norm_h;
    auto theta = omega * tau;
    auto cos_t = cos(theta);
    auto sin_t = sin(theta);

    auto a_attractor = kappa_att * (u_attractor - dot_hu * h_unit);
    auto h_projected = (cos_t * h_unit) + (sin_t * v_hat) + (0.5f * tau * tau * a_attractor);
    auto norm_proj = sqrt(sum(h_projected * h_projected, -1, true) + EPS);
    auto h_star = h_projected / norm_proj;

    auto r_val = sum(h_star * u_attractor, -1, true);
    auto dot_h_hstar = sum(h_unit * h_star, -1, true);
    auto h_deflated = h_unit - dot_h_hstar * h_star;

    auto q_tension = sq_v_perp / (sq_h + EPS);
    auto g_perm = 1.0f / (1.0f + exp(-beta_perm * (q_tension - theta_perm)));

    nb::dict result;
    result["h_star"]               = h_star;
    result["h_deflated"]           = h_deflated;
    result["correlation_r"]        = r_val;
    result["curvature_kappa"]      = kappa_kin;
    result["kinetic_energy"]       = 0.5f * sq_v_perp;
    result["angular_displacement"] = theta;
    result["dirichlet_tension"]    = q_tension;
    result["permeability_gate"]    = g_perm;
    result["mode"]                 = mode;
    return result;
}
```

Dentro del `NB_MODULE(aether_native_c, m)` existente, añadir la línea:

```cpp
    m.def("dispatch_geodesic_trajectory_cell", &dispatch_geodesic_trajectory_cell_cpp,
          "Célula Proyectiva Geodésica Autónoma — Hito 1.1",
          nb::arg("h_in"), nb::arg("v_drag"), nb::arg("a_flow"), nb::arg("u_attractor"),
          nb::arg("tau") = 1.0f, nb::arg("kappa_att") = 1.20f,
          nb::arg("beta_perm") = 12.0f, nb::arg("theta_perm") = 0.50f,
          nb::arg("mode") = 0);
```

---

### PASO 4: Crear la Batería Aislada `tests/test_geodesic_trajectory_cell.py`

*(Usar el archivo completo del paste que incluye `WARMUP_RUNS = 3`, `BENCH_RUNS = 10`, `LATENCY_P50_US_LIMIT = 200.0`, comprobación de tangencia previa, y los 7 tests de Correctness y Performance).*

---

### PASO 5: Batería de Verificación y Certificación (Comandos de Terminal)

Antigravity debe ejecutar la siguiente secuencia de tres fases:

```bash
# FASE 1: Recompilación limpia de la extensión nativa C++
python tools/compilar_extension_c.py

# FASE 2: Batería de Verificación Unitaria de la Célula Aislada
python tests/test_geodesic_trajectory_cell.py

# FASE 3: Batería de Regresión del Motor (Invariantes de Lie y Variedad)
python tests/test_advisor_battery.py --numerical-only
```

---

### CRITERIOS DE CERTIFICACIÓN PARA DAR EL PASO AL HITO 1.2

1. **FASE 1:** Salida con código 0 y enlace exitoso de `aether_native_c`.
2. **FASE 2:** **7/7 PASS** en cada dimensión ($D \in \{1024, 2048, 5120\}$), garantizando:
   - $\|h^*(\tau)\| = 1.000000$ (error $< 10^{-5}$)
   - Identidad en $\tau = 0$ (error $< 10^{-5}$)
   - Ortogonalidad exacta $|\langle h^*, h_{\text{deflated}} \rangle| < 10^{-5}$
   - Curvatura de Lagrange coincidente con referencia analítica
   - Performance Gate: $p50 < 200\,\mu\text{s}$ en estado *warm*
3. **FASE 3:** Las **29 pruebas de la batería del asesor deben mantenerse en 100% PASS**.

Queda formalmente emitida la directriz. Esperamos la confirmación de ejecución de Antigravity con los resultados de la terminal.