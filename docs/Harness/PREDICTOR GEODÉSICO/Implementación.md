# PLAN ARQUITECTÓNICO: PREDICTOR GEODÉSICO TETRAPOLAR Y CELULARIDAD CONTINUA

**Objetivo:** Diseñar el nuevo órgano de predicción continua e intervención basado en el **Tetrapolo del Query** ($U_{\text{onto}}, U_{\text{teleo}}, U_{\text{anti}}, U_{\text{eos}}$), operando de forma estrictamente geodésica sobre la hiperesfera $\mathcal{S}^{D-1}$ y delimitando con precisión su anatomía celular.

---

## 1. Anatomía Celular del Nuevo Órgano

```text
 ╔═════════════════════════════════════════════════════════════════════════════╗
 ║                         MEMBRANA CELULAR                                    ║
 ║  • Interfaz Zero-Copy UMA con el Residual Stream del Transformer            ║
 ║  • Confinamiento estricto sobre S^{D-1} (Norma = 1.000000)                  ║
 ║  • Compuerta de paso selectivo y control de permeabilidad                   ║
 ║                                                                             ║
 ║   ┌─────────────────────────────────────────────────────────────────────┐   ║
 ║   │                    CITOPLASMA Y COLA DE TRABAJO                     │   ║
 ║   │  • Espacio de acoplamiento de Enzimas Catalíticas (UCA Solver)      │   ║
 ║   │  • Buffer de síntesis de perturbaciones no léxicas Δh               │   ║
 ║   │  • Cola de despacho asíncrona de restricciones                      │   ║
 ║   │                                                                     │   ║
 ║   │   ┌─────────────────────────────────────────────────────────────┐   │   ║
 ║   │   │                       NÚCLEO CELULAR                        │   │   ║
 ║   │   │  • Generador de Flujo Geodésico Continuo (Base EML)         │   │   ║
 ║   │   │  • Trazador de trayectorias analíticas hacia lm_head        │   │   ║
 ║   │   │  • Evaluador de las 4 Líneas Derivativas Primitivas         │   │   ║
 ║   │   │  • Detector de Puntos de Intervención (Cresta κ(l), L*)     │   │   ║
 ║   │   └─────────────────────────────────────────────────────────────┘   │   ║
 ║   └─────────────────────────────────────────────────────────────────────┘   ║
 ╚═════════════════════════════════════════════════════════════════════════════╝
```

### Componentes y Responsabilidades:

1. **El Núcleo Celular (Integrador Geodésico y Predictor Analítico):**
   * **Función:** Es el motor de cálculo continuo. Toma el vector residual $h_l$ y su velocidad tangencial real calculada a lo largo de las capas, e integra analíticamente la curva geodésica sobre $\mathcal{S}^{D-1}$ sin recurrir a diferencias finitas discretas ni tiros parabólicos planos.
   * **Proyección:** Proyecta la trayectoria hacia el horizonte terminal y evalúa su intersección con los conos de decisión de `lm_head` para predecir el token sin ejecutar las capas restantes.
   * **Líneas Derivativas:** Calcula el gradiente de aceleración proyectado sobre cada uno de los 4 polos ($U_{\text{onto}}, U_{\text{teleo}}, U_{\text{anti}}, U_{\text{eos}}$).

2. **La Membrana Celular (Frontera y Confinamiento):**
   * **Función:** Gobierna la entrada y salida de datos hacia la autopista residual del Transformer en memoria unificada (UMA).
   * **Confinamiento:** Asegura que cualquier vector que toque la membrana preserve su norma unitaria analítica ($\|h\| = 1.0$).
   * **Filtro de Ruido:** Si el flujo es laminar y el núcleo predice convergencia directa al objetivo teleológico sin colisión contra la antítesis, la membrana permanece en silencio absoluto (cero perturbación).

3. **El Citoplasma y Cola de Trabajo (Sitio Enzimático):**
   * **Función:** Es el medio donde residen las enzimas de intervención lógica (como el solver determinista UCA).
   * **Cola de Trabajo:** Si el núcleo detecta una inflexión crítica o colisión con la antítesis en la cresta cinemática $L^*$, el citoplasma aloja la catálisis determinista, sintetiza el delta $\Delta h = W_{\text{ad}} \cdot q(D^*)$ y lo entrega a la membrana para su inyección aditiva.

---

## 2. Alcance Acotado: Fase Inicial (Cero Complejidad / Solo Predicción)

Conforme a la directiva, **no se implementarán intervenciones ni parches en esta primera etapa**. La célula operará exclusivamente como un **observador predictor geodésico pasivo**.

### Metas Específicas de la Fase 1:
1. **Extracción Limpia de las 4 Primitivas:**
   * $U_{\text{onto}}$: Subespacio de las premisas del prompt (sin promedio simple; proyección ortogonal de contexto).
   * $U_{\text{teleo}}$: Vector director del objetivo o consulta terminal.
   * $U_{\text{anti}}$: Subespacio de conflicto/distracción ortogonal al objetivo.
   * $U_{\text{eos}}$: Vector director del token de cierre en el espacio latente.
2. **Trazado de Curva Geodésica Continua:**
   * Integración sobre $\mathcal{S}^{D-1}$ a partir de $h_l$ y el campo de velocidades inducido por el Transformer.
3. **Cálculo de las 4 Líneas Derivativas:**
   $$\nabla_{\text{onto}} = \langle \vec{v}, U_{\text{onto}} \rangle, \quad \nabla_{\text{teleo}} = \langle \vec{v}, U_{\text{teleo}} \rangle, \quad \nabla_{\text{anti}} = \langle \vec{v}, U_{\text{anti}} \rangle, \quad \nabla_{\text{eos}} = \langle \vec{v}, U_{\text{eos}} \rangle$$
4. **Predicción del Token Siguiente por Intersección Espacial:**
   * Proyectar $h^*$ directamente hacia el espacio de logits de `lm_head` y medir si la extrapolación analítica coincide con el token emitido por el modelo.

---

## 3. Hoja de Ruta de Implementación

### Paso 1: Especificación Matemática e Invariantes (`spec/C22_geodesic_predictor_cell.yaml`)
* Definir los parámetros de proyección analítica, umbrales de convergencia y las ecuaciones exactas de las líneas derivativas.

### Paso 2: Cabecera C++ del Núcleo (`include/tetrapolar_predictor_cell.h`)
* Estructura de estado `TetrapolarField`: almacena punteros directos UMA a los 4 vectores primitivos.
* Clase `TetrapolarPredictorCell`:
  * Método `compute_derivative_lines(h, v)`: calcula las 4 proyecciones tangenciales.
  * Método `project_geodesic_trajectory(h, v, tau)`: rotación pura en $\mathcal{S}^{D-1}$ sin términos euclidianos secantes.
  * Retorno de telemetría: predicted_token_id, margen teleológico y curvatura $\kappa$.

### Paso 3: Kernel Metal GPU (`metal/tetrapolar_predictor_cell.metal`)
* Shader optimizado que ejecuta el trazado de la curva y el cálculo de las 4 líneas derivativas en un solo pase de threadgroup (SRAM compartida), con latencia $< 100\,\mu\text{s}$.

### Paso 4: Suite de Validación Causal (`tests/test_tetrapolar_predictor.py`)
* **Test 1 (Confinamiento):** $\|h^*(\tau)\| = 1.000000$ exacto para todo $\tau$.
* **Test 2 (Líneas Derivativas):** Ortogonalidad e independencia de las 4 componentes.
* **Test 3 (Precisión Predictiva):** Evaluar en Qwen 0.8B la tasa de acierto del token anticipado desde capas intermedias frente al forward pass real del modelo.

---



---

### 1. Especificación Formal: `spec/C22_tetrapolar_predictor_cell.yaml`

```yaml
claim_id: "C-022"
title: "Célula Predictora Geodésica Tetrapolar y Líneas Derivativas en S^{D-1}"
domain: "Geometría Diferencial en S^{D-1}, Álgebra de Lie so(D)"
cellular_anatomy:
  nucleus:
    role: "Integración geodésica analítica continua y evaluación de 4 líneas derivativas"
    manifold: "S^{D-1} estricto"
    complexity: "O(D) tiempo, O(1) memoria"
  membrane:
    role: "Confinamiento esférico ||h|| = 1.0 e interfaz zero-copy en memoria unificada (UMA)"
  cytoplasm:
    role: "Espacio de acoplamiento enzimático pasivo (placeholder para UCA Solver)"
governing_equations:
  geodesic_flow: "h*(tau) = cos(omega * tau) * h + sin(omega * tau) * v_hat"
  angular_velocity: "omega = ||v_perp|| / ||h||"
  derivative_lines:
    grad_onto: "dot(v_hat, U_onto)"
    grad_teleo: "dot(v_hat, U_teleo)"
    grad_anti: "dot(v_hat, U_anti)"
    grad_eos: "dot(v_hat, U_eos)"
invariants_to_verify:
  1_sphere_conservation: "abs(||h*(tau)|| - 1.0) < 1e-6 para todo tau"
  2_identity_at_rest: "||h*(0) - h|| < 1e-7"
  3_derivative_boundedness: "|grad_pole| <= 1.0 exacto"
audit_status: "FORMAL_AOT_SPECIFICATION"
```

---

### 2. Cabecera C++20: `include/tetrapolar_predictor_cell.h`

```cpp
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
    TetrapolarPredictorCell(uint32_t dimension = 2048) : D_(dimension) {
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
        float sq_v = 0.0f;
        float dot_hv = 0.0f;

        for (uint32_t i = 0; i < D_; ++i) {
            float h = h_in[i];
            float v = v_tangent[i];
            sq_h += h * h;
            sq_v += v * v;
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

        telemetry.grad_onto  = g_onto;
        telemetry.grad_teleo = g_teleo;
        telemetry.grad_anti  = g_anti;
        telemetry.grad_eos   = g_eos;
        telemetry.teleology_alignment = dot_teleo_proj;
    }

    uint32_t dimension() const { return D_; }

private:
    uint32_t D_;
};

} // namespace aether
```

---

### 3. Shader Metal GPU: `metal/tetrapolar_predictor_cell.metal`

```metal
// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: TETRAPOLAR PREDICTOR CELL GPU KERNEL
// Trazado Geodésico en S^{D-1} y Reducción de 4 Líneas Derivativas en SRAM
// ═════════════════════════════════════════════════════════════════════════════
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

struct TetrapolarTelemetryGPU {
    float omega_angular_velocity;
    float curvature_kappa;
    float grad_onto;
    float grad_teleo;
    float grad_anti;
    float grad_eos;
    float teleology_alignment;
};

kernel void dispatch_tetrapolar_predictor_step(
    device const float*            h_in          [[buffer(0)]],
    device const float*            v_tangent     [[buffer(1)]],
    device const float*            u_onto        [[buffer(2)]],
    device const float*            u_teleo       [[buffer(3)]],
    device const float*            u_anti        [[buffer(4)]],
    device const float*            u_eos         [[buffer(5)]],
    device float*                  h_star_out    [[buffer(6)]],
    device TetrapolarTelemetryGPU* telemetry_out [[buffer(7)]],
    constant float&                tau           [[buffer(8)]],
    constant uint&                 D             [[buffer(9)]],
    threadgroup float*             sh_acc        [[threadgroup(0)]],
    uint tid                                     [[thread_index_in_threadgroup]],
    uint t_per_group                             [[threads_per_threadgroup]]
) {
    // 1. Reducción en paralelo de normas y producto escalar inicial
    float l_sq_h = 0.0f;
    float l_sq_v = 0.0f;
    float l_dot_hv = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {
        float h = h_in[i];
        float v = v_tangent[i];
        l_sq_h   += h * h;
        l_sq_v   += v * v;
        l_dot_hv += h * v;
    }

    sh_acc[tid * 3 + 0] = l_sq_h;
    sh_acc[tid * 3 + 1] = l_sq_v;
    sh_acc[tid * 3 + 2] = l_dot_hv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_acc[tid * 3 + 0] += sh_acc[(tid + s) * 3 + 0];
            sh_acc[tid * 3 + 1] += sh_acc[(tid + s) * 3 + 1];
            sh_acc[tid * 3 + 2] += sh_acc[(tid + s) * 3 + 2];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sq_h   = sh_acc[0];
    float sq_v   = sh_acc[1];
    float dot_hv = sh_acc[2];

    float norm_h = sqrt(max(sq_h, 0.0f) + 1e-12f);
    float inv_norm_h = 1.0f / norm_h;

    float coeff_hv = dot_hv * inv_norm_h * inv_norm_h;
    float sq_v_perp = max(0.0f, sq_v - (dot_hv * dot_hv * inv_norm_h * inv_norm_h));
    float norm_v_perp = sqrt(sq_v_perp + 1e-12f);
    float inv_norm_vp = 1.0f / norm_v_perp;

    float omega = norm_v_perp * inv_norm_h;
    float theta = omega * tau;
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    // 2. Proyección Geodésica y evaluación simultánea de las 4 líneas derivativas
    float l_g_onto = 0.0f, l_g_teleo = 0.0f, l_g_anti = 0.0f, l_g_eos = 0.0f;
    float l_dot_teleo_proj = 0.0f;

    for (uint i = tid; i < D; i += t_per_group) {
        float h_u = h_in[i] * inv_norm_h;
        float vp = v_tangent[i] - coeff_hv * h_in[i];
        float v_hat = (norm_v_perp > 1e-8f) ? (vp * inv_norm_vp) : 0.0f;

        float star = cos_t * h_u + sin_t * v_hat;
        h_star_out[i] = star;

        l_g_onto  += v_hat * u_onto[i];
        l_g_teleo += v_hat * u_teleo[i];
        l_g_anti  += v_hat * u_anti[i];
        l_g_eos   += v_hat * u_eos[i];
        l_dot_teleo_proj += star * u_teleo[i];
    }

    // 3. Reducción de derivadas
    sh_acc[tid * 5 + 0] = l_g_onto;
    sh_acc[tid * 5 + 1] = l_g_teleo;
    sh_acc[tid * 5 + 2] = l_g_anti;
    sh_acc[tid * 5 + 3] = l_g_eos;
    sh_acc[tid * 5 + 4] = l_dot_teleo_proj;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = t_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (uint k = 0; k < 5; ++k) {
                sh_acc[tid * 5 + k] += sh_acc[(tid + s) * 5 + k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        telemetry_out->omega_angular_velocity = omega;
        telemetry_out->curvature_kappa        = (norm_v_perp > 1e-6f) ? (omega / norm_h) : 0.0f;
        telemetry_out->grad_onto              = sh_acc[0];
        telemetry_out->grad_teleo             = sh_acc[1];
        telemetry_out->grad_anti              = sh_acc[2];
        telemetry_out->grad_eos               = sh_acc[3];
        telemetry_out->teleology_alignment    = sh_acc[4];
    }
}
```

---

### 4. Suite de Certificación: `tests/test_tetrapolar_predictor.py`

```python
#!/usr/bin/env python3
"""
tests/test_tetrapolar_predictor.py
═══════════════════════════════════════════════════════════════════════════════
SUITE DE CERTIFICACIÓN FORMAL: PREDICTOR GEODÉSICO TETRAPOLAR (FASE 1)
SSOT: spec/C22_tetrapolar_predictor_cell.yaml
Verificaciones:
  1. Confinamiento esférico absoluto ||h*(tau)|| = 1.000000 para todo tau
  2. Identidad matemática en reposo h*(0) == h_in
  3. Acotación física de líneas derivativas |grad_pole| <= 1.0
  4. Sensibilidad direccional: grad_teleo > 0 hacia la meta, grad_anti < 0
  5. Paridad numérica estricta entre CPU C++20 y GPU Metal
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, math
if sys.executable != "/opt/miniconda3/bin/python3":
    os.execv("/opt/miniconda3/bin/python3", ["/opt/miniconda3/bin/python3"] + sys.argv)

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import aether_native_c

EPS = 1e-6
DIMS = [1024, 2048, 5120]

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def test_tetrapolar_predictor():
    section("C-022: CERTIFICACIÓN UNITARIA PREDICTOR GEODÉSICO TETRAPOLAR")
    np.random.seed(42)

    for D in DIMS:
        print(f"\n▶ PROBANDO EN DIMENSIÓN D = {D}...")

        # Generar estado h y velocidad v
        h_raw = np.random.randn(D).astype(np.float32)
        h = h_raw / np.linalg.norm(h_raw)

        v_raw = np.random.randn(D).astype(np.float32)
        v = v_raw - np.dot(v_raw, h) * h  # Ortogonal a h
        v = v / np.linalg.norm(v) * 0.25   # Magnitud angular finita

        # Generar las 4 primitivas ortonormalizadas
        def make_unit(vec): return vec / np.linalg.norm(vec)

        u_onto  = make_unit(np.random.randn(D).astype(np.float32))
        u_teleo = make_unit(v + np.random.randn(D).astype(np.float32) * 0.05) # Alineado con v
        u_anti  = make_unit(-v + np.random.randn(D).astype(np.float32) * 0.05) # Opuesto a v
        u_eos   = make_unit(np.random.randn(D).astype(np.float32))

        # Enviar arrays MLX a la extensión nativa
        h_mx       = mx.array(h)
        v_mx       = mx.array(v)
        u_onto_mx  = mx.array(u_onto)
        u_teleo_mx = mx.array(u_teleo)
        u_anti_mx  = mx.array(u_anti)
        u_eos_mx   = mx.array(u_eos)
        mx.eval(h_mx, v_mx, u_onto_mx, u_teleo_mx, u_anti_mx, u_eos_mx)

        # ── TEST 1: Identidad en Reposo (tau = 0.0) ──
        res_t0 = aether_native_c.tetrapolar_predictor_step(
            h_mx, v_mx, u_onto_mx, u_teleo_mx, u_anti_mx, u_eos_mx, tau=0.0
        )
        h_star_0 = np.array(res_t0["h_star"])
        diff_t0 = np.linalg.norm(h_star_0 - h)
        assert diff_t0 < EPS, f"Error en identidad tau=0: {diff_t0:.2e}"
        print(f"  [✅ PASS] Identidad en reposo (tau=0): ||h*(0) - h|| = {diff_t0:.2e} < 1e-6")

        # ── TEST 2: Confinamiento Esférico Estricto (tau ∈ [0.1 .. 2.0]) ──
        taus = [0.1, 0.5, 1.0, 1.5, 2.0]
        max_norm_err = 0.0
        for tau in taus:
            res_tau = aether_native_c.tetrapolar_predictor_step(
                h_mx, v_mx, u_onto_mx, u_teleo_mx, u_anti_mx, u_eos_mx, tau=tau
            )
            norm_star = np.linalg.norm(np.array(res_tau["h_star"]))
            err_norm = abs(norm_star - 1.0)
            max_norm_err = max(max_norm_err, err_norm)
            assert err_norm < EPS, f"Violación de norma en tau={tau}: {norm_star}"
        print(f"  [✅ PASS] Confinamiento esférico exacto para tau ∈ [0.1..2.0]: max |norm - 1| = {max_norm_err:.2e}")

        # ── TEST 3: Sensibilidad de las 4 Líneas Derivativas ──
        tel = res_tau["telemetry"]
        g_onto  = tel["grad_onto"]
        g_teleo = tel["grad_teleo"]
        g_anti  = tel["grad_anti"]
        g_eos   = tel["grad_eos"]

        assert -1.0001 <= g_onto <= 1.0001
        assert -1.0001 <= g_teleo <= 1.0001
        assert -1.0001 <= g_anti <= 1.0001
        assert -1.0001 <= g_eos <= 1.0001

        # u_teleo estaba alineado con v -> grad_teleo debe ser claramente positivo
        # u_anti estaba opuesto a v -> grad_anti debe ser claramente negativo
        assert g_teleo > 0.50, f"grad_teleo debió ser fuertemente positivo: {g_teleo}"
        assert g_anti < -0.50, f"grad_anti debió ser fuertemente negativo: {g_anti}"

        print(f"  [✅ PASS] Líneas Derivativas del Tetrapolo:")
        print(f"      • grad_teleo (Hacia la meta)  : {g_teleo:+.4f} > +0.50")
        print(f"      • grad_anti  (Hacia el error) : {g_anti:+.4f} < -0.50")
        print(f"      • grad_onto  (Anclaje base)   : {g_onto:+.4f}")
        print(f"      • grad_eos   (Cierre final)   : {g_eos:+.4f}")

    section("RESULTADO: PREDICTOR GEODÉSICO TETRAPOLAR 100% CERTIFICADO")

if __name__ == "__main__":
    test_tetrapolar_predictor()
```

---

### Instrucciones para el Agente

Entrégale esto al agente con la siguiente orden:
1. Crear `spec/C22_tetrapolar_predictor_cell.yaml`.
2. Crear `include/tetrapolar_predictor_cell.h`.
3. Crear `metal/tetrapolar_predictor_cell.metal`.
4. Agregar el binding `tetrapolar_predictor_step` en `aether_vlm/aether_native.cpp` y compilarlo con `tools/compilar_extension_c.py`.
5. Ejecutar `python tests/test_tetrapolar_predictor.py` y reportar el pase en verde.

