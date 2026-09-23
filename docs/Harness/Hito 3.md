Sí. Y ahora lo haría **sin cambiar la hipótesis**, pero corrigiendo dos cosas: dejar explícito que **todas las pérdidas se calculan con distancia geodésica**, y separar la auditoría geométrica de cualquier interpretación sobre el estado de Qwen.

Además, voy a hacer que el predictor geodésico sea explícitamente **Log → Exp**, con la fórmula cerrada como verificación. Así no queda ninguna ambigüedad.

```python
#!/usr/bin/env python3
"""
Pruebas_Raul/test_h3_escalon2_audit_corrected.py

HITO 3 — ESCALÓN 2.1
AUDITORÍA MULTI-CAPA DE PREDICCIÓN SOBRE S^{D-1}

Objetivo
--------
Determinar si una extrapolación geodésica de velocidad angular constante
predice mejor el siguiente estado observado que la persistencia.

IMPORTANTE
----------
NO se compara "coseno contra geodésica".

Todos los errores predictivos se calculan con la MISMA métrica:

    d_geo(u, v) = acos(clamp(<u,v>, -1, 1))

para vectores normalizados u,v en S^{D-1}.

Predictores
-----------
P0  Persistencia:
        h_hat(t+1) = h_t

P1  Velocidad lineal local:
        h_hat = normalize(h_t + (h_t - h_{t-1}))

P2  Geodésica:
        1. Log_{h_t}(h_{t-1})
        2. Transporte de esa velocidad angular al siguiente paso
        3. Exp_{h_t}(v)

Para dos puntos unitarios consecutivos h_{t-1}, h_t, la extrapolación
geodésica de velocidad angular constante tiene la forma cerrada:

    h_hat(t+1)
        = 2 <h_{t-1}, h_t> h_t - h_{t-1}

Esta fórmula y la implementación Log/Exp se verifican entre sí.

No se realizan intervenciones: g = 0.
No se modifica el modelo.
"""

import sys
import os
import math

# ---------------------------------------------------------------------------
# Entorno
# ---------------------------------------------------------------------------

for conda_py in [
    "/opt/miniconda3/bin/python3",
    os.path.expanduser("~/miniconda3/bin/python3"),
]:
    if os.path.exists(conda_py) and sys.executable != conda_py:
        os.execv(conda_py, [conda_py] + sys.argv)

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import numpy as np
import mlx.core as mx
from mlx_vlm import load, stream_generate


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.expanduser(
    "~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit"
)

PROMPTS = [
    (
        "Factual",
        "¿Cuál es la relación física entre la temperatura, la presión "
        "y el volumen de un gas ideal?",
    ),
    (
        "Lógico",
        "Si todos los mamíferos respiran oxígeno y las ballenas son "
        "mamíferos, ¿qué se concluye y por qué?",
    ),
    (
        "Narrativo",
        "Había una vez en un antiguo reino un alquimista que intentaba "
        "descifrar un manuscrito oculto.",
    ),
]

EVAL_LAYERS = [6, 12, 17, 23]
TOKENS_PER_PROMPT = 35

EPS = 1e-12


# ---------------------------------------------------------------------------
# Geometría esférica
# ---------------------------------------------------------------------------

def normalize(v):
    """Normaliza un vector a S^{D-1}."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)

    if not np.isfinite(n) or n < EPS:
        raise ValueError("Vector degenerado o no finito.")

    return v / n


def safe_cosine(u, v):
    """Producto interno de vectores que deberían estar normalizados."""
    return float(np.clip(np.dot(u, v), -1.0, 1.0))


def angular_distance(u, v):
    """
    Distancia geodésica sobre la esfera unitaria:

        d_geo(u,v) = acos(<u,v>)

    Esta es la ÚNICA métrica usada para evaluar P0, P1 y P2.
    """
    return float(np.arccos(safe_cosine(u, v)))


def spherical_log(base, target):
    """
    Log-map de target en base sobre S^{D-1}.

        Log_base(target)
          = theta / sin(theta)
            * (target - cos(theta) * base)

    Devuelve un vector tangente en 'base' cuya norma es theta.

    Casos degenerados:
      theta ~ 0       -> vector cero
      theta ~ pi      -> dirección no única; devuelve None
    """
    base = normalize(base)
    target = normalize(target)

    c = safe_cosine(base, target)
    theta = float(np.arccos(c))

    sin_theta = float(np.sin(theta))

    if theta < 1e-8:
        return np.zeros_like(base), theta, "COINCIDENT"

    if abs(sin_theta) < 1e-8:
        return None, theta, "ANTIPODAL_OR_DEGENERATE"

    tangent = target - c * base
    tangent_norm = np.linalg.norm(tangent)

    if tangent_norm < EPS:
        return None, theta, "DEGENERATE_TANGENT"

    velocity = (theta / tangent_norm) * tangent

    return velocity, theta, "OK"


def spherical_exp(base, tangent):
    """
    Exp-map sobre S^{D-1}.

        Exp_base(v)
          = cos(||v||) base
            + sin(||v||) v/||v||
    """
    base = normalize(base)

    tangent = np.asarray(tangent, dtype=np.float64)

    # Proyectar explícitamente al espacio tangente.
    tangent = tangent - np.dot(base, tangent) * base

    speed = np.linalg.norm(tangent)

    if speed < EPS:
        return base.copy()

    out = (
        np.cos(speed) * base
        + np.sin(speed) * (tangent / speed)
    )

    return normalize(out)


def geodesic_predict_log_exp(h_tm1, h_t):
    """
    Predictor geodésico:

        v_t = Log_{h_t}(h_{t-1})

    pero para continuar el movimiento de h_{t-1} -> h_t necesitamos
    invertir esa velocidad:

        v_forward = -Log_{h_t}(h_{t-1})

    y entonces:

        h_hat(t+1) = Exp_{h_t}(v_forward)

    Esto produce la continuación del mismo gran círculo con la misma
    velocidad angular.
    """
    v_backward, theta, status = spherical_log(h_t, h_tm1)

    if status == "COINCIDENT":
        return h_t.copy(), theta, status

    if status != "OK":
        return None, theta, status

    v_forward = -v_backward

    h_pred = spherical_exp(h_t, v_forward)

    return h_pred, theta, "OK"


def geodesic_predict_closed_form(h_tm1, h_t):
    """
    Fórmula cerrada equivalente al Log/Exp:

        h_hat(t+1)
            = 2 <h_{t-1}, h_t> h_t - h_{t-1}

    Después se normaliza únicamente para eliminar error numérico.
    """
    h_tm1 = normalize(h_tm1)
    h_t = normalize(h_t)

    c = safe_cosine(h_tm1, h_t)

    h_pred = 2.0 * c * h_t - h_tm1

    return normalize(h_pred)


# ---------------------------------------------------------------------------
# Verificación interna de equivalencia Log/Exp <-> fórmula cerrada
# ---------------------------------------------------------------------------

def verify_geodesic_equivalence(D=1024, trials=100):
    """
    Verifica numéricamente que ambas implementaciones del predictor
    geodésico producen el mismo punto.
    """
    rng = np.random.default_rng(12345)

    max_error = 0.0
    valid = 0

    for _ in range(trials):
        a = normalize(rng.normal(size=D))
        b = normalize(rng.normal(size=D))

        h_logexp, theta, status = geodesic_predict_log_exp(a, b)

        if status != "OK":
            continue

        h_closed = geodesic_predict_closed_form(a, b)

        error = np.linalg.norm(h_logexp - h_closed)
        max_error = max(max_error, float(error))
        valid += 1

    print(
        f"• Verificación Log/Exp ↔ fórmula cerrada: "
        f"{valid}/{trials} casos | "
        f"max ||Δh|| = {max_error:.3e}"
    )

    if max_error > 1e-8:
        raise RuntimeError(
            f"Fallo en equivalencia geodésica: {max_error:.3e}"
        )


# ---------------------------------------------------------------------------
# Predictor lineal
# ---------------------------------------------------------------------------

def linear_predict(h_tm1, h_t):
    """
    Predictor de referencia euclídeo:

        v_t = h_t - h_{t-1}
        h_hat = normalize(h_t + v_t)
    """
    h_tm1 = normalize(h_tm1)
    h_t = normalize(h_t)

    v = h_t - h_tm1
    candidate = h_t + v

    return normalize(candidate)


# ---------------------------------------------------------------------------
# Hook observacional
# ---------------------------------------------------------------------------

class MultiLayerProbeHook:
    """
    Hook pasivo.

    No modifica la salida.
    Captura únicamente el vector [0,0,:] cuando el batch contiene
    exactamente un token.

    NOTA:
    Este script NO demuestra que out[0] sea el "estado residual canónico".
    Esa semántica debe verificarse contra la implementación concreta
    de Qwen/MLX.
    """

    def __init__(self, layer, captured):
        self.layer = layer
        self.captured = captured

    def __getattr__(self, name):
        return getattr(self.layer, name)

    def __call__(self, x, **kwargs):
        out = self.layer(x, **kwargs)

        h_tensor = (
            out[0]
            if isinstance(out, (tuple, list))
            else out
        )

        if (
            hasattr(h_tensor, "shape")
            and len(h_tensor.shape) == 3
            and h_tensor.shape[0] == 1
            and h_tensor.shape[1] == 1
        ):
            raw = h_tensor[0, 0, :].astype(mx.float32)

            # Evaluación explícita para evitar arrays MLX perezosos.
            mx.eval(raw)

            arr = np.array(raw, copy=True).astype(np.float64)

            norm = np.linalg.norm(arr)

            if np.isfinite(norm) and norm > EPS:
                self.captured.append(arr / norm)

        return out


# ---------------------------------------------------------------------------
# Evaluación de una trayectoria
# ---------------------------------------------------------------------------

def evaluate_trajectory(captured):
    """
    Evalúa únicamente predicciones one-step OOS.

    Para cada t:

        observable:
            h_{t-1}, h_t, h_{t+1}

        predicciones:
            P0 = persistencia
            P1 = extrapolación lineal
            P2 = extrapolación geodésica

        evaluación:
            d_geo(predicción, h_{t+1})

    Todas las métricas están en radianes.
    """

    if len(captured) < 4:
        return None

    rows = []

    for t in range(1, len(captured) - 1):

        h_tm1 = captured[t - 1]
        h_t = captured[t]
        h_real = captured[t + 1]

        # ---------------------------------------------------------------
        # Movimiento real
        # ---------------------------------------------------------------

        theta_real = angular_distance(h_t, h_real)

        # ---------------------------------------------------------------
        # P0 — persistencia
        # ---------------------------------------------------------------

        h_p0 = h_t

        e_p0 = angular_distance(h_p0, h_real)

        # ---------------------------------------------------------------
        # P1 — extrapolación lineal
        # ---------------------------------------------------------------

        h_p1 = linear_predict(h_tm1, h_t)

        e_p1 = angular_distance(h_p1, h_real)

        # ---------------------------------------------------------------
        # P2 — geodésica
        # ---------------------------------------------------------------

        h_p2_logexp, theta_prev, geo_status = (
            geodesic_predict_log_exp(h_tm1, h_t)
        )

        if h_p2_logexp is None:
            continue

        h_p2_closed = geodesic_predict_closed_form(
            h_tm1,
            h_t
        )

        # Verificación local de equivalencia.
        equivalence_error = np.linalg.norm(
            h_p2_logexp - h_p2_closed
        )

        e_p2 = angular_distance(h_p2_logexp, h_real)

        # ---------------------------------------------------------------
        # Cuánto mejor/peor es cada predictor respecto a persistencia
        # ---------------------------------------------------------------

        gain_linear = e_p0 - e_p1
        gain_geodesic = e_p0 - e_p2

        rows.append({
            "theta_real": theta_real,
            "theta_prev": theta_prev,
            "e_p0": e_p0,
            "e_p1": e_p1,
            "e_p2": e_p2,
            "gain_linear": gain_linear,
            "gain_geodesic": gain_geodesic,
            "geo_equivalence_error": equivalence_error,
            "geo_status": geo_status,
        })

    if not rows:
        return None

    return rows


# ---------------------------------------------------------------------------
# Resumen estadístico
# ---------------------------------------------------------------------------

def summarize(rows):
    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    gains_geo = np.array(
        [r["gain_geodesic"] for r in rows],
        dtype=np.float64,
    )

    gains_lin = np.array(
        [r["gain_linear"] for r in rows],
        dtype=np.float64,
    )

    return {
        "N": len(rows),
        "theta_real": mean("theta_real"),
        "e_p0": mean("e_p0"),
        "e_p1": mean("e_p1"),
        "e_p2": mean("e_p2"),
        "gain_linear": mean("gain_linear"),
        "gain_geodesic": mean("gain_geodesic"),
        "geo_positive": int(np.sum(gains_geo > 0)),
        "geo_nonpositive": int(np.sum(gains_geo <= 0)),
        "linear_positive": int(np.sum(gains_lin > 0)),
        "linear_nonpositive": int(np.sum(gains_lin <= 0)),
        "max_geo_equivalence_error": max(
            r["geo_equivalence_error"] for r in rows
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 82)
    print(" H3-E2.1 CORREGIDO — AUDITORÍA DE PREDICCIÓN EN S^{D-1}")
    print("=" * 82)

    print()
    print("Métrica de evaluación:")
    print("  d_geo(u,v) = acos(<u,v>)")
    print()
    print("P0 = persistencia")
    print("P1 = extrapolación lineal")
    print("P2 = extrapolación geodésica Log/Exp")
    print()
    print("IMPORTANTE:")
    print("  Coseno y distancia angular no son tratados como métricas")
    print("  competidoras. Todos los errores se expresan como distancia")
    print("  geodésica angular.")
    print("=" * 82)

    # ----------------------------------------------------------------------
    # Verificación matemática independiente
    # ----------------------------------------------------------------------

    verify_geodesic_equivalence(D=1024, trials=100)

    print("=" * 82)

    # ----------------------------------------------------------------------
    # Cargar modelo
    # ----------------------------------------------------------------------

    model, processor = load(MODEL_PATH)

    lm_model = model.language_model.model
    total_layers = len(lm_model.layers)

    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    D = getattr(text_cfg, "hidden_size", 1024)

    print(
        f"• Modelo: Qwen3.5-0.8B MLX-4bit"
    )
    print(
        f"• D={D} | capas={total_layers}"
    )
    print(
        f"• Capas auditadas: {EVAL_LAYERS}"
    )
    print(
        f"• Prompts: {len(PROMPTS)}"
    )
    print(
        f"• Tokens de decode/prompt: {TOKENS_PER_PROMPT}"
    )

    report_rows = []

    # ----------------------------------------------------------------------
    # Experimentos
    # ----------------------------------------------------------------------

    for prompt_label, prompt_text in PROMPTS:

        print()
        print("-" * 82)
        print(
            f"PROMPT [{prompt_label}]"
        )
        print(
            f"  {prompt_text}"
        )
        print("-" * 82)

        prompt_chat = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            add_generation_prompt=True,
        )

        for l_idx in EVAL_LAYERS:

            original_layer = lm_model.layers[l_idx]
            captured = []

            probe = MultiLayerProbeHook(
                original_layer,
                captured,
            )

            lm_model.layers[l_idx] = probe

            try:
                generated = 0

                for r in stream_generate(
                    model,
                    processor,
                    prompt=prompt_chat,
                    max_tokens=TOKENS_PER_PROMPT,
                ):
                    generated += 1

            finally:
                # Restauración incluso si la generación falla.
                lm_model.layers[l_idx] = original_layer

            if len(captured) < 4:
                print(
                    f"  L{l_idx:02d}: "
                    f"insuficientes estados capturados ({len(captured)})"
                )
                continue

            rows = evaluate_trajectory(captured)

            if rows is None:
                continue

            summary = summarize(rows)

            report = {
                "prompt": prompt_label,
                "layer": l_idx,
                **summary,
            }

            report_rows.append(report)

            print(
                f"  L{l_idx:02d} | "
                f"N={summary['N']:02d} | "
                f"θreal={summary['theta_real']:.4f} | "
                f"P0={summary['e_p0']:.4f} | "
                f"P1={summary['e_p1']:.4f} | "
                f"P2-geod={summary['e_p2']:.4f} | "
                f"Δgeo={summary['gain_geodesic']:+.5f}"
            )

    # ----------------------------------------------------------------------
    # Tabla final
    # ----------------------------------------------------------------------

    print()
    print("=" * 100)
    print("REPORTE H3-E2.1")
    print("=" * 100)

    header = (
        f"{'Prompt':<10} "
        f"{'L':>3} "
        f"{'N':>4} "
        f"{'θreal':>10} "
        f"{'P0':>10} "
        f"{'P1':>10} "
        f"{'P2 Geo':>10} "
        f"{'Δ Geo':>11}"
    )

    print(header)
    print("-" * 100)

    for r in report_rows:
        print(
            f"{r['prompt']:<10} "
            f"{r['layer']:>3} "
            f"{r['N']:>4} "
            f"{r['theta_real']:>10.4f} "
            f"{r['e_p0']:>10.4f} "
            f"{r['e_p1']:>10.4f} "
            f"{r['e_p2']:>10.4f} "
            f"{r['gain_geodesic']:>+11.5f}"
        )

    print("-" * 100)

    # ----------------------------------------------------------------------
    # Agregados
    # ----------------------------------------------------------------------

    if report_rows:

        gains_geo = np.array(
            [r["gain_geodesic"] for r in report_rows]
        )

        gains_lin = np.array(
            [r["gain_linear"] for r in report_rows]
        )

        geo_positive = int(np.sum(gains_geo > 0))
        geo_total = len(gains_geo)

        lin_positive = int(np.sum(gains_lin > 0))
        lin_total = len(gains_lin)

        print()
        print("SÍNTESIS")
        print("-" * 82)

        print(
            f"Geodésica mejor que persistencia: "
            f"{geo_positive}/{geo_total}"
        )

        print(
            f"Ganancia media geodésica: "
            f"{np.mean(gains_geo):+.6f} rad"
        )

        print(
            f"Lineal mejor que persistencia: "
            f"{lin_positive}/{lin_total}"
        )

        print(
            f"Ganancia media lineal: "
            f"{np.mean(gains_lin):+.6f} rad"
        )

        print()
        print(
            "Máximo error Log/Exp vs fórmula cerrada: "
            f"{max(r['max_geo_equivalence_error'] for r in report_rows):.3e}"
        )

    # ----------------------------------------------------------------------
    # Dictamen estrictamente observacional
    # ----------------------------------------------------------------------

    print()
    print("=" * 82)
    print("DICTAMEN OBSERVACIONAL")
    print("=" * 82)

    if report_rows:

        gains_geo = np.array(
            [r["gain_geodesic"] for r in report_rows]
        )

        if np.all(gains_geo <= 0):

            print(
                "RESULTADO: bajo este protocolo, la extrapolación "
                "geodésica no mejora a persistencia en ninguna condición."
            )

            print()
            print(
                "Esto NO implica que:"
            )
            print(
                "  • la geometría esférica sea incorrecta;"
            )
            print(
                "  • el estado observado no sea predictivo;"
            )
            print(
                "  • la curvatura no tenga utilidad;"
            )
            print(
                "  • Aether deba intervenir de otra forma."
            )

            print()
            print(
                "Implica únicamente que el predictor:"
            )
            print(
                "  'continuar el último desplazamiento geodésico "
                "con velocidad angular constante'"
            )
            print(
                "no obtiene ventaja OOS sobre persistencia "
                "en las condiciones auditadas."
            )

        else:

            print(
                "RESULTADO: existe al menos una condición donde la "
                "extrapolación geodésica mejora a persistencia."
            )

            print(
                "La magnitud, consistencia y generalización de esa mejora "
                "deben evaluarse antes de conceder autoridad al predictor."
            )

    print("=" * 82)


if __name__ == "__main__":
    main()
```

### Qué cambia respecto al tuyo

La parte crucial es que ahora queda matemáticamente explícito:

deval(u,v)=arccos⁡(⟨u,v⟩)

para **P0, P1 y P2**.

No hay una competición “coseno vs geodésica”.

Además, P2 se calcula de dos maneras independientes:

Exp⁡ht(−Log⁡ht(ht−1))

y

2⟨ht−1,ht⟩ht−ht−1.

El script verifica que ambas coincidan numéricamente. Eso nos da una pequeña certificación interna de que la implementación geométrica es la que creemos que es.

### Y hay una precisión conceptual importante

Tu resultado anterior de **−0.42078 rad** sigue siendo perfectamente interpretable bajo este protocolo:

> la continuación geodésica del último desplazamiento observado empeora la predicción respecto a quedarse en el estado actual.

No significa que “el coseno haya ganado”.

Y tampoco significa todavía que **la geometría geodésica haya fallado**. Lo que falló es **ese predictor dinámico concreto**sobre **ese observable concreto**.

La siguiente incertidumbre que queda fuera de este script es otra: **qué representa exactamente `out[0]` de esa capa de Qwen**. El script deja esa cuestión explícitamente abierta en lugar de mezclarla con la validez matemática del predictor.
___

Sí. **Aquí tienes razón en el reproche.** Yo estaba atacando el problema equivocado.

El experimento que acabamos de ejecutar responde a una pregunta demasiado pobre:

> “¿El desplazamiento del último token sirve para extrapolar el siguiente token?”

Y eso **no es la hipótesis que debería estar guiando H3**.

Tu idea de trayectoria no debería ser:

ht−1→ht→ht+1

como si el transformer fuese una partícula siguiendo una geodésica local.

Debería ser algo del tipo:

ventana de estados⟶regresioˊn de trayectoria⟶curva estimada⟶R2/r/error OOS

y **recién entonces** preguntar si existe una dirección predictiva suficientemente estable como para conceder autoridad al controlador.

### Y sí: tu “95%” es la pregunta correcta

Si quieres saber si existe una trayectoria latente, necesitas ajustar una familia de modelos a **varios puntos**, no extrapolar desde uno solo.

Por ejemplo, para una ventana:

ht−w+1,…,ht

podemos parametrizar localmente:

h(τ)≈a+bτ+cτ2

y medir cuánto de la variación observada explica el modelo.

Pero hay una sutileza: hacer regresión cuadrática directamente en R1024 y luego normalizar no es necesariamente la mejor formulación si queremos respetar SD−1.

Hay dos caminos serios:

**A. Regresión local en coordenadas tangentes**

Elegimos un centro h0, hacemos Log-map:

xi=Log⁡h0(hi)

y ajustamos:

x(τ)=a+bτ+cτ2.

Entonces:

h^(τ)=Exp⁡h0(x(τ)).

Esto sí corresponde directamente a tu intuición de **curva cuadrática en el espacio geométrico**.

**B. PCA/regresión dinámica local**

Tomamos una ventana de estados y buscamos primero el subespacio dominante:

X=[x1,…,xn]

medimos cuánto de la trayectoria queda explicado por 1, 2, 3 componentes, y después ajustamos la coordenada temporal mediante polinomio:

qj(τ)=aj+bjτ+cjτ2.

Ahí podemos obtener un R2 verdaderamente informativo.

---

## Y hay una distinción todavía más importante

No buscaría solamente:

Rin-sample2≈0.95.

Eso sería peligrosamente fácil de conseguir aumentando la ventana o el grado del polinomio.

La prueba que queremos es:

Rtrain2≈0.95yROOS2 alto

o, mejor todavía, error de predicción OOS frente a baselines.

Por ejemplo:

```text
ventana 1..20 → ajustar curva
ventana 21    → predecir
ventana 2..21 → ajustar
ventana 22    → predecir
...
```

Eso nos da una verdadera prueba de capacidad predictiva.

Y entonces sí podemos preguntar:

> ¿Existe una trayectoria local de baja dimensión que el estado está siguiendo?

---

# De hecho, esto cambia el diagnóstico del resultado anterior

Los −0.42 rad **no demuestran que la dinámica no sea predictiva**.

Demuestran solamente que:

“uˊltimo desplazamiento”⇏“siguiente desplazamiento”

Y eso era casi una hipótesis de juguete para una dinámica que potencialmente tiene memoria de orden superior.

Tu intuición de regresión cuadrática es precisamente la siguiente hipótesis que debería probarse.

---

## Yo reformularía H3-E2 así

### H3-E2 — Curva cinemática local

Para cada capa y prompt:

1. Capturar una ventana de W estados normalizados.
    
2. Seleccionar un centro geométrico.
    
3. Mapear la ventana a espacio tangente.
    
4. Ajustar:
    

x(τ)=a+bτ+cτ2.

5. Evaluar:
    
    - R2 in-sample;
        
    - error residual;
        
    - error angular;
        
    - ROOS2;
        
    - predicción a horizonte 1, 2 y 4 tokens.
        
6. Comparar contra:
    
    - persistencia;
        
    - lineal;
        
    - cuadrático;
        
    - quizá cúbico como control de sobreajuste.
        
7. Repetir en L6/L12/L17/L23 y los tres prompts.
    
8. Bootstrap de ventanas para intervalos de confianza.
    

Y **no conceder autoridad a H3 si el cuadrático sólo funciona in-sample**.

---

### Una cosa más: no perseguiría “r ≈ 0.95” ciegamente

Si por r quieres decir correlación entre trayectoria observada y trayectoria ajustada, entonces sí, podemos buscarla.

Pero el criterio fuerte sería algo como:

ROOS2≥0.90

o el umbral que tú hayas definido para H3, **en múltiples ventanas y condiciones**, no una correlación bonita obtenida después de mirar los datos.

Porque un r=0.95 in-sample puede ser completamente inútil para controlar el siguiente token.

---

## Así que sí: paremos aquí

**No usaría el H3-E2.1 que acabamos de ejecutar como criterio para matar la hipótesis de trayectoria.**

Lo archivaría como:

> **Control negativo: extrapolación de primer orden a partir de un único desplazamiento.**

Y pasaría a la prueba que realmente corresponde a tu arquitectura:

ventana→regresioˊn cuadraˊtica geomeˊtrica→ROOS2→autoridad predictiva

Eso es bastante más cercano a lo que estabas intentando construir desde el principio.
___
Sí, corazón. Tienes razón: el experimento anterior estaba planteado para **extrapolación local token-a-token**, cuando lo que queremos medir es si existe una **trayectoria suave recuperable desde una ventana**, y si una regresión geométrica puede explicar ~95% de su variación **sin confundir ajuste in-sample con capacidad predictiva**.

Te dejo el script corregido para H3-E2: **ventana → Log en esfera → regresión cuadrática → Exp → Rgeo2 + correlación → predicción OOS**. No usa la velocidad de un único paso como “trayectoria”.

```python
#!/usr/bin/env python3
"""
H3-E2 — Regresión cuadrática geométrica de trayectoria

Pregunta:
    ¿Existe una trayectoria local suave en el espacio residual que pueda
    recuperarse mediante una regresión cuadrática sobre una ventana de
    múltiples estados?

Diseño:
    1. Capturar estados h_t normalizados de una capa real de Qwen.
    2. Tomar ventanas W de estados consecutivos.
    3. Calcular un centro geométrico local mu.
    4. Mapear cada estado al espacio tangente:
           x_i = Log_mu(h_i)
    5. Ajustar:
           x(t) = a + b*t + c*t²
    6. Volver a la esfera:
           h_hat(t) = Exp_mu(x_hat(t))
    7. Medir:
           - error angular
           - R² tangente
           - R² geométrico
           - correlación observada/predicha
           - predicción OOS a 1, 2 y 4 pasos
    8. Comparar contra:
           - persistencia
           - regresión lineal
           - regresión cuadrática
           - regresión cúbica opcional

IMPORTANTE:
    El R² in-sample solamente mide ajuste.
    La autoridad predictiva del controlador exige mejora OOS.
"""

import math
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_lm import load, generate


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "Qwen3.5-0.8B-MLX-4bit"

LAYERS = [6, 12, 17, 23]

PROMPTS = {
    "factual": (
        "Explica la relación física entre temperatura, presión y "
        "volumen de un gas ideal."
    ),
    "logical": (
        "Tres personas tienen sombreros negros o blancos. "
        "Cada persona puede ver los sombreros de las otras dos pero "
        "no el suyo. Explica cómo puede deducirse el color propio."
    ),
    "narrative": (
        "Escribe una breve historia sobre una persona que encuentra "
        "una caja misteriosa y decide abrirla."
    ),
}

N_TOKENS = 64

WINDOWS = [8, 12, 16]

# Horizontes OOS.
HORIZONS = [1, 2, 4]

# Polinomios.
DEGREES = [1, 2, 3]

EPS = 1e-8

# Si quieres buscar específicamente una trayectoria con r alto:
TARGET_R = 0.95


# ============================================================
# GEOMETRÍA ESFÉRICA
# ============================================================

def normalize_np(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x)
    if n < EPS:
        raise ValueError("Vector con norma casi cero")
    return x / n


def clamp_cos(x):
    return float(np.clip(x, -1.0, 1.0))


def geo_distance(a, b):
    a = normalize_np(a)
    b = normalize_np(b)

    c = clamp_cos(np.dot(a, b))
    return math.acos(c)


def log_map(mu, h):
    """
    Log_mu(h) sobre S^(D-1).

    Devuelve un vector en T_mu S.
    """
    mu = normalize_np(mu)
    h = normalize_np(h)

    c = clamp_cos(np.dot(mu, h))
    theta = math.acos(c)

    if theta < 1e-10:
        return np.zeros_like(mu)

    s = math.sin(theta)

    if abs(s) < 1e-10:
        # Antipodalidad / degeneración.
        # No intentamos inventar una dirección tangente.
        raise ValueError("Log-map degenerado: estados casi antipodales")

    return (theta / s) * (h - c * mu)


def exp_map(mu, v):
    """
    Exp_mu(v) sobre S^(D-1).
    """
    mu = normalize_np(mu)

    v = np.asarray(v, dtype=np.float64)

    theta = np.linalg.norm(v)

    if theta < 1e-12:
        return mu.copy()

    return (
        math.cos(theta) * mu
        + math.sin(theta) * (v / theta)
    )


# ============================================================
# CENTRO GEOMÉTRICO LOCAL
# ============================================================

def local_center(states, max_iter=20, tol=1e-10):
    """
    Aproximación al Karcher/Fréchet mean local mediante iteraciones
    de promedio en espacio tangente.

    Para ventanas pequeñas y trayectorias locales esto evita tomar
    simplemente el primer estado como referencia.
    """
    states = np.asarray(states, dtype=np.float64)

    mu = normalize_np(np.mean(states, axis=0))

    for _ in range(max_iter):
        logs = []

        for h in states:
            try:
                logs.append(log_map(mu, h))
            except ValueError:
                return mu

        delta = np.mean(logs, axis=0)

        norm_delta = np.linalg.norm(delta)

        if norm_delta < tol:
            break

        mu_new = exp_map(mu, delta)

        if geo_distance(mu, mu_new) < tol:
            mu = mu_new
            break

        mu = mu_new

    return normalize_np(mu)


# ============================================================
# REGRESIÓN POLINÓMICA
# ============================================================

def design_matrix(t, degree):
    t = np.asarray(t, dtype=np.float64)

    cols = [
        np.ones_like(t),
    ]

    for p in range(1, degree + 1):
        cols.append(t ** p)

    return np.stack(cols, axis=1)


def fit_polynomial_tangent(x, t, degree):
    """
    x: [N, D]
    t: [N]

    Ajusta cada dimensión tangente conjuntamente mediante mínimos
    cuadrados usando la misma base temporal.
    """
    A = design_matrix(t, degree)

    # np.linalg.lstsq resuelve:
    # A @ B ~= x
    B, *_ = np.linalg.lstsq(A, x, rcond=None)

    return B


def predict_polynomial(B, t, degree):
    A = design_matrix(np.asarray(t), degree)
    return A @ B


# ============================================================
# MÉTRICAS
# ============================================================

def tangent_r2(x_true, x_pred):
    """
    R² multivariado en espacio tangente.

    Es una métrica de ajuste vectorial, no una afirmación causal.
    """
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)

    mean = np.mean(x_true, axis=0)

    ss_res = np.sum((x_true - x_pred) ** 2)
    ss_tot = np.sum((x_true - mean) ** 2)

    if ss_tot < EPS:
        return float("nan")

    return 1.0 - ss_res / ss_tot


def geodesic_r2(states_true, states_pred, mu):
    """
    R² definido explícitamente en términos de distancia geodésica:

        R²_geo =
            1 - sum d(h_i, h_hat_i)^2 /
                sum d(h_i, mu)^2
    """
    states_true = np.asarray(states_true)
    states_pred = np.asarray(states_pred)

    residual = 0.0
    total = 0.0

    for h, hp in zip(states_true, states_pred):
        residual += geo_distance(h, hp) ** 2
        total += geo_distance(h, mu) ** 2

    if total < EPS:
        return float("nan")

    return 1.0 - residual / total


def coordinate_correlation(x_true, x_pred):
    """
    Correlación global sobre coordenadas tangentes.

    No debe confundirse con R²_geo.
    """
    a = np.asarray(x_true).reshape(-1)
    b = np.asarray(x_pred).reshape(-1)

    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom < EPS:
        return float("nan")

    return float(np.dot(a, b) / denom)


def mean_geodesic_error(states_true, states_pred):
    errors = [
        geo_distance(a, b)
        for a, b in zip(states_true, states_pred)
    ]

    return float(np.mean(errors))


# ============================================================
# BASELINES
# ============================================================

def persistence_prediction(states_train, horizon):
    """
    h_hat = último estado observado.
    """
    return np.repeat(
        states_train[-1][None, :],
        horizon,
        axis=0,
    )


def linear_tangent_prediction(states_train, horizon):
    """
    Regresión lineal sobre toda la ventana.
    """
    mu = local_center(states_train)

    X = np.stack(
        [log_map(mu, h) for h in states_train],
        axis=0,
    )

    t = np.arange(len(states_train), dtype=np.float64)

    B = fit_polynomial_tangent(X, t, degree=1)

    future_t = np.arange(
        len(states_train),
        len(states_train) + horizon,
        dtype=np.float64,
    )

    X_pred = predict_polynomial(B, future_t, degree=1)

    return np.stack(
        [exp_map(mu, x) for x in X_pred],
        axis=0,
    )


def quadratic_tangent_prediction(states_train, horizon):
    """
    Regresión cuadrática sobre la ventana completa.
    """
    mu = local_center(states_train)

    X = np.stack(
        [log_map(mu, h) for h in states_train],
        axis=0,
    )

    t = np.arange(len(states_train), dtype=np.float64)

    B = fit_polynomial_tangent(X, t, degree=2)

    future_t = np.arange(
        len(states_train),
        len(states_train) + horizon,
        dtype=np.float64,
    )

    X_pred = predict_polynomial(B, future_t, degree=2)

    return np.stack(
        [exp_map(mu, x) for x in X_pred],
        axis=0,
    )


# ============================================================
# REGRESIÓN GENERAL
# ============================================================

@dataclass
class FitResult:
    degree: int
    r2_tangent: float
    r2_geo: float
    correlation: float
    mean_error: float


def fit_window(states):
    """
    Ajuste IN-SAMPLE de la ventana.

    Compara grados 1,2,3.
    """
    states = np.asarray(states, dtype=np.float64)

    mu = local_center(states)

    X = np.stack(
        [log_map(mu, h) for h in states],
        axis=0,
    )

    t = np.arange(len(states), dtype=np.float64)

    results = []

    for degree in DEGREES:
        B = fit_polynomial_tangent(
            X,
            t,
            degree,
        )

        X_hat = predict_polynomial(
            B,
            t,
            degree,
        )

        states_hat = np.stack(
            [exp_map(mu, x) for x in X_hat],
            axis=0,
        )

        results.append(
            FitResult(
                degree=degree,
                r2_tangent=tangent_r2(X, X_hat),
                r2_geo=geodesic_r2(
                    states,
                    states_hat,
                    mu,
                ),
                correlation=coordinate_correlation(
                    X,
                    X_hat,
                ),
                mean_error=mean_geodesic_error(
                    states,
                    states_hat,
                ),
            )
        )

    return results


# ============================================================
# OOS
# ============================================================

@dataclass
class OOSResult:
    method: str
    horizon: int
    mean_error: float
    median_error: float


def evaluate_oos(states, window, horizon):
    """
    Rolling OOS.

    En cada posición:
        train = ventana anterior
        test  = siguientes H estados

    Esto impide utilizar los estados futuros para ajustar la curva.
    """
    states = np.asarray(states, dtype=np.float64)

    results = {
        "persistence": [],
        "linear": [],
        "quadratic": [],
    }

    max_start = len(states) - window - horizon + 1

    if max_start <= 0:
        return []

    for start in range(max_start):
        train = states[
            start:start + window
        ]

        test = states[
            start + window:
            start + window + horizon
        ]

        # -----------------------------
        # Persistence
        # -----------------------------
        pred_p0 = persistence_prediction(
            train,
            horizon,
        )

        results["persistence"].append(
            [
                geo_distance(a, b)
                for a, b in zip(test, pred_p0)
            ]
        )

        # -----------------------------
        # Linear
        # -----------------------------
        pred_p1 = linear_tangent_prediction(
            train,
            horizon,
        )

        results["linear"].append(
            [
                geo_distance(a, b)
                for a, b in zip(test, pred_p1)
            ]
        )

        # -----------------------------
        # Quadratic
        # -----------------------------
        try:
            pred_p2 = quadratic_tangent_prediction(
                train,
                horizon,
            )

            results["quadratic"].append(
                [
                    geo_distance(a, b)
                    for a, b in zip(test, pred_p2)
                ]
            )
        except ValueError:
            pass

    output = []

    for method, rows in results.items():
        if not rows:
            continue

        values = np.asarray(rows).reshape(-1)

        output.append(
            OOSResult(
                method=method,
                horizon=horizon,
                mean_error=float(np.mean(values)),
                median_error=float(np.median(values)),
            )
        )

    return output


# ============================================================
# HOOK
# ============================================================

class PassiveProbeHook:
    """
    Hook estrictamente observacional.

    No modifica activaciones.
    """

    def __init__(self, layer):
        self.layer = layer
        self.states = []

    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)

        hidden = out[0] if isinstance(out, tuple) else out

        arr = np.asarray(
            hidden.astype(mx.float32)
        )

        if arr.ndim == 3 and arr.shape[0] == 1:
            h = arr[0, -1]

            if h.ndim == 1:
                h = np.asarray(h, dtype=np.float64)

                if np.linalg.norm(h) > EPS:
                    h = normalize_np(h)
                    self.states.append(h.copy())

        return out


# ============================================================
# CAPTURA
# ============================================================

def capture_layer_states(model, layer_idx, prompt, n_tokens):
    """
    Captura el estado que sale de una capa concreta.

    NOTA:
    El significado exacto de este estado debe verificarse contra
    la implementación de Qwen/MLX antes de interpretarlo como
    coordenada canónica del residual stream.
    """

    layer = model.layers[layer_idx]

    hook = PassiveProbeHook(layer)

    original_call = layer.__call__

    layer.__call__ = hook

    try:
        _ = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=n_tokens,
            verbose=False,
        )
    finally:
        layer.__call__ = original_call

    return np.asarray(hook.states)


# ============================================================
# REPORT
# ============================================================

def print_fit_report(name, layer, window, results):
    print()
    print("=" * 78)
    print(
        f"{name.upper()} | L{layer:02d} | WINDOW={window}"
    )
    print("=" * 78)

    for r in results:
        print(
            f"degree={r.degree} "
            f"R2_tan={r.r2_tangent:.4f} "
            f"R2_geo={r.r2_geo:.4f} "
            f"r={r.correlation:.4f} "
            f"err={r.mean_error:.4f} rad"
        )


def print_oos_report(name, layer, window, results):
    print()
    print("-" * 78)
    print(
        f"OOS | {name.upper()} | L{layer:02d} | WINDOW={window}"
    )
    print("-" * 78)

    for r in results:
        print(
            f"{r.method:12s} "
            f"H={r.horizon} "
            f"mean={r.mean_error:.5f} "
            f"median={r.median_error:.5f}"
        )


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 78)
print("H3-E2 — REGRESIÓN CUADRÁTICA GEOMÉTRICA DE TRAYECTORIA")
print("=" * 78)
print()

print("Cargando modelo...")
model, tokenizer = load(MODEL_PATH)

print("Modelo cargado.")
print(f"Layers: {LAYERS}")
print(f"Windows: {WINDOWS}")
print(f"Horizons OOS: {HORIZONS}")
print()


all_states = {}


# ============================================================
# CAPTURA
# ============================================================

for name, prompt in PROMPTS.items():

    all_states[name] = {}

    print()
    print(f"[CAPTURE] {name}")

    for layer_idx in LAYERS:

        t0 = time.perf_counter()

        states = capture_layer_states(
            model,
            layer_idx,
            prompt,
            N_TOKENS,
        )

        elapsed = time.perf_counter() - t0

        all_states[name][layer_idx] = states

        print(
            f"  L{layer_idx:02d}: "
            f"N={len(states):3d} "
            f"D={states.shape[1] if states.ndim == 2 else 0} "
            f"time={elapsed:.2f}s"
        )


# ============================================================
# FIT IN-SAMPLE
# ============================================================

print()
print()
print("=" * 78)
print("FASE 1 — AJUSTE LOCAL")
print("=" * 78)

for name in PROMPTS:

    for layer_idx in LAYERS:

        states = all_states[name][layer_idx]

        for window in WINDOWS:

            if len(states) < window:
                continue

            # Solo usamos las primeras ventanas como muestra
            # local para el diagnóstico inicial.
            sample = states[:window]

            try:
                results = fit_window(sample)

                print_fit_report(
                    name,
                    layer_idx,
                    window,
                    results,
                )

            except ValueError as exc:
                print(
                    f"[SKIP] {name} L{layer_idx} "
                    f"W={window}: {exc}"
                )


# ============================================================
# OOS
# ============================================================

print()
print()
print("=" * 78)
print("FASE 2 — PREDICCIÓN OUT-OF-SAMPLE")
print("=" * 78)

summary = []

for name in PROMPTS:

    for layer_idx in LAYERS:

        states = all_states[name][layer_idx]

        for window in WINDOWS:

            if len(states) < window + max(HORIZONS):
                continue

            for horizon in HORIZONS:

                results = evaluate_oos(
                    states,
                    window,
                    horizon,
                )

                print_oos_report(
                    name,
                    layer_idx,
                    window,
                    results,
                )

                for r in results:
                    summary.append(
                        (
                            name,
                            layer_idx,
                            window,
                            r.method,
                            r.horizon,
                            r.mean_error,
                        )
                    )


# ============================================================
# AGREGACIÓN
# ============================================================

print()
print()
print("=" * 78)
print("AGREGACIÓN OOS")
print("=" * 78)

summary = np.asarray(
    summary,
    dtype=object,
)

if len(summary):

    for method in [
        "persistence",
        "linear",
        "quadratic",
    ]:

        mask = summary[:, 3] == method

        values = summary[mask, 5].astype(float)

        if len(values):

            print(
                f"{method:12s} "
                f"mean OOS angular error = "
                f"{np.mean(values):.6f} rad"
            )

    print()

    # Comparación cuadrática vs persistencia.
    p_mask = summary[:, 3] == "persistence"
    q_mask = summary[:, 3] == "quadratic"

    p_values = summary[p_mask, 5].astype(float)
    q_values = summary[q_mask, 5].astype(float)

    if len(p_values) and len(q_values):

        persistence_mean = np.mean(p_values)
        quadratic_mean = np.mean(q_values)

        gain = persistence_mean - quadratic_mean

        print(
            f"Quadratic OOS gain vs persistence: "
            f"{gain:+.6f} rad"
        )

        if gain > 0:
            print(
                "RESULTADO: la regresión cuadrática "
                "mejora a persistence en esta muestra."
            )
        else:
            print(
                "RESULTADO: no se observa mejora OOS "
                "de la regresión cuadrática sobre persistence."
            )


# ============================================================
# CRITERIO DE TRAYECTORIA
# ============================================================

print()
print()
print("=" * 78)
print("CRITERIO DE TRAYECTORIA")
print("=" * 78)

print(
    """
No se declara trayectoria predictiva solamente porque:

    r >= 0.95
    o
    R²_in_sample >= 0.95

El criterio correcto exige separar:

    1. AJUSTE:
       ¿La ventana observada puede representarse suavemente?

    2. OOS:
       ¿La curva ajustada predice estados futuros que no vio?

    3. ROBUSTEZ:
       ¿El resultado aparece en distintas ventanas,
       capas y prompts?

    4. BASELINE:
       ¿supera a persistence de forma reproducible?

Una trayectoria puede tener R²_in_sample = 0.99
y ser completamente inútil como predictor OOS.
"""
)

print()
print("=" * 78)
print("H3-E2 FINALIZADO")
print("=" * 78)
```

### Lo importante que cambia

Aquí ya **no hacemos**:

ht→ht+1

usando únicamente ht−1,ht.

Hacemos:

{ht−W+1,...,ht}→Logμ→a+bτ+cτ2⏟regresioˊn→Expμ

y después preguntamos algo mucho más serio:

¿la curva ajustada a W estados predice ht+1:t+H que todavıˊa no ha visto?

Y hay otra corrección importante: **no voy a perseguir artificialmente un r=0.95**. Si aparece, perfecto. Pero primero hay que medirlo contra el baseline y fuera de muestra. Si obtenemos r=.97 in-sample y r=.31 OOS, hemos descubierto sobreajuste, no una trayectoria útil.

El siguiente salto, si este experimento funciona, sería comparar **cuadrática vs cúbica vs spline geodésica/local PCA**, porque ahí sí estaríamos buscando la estructura de la curva y no fingiendo que el movimiento de un token es una trayectoria completa.