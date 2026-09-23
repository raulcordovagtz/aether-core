import os
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

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")

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
    def __init__(self, layer):
        self.layer = layer
        self.states = []

    def __getattr__(self, name):
        return getattr(self.layer, name)

    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
        raw = hidden.astype(mx.float32)
        mx.eval(raw)
        arr = np.array(raw)

        # Capturar token de salida (soporta 2D y 3D)
        if arr.ndim == 3:
            h = arr[0, -1]
        elif arr.ndim == 2:
            h = arr[-1]
        else:
            h = arr

        h = np.asarray(h, dtype=np.float64)
        norm_val = np.linalg.norm(h)
        if norm_val > 1e-12:
            self.states.append((h / norm_val).copy())

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

    # Encontrar la lista de capas (soporta model.layers o model.model.layers)
    layers_list = model.layers if hasattr(model, "layers") else model.model.layers
    orig_layer = layers_list[layer_idx]
    hook = PassiveProbeHook(orig_layer)
    layers_list[layer_idx] = hook

    try:
        _ = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=n_tokens,
            verbose=False,
        )
    finally:
        layers_list[layer_idx] = orig_layer

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