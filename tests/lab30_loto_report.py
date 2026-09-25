#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

DATA_FILE = Path("results/lab30_trajectory_battery.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

records = []

with DATA_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print("=" * 82)
print("LAB30 — LOTO POR TASK_ID + DIAGNÓSTICO DE TRUNCAMIENTO")
print("=" * 82)

print(f"\nN={len(records)}")
print(f"task_id únicos={len(set(r['task_id'] for r in records))}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def feat(r, name, mode):
    x = np.array(
        [float(s.get(name, 0.0)) for s in r["trajectory"]],
        dtype=float
    )

    if len(x) == 0:
        return 0.0

    if mode == "max":
        return float(np.max(x))
    if mode == "min":
        return float(np.min(x))
    if mode == "mean":
        return float(np.mean(x))

    raise ValueError(mode)


y = np.array([not r["correct"] for r in records], dtype=int)
length = np.array([len(r["trajectory"]) for r in records], dtype=float)

A = np.array([feat(r, "a_norm", "max") for r in records])
Q = np.array([feat(r, "q", "max") for r in records])
K = np.array([feat(r, "kappa", "max") for r in records])
V = np.array([feat(r, "v_norm", "max") for r in records])
M = np.array([feat(r, "margin", "min") for r in records])
H = np.array([feat(r, "entropy", "mean") for r in records])

task_ids = np.array([r["task_id"] for r in records])
families = np.array([r["family"] for r in records])


# ─────────────────────────────────────────────────────────────────────────────
# AUROC
# ─────────────────────────────────────────────────────────────────────────────

def auroc(y, score):

    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)

    pos = score[y == 1]
    neg = score[y == 0]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg)
        wins += 0.5 * np.sum(p == neg)

    return wins / (len(pos) * len(neg))


def zscore_train(x_train, x_test):

    mu = np.mean(x_train)
    sd = np.std(x_train)

    if sd < 1e-12:
        sd = 1.0

    return (
        (x_train - mu) / sd,
        (x_test - mu) / sd
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. TASK STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 82)
print("1. TASK_ID — ESTRUCTURA COMPLETA")
print("=" * 82)

for task in sorted(set(task_ids)):

    idx = task_ids == task
    yy = y[idx]
    ll = length[idx]
    aa = A[idx]

    l_corr = np.mean(ll[yy==0]) if np.sum(yy==0) > 0 else np.nan
    l_err  = np.mean(ll[yy==1]) if np.sum(yy==1) > 0 else np.nan
    a_corr = np.mean(aa[yy==0]) if np.sum(yy==0) > 0 else np.nan
    a_err  = np.mean(aa[yy==1]) if np.sum(yy==1) > 0 else np.nan

    print(
        f"{task:<6} "
        f"fam={families[idx][0]:<12} "
        f"N={len(yy):<3} "
        f"err={np.sum(yy):<2} "
        f"acc={np.sum(yy==0):<2} "
        f"err_rate={np.mean(yy):.3f} "
        f"Lcorr={l_corr:5.1f} "
        f"Lerr={l_err:5.1f} "
        f"Acorr={a_corr:6.3f} "
        f"Aerr={a_err:6.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRUNCAMIENTO / RESPUESTAS MUY CORTAS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 82)
print("2. DIAGNÓSTICO DE LONGITUD / TRUNCAMIENTO")
print("=" * 82)

for threshold in [3, 5, 10, 64, 100, 120, 127, 128, 129]:

    n = np.sum(length <= threshold)
    err = np.sum(y[length <= threshold])

    print(
        f"L <= {threshold:3d} : "
        f"N={n:3d} "
        f"errors={err:3d} "
        f"error_rate={(err/n if n else np.nan):.4f}"
    )

print("\nExtremo superior:")

for lo in [118, 120, 125, 127, 128, 129]:

    idx = length >= lo

    print(
        f"L >= {lo:3d} : "
        f"N={np.sum(idx):3d} "
        f"errors={np.sum(y[idx]):3d} "
        f"error_rate={np.mean(y[idx]) if np.sum(idx) else np.nan:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOTO TASK_ID
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 82)
print("3. LEAVE-ONE-TASK-OUT")
print("=" * 82)

tasks = sorted(set(task_ids))
results = []

for heldout in tasks:

    test = task_ids == heldout
    train = ~test

    ytr = y[train]
    yte = y[test]

    if len(np.unique(ytr)) < 2:
        continue

    test_has_both = len(np.unique(yte)) == 2

    train_features = {
        "length": length[train],
        "a_max": A[train],
        "q_max": Q[train],
        "kappa": K[train],
        "v_max": V[train],
        "-margin": -M[train],
        "entropy": H[train],
    }

    test_features = {
        "length": length[test],
        "a_max": A[test],
        "q_max": Q[test],
        "kappa": K[test],
        "v_max": V[test],
        "-margin": -M[test],
        "entropy": H[test],
    }

    row = {
        "task": heldout,
        "family": families[test][0],
        "N": len(yte),
        "errors": int(np.sum(yte)),
        "test_has_both": test_has_both,
    }

    for name in train_features:
        tr_z, te_z = zscore_train(
            train_features[name],
            test_features[name]
        )
        if test_has_both:
            row[name] = auroc(yte, te_z)
        else:
            row[name] = np.nan

    feature_names = [
        "length",
        "a_max",
        "q_max",
        "kappa",
        "v_max",
        "-margin",
        "entropy",
    ]

    Ztr = []
    Zte = []

    for name in feature_names:
        tr_z, te_z = zscore_train(
            train_features[name],
            test_features[name]
        )
        Ztr.append(tr_z)
        Zte.append(te_z)

    Ztr = np.column_stack(Ztr)
    Zte = np.column_stack(Zte)

    mu_err = np.mean(Ztr[ytr == 1], axis=0)
    mu_ok  = np.mean(Ztr[ytr == 0], axis=0)

    w = mu_err - mu_ok

    score_tr = Ztr @ w
    score_te = Zte @ w

    row["joint"] = auroc(yte, score_te) if test_has_both else np.nan

    keep = [
        feature_names.index("a_max"),
        feature_names.index("q_max"),
        feature_names.index("kappa"),
        feature_names.index("v_max"),
        feature_names.index("-margin"),
        feature_names.index("entropy"),
    ]

    w_no_length = w[keep]
    score_no_length = Zte[:, keep] @ w_no_length

    row["joint_no_length"] = (
        auroc(yte, score_no_length)
        if test_has_both else np.nan
    )

    results.append(row)


# ─────────────────────────────────────────────────────────────────────────────
# IMPRIMIR LOTO
# ─────────────────────────────────────────────────────────────────────────────

print(
    f"\n{'TASK':<6} "
    f"{'FAM':<12} "
    f"{'ERR':<4} "
    f"{'a':>7} "
    f"{'q':>7} "
    f"{'k':>7} "
    f"{'v':>7} "
    f"{'M':>7} "
    f"{'L':>7} "
    f"{'JOINT':>8} "
    f"{'NO-L':>8}"
)

for r in results:
    print(
        f"{r['task']:<6} "
        f"{r['family']:<12} "
        f"{r['errors']:<4} "
        f"{r['a_max']:7.4f} "
        f"{r['q_max']:7.4f} "
        f"{r['kappa']:7.4f} "
        f"{r['v_max']:7.4f} "
        f"{r['-margin']:7.4f} "
        f"{r['length']:7.4f} "
        f"{r['joint']:8.4f} "
        f"{r['joint_no_length']:8.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. RESUMEN LOTO
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 82)
print("4. RESUMEN LOTO")
print("=" * 82)

valid = [
    r for r in results
    if not np.isnan(r["joint"])
]

print(f"Tareas con AUROC evaluable : {len(valid)}/{len(results)}")

for name in [
    "a_max",
    "q_max",
    "kappa",
    "v_max",
    "-margin",
    "length",
    "joint",
    "joint_no_length",
]:
    vals = np.array([
        r[name] for r in valid
        if not np.isnan(r[name])
    ])

    if len(vals):
        print(
            f"{name:<18} "
            f"mean={np.mean(vals):.4f} "
            f"median={np.median(vals):.4f} "
            f"min={np.min(vals):.4f} "
            f"max={np.max(vals):.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. ONLINE LOTO: t=10,15,20
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 82)
print("5. ONLINE LOTO — t=10,15,20")
print("=" * 82)

def online_feature(t, field, mode):
    vals = []
    for r in records:
        tr = r["trajectory"][:t]
        if not tr:
            vals.append(0.0)
            continue
        x = np.array([float(s.get(field,0.0)) for s in tr])
        if mode == "max":
            vals.append(np.max(x))
        elif mode == "min":
            vals.append(np.min(x))
        elif mode == "mean":
            vals.append(np.mean(x))
    return np.array(vals)


for t in [10, 15, 20]:
    AA = online_feature(t, "a_norm", "max")
    QQ = online_feature(t, "q", "max")
    KK = online_feature(t, "kappa", "max")
    VV = online_feature(t, "v_norm", "max")
    MM = -online_feature(t, "margin", "min")

    print(f"\nt={t}")
    online_rows = []

    for heldout in tasks:
        test = task_ids == heldout
        train = ~test

        ytr = y[train]
        yte = y[test]

        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue

        feats_tr = np.column_stack([
            AA[train],
            QQ[train],
            KK[train],
            VV[train],
            MM[train],
        ])

        feats_te = np.column_stack([
            AA[test],
            QQ[test],
            KK[test],
            VV[test],
            MM[test],
        ])

        for j in range(feats_tr.shape[1]):
            mu = np.mean(feats_tr[:,j])
            sd = np.std(feats_tr[:,j])
            if sd < 1e-12:
                sd = 1.0
            feats_tr[:,j] = (feats_tr[:,j]-mu)/sd
            feats_te[:,j] = (feats_te[:,j]-mu)/sd

        w = (
            np.mean(feats_tr[ytr==1],axis=0)
            - np.mean(feats_tr[ytr==0],axis=0)
        )

        score = feats_te @ w
        online_rows.append(auroc(yte,score))

    if online_rows:
        print(
            f"  joint trajectory detector: "
            f"mean={np.mean(online_rows):.4f} "
            f"median={np.median(online_rows):.4f} "
            f"N={len(online_rows)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONCLUSIÓN MECÁNICA
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 82)
print("6. DATOS PARA DECISIÓN")
print("=" * 82)
print("""
Preguntas que este reporte contesta:
  1. ¿Sobrevive la señal cuando task_id queda completamente fuera?
  2. ¿Sobrevive sin longitud (joint_no_length)?
  3. ¿Sobrevive online en t=10/15/20?
  4. ¿Qué tareas aportan realmente la señal?
  5. ¿Los errores están dominados por truncamiento?
""")
print("=" * 82)
