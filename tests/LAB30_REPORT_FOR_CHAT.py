#!/usr/bin/env python3
"""
LAB30_REPORT_FOR_CHAT.py

Lee el JSONL completo localmente y produce SOLO un reporte compacto.
No imprime trayectorias ni datos por token.
"""

import json
from pathlib import Path
import numpy as np

DATA_FILE = Path("results/lab30_trajectory_battery.jsonl")

def auroc(y, score):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(score, dtype=float)
    pos = s[y]
    neg = s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = np.argsort(np.argsort(s)) + 1
    rp = np.sum(ranks[y])
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def zdiff(a, y):
    a = np.asarray(a, dtype=float)
    y = np.asarray(y, dtype=bool)
    c = a[~y]
    e = a[y]
    if len(c) == 0 or len(e) == 0:
        return np.nan
    se = np.sqrt(np.var(c, ddof=1) / len(c) + np.var(e, ddof=1) / len(e) + 1e-12)
    return float((np.mean(e) - np.mean(c)) / se)

def residualize(y, x):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept), slope

def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)

records = []
with DATA_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

N = len(records)
y = np.array([not r["correct"] for r in records], dtype=bool)
length = np.array([len(r["trajectory"]) for r in records], dtype=float)
families = [r.get("family", "UNKNOWN") for r in records]

def traj_feature(r, name, mode):
    x = np.array([float(s.get(name, 0.0)) for s in r["trajectory"]], dtype=float)
    if len(x) == 0: return 0.0
    if mode == "max": return np.max(x)
    if mode == "mean": return np.mean(x)
    if mode == "min": return np.min(x)
    raise ValueError(mode)

a_max = np.array([traj_feature(r, "a_norm", "max") for r in records])
q_max = np.array([traj_feature(r, "q", "max") for r in records])
k_max = np.array([traj_feature(r, "kappa", "max") for r in records])
v_max = np.array([traj_feature(r, "v_norm", "max") for r in records])
margin_min = np.array([traj_feature(r, "margin", "min") for r in records])
entropy_mean = np.array([traj_feature(r, "entropy", "mean") for r in records])

print()
print("=" * 78)
print("LAB30 — REPORTE COMPACTO PARA AUDITORÍA AETHER")
print("=" * 78)
print(f"N rollouts              : {N}")
print(f"Errores                 : {np.sum(y)}")
print(f"Aciertos                : {np.sum(~y)}")
print(f"Longitud media          : {np.mean(length):.3f}")
print(f"Longitud mediana        : {np.median(length):.3f}\n")

print("1. COMPOSICIÓN POR FAMILIA")
print("-" * 78)
for fam in sorted(set(families)):
    idx = np.array([f == fam for f in families])
    n = np.sum(idx)
    ne = np.sum(y[idx])
    print(f"{fam:<16} N={n:<4} err={ne:<4} acc={n-ne:<4} error_rate={ne/max(n,1):.4f}")
print()

print("2. CONTROL DE LONGITUD")
print("-" * 78)
a_res, slope = residualize(a_max, length)
print(f"corr(length,a_max)       : {np.corrcoef(length, a_max)[0,1]:+.5f}")
print(f"slope a_max~length       : {slope:+.6f}")
print(f"AUROC raw a_max          : {auroc(y,a_max):.5f}")
print(f"AUROC residual a_max     : {auroc(y,a_res):.5f}")
print(f"Z raw                     : {zdiff(a_max,y):+.3f}")
print(f"Z residual                : {zdiff(a_res,y):+.3f}\n")

print("3. SENSORES — ROLLOUT LEVEL")
print("-" * 78)
features = {
    "a_max": a_max, "q_max": q_max, "kappa_max": k_max,
    "v_max": v_max, "-margin_min": -margin_min,
    "entropy_mean": entropy_mean, "length": length,
}
for name, x in features.items():
    print(f"{name:<18} AUROC={auroc(y,x):.5f} Z={zdiff(x,y):+.3f}")
print()

print("4. CORRELACIONES")
print("-" * 78)
corr_features = {
    "length": length, "a_max": a_max, "q_max": q_max,
    "kappa_max": k_max, "v_max": v_max, "-margin": -margin_min,
    "entropy": entropy_mean,
}
names = list(corr_features.keys())
print("             " + " ".join(f"{n:>10}" for n in names))
for n1 in names:
    row = [f"{np.corrcoef(corr_features[n1],corr_features[n2])[0,1]:+10.3f}" for n2 in names]
    print(f"{n1:<10}" + "".join(row))
print()

print("5. AUROC DENTRO DE CADA FAMILIA")
print("-" * 78)
for fam in sorted(set(families)):
    idx = np.array([f == fam for f in families])
    yf = y[idx]
    if np.sum(yf) == 0 or np.sum(~yf) == 0:
        print(f"{fam:<16} insuficiente (errores/aciertos = {np.sum(yf)}/{np.sum(~yf)})")
        continue
    print(f"{fam:<16} N={np.sum(idx):<4} a={auroc(yf,a_max[idx]):.4f} margin={auroc(yf,-margin_min[idx]):.4f} q={auroc(yf,q_max[idx]):.4f} kappa={auroc(yf,k_max[idx]):.4f}")
print()

print("6. DETECTOR ONLINE")
print("-" * 78)
for t in [5, 10, 15, 20, 25, 30]:
    a_t, q_t, k_t, m_t, h_t = [], [], [], [], []
    for r in records:
        tr = r["trajectory"][:t]
        if not tr:
            a_t.append(0); q_t.append(0); k_t.append(0); m_t.append(0); h_t.append(0)
            continue
        a_t.append(max(s.get("a_norm",0) for s in tr))
        q_t.append(max(s.get("q",0) for s in tr))
        k_t.append(max(s.get("kappa",0) for s in tr))
        m_t.append(-min(s.get("margin",0) for s in tr))
        h_t.append(np.mean([s.get("entropy",0) for s in tr]))
    a_t, q_t, k_t, m_t = np.asarray(a_t), np.asarray(q_t), np.asarray(k_t), np.asarray(m_t)
    combo = zscore(a_t) + zscore(m_t)
    print(f"t={t:<3} a={auroc(y,a_t):.4f} q={auroc(y,q_t):.4f} k={auroc(y,k_t):.4f} margin={auroc(y,m_t):.4f} combo(a+m)={auroc(y,combo):.4f}")
print()

print("7. COMPLEMENTARIEDAD")
print("-" * 78)
M = zscore(-margin_min)
A = zscore(a_max)
Q = zscore(q_max)
K = zscore(k_max)
print(f"margin                  : {auroc(y,M):.5f}")
print(f"a_max                   : {auroc(y,A):.5f}")
print(f"margin + a_max          : {auroc(y,M+A):.5f}")
print(f"margin + q_max          : {auroc(y,M+Q):.5f}")
print(f"margin + kappa_max      : {auroc(y,M+K):.5f}\n")

print("8. RESUMEN PARA DECISIÓN")
print("-" * 78)
best_t, best_auc = None, -1
for t in [5,10,15,20,25,30]:
    a_t = np.array([max([s.get("a_norm",0.0) for s in r["trajectory"][:t]], default=0.0) for r in records])
    auc = auroc(y,a_t)
    if auc > best_auc:
        best_auc, best_t = auc, t

print(f"best_online_a_max_t     : t={best_t}, AUROC={best_auc:.5f}")
print(f"raw_a_max_AUROC         : {auroc(y,a_max):.5f}")
print(f"residual_a_AUROC        : {auroc(y,a_res):.5f}")
print(f"margin_AUROC             : {auroc(y,-margin_min):.5f}")
print(f"joint_AUROC              : {auroc(y,M+A):.5f}")
print(f"a_vs_length_r            : {np.corrcoef(a_max,length)[0,1]:+.5f}")
print(f"a_vs_margin_r            : {np.corrcoef(a_max,margin_min)[0,1]:+.5f}")
print("\nFIN DEL REPORTE\n" + "=" * 78)
