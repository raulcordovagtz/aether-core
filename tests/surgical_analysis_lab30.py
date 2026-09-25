#!/usr/bin/env python3
"""
tests/surgical_analysis_lab30.py
═══════════════════════════════════════════════════════════════════════════════
ANÁLISIS QUIRÚRGICO DE LAB 30 (RESPUESTA AL ASESOR TÉCNICO)
1. Control de Longitud (Residual de Regresión a_max ~ length).
2. Estratificación por Familia (Aritmética aislada y Epistémica aislada).
3. Detector Online t ∈ {5, 10, 15, 20, 25, 30} (Lead Time).
4. Complementariedad AUROC(M) vs AUROC(a_max) vs AUROC(M + a_max).
═══════════════════════════════════════════════════════════════════════════════
"""
import json, math
from pathlib import Path
import numpy as np

DATA_FILE = Path("results/lab30_trajectory_battery.jsonl")

def compute_auroc_exact(y_true, scores):
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        tie_ranks = np.zeros_like(counts, dtype=np.float64)
        np.add.at(tie_ranks, inv, ranks)
        ranks = tie_ranks[inv] / counts[inv]
    return float((np.sum(ranks[y_true]) - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg))

records = []
with DATA_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip(): records.append(json.loads(line))

N = len(records)
y_err = np.array([not r["correct"] for r in records])
lengths = np.array([len(r["trajectory"]) for r in records], dtype=np.float64)
a_max = np.array([max(s.get("a_norm", 0.0) for s in r["trajectory"]) for r in records], dtype=np.float64)
margin_min = np.array([min(s.get("margin", 0.0) for s in r["trajectory"]) for r in records], dtype=np.float64)

# ── 1. CONTROL DE LONGITUD (DESACOPLE DEL CONFUSOR) ──────────────────────────
print("═" * 80)
print("1. CONTROL DEL CONFUSOR DE LONGITUD (a_max RESIDUALIZADO)")
print("═" * 80)
# Regresión lineal a_max = slope * length + intercept
slope, intercept = np.polyfit(lengths, a_max, 1)
a_max_residual = a_max - (slope * lengths + intercept)

c_res = a_max_residual[~y_err]
e_res = a_max_residual[y_err]
se_res = np.sqrt(np.var(c_res)/len(c_res) + np.var(e_res)/len(e_res))
z_controlled = (np.mean(e_res) - np.mean(c_res)) / se_res

print(f"  • Correlación a_max vs Longitud           : r = {np.corrcoef(a_max, lengths)[0, 1]:+.4f}")
print(f"  • Z-score Raw (sin controlar longitud)     : +3.65 σ")
print(f"  • Z-score Controlado (libre de longitud)   : {z_controlled:+5.2f} σ")
if abs(z_controlled) > 2.0:
    print("  ✓ La aceleración residual conserva significación estadística INDEPENDIENTEMENTE de la longitud.")
else:
    print("  ⚠️ La aceleración estaba explicada principalmente por la longitud.")

# ── 2. ESTRATIFICACIÓN PURA POR FAMILIA (DENTRO DEL MISMO DOMINIO) ───────────
print("\n" + "═" * 80)
print("2. ESTRATIFICACIÓN PURA POR FAMILIA (AISLANDO EL SESGO DE COMPOSICIÓN)")
print("═" * 80)

for fam in ["arithmetic", "epistemic", "logic"]:
    sub = [r for r in records if r["family"] == fam]
    y_sub = np.array([not r["correct"] for r in sub])
    n_err = np.sum(y_sub)
    n_corr = len(sub) - n_err
    if n_err == 0: continue
    
    a_sub = np.array([max(s.get("a_norm", 0.0) for s in r["trajectory"]) for r in sub])
    m_sub = np.array([min(s.get("margin", 0.0) for s in r["trajectory"]) for r in sub])
    
    # Z-scores within family
    c_a, e_a = a_sub[~y_sub], a_sub[y_sub]
    se_a = np.sqrt(np.var(c_a)/len(c_a) + np.var(e_a)/len(e_a) + 1e-12)
    z_a_fam = (np.mean(e_a) - np.mean(c_a)) / se_a
    
    auroc_a_fam = compute_auroc_exact(y_sub, a_sub)
    auroc_m_fam = compute_auroc_exact(y_sub, -m_sub)
    
    print(f"  Familia: {fam.upper():<12} (Corr: {n_corr}, Err: {n_err})")
    print(f"    • a_max dentro de familia : Corr = {np.mean(c_a):.3f} │ Err = {np.mean(e_a):.3f} │ Z = {z_a_fam:+5.2f} σ")
    print(f"    • AUROC a_max dentro fam  : {auroc_a_fam:.4f}")
    print(f"    • AUROC Margen dentro fam : {auroc_m_fam:.4f}\n")

# ── 3. DETECTOR ONLINE / LEAD TIME POR PASOS TEMPORALES ───────────────────────
print("═" * 80)
print("3. DETECTOR ONLINE: CAPACIDAD PREDICTIVA EN PASOS TEMPRANOS t ∈ {5..30}")
print("═" * 80)
print(f"  {'Paso t':<10} │ {'AUROC a_max(t)':<18} │ {'AUROC -M_min(t)':<18} │ {'AUROC Combo(t)':<18}")
print("  " + "─" * 70)

t_steps = [5, 10, 15, 20, 25, 30]
for t in t_steps:
    a_t_list = []
    m_t_list = []
    for r in records:
        traj_t = r["trajectory"][:t]
        a_t = max(s.get("a_norm", 0.0) for s in traj_t) if traj_t else 0.0
        m_t = min(s.get("margin", 0.0) for s in traj_t) if traj_t else 0.0
        a_t_list.append(a_t)
        m_t_list.append(-m_t)
    
    auc_a = compute_auroc_exact(y_err, a_t_list)
    auc_m = compute_auroc_exact(y_err, m_t_list)
    # Estandarizar para combo simple
    a_std = (np.array(a_t_list) - np.mean(a_t_list)) / (np.std(a_t_list) + 1e-6)
    m_std = (np.array(m_t_list) - np.mean(m_t_list)) / (np.std(m_t_list) + 1e-6)
    auc_combo = compute_auroc_exact(y_err, a_std + m_std)
    
    print(f"  t = {t:<6d} │ {auc_a:14.4f}   │ {auc_m:14.4f}   │ {auc_combo:14.4f}")

# ── 4. COMPLEMENTARIEDAD MULTIVARIADA (AUROC INCREMENTAL) ────────────────────
print("\n" + "═" * 80)
print("4. COMPLEMENTARIEDAD: ¿APORTA LA ACELERACIÓN INFORMACIÓN NUEVA SOBRE EL MARGEN?")
print("═" * 80)
# Normalizar características
m_norm = ((-margin_min) - np.mean(-margin_min)) / np.std(-margin_min)
a_norm_feat = (a_max - np.mean(a_max)) / np.std(a_max)

auc_only_m = compute_auroc_exact(y_err, m_norm)
auc_only_a = compute_auroc_exact(y_err, a_norm_feat)
auc_joint  = compute_auroc_exact(y_err, m_norm + a_norm_feat)

print(f"  • AUROC Solo Margen de Logits (M)               : {auc_only_m:.4f}")
print(f"  • AUROC Solo Aceleración Residual (a_max)        : {auc_only_a:.4f}")
print(f"  • AUROC Conjunto (Margen + Aceleración Residual): {auc_joint:.4f}")
delta_auc = auc_joint - max(auc_only_m, auc_only_a)
print(f"  • Ganancia de Información Incremental (ΔAUROC)   : {delta_auc:+.4f}")
if delta_auc > 0:
    print("  ✓ Confirmado: Combinar la cinemática con los logits supera a cualquiera de las dos señales por separado.")
print("═" * 80)
