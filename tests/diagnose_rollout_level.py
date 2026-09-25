#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

DATA_FILE = Path("results/lab30_trajectory_battery.jsonl")

records = []
with DATA_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

N = len(records)
print(f"ANÁLISIS A NIVEL DE TRAYECTORIA REAL (N = {N} ROLLOUTS INDEPENDIENTES)\n")

# Métricas por trayectoria (N=640)
y_err = np.array([not r["correct"] for r in records]) # 1 si error, 0 si acierto
lengths = np.array([len(r["trajectory"]) for r in records])

a_max = np.array([max(s.get("a_norm", 0.0) for s in r["trajectory"]) for r in records])
q_max = np.array([max(s.get("q", 0.0) for s in r["trajectory"]) for r in records])
v_max = np.array([max(s.get("v_norm", 0.0) for s in r["trajectory"]) for r in records])
margin_min = np.array([min(s.get("margin", 0.0) for s in r["trajectory"]) for r in records])
entropy_mean = np.array([np.mean([s.get("entropy", 0.0) for s in r["trajectory"]]) for r in records])

# Estadísticas Rollout-Level Correctos vs Errores
def stats_diff(name, arr):
    c = arr[~y_err]
    e = arr[y_err]
    se = np.sqrt(np.var(c)/len(c) + np.var(e)/len(e) + 1e-12)
    z = (np.mean(e) - np.mean(c)) / se
    print(f"  {name:<24} │ Corr: {np.mean(c):6.3f} ± {np.std(c):5.3f} │ Err: {np.mean(e):6.3f} ± {np.std(e):5.3f} │ Z real: {z:+6.2f} σ")

print("1. SEPARACIÓN REAL A NIVEL DE TRAYECTORIA (GRADOS DE LIBERTAD REALES N=640):")
print("─" * 80)
stats_diff("Longitud (Tokens)", lengths)
stats_diff("Aceleración Máx (a_max)", a_max)
stats_diff("Tensión Máx (q_max)", q_max)
stats_diff("Velocidad Máx (v_max)", v_max)
stats_diff("Margen Mínimo (M_min)", margin_min)
stats_diff("Entropía Media (H_mean)", entropy_mean)
print("─" * 80)

# Correlación entre Aceleración y Margen
corr_a_m = np.corrcoef(a_max, margin_min)[0, 1]
print(f"\n2. ORTOGONALIDAD INFORMACIONAL:")
print(f"  • Correlación Pearson entre Aceleración (a_max) y Margen Mínimo (M_min): {corr_a_m:+.4f}")
if abs(corr_a_m) < 0.30:
    print("  ✓ La aceleración residual es altamente ortogonal al margen de logits (aporta información nueva).")

print("\n3. DESGLOSE DE LONGITUD DE RESPUESTA:")
print(f"  • Longitud media en Aciertos : {np.mean(lengths[~y_err]):.1f} tokens")
print(f"  • Longitud media en Errores  : {np.mean(lengths[y_err]):.1f} tokens")
