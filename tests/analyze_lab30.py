#!/usr/bin/env python3
"""
tests/analyze_lab30.py
═══════════════════════════════════════════════════════════════════════════════
ANALIZADOR ESTADÍSTICO DE LAS 640 TRAYECTORIAS DE LAB 30
Procesa results/lab30_trajectory_battery.jsonl y genera los 4 paneles:
  Panel 1: Desglose por Familia y Precisión Base.
  Panel 2: Separación Estadística Global (Aciertos vs Errores).
  Panel 3: Anticipación Temprana (Tokens t <= 10).
  Panel 4: Discriminación Formal AUROC (Logits vs Cinemática vs Completo).
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, json, math
from pathlib import Path
import numpy as np

DATA_FILE = Path("results/lab30_trajectory_battery.jsonl")

def section(title):
    print("\n" + "═" * 80)
    print(f"  {title}")
    print("═" * 80)

def compute_auroc_exact(y_true, scores):
    """Calcula el AUROC exacto vía suma de rangos de Mann-Whitney (cero dependencias)."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Ordenar por score ascendente
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Manejo de empates
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        tie_ranks = np.zeros_like(counts, dtype=np.float64)
        np.add.at(tie_ranks, inv, ranks)
        ranks = tie_ranks[inv] / counts[inv]
    sum_ranks_pos = np.sum(ranks[y_true])
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))

def main():
    if not DATA_FILE.exists():
        print(f"❌ Error: No se encuentra el archivo {DATA_FILE}")
        sys.exit(1)

    section("CARGANDO DATASET CINEMÁTICO DE LAB 30")
    records = []
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    N_total = len(records)
    total_tokens = sum(len(r["trajectory"]) for r in records)
    print(f"  • Total de rollouts cargados : {N_total}")
    print(f"  • Total de pasos temporales  : {total_tokens:,} tokens analizados")

    # ── PANEL 1: RENDIMIENTO GLOBAL Y DESGLOSE POR FAMILIA ───────────────────
    section("PANEL 1: RENDIMIENTO POR FAMILIA DE PROBLEMAS")
    families = {}
    for r in records:
        fam = r["family"]
        if fam not in families:
            families[fam] = {"total": 0, "correct": 0}
        families[fam]["total"] += 1
        if r["correct"]:
            families[fam]["correct"] += 1

    total_correct = sum(1 for r in records if r["correct"])
    acc_global = (total_correct / N_total) * 100.0

    print(f"  {'Familia':<20} │ {'Total':<8} │ {'Correctos':<10} │ {'Errores':<10} │ {'Precisión Base'}")
    print("  " + "─" * 70)
    for fam, stats in sorted(families.items()):
        acc = (stats["correct"] / stats["total"]) * 100.0
        errs = stats["total"] - stats["correct"]
        print(f"  {fam.capitalize():<20} │ {stats['total']:<8d} │ {stats['correct']:<10d} │ {errs:<10d} │ {acc:6.1f}%")
    print("  " + "─" * 70)
    print(f"  {'GLOBAL TOTAL':<20} │ {N_total:<8d} │ {total_correct:<10d} │ {N_total - total_correct:<10d} │ {acc_global:6.1f}%\n")

    # ── PANEL 2: SEPARACIÓN ESTADÍSTICA CINEMÁTICA (TOKEN-LEVEL) ─────────────
    section("PANEL 2: SEPARACIÓN ESTADÍSTICA (CORRECTOS vs INCORRECTOS)")
    print("  Evaluado a nivel de token sobre todas las trayectorias:\n")
    
    metrics = ["q", "kappa", "v_norm", "a_norm", "entropy", "margin"]
    metric_labels = {
        "q": "Tensión Dirichlet (q)",
        "kappa": "Curvatura Lagrange (κ)",
        "v_norm": "Velocidad (||v||)",
        "a_norm": "Aceleración (||a||)",
        "entropy": "Entropía Shannon (H)",
        "margin": "Margen Top-1/Top-2"
    }

    correct_data = {m: [] for m in metrics}
    incorrect_data = {m: [] for m in metrics}

    for r in records:
        target_dict = correct_data if r["correct"] else incorrect_data
        for step in r["trajectory"]:
            for m in metrics:
                target_dict[m].append(step.get(m, 0.0))

    print(f"  {'Variable':<26} │ {'Correctos (μ ± σ)':<22} │ {'Errores (μ ± σ)':<22} │ {'Separación Z-Score'}")
    print("  " + "─" * 80)

    for m in metrics:
        arr_c = np.array(correct_data[m], dtype=np.float64)
        arr_i = np.array(incorrect_data[m], dtype=np.float64)

        mu_c, std_c = np.mean(arr_c), np.std(arr_c)
        mu_i, std_i = np.mean(arr_i), np.std(arr_i)

        # Z-Score de separación entre medias (Welch)
        se_diff = np.sqrt((std_c**2 / len(arr_c)) + (std_i**2 / len(arr_i)) + 1e-12)
        z_score = (mu_i - mu_c) / se_diff

        label = metric_labels[m]
        str_c = f"{mu_c:6.3f} ± {std_c:5.3f}"
        str_i = f"{mu_i:6.3f} ± {std_i:5.3f}"
        print(f"  {label:<26} │ {str_c:<22} │ {str_i:<22} │ {z_score:+8.2f} σ")

    # ── PANEL 3: ANTICIPACIÓN TEMPRANA (PRIMEROS 10 TOKENS) ──────────────────
    section("PANEL 3: ANTICIPACIÓN TEMPRANA (FÍSICA EN TOKENS t <= 10)")
    print("  ¿Se detecta la divergencia en los primeros 10 tokens antes del cierre?\n")

    early_c = {m: [] for m in metrics}
    early_i = {m: [] for m in metrics}

    for r in records:
        target_dict = early_c if r["correct"] else early_i
        for step in r["trajectory"][:10]: # Solo t <= 10
            for m in metrics:
                target_dict[m].append(step.get(m, 0.0))

    print(f"  {'Variable (t <= 10)':<26} │ {'Correctos (μ)':<16} │ {'Errores (μ)':<16} │ {'Ratio (Err / Corr)'}")
    print("  " + "─" * 76)

    for m in metrics:
        mc = np.mean(early_c[m])
        mi = np.mean(early_i[m])
        ratio = mi / (mc + 1e-12)
        print(f"  {metric_labels[m]:<26} │ {mc:12.4f}     │ {mi:12.4f}     │ {ratio:8.2f}x")

    # ── PANEL 4: DISCRIMINACIÓN AUROC (CAPACIDAD PREDICTIVA) ─────────────────
    section("PANEL 4: DISCRIMINACIÓN PREDICTIVA FORMAL (AUROC DE DETECCIÓN)")
    print("  Evaluando qué señal predice mejor el error a nivel de trayectoria:\n")

    # Características agregadas por trayectoria
    y_error = [not r["correct"] for r in records] # 1 si falló, 0 si acertó

    feat_q_max = [max(s.get("q", 0.0) for s in r["trajectory"]) for r in records]
    feat_q_mean = [np.mean([s.get("q", 0.0) for s in r["trajectory"]]) for r in records]
    feat_a_max = [max(s.get("a_norm", 0.0) for s in r["trajectory"]) for r in records]
    feat_ent_mean = [np.mean([s.get("entropy", 0.0) for s in r["trajectory"]]) for r in records]
    feat_margin_min = [-min(s.get("margin", 0.0) for s in r["trajectory"]) for r in records] # Invertido: menor margen -> mayor riesgo

    # Combinación lineal cinemática pura: q_max + a_max
    feat_kinematic_combo = np.array(feat_q_max) + 0.5 * np.array(feat_a_max)

    # Combinación total: Cinemática + Entropía
    feat_total_combo = feat_kinematic_combo + 0.5 * np.array(feat_ent_mean)

    auroc_q_max = compute_auroc_exact(y_error, feat_q_max)
    auroc_q_mean = compute_auroc_exact(y_error, feat_q_mean)
    auroc_a_max = compute_auroc_exact(y_error, feat_a_max)
    auroc_ent = compute_auroc_exact(y_error, feat_ent_mean)
    auroc_margin = compute_auroc_exact(y_error, feat_margin_min)
    auroc_kin_combo = compute_auroc_exact(y_error, feat_kinematic_combo)
    auroc_total = compute_auroc_exact(y_error, feat_total_combo)

    print(f"  {'Señal Predictiva Evaluada':<35} │ {'Tipo de Señal':<20} │ {'AUROC (Área Bajo la Curva)'}")
    print("  " + "─" * 78)
    print(f"  {'1. Margen Mínimo (-margin_min)':<35} │ {'Logits (Probab)':<20} │ {auroc_margin:8.4f}")
    print(f"  {'2. Entropía Media (H_mean)':<35} │ {'Logits (Probab)':<20} │ {auroc_ent:8.4f}")
    print(f"  {'3. Aceleración Máxima (a_max)':<35} │ {'Cinemática Pura':<20} │ {auroc_a_max:8.4f}")
    print(f"  {'4. Tensión Media (q_mean)':<35} │ {'Cinemática Pura':<20} │ {auroc_q_mean:8.4f}")
    print(f"  {'5. Tensión Máxima (q_max)':<35} │ {'Cinemática Pura':<20} │ {auroc_q_max:8.4f}")
    print(f"  {'6. Combo Cinemático (q_max + a_max)':<35} │ {'Cinemática Pura':<20} │ {auroc_kin_combo:8.4f}")
    print(f"  {'7. AETHER Total (Cinemática + Entropía)':<35} │ {'Multimodal Integrada':<20} │ {auroc_total:8.4f}")

    section("DICTAMEN Y CONCLUSIÓN CIENTÍFICA")
    best_kin = max(auroc_q_max, auroc_kin_combo)
    if best_kin > auroc_ent:
        print(f"  🏆 DESCUBRIMIENTO CONFIRMADO:")
        print(f"     La cinemática del espacio residual (AUROC={best_kin:.4f}) SUPERA a las señales")
        print(f"     de probabilidad superficial (Entropía={auroc_ent:.4f}) detectando errores.")
    else:
        print(f"  ✓ Análisis completado: Cinemática AUROC={best_kin:.4f} vs Logits AUROC={auroc_ent:.4f}.")
    print("═" * 80)

if __name__ == "__main__":
    main()
