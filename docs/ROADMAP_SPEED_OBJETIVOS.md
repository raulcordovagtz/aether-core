# HOJA DE RUTA CANÓNICA: BRANCH SPEED (OBJETIVOS GENUINOS)

Este documento reemplaza formalmente las simulaciones descartadas en la auditoría forense.
Los siguientes objetivos se construirán sobre tensores reales en el runtime de Apple Silicon (`core_vlm`).

## 1. TIMÓN ACTIVO GEODÉSICO NATIVO (C-018)
- **Objetivo:** Intervención en el espacio latente $S^{2D-1}$ en la capa causal durante el decode autorregresivo.
- **Mecanismo:** Operador riemanniano intrínseco $\operatorname{Exp}_\Phi(v)$ evaluado en GPU en registros.
- **Criterio de Aceptación:** Modulación real de logits y desvío semántico comprobable usando el MISMO prompt que Vanilla (Cero prompt-hacking).
- **Meta de Velocidad:** $> 12\text{ tok/s}$ sostenidos en Qwen 27B en Apple M2 Max.

## 2. INTERVENCIÓN Y CAUSAL PATCHING MECANICISTA REAL (T14)
- **Objetivo:** Localización del locus causal de factualidad visual en las 64 capas de Qwen 27B.
- **Mecanismo:** Intercepción de activación de capa real ($h_l$) en el grafo de MLX, conmutando el estado latente por el vector de memoria visual fáctica.
- **Criterio de Aceptación:** Medición de probabilidad directa en el token de salida sin bucles sintéticos ni `np.random`.

## 3. CONTROL ÓPTIMO VARIACIONAL GEODÉSICO (C-020)
- **Objetivo:** Detección de caída en cuencas de contradicción o alucinación.
- **Mecanismo:** Sonda de energía de coherencia $\mathcal{E}(\Phi)$ y aplicación de vector de recuperación ortogonal $\Pi_\perp$.
- **Criterio de Aceptación:** Ruptura comprobable de trampas semánticas adversariales.

## 4. INTEGRACIÓN DE MICRO-CELDAS DETERMINISTAS (HARNESS C-013)
- **Objetivo:** Co-procesador lógico en silicio para invariantes formales.
- **Mecanismo:** Activación en Metal (`cell_alu.metal`) para operaciones exactas (aritmética, paridad, álgebra booleana) en $< 150\ \mu\text{s}$.
- **Criterio de Aceptación:** Cero alucinación en respuestas lógicas estrictas.
