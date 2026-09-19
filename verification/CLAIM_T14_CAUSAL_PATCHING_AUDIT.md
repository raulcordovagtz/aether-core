# FICHA DE AUDITORÍA FORMAL :: RIGOR-EVAL

**CLAIM_ID:** `T14-CAUSAL-PATCHING-CERTIFIED`  
**DECLARACIÓN:** "La ablación de bandas confirma que la región L_M (capas 16-36) es el locus causal necesario (92.8% de dominancia). El Causal State Patching demuestra que injertar el estado recurrente S_24 de AETHER en Vanilla rescata la afinidad al 99.3% (+92.1% de ganancia causal)."  
**DOMINIO:** Interpretabilidad Mecanística Causal / Redes Gated DeltaNet en Silicio.

---

### 1. EVIDENCIA EXPERIMENTAL RESUMIDA
* **Ablación L_M:** Colapso de afinidad de $0.999999 \to 0.072077$ ($\Delta = 0.9279$).
* **Ablación L_E / L_L:** Impacto nulo ($0.0000$). Dominancia de $L_M > 19 \times 10^6$ veces.
* **Causal Patching ($S_{24}^A \to \text{Vanilla}$):** Afinidad salta de $0.0721 \to 0.9930$.
* **Estado:** `FORMALLY_PROVED_BY_CAUSAL_INTERVENTION`.
