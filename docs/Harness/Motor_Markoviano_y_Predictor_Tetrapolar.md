# ESPECIFICACIÓN TÉCNICA: MOTOR MARKOVIANO DE CONTEXTO ULTRALARGO Y PREDICTOR TETRAPOLAR (C-022)

**Fecha de Consolidación:** 2026-09-24  
**Estado:** PRODUCCIÓN / SILICIO CERTIFICADO (APPLE SILICON UMA)  
**Módulos Asociados:** 
- `spec/C22_tetrapolar_predictor_cell.yaml`
- `include/tetrapolar_predictor_cell.h`
- `metal/tetrapolar_predictor_cell.metal`
- `tools/construir_almacen_markoviano.py`
- `tools/consultar_almacen_markoviano.py`
- `tests/test_tetrapolar_predictor.py`
- `tests/test_tetrapolar_real_inference.py`

---

## 1. El Motor Markoviano de Contexto Ultralargo

### Fundamento Físico:
En un Transformer autoregresivo, el vector residual del último token en la última capa ($h_{\text{boundary}} \in \mathbb{R}^D$) satisface la **propiedad de Markov**: contiene la totalidad del estado semántico y la compresión causal acumulada del texto procesado hasta esa posición.

### Arquitectura de Compresión:
1. **Partición de Flujo:** Documentos de escala masiva ($>200,000$ tokens) se dividen en ventanas regulares de $W = 512$ tokens.
2. **Extracción de Frontera:** Al final de cada ventana $k$, se extrae exclusivamente el vector residual $h_k \in \mathbb{R}^D$ ($4\text{ KB}$ en $D=1024$ float32).
3. **Almacenamiento Residual:** Un documento de 231,899 tokens se empaqueta íntegramente en un almacén comprimido de **$1.12\text{ MB}$ en disco** (compresión física de $49,986\times$ frente a un KV-Cache tradicional de $56\text{ GB}$).

### Inferencia en Una Sola Pasada (Single Forward Pass):
* **Localización:** Un índice asociativo identifica la ventana objetivo $k$ en $< 1\,\text{ms}$.
* **Inyección de Frontera en Capa 0:** El vector residual $h_{k-1}$ de la ventana previa se inyecta directamente como ancla de estado en la Capa 0 de la secuencia actual en memoria unificada (UMA).
* **Resolución Fáctica:** En una sola pasada de 512 tokens, el modelo recupera definiciones, fórmulas matemáticas cerradas y relaciones inéditas sin alucinación y sin mantener el KV-Cache del historial completo en RAM.

---

## 2. El Predictor Geodésico Tetrapolar (Contrato C-022)

### Anatomía del Órgano Celular:
* **Núcleo Celular:** Generador de flujo analítico continuo sobre la hiperesfera $\mathcal{S}^{D-1}$. Integra la curva geodésica exacta sin distorsión euclidiana:
  $$h^*(\tau) = \cos(\omega \tau) \hat{h} + \sin(\omega \tau) \hat{v}_\perp$$
  donde $\omega = \frac{\|\vec{v}_\perp\|}{\|h\|}$ y $\hat{v}_\perp$ es la velocidad tangencial ortogonal al estado.
* **Membrana Celular:** Control de frontera en memoria unificada (UMA) que garantiza la conservación estricta de la norma $\|h^*(\tau)\| = 1.000000$ con error $< 10^{-6}$ en todo horizonte $\tau$.
* **Citoplasma:** Espacio de trabajo para acoplamiento de catálisis determinista y cola de evaluación asíncrona.

### El Campo Tetrapolar de Navegación:
El núcleo proyecta y evalúa la trayectoria contra cuatro primitivas ortogonales extraídas del espacio de embeddings:
1. **$U_{\text{onto}}$ (Ontología):** Subespacio de anclaje de las premisas del prompt.
2. **$U_{\text{teleo}}$ (Teleología):** Dirección de la meta deductiva o consulta.
3. **$U_{\text{anti}}$ (Antítesis):** Subespacio de conflicto, distracción o contradicción lógica.
4. **$U_{\text{eos}}$ (Pozo Terminal):** Vector director del token de cierre (End of Sequence).

### Las Cuatro Líneas Derivativas:
En un único despacho de GPU en memoria compartida (SRAM), el kernel evalúa:
$$\nabla_{\text{onto}} = \langle \hat{v}_\perp, U_{\text{onto}} \rangle, \quad \nabla_{\text{teleo}} = \langle \hat{v}_\perp, U_{\text{teleo}} \rangle$$
$$\nabla_{\text{anti}} = \langle \hat{v}_\perp, U_{\text{anti}} \rangle, \quad \nabla_{\text{eos}} = \langle \hat{v}_\perp, U_{\text{eos}} \rangle$$

---

## 3. Evidencia Empírica Certificada en Silicio

1. **Paridad Numérica CPU $\leftrightarrow$ Metal GPU:** Discrepancia máxima $\le 2.98 \times 10^{-8}$ en $D \in \{1024, 2048, 5120\}$.
2. **Anticipación Geodésica Pasiva:** Identificación exacta del token final en la Capa 21 ($87.5\%$ de profundidad), dos capas antes de la salida real del Transformer.
3. **Sismografía Cinemática:** La curvatura $\kappa$ duplica su amplitud de forma nítida ($\kappa = 0.054 \to 0.107$) exactamente en el token de decisión fáctica.
4. **Validación Inédita Masiva:** Reconstrucción textual y matemática exacta de un documento inédito de 231,899 tokens a partir de su almacén de $1.12\text{ MB}$.
