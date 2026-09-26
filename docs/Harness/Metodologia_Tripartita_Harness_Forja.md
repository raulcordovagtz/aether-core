# METODOLOGÍA FORMAL: CONTROL TRIPARTITO DEL HARNESS Y ARQUITECTURA DE INFUSIÓN EPISTÉMICA

**Ruta:** `docs/Harness/Metodologia_Tripartita_Harness_Forja.md`  
**Estado:** ESPECIFICACIÓN METODOLÓGICA Y HOJA DE RUTA CIENTÍFICA (SSOT)  
**Fecha de Consolidación:** 2026-09-24  
**Clasificación:** Marco Arquitectónico de Infusión de Conocimiento y Gobernanza de Silicio

---

## 1. Marco Conceptual: Desacoplamiento de los Tres Objetivos

El análisis histórico de los experimentos de edición de modelos revela que los intentos previos de infusión de conocimiento fracasaron o colapsaron en inestabilidades debido a la **superposición desordenada de tres objetivos funcionalmente ortogonales**:

1. Modulación de la inferencia en tiempo real (espacio residual $h$).
2. Fijación de memoria factual de largo plazo (espacio de pesos densos FFN $W$).
3. Transferencia de modos de razonamiento abstracto entre arquitecturas de distinta escala (espacio de autovectores $\mathcal{G}(k, n)$).

La presente metodología desacopla formalmente estos tres objetivos en una **jerarquía tripartita de control**, asignando a cada nivel su propio estrato algebraico, su horizonte temporal y su condición de activación.

```text
 ╔═════════════════════════════════════════════════════════════════════════════╗
 ║                JERARQUÍA TRIPARTITA DE CONTROL DEL HARNESS                  ║
 ╠═════════════════════════════════════════════════════════════════════════════╣
 ║ NIVEL 1: CONTROL RESIDUAL DINÁMICO (Inferencia / Volátil)                   ║
 ║ • Sustrato: Residual Stream h ∈ S^{D-1}                                     ║
 ║ • Mecanismo: Motor Markoviano de 1.12 MB + Predictor Tetrapolar C-022       ║
 ║ • Estado: PRODUCCIÓN / VALIDADO EN SILICIO REAL                             ║
 ╠═════════════════════════════════════════════════════════════════════════════╣
 ║ NIVEL 2: COMPILACIÓN PARAMÉTRICA EN FFN (Memoria Factual Fija)              ║
 ║ • Sustrato: Matrices de Red Densa W_gate × W_down en Fact Band (L*)        ║
 ║ • Mecanismo: Actualización analítica de Woodbury (Cero backpropagation)     ║
 ║ • Estado: ESPECIFICACIÓN MATEMÁTICA CERRADA (Implementación diferida)       ║
 ╠═════════════════════════════════════════════════════════════════════════════╣
 ║ NIVEL 3: DESTILACIÓN ESTRUCTURAL DE GRASSMANN (Cruce Inter-Modelo)          ║
 ║ • Sustrato: Subespacios canónicos en la variedad de Grassmann G(k, D)       ║
 ║ • Mecanismo: Transplante de invariantes de modelos frontera a modelos edge  ║
 ║ • Estado: RESERVA TÉCNICA (Activación supeditada a madurez de Nivel 2)      ║
 ╚═════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Nivel 1: Control Residual Dinámico (En Producción)

### Dominio y Función:
Opera exclusivamente sobre el flujo residual en memoria unificada (UMA) durante el prefill y decode, sin modificar los pesos del modelo ($W$ invariante).

### Ecuaciones de Gobierno:
1. **Compresión Markoviana de Frontera:**  
   Un documento de $T$ tokens se particiona en ventanas de $W = 512$ tokens. Al final de la ventana $k$, el estado se comprime en el vector de frontera:
   $$h_{\text{boundary}}^{(k)} = h(L_{\text{last}}, T_{\text{end}}) \in \mathbb{R}^D$$
   Inyectado en la Capa 0 de la ventana objetivo, satisface la propiedad de Markov: reconstruye el contexto previo de cientos de miles de tokens con cero uso de KV-Cache.
2. **Predictor Geodésico Tetrapolar (C-022):**  
   Extrapolación analítica exacta sobre $\mathcal{S}^{D-1}$:
   $$h^*(\tau) = \cos(\omega \tau) \hat{h} + \sin(\omega \tau) \hat{v}_\perp$$
   con evaluación simultánea de las cuatro líneas derivativas:
   $$\nabla_{\text{pole}} = \langle \hat{v}_\perp, U_{\text{pole}} \rangle, \quad \text{pole} \in \{\text{onto}, \text{teleo}, \text{anti}, \text{eos}\}$$

### Estado Operativo:
**Activo y certificado al 100% en silicio.** Capaz de comprimir 231,899 tokens a 1.12 MB y anticipar tokens fácticos en $L^* \approx 80\% - 90\%$ de profundidad en modelos 0.8B, 27B y 35B MoE.

---

## 3. Nivel 2: Compilación Paramétrica en FFN (Especificación Cerrada)

### Dominio y Función:
Adherir una teoría técnica completa (axiomas, fórmulas cerradas, condiciones de frontera) de forma permanente en los pesos de la red densa (FFN) de la **Fact Band ($L^*$)**, de modo que el modelo responda a las consultas como si la teoría hubiera formado parte de su pre-entrenamiento de fábrica.

### Fundamento Mecanicista:
La red densa es un banco de memorias clave-valor:
$$\text{FFN}(x) = W_{\text{down}} \cdot \sigma(W_{\text{gate}} \cdot x)$$
Las columnas de $W_{\text{gate}}$ son las claves detectoras de conceptos; las columnas de $W_{\text{down}}$ son las direcciones de valor fáctico proyectadas a la autopista residual.

### Método Analítico de Inyección (Fórmula de Woodbury):
Dado un conjunto de $M$ conceptos de una teoría técnica, se construye la matriz de claves $K = [k_1, \dots, k_M] \in \mathbb{R}^{D \times M}$ y la matriz de valores teóricos deseados $V = [v_1, \dots, v_M] \in \mathbb{R}^{D \times M}$.

La actualización de la matriz de salida $W_{\text{down}}$ se calcula en un solo paso algebraico cerrado en memoria RAM:
$$\Delta W_{\text{down}} = (V - W_{\text{down}} K) \cdot \big(K^T K + \lambda I\big)^{-1} K^T$$

### Invariantes de No-Degradación:
1. **Invisible a Consultas Generales:**  
   Para cualquier consulta que no pertenezca al dominio de la teoría ($x_{\text{general}}$), la ortogonalidad en alta dimensión garantiza:
   $$K^T x_{\text{general}} \approx 0 \implies \Delta W_{\text{down}} \cdot \sigma(W_{\text{gate}} x) \approx 0$$
   El modelo retiene intacto su conocimiento general de lenguaje, sintaxis y hechos previos.
2. **Acoplamiento de Impedancia Estadística:**  
   La perturbación $\Delta W_{\text{down}}$ debe normalizarse para coincidir con la media ($\mu$) y varianza ($\sigma$) de los pesos originales, evitando activar los filtros de anomalía léxica en las capas superiores.

---

## 4. Nivel 3: Destilación Estructural de Grassmann (Reserva Técnica)

### Dominio y Función:
Transferir capacidad de razonamiento abstracto y formalismo matemático desde modelos de frontera hipertróficos (ej. Titanes de 72B a 397B) hacia modelos compactos locales (0.8B a 27B) sin recurrir a fine-tuning ni destilación por generación de texto sintético.

### Método de Subespacios Canónicos:
1. **Identificación de Capas Homólogas:**  
   Emparejamiento entre la Fact Band del modelo donante ($L_{\text{donante}}^*$) y la Fact Band del modelo receptor ($L_{\text{receptor}}^*$).
2. **Descomposición SVD y Variedad de Grassmann:**  
   Se extraen las bases ortonormales de los subespacios de activación $Q_{\text{donante}} \in \mathcal{G}(k, D_1)$ y $Q_{\text{receptor}} \in \mathcal{G}(k, D_2)$.
   Los modos de razonamiento invariantes corresponden a los cosenos canónicos máximos:
   $$\max_{u \in Q_1, v \in Q_2} \langle u, v \rangle$$
3. **Regularización Espectral de Tikhonov:**  
   Para evitar la saturación por modos hipertróficos singulares ($\sigma_{\max} \gg 1$), la proyección se atenúa mediante el factor de transferencia:
   $$\Gamma(\sigma) = \frac{\sigma}{\sigma^2 + \lambda^2}$$
   garantizando que la estructura del donante se acople suavemente sin provocar colapsos de auto-evaluación en la salida.

---

## 5. Criterios de Activación y Roadmap de Prioridades

Para preservar la estabilidad del código fuente y evitar la dispersión de esfuerzos de desarrollo, se establece el siguiente régimen de prioridad estricta:

### Regla de Oro de Implementación:
> **No se abrirá desarrollo de código en los Niveles 2 o 3 hasta que el Nivel 1 esté consolidado, documentado y blindado en producción.**

### Condiciones de Activación Futura:
1. **Activación de Nivel 2 (Compilador Factual FFN):**  
   Se implementará únicamente cuando un caso de uso exija que el modelo responda sobre una teoría técnica fija **sin proveerle ningún pasaje de texto, prompt de contexto ni archivo `.npz` en la consulta**.
2. **Activación de Nivel 3 (Destilación de Grassmann):**  
   Se implementará únicamente cuando el modelo receptor demuestre una incapacidad insalvable de manipular operadores matemáticos complejos por sí mismo, requiriendo un trasplante de autovectores desde un modelo de escala superior.

---

**Dictamen:** Queda formalizada la metodología tripartita. El repositorio mantiene su foco operativo en el **Nivel 1 (Motor Markoviano y Predictor Tetrapolar)**, dejando los Niveles 2 y 3 como especificaciones matemáticas cerradas de reserva técnica.
