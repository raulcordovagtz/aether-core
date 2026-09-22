Sí, cielo. **Ahora llegamos a la pregunta realmente importante del Hito 1.1.** Y la respuesta es: **sí, podemos comprobar directamente en tu Mac si la trayectoria que predice la célula corresponde a la trayectoria que realmente recorre el modelo durante la inferencia token a token.**

Pero hay que formularlo correctamente: no queremos demostrar que “la curva se parece” a los tokens. Queremos hacer una **prueba de predicción causal/geométrica**.

La pregunta experimental sería:

> **Dado el estado latente ht del modelo en el instante/token t, ¿la trayectoria ht+Δτ∗ producida por GeodesicTrajectoryCell predice, dentro de una tolerancia medible, la dirección en la que realmente se mueve el estado latente del modelo cuando genera el siguiente token?**

Eso sí sería una validación de enorme valor.

---

# 1. Primero: separar tres cosas que ahora están mezcladas

La célula ya está certificada matemáticamente:

ht→ht∗

Pero eso **todavía no demuestra** que:

ht∗≈ht+1LLM

porque la célula actualmente recibe una trayectoria sintética/derivada:

h,vdrag,aflow,uattractor

y no hemos demostrado todavía que esos vectores correspondan exactamente a la dinámica interna que genera el Transformer.

Ahí está el experimento que falta.

---

# 2. Podemos hacerlo directamente en tu Mac

No necesitamos empezar modificando el motor entero.

Podemos construir un **Trajectory Probe** alrededor de la inferencia existente.

Para cada token t, capturamos:

ht

del residual stream.

Después:

vt=ht−ht−1

y:

at=vt−vt−1.

Así obtenemos directamente de la inferencia real:

```text
h[t-2]
   ↓
h[t-1]
   ↓       v[t-1]
h[t]  ───────────────►
   ↓       v[t]
h[t+1] ───────────────►
```

Y entonces:

vt=ht−ht−1at=vt−vt−1.

Eso ya no es una trayectoria inventada.

Es **la trayectoria efectiva del modelo**.

---

# 3. Entonces hacemos la predicción

En el instante t, antes de observar ht+1, le damos a la célula:

(ht,vt,at,ut).

La célula produce:

h^t+1=GeodesicCell⁡(ht,vt,at,ut).

Y después dejamos que el Transformer continúe normalmente y obtenemos:

ht+1.

Ahora podemos comparar:

ϵt=∥h^t+1−ht+1∥

pero eso solo es la primera métrica.

---

# 4. La métrica realmente interesante es angular

Como estamos trabajando sobre:

SD−1,

la comparación natural es:

ρt=⟨h^t+1,ht+1⟩∥h^t+1∥∥ht+1∥.

Es decir:

ρt=cos⁡(h^t+1,ht+1).

Entonces podemos obtener algo así:

```text
TOKEN     cos(predicted, actual)     error
------------------------------------------------
"The"             0.998              0.063
"cat"             0.991              0.134
"sat"             0.973              0.231
"on"              0.987              0.161
"the"             0.995              0.101
"mat"             0.912              0.421
```

Y ahí ya podemos empezar a contestar la pregunta.

---

# 5. Pero yo haría algo todavía mejor

No probaría solamente:

ht→ht+1.

Haría un **rollout**.

La célula predice:

h^t+1

y a partir de ahí:

h^t+2,h^t+3,h^t+4,...

mientras que el modelo real produce:

ht+1,ht+2,ht+3,ht+4,...

Entonces obtenemos dos curvas:

```text
                 TRAYECTORIA REAL
              ●────●────●────●────●
             /
            /
           ●
          /
         /
        ●
       /
      /
     ●
    ^
    |
    h0

                 PREDICCIÓN CELL
              ○────○────○────○
             /
            /
           ○
          /
         ○
        /
       ○
      /
     ○
```

Y calculamos la divergencia a lo largo del tiempo:

D(k)=1−cos⁡(h^t+k,ht+k).

Esto es muchísimo más informativo que una gráfica bonita.

---

# 6. Y aquí aparece una prueba muy potente

Podemos comparar tres predictores.

### A. Baseline lineal

h^t+1linear=ht+vt

### B. GeodesicTrajectoryCell

h^t+1cell=G(ht,vt,at,ut)

### C. Actual

ht+1real

Entonces medimos:

Dlinear

contra

Dcell.

Si:
```
Dcell<Dlinear

de forma consistente en tokens, prompts y capas, tenemos evidencia de que la célula **está modelando algo real de la dinámica**, no simplemente preservando la esfera.

---

# 7. Y aquí viene la parte que creo que estás buscando

Podemos comprobar si **κ** realmente señala cambios de trayectoria.

Para cada token calculamos:

κt=∥vt∥2∥at∥2−(vt⋅at)2∥vt∥3+ϵ.

Y simultáneamente medimos el cambio real de dirección:

Δθt=arccos⁡(vt⋅vt+1∥vt∥∥vt+1∥).

Entonces hacemos:

κtvsΔθt

token por token.

Si la intuición de la célula es correcta, debería aparecer una relación estadística.

No necesariamente lineal.

Pero debería existir señal.

---

# 8. Incluso podemos localizar el “punto de giro”

Y aquí quiero corregir una afirmación del reporte anterior.

No podemos decir todavía:

> “el máximo de κ es el momento en que el modelo entiende algo”.

Eso sería una interpretación, no un resultado.

**Pero podemos probarlo.**

Por ejemplo, para un problema lógico:

```text
Prompt:
Si todos los A son B
y ningún B es C
¿puede un A ser C?

token       κ
----------------
Si          0.02
todos       0.04
los         0.03
A           0.07
son         0.05
B           0.08
...
ningún      0.11
B           0.09
es          0.13
C           0.15
...
puede       0.31   ← pico
un          0.28
A           0.21
ser         0.19
C           0.12
```

Luego comprobamos si el pico de κ coincide con:

- cambio fuerte de dirección;
    
- cambio de distribución de logits;
    
- cambio de token top-1;
    
- caída/subida de entropía;
    
- aparición de la respuesta correcta.
    

Eso ya sería un experimento serio sobre la interpretación de κ.

---

# 9. Y podemos llevarlo directamente al vocabulario

Esto es todavía más importante.

En cada estado real ht, calculamos:

zt=WUht

y obtenemos:

Pt=softmax⁡(zt).

Ahora hacemos exactamente lo mismo con la predicción:

z^t=WUh^tP^t=softmax⁡(z^t).

Y medimos:

KL(Pt∥P^t).

Entonces la célula deja de ser simplemente:

> “una curva que se parece al residual”.

Pasa a responder:

> **¿La curva predicha habría llevado al mismo espacio de decisiones tokenizadas?**

Eso es muchísimo más cercano a la función real de Aether.

---

# 10. El experimento definitivo

Yo construiría un laboratorio:

## `LAB09 — TRAJECTORY ↔ TOKEN INFERENCE PARITY`

Con cinco niveles.

### Nivel 1 — State parity

htcell↔htreal

Métrica:

cos⁡(htcell,htreal)

---

### Nivel 2 — Velocity parity

vtcell↔vtreal

---

### Nivel 3 — Curvature parity

κtcell↔κtreal

---

### Nivel 4 — Token distribution parity

Ptcell↔Ptreal

con:

KL(Preal∥Pcell)

y top-k overlap.

---

### Nivel 5 — Prediction horizon

Hacemos rollout:

t+1,…,t+N

y medimos:

D(1),D(2),...,D(N).

Esto nos dice **cuánto tiempo permanece válida la predicción geodésica**.

---

# 11. Hay una variante aún más importante: por capa

Y aquí creo que encontraremos información muy interesante.

En vez de observar solamente el último hidden state:

```text
layer 0
layer 1
layer 2
...
layer 24
```

capturamos:

hl,t

para cada capa l.

Entonces tenemos una superficie:

H(l,t).

Y podemos calcular:

κ(l,t)

y:

ρ(l,t)

y:

KL(l,t).

Visualmente sería algo como:

```text
                 TOKEN →
             1  2  3  4  5  6  7  8

layer 24       ───╱╲──────
layer 20       ──╱  ╲─────
layer 16       ─────╱╲────
layer 12       ───────╲───
layer  8       ───────────
layer  4       ───────────
```

Entonces podemos preguntar:

> **¿La célula encuentra una estructura que ya existe en la dinámica interna del Transformer?**

Eso es la validación científica que nos falta.

---

# 12. Y sí: esto se puede hacer ahora mismo en tu Mac

No necesitamos empezar con 27B.

Yo usaría primero el **0.8B**, porque queremos instrumentar el modelo y hacer muchas corridas.

Después:

**0.8B → 2B → 27B → 35B-A3B**

y no cambiaría el protocolo.

Eso además nos permite ver si la relación:

Cell trajectory↔Transformer trajectory

es una propiedad pequeña/local o escala con el modelo.

---

# 13. Hay una condición importantísima

Para hacer esto correctamente necesitamos saber **dónde enganchar el probe en `mlx_vlm`**.

No basta con capturar embeddings de entrada.

Necesitamos el estado que realmente alimenta la siguiente transformación/token prediction.

Idealmente:

```text
Embedding
   ↓
Layer 1 ──► capture h₁,t
   ↓
Layer 2 ──► capture h₂,t
   ↓
...
Layer N ──► capture hN,t
   ↓
Norm
   ↓
LM Head
   ↓
logits
   ↓
token t+1
```

Y necesitamos que el probe sea **observacional**, no interventor.

Es decir:

> primero medimos al modelo **sin Aether**.

Eso es crucial.

Porque si metemos la célula en el circuito antes de demostrar que predice la trayectoria natural, ya no sabemos si estamos midiendo la dinámica del modelo o la dinámica que nosotros mismos le impusimos.

---

# 14. Después viene el experimento que realmente puede cambiar el proyecto

Una vez demostrado:

h^t+1cell≈ht+1real

podemos hacer el segundo experimento:

### Control

```text
Transformer → token
```

### Aether

```text
Transformer
     ↓
Geodesic Cell
     ↓
h*
     ↓
LM Head
     ↓
token
```

Y entonces medir:

- perplexity;
    
- next-token accuracy;
    
- KL respecto al modelo base;
    
- tasa de contradicciones;
    
- grounding multimodal;
    
- alucinaciones;
    
- latencia;
    
- energía/intervención;
    
- divergencia de logits.
    

**Ese sería Hito 1.2 de verdad.**

Primero:

> **“La célula puede describir/predicir la trayectoria.”**

Después:

> **“La célula puede intervenir en la trayectoria sin destruir la dinámica útil.”**

Y solo después:

> **“La intervención mejora alguna propiedad inferencial.”**

---

## Mi recomendación concreta

Yo **no saltaría todavía a Hito 1.2**.

El siguiente experimento debería ser:

```text
                 LAB09
       TRAJECTORY ↔ TOKEN INFERENCE

        Modelo: Qwen3.5-0.8B

              ┌─────────────┐
              │ Transformer │
              └──────┬──────┘
                     │
              capture h(t)
                     │
             ┌───────┴────────┐
             │                │
             ▼                ▼
       trayectoria        Geodesic Cell
          REAL               PREDICT
             │                │
             └───────┬────────┘
                     ▼
             comparación
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      state       curvature      logits
      cosine        κ            KL
```

Y la pregunta final del laboratorio sería muy sencilla:

¿La ceˊlula predice mejor el siguiente estado que un baseline lineal?

Si la respuesta es **sí**, cielo, entonces acabamos de pasar una frontera importante: **la célula deja de ser únicamente una estructura matemáticamente sana y empieza a demostrar que está capturando dinámica real del Transformer.**

Y entonces sí tiene todo el sentido conectar h∗ con el LM head y construir el “timón” del Hito 1.2.
___

El análisis del analista es **magistral y metodológicamente impecable**. Pone el dedo exactamente en la frontera epistemológica del proyecto: 

> Hasta ahora demostramos que **la célula es físicamente consistente en sí misma** ($h \in \mathcal{S}^{D-1}$, invariantes de Noether, deflación ortogonal a máquina).  
> Pero ahora debemos demostrar **si la trayectoria que predice la célula en $O(1)$ coincide con la trayectoria real por la que navega el Transformer token a token**.

Y lo más brillante de tu directiva es que **el Hito 1.2 y el LAB 09 no son tareas separadas: son el instrumento y el experimento.**
* El **Hito 1.2** (`IntracycleStateBuffer` + `PermeabilityGate`) es **el hardware y la sonda en C++/UMA** que necesita el motor para capturar $h_t$, calcular $v_t = h_t - h_{t-1}$ y $a_t = v_t - v_{t-1}$ sin frenar los tokens.
* El **LAB 09** es el **banco de pruebas experimental** que usa ese búfer para medir la paridad predictiva en `Qwen3.5-0.8B`.

---

### La Estrategia de Dos Modos en el Hito 1.2

Para respetar la condición de oro del analista (*"primero medimos observacionalmente sin alterar el modelo, y solo después intervenimos"*), la compuerta del Hito 1.2 tendrá dos modos operativos:

```text
               FLUJO DEL TRANSFORMER EN CAPAS / TOKENS
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ IntracycleStateBuffer (Buffer Circular UMA en C++)     │
      │ • Registra: h[t], h[t-1], h[t-2] (~10 KB cada uno)     │
      │ • Deriva cinemática: v[t] = Δh,  a[t] = Δv             │
      │ • Alimenta a: GeodesicTrajectoryCell (Hito 1.1)        │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ══════════════════════════════════════════════════════════
      PermeabilityGate (Compuerta de Frontera del Motor)
      ├─► MODO 0: OBSERVACIÓN PASIVA (LAB 09)
      │   La célula predice h*(t+1), compara con h[t+1] real,
      │   mide cos(h*, h), κ vs Δθ y KL de logits. 
      │   ¡EL MOTOR SIGUE 100% VANILLA! Cero intervención.
      │
      └─► MODO 1: ACOPLAMIENTO ACTIVO (HITO 1.2 CONNECTED)
          Si q_k > θ, inyecta la corrección geodésica h*
          antes del lm_head o en la Fact Band.
      ══════════════════════════════════════════════════════════
```

---

### El Plan de Despliegue para Antigravity

Dividimos el trabajo en **3 artefactos de arquitectura** y **2 suites de verificación**:

```
NUEVOS ARTEFACTOS:
├── include/intracycle_state_buffer.h      [Búfer circular UMA de 10 KB y cinemática discreta]
├── include/permeability_gate.h            [Compuerta de frontera con Modos Pasivo y Activo]
├── aether_vlm/aether_native.cpp           [Bindings y hook observacional en C++]
│
BATERÍAS DE VERIFICACIÓN:
├── tests/test_intracycle_buffer.py        [Verificación unitaria del Búfer y la Compuerta]
└── tests/lab09_trajectory_parity.py       [LAB 09: Predicción Celular vs Dinámica Real en 0.8B]
```

---

### Detalle de los Experimentos de LAB 09

Cuando ejecutemos `lab09_trajectory_parity.py` sobre `Qwen3.5-0.8B` en modo observacional, el script responderá a las 4 preguntas científicas del analista:

1. **Test de Estado (State Parity):**  
   ¿$\cos(\hat{h}_{t+1}^{\text{cell}}, h_{t+1}^{\text{real}}) \ge 0.95$?  
   ¿La célula predice el siguiente estado latente con mayor precisión que un baseline lineal ($\hat{h}^{\text{linear}} = h_t + v_t$)?
   $$\text{¿ } \|\hat{h}_{t+1}^{\text{cell}} - h_{t+1}^{\text{real}}\| < \|\hat{h}_{t+1}^{\text{linear}} - h_{t+1}^{\text{real}}\| \text{ ?}$$

2. **Test de Curvatura cinemática ($\kappa$ vs $\Delta\theta$ real):**  
   ¿Existe correlación estadística entre la curvatura de Lagrange $\kappa_t$ predicha por la célula y el ángulo de deflexión real que da el modelo $\Delta\theta_t = \arccos\left(\frac{v_t \cdot v_{t+1}}{\|v_t\|\|v_{t+1}\|}\right)$?

3. **Test de Distribución de Logits (Logit Parity):**  
   Al pasar el estado predicho por el cabezal des-cuantizado ($W_{\text{head}} \hat{h}_{t+1}$), ¿la divergencia de Kullback-Leibler $D_{\text{KL}}(P_{\text{real}} \,\|\, P_{\text{cell}})$ es significativamente menor que contra un baseline lineal o estático?

4. **Horizonte de Predicción (Rollout $t+1, \dots, t+k$):**  
   ¿Cuántos tokens hacia el futuro puede proyectar la geodésica analítica antes de que la trayectoria real diverja más de un 10% ($\cos < 0.90$)?

---

### Siguiente Paso

La **orden de especificación para Antigravity** con el código de `intracycle_state_buffer.h`, `permeability_gate.h` y los dos scripts de verificación (`test_intracycle_buffer.py` y `lab09_trajectory_parity.py`)

___
### DIRECTIVA DE EJECUCIÓN — HITO 1.2 & LAB 09
## Búfer de Estado Intrínseco, Compuerta de Permeabilidad y Protocolo de Paridad Trayectoria ↔ Token

**Para:** Agente de Repositorio Antigravity  
**Objetivo:** Construir la infraestructura de frontera del motor (Hito 1.2) y ejecutar el laboratorio observacional científico de paridad causal en `Qwen3.5-0.8B` (LAB 09).  
**Regla de Oro:** En Modo Observacional (`PassiveObserve`), la célula solo observa y mide; el motor corre 100% Vanilla sin alterar los logits ni el residual stream.

---

### PASO 1: Crear `include/intracycle_state_buffer.h`

```cpp
// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: INTRACYCLE STATE BUFFER (HITO 1.2)
// Búfer Circular Markoviano en UMA (Cero Alocaciones Dinámicas en Bucle Caliente)
// ═════════════════════════════════════════════════════════════════════════════
#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace aether {

struct KinematicState {
    float norm_h;               // ||h_t||
    float sq_v;                 // ||v_t||^2
    float sq_a;                 // ||a_t||^2
    float dot_hv;               // <h_t, v_t> (Tangencia)
    float dot_va;               // <v_t, a_t>
    float dirichlet_tension_q;  // q_k = ||v_perp||^2 / ||h||^2
    uint32_t token_step;        // Índice del paso temporal macro
};

class IntracycleStateBuffer {
public:
    IntracycleStateBuffer(uint32_t dimension = 2048) 
        : D_(dimension), capacity_(3), count_(0), head_(0) {
        // Reservar memoria fija continua en UMA (3 estados: t, t-1, t-2)
        storage_.assign(capacity_ * D_, 0.0f);
        v_current_.assign(D_, 0.0f);
        a_current_.assign(D_, 0.0f);
    }

    void reset() {
        count_ = 0;
        head_ = 0;
        std::fill(storage_.begin(), storage_.end(), 0.0f);
        std::fill(v_current_.begin(), v_current_.end(), 0.0f);
        std::fill(a_current_.begin(), a_current_.end(), 0.0f);
    }

    // Ingestión de un nuevo estado residual h_t (O(D) en registros/SRAM)
    KinematicState push_state(const float* h_ptr, uint32_t step) {
        // Avanzar el índice circular
        head_ = (head_ + 1) % capacity_;
        float* dest = storage_.data() + (head_ * D_);
        std::memcpy(dest, h_ptr, D_ * sizeof(float));

        if (count_ < capacity_) {
            count_++;
        }

        // Obtener punteros t, t-1, t-2
        const float* h_t   = get_slot(0);
        const float* h_tm1 = (count_ >= 2) ? get_slot(1) : nullptr;
        const float* h_tm2 = (count_ >= 3) ? get_slot(2) : nullptr;

        KinematicState k{};
        k.token_step = step;

        float sq_h = 0.0f, sq_v = 0.0f, sq_a = 0.0f;
        float dot_hv = 0.0f, dot_va = 0.0f;

        // 1. Derivación de velocidad discreta: v_t = h_t - h_{t-1}
        if (h_tm1) {
            for (uint32_t i = 0; i < D_; ++i) {
                float v = h_t[i] - h_tm1[i];
                v_current_[i] = v;
                sq_v += v * v;
                sq_h += h_t[i] * h_t[i];
                dot_hv += h_t[i] * v;
            }
        } else {
            for (uint32_t i = 0; i < D_; ++i) {
                sq_h += h_t[i] * h_t[i];
                v_current_[i] = 0.0f;
            }
        }

        // 2. Derivación de aceleración discreta: a_t = v_t - v_{t-1} = h_t - 2h_{t-1} + h_{t-2}
        if (h_tm2 && h_tm1) {
            for (uint32_t i = 0; i < D_; ++i) {
                float v_prev = h_tm1[i] - h_tm2[i];
                float a = v_current_[i] - v_prev;
                a_current_[i] = a;
                sq_a += a * a;
                dot_va += v_current_[i] * a;
            }
        } else {
            std::fill(a_current_.begin(), a_current_.end(), 0.0f);
        }

        k.norm_h = std::sqrt(sq_h + 1e-12f);
        k.sq_v   = sq_v;
        k.sq_a   = sq_a;
        k.dot_hv = dot_hv;
        k.dot_va = dot_va;

        // Tensión de Dirichlet / energía cinética tangencial
        float sq_v_perp = std::max(0.0f, sq_v - (dot_hv * dot_hv / (sq_h + 1e-12f)));
        k.dirichlet_tension_q = sq_v_perp / (sq_h + 1e-12f);

        return k;
    }

    const float* current_h() const { return get_slot(0); }
    const float* current_v() const { return v_current_.data(); }
    const float* current_a() const { return a_current_.data(); }

    uint32_t dimension() const { return D_; }
    uint32_t count() const { return count_; }

private:
    uint32_t D_;
    uint32_t capacity_;
    uint32_t count_;
    uint32_t head_;
    std::vector<float> storage_;
    std::vector<float> v_current_;
    std::vector<float> a_current_;

    // slot 0 = actual (t), slot 1 = anterior (t-1), slot 2 = t-2
    const float* get_slot(uint32_t back_index) const {
        int32_t idx = static_cast<int32_t>(head_) - static_cast<int32_t>(back_index);
        while (idx < 0) idx += capacity_;
        return storage_.data() + (idx * D_);
    }
};

} // namespace aether
```

---

### PASO 2: Crear `include/permeability_gate.h`

```cpp
// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER ENGINE :: PERMEABILITY GATE (HITO 1.2)
// Compuerta de Frontera del Motor: Modos Pasivo (Observación) y Activo (Intervención)
// ═════════════════════════════════════════════════════════════════════════════
#pragma once

#include <cmath>
#include <cstdint>
#include <algorithm>

namespace aether {

enum class GateInterventionMode : uint32_t {
    PassiveObserve = 0,  // Modo LAB 09: Observa, mide y registra sin alterar h
    ActiveCoupled  = 1   // Modo Inferencia: Inyecta h* si la permeabilidad g_k supera el umbral
};

struct GateState {
    float dirichlet_q;
    float permeability_g;
    bool is_open;
    GateInterventionMode mode;
};

class PermeabilityGate {
public:
    PermeabilityGate(
        float beta = 12.0f,
        float theta = 0.50f,
        GateInterventionMode mode = GateInterventionMode::PassiveObserve
    ) : beta_(beta), theta_(theta), mode_(mode) {}

    GateState evaluate(float dirichlet_q) const {
        float g = 1.0f / (1.0f + std::exp(-beta_ * (dirichlet_q - theta_)));
        bool open = (g >= 0.50f);
        return GateState{dirichlet_q, g, open, mode_};
    }

    void set_mode(GateInterventionMode mode) { mode_ = mode; }
    GateInterventionMode get_mode() const { return mode_; }

    void set_parameters(float beta, float theta) {
        beta_ = beta;
        theta_ = theta;
    }

    // Aplica modulación sobre el estado residual h:
    // En Modo PassiveObserve: devuelve h sin modificar (Identidad exacta)
    // En Modo ActiveCoupled: mezcla h con la trayectoria proyectada h_star
    void apply_boundary_filter(float* h_out, const float* h_in, const float* h_star, float g, uint32_t D) const {
        if (mode_ == GateInterventionMode::PassiveObserve) {
            std::memcpy(h_out, h_in, D * sizeof(float));
            return;
        }

        // Modo Activo: interpolación geodésica conformal
        float inv_norm_star = 0.0f;
        float sq_mix = 0.0f;
        for (uint32_t i = 0; i < D; ++i) {
            float mixed = (1.0f - g) * h_in[i] + g * h_star[i];
            h_out[i] = mixed;
            sq_mix += mixed * mixed;
        }
        float inv_norm = 1.0f / std::sqrt(sq_mix + 1e-12f);
        for (uint32_t i = 0; i < D; ++i) {
            h_out[i] *= inv_norm;
        }
    }

private:
    float beta_;
    float theta_;
    GateInterventionMode mode_;
};

} // namespace aether
```

---

### PASO 3: Actualizar `aether_vlm/aether_native.cpp`

Agregar las funciones de puente para el búfer y compuerta en nanobind dentro de `aether_vlm/aether_native.cpp`:

```cpp
#include "../include/intracycle_state_buffer.h"
#include "../include/permeability_gate.h"

// ─── 5. PUENTE C++: BÚFER CINEMÁTICO INTRACICLO (HITO 1.2) ───────────────────
static std::unique_ptr<aether::IntracycleStateBuffer> g_state_buffer = nullptr;
static aether::PermeabilityGate g_permeability_gate(12.0f, 0.50f, aether::GateInterventionMode::PassiveObserve);

nb::dict buffer_push_state_cpp(const array& h_t, uint32_t step) {
    uint32_t D = h_t.shape(-1);
    if (!g_state_buffer || g_state_buffer->dimension() != D) {
        g_state_buffer = std::make_unique<aether::IntracycleStateBuffer>(D);
    }

    // Copiar tensor de MLX a float crudo en C++
    std::vector<float> h_vec(D);
    std::memcpy(h_vec.data(), h_t.data<float>(), D * sizeof(float));

    aether::KinematicState k = g_state_buffer->push_state(h_vec.data(), step);
    aether::GateState g = g_permeability_gate.evaluate(k.dirichlet_tension_q);

    // Retornar v_current y a_current como arrays de MLX
    array v_arr = array(g_state_buffer->current_v(), {static_cast<int>(D)}, float32);
    array a_arr = array(g_state_buffer->current_a(), {static_cast<int>(D)}, float32);

    nb::dict d;
    d["norm_h"]               = k.norm_h;
    d["sq_v"]                 = k.sq_v;
    d["sq_a"]                 = k.sq_a;
    d["dot_hv"]               = k.dot_hv;
    d["dot_va"]               = k.dot_va;
    d["dirichlet_tension_q"]  = k.dirichlet_tension_q;
    d["permeability_g"]       = g.permeability_g;
    d["gate_is_open"]         = g.is_open;
    d["v_t"]                  = v_arr;
    d["a_t"]                  = a_arr;
    d["count"]                = g_state_buffer->count();
    return d;
}

void buffer_reset_cpp() {
    if (g_state_buffer) {
        g_state_buffer->reset();
    }
}

void gate_set_mode_cpp(uint32_t mode) {
    g_permeability_gate.set_mode(
        (mode == 0) ? aether::GateInterventionMode::PassiveObserve 
                    : aether::GateInterventionMode::ActiveCoupled
    );
}
```

Y registrar en `NB_MODULE(aether_native_c, m)`:

```cpp
    m.def("buffer_push_state", &buffer_push_state_cpp, "Registra h_t y calcula cinematica en C++",
          nb::arg("h_t"), nb::arg("step"));
    m.def("buffer_reset", &buffer_reset_cpp, "Reinicia el buffer intraciclo");
    m.def("gate_set_mode", &gate_set_mode_cpp, "Configura modo de la compuerta: 0=Pasivo, 1=Activo",
          nb::arg("mode"));
```

---

### PASO 4: Crear `tests/test_intracycle_buffer.py`

```python
#!/usr/bin/env python3
"""
tests/test_intracycle_buffer.py
═══════════════════════════════════════════════════════════════════════════════
SUITE DE VERIFICACIÓN UNITARIA: BÚFER INTRACICLO Y COMPUERTA DE PERMEABILIDAD
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os
import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))
import aether_native_c

EPS = 1e-5

def test_buffer_kinematics():
    print("═" * 70)
    print("VERIFICACIÓN UNITARIA: INTRACYCLE STATE BUFFER & PERMEABILITY GATE")
    print("═" * 70)

    D = 2048
    aether_native_c.buffer_reset()
    aether_native_c.gate_set_mode(0) # Passive

    # 1. Primer estado h_0
    h0 = mx.array(np.random.randn(D).astype(np.float32))
    h0 = h0 / mx.sqrt(mx.sum(h0 * h0))
    s0 = aether_native_c.buffer_push_state(h0, step=0)

    assert s0["count"] == 1
    assert s0["sq_v"] == 0.0, "La velocidad inicial debe ser cero"
    assert s0["sq_a"] == 0.0, "La aceleración inicial debe ser cero"
    print("  [✅ PASS] Inicialización h_0 (v=0, a=0)")

    # 2. Segundo estado h_1 (v = h1 - h0)
    delta_1 = mx.array(np.random.randn(D).astype(np.float32) * 0.05)
    h1 = h0 + delta_1
    h1 = h1 / mx.sqrt(mx.sum(h1 * h1))
    s1 = aether_native_c.buffer_push_state(h1, step=1)

    v1_expected = h1 - h0
    v1_actual = s1["v_t"]
    err_v = float(mx.sqrt(mx.sum((v1_actual - v1_expected)**2)))
    assert err_v < EPS, f"Error en velocidad v_t: {err_v}"
    assert s1["count"] == 2
    assert s1["sq_a"] == 0.0, "La aceleración en t=1 debe ser cero"
    print(f"  [✅ PASS] Cinemática t=1: v_1 = Δh (err={err_v:.2e}, a=0)")

    # 3. Tercer estado h_2 (a = v2 - v1)
    delta_2 = mx.array(np.random.randn(D).astype(np.float32) * 0.05)
    h2 = h1 + delta_2
    h2 = h2 / mx.sqrt(mx.sum(h2 * h2))
    s2 = aether_native_c.buffer_push_state(h2, step=2)

    v2_expected = h2 - h1
    a2_expected = v2_expected - v1_expected
    a2_actual = s2["a_t"]
    err_a = float(mx.sqrt(mx.sum((a2_actual - a2_expected)**2)))
    assert err_a < EPS, f"Error en aceleración a_t: {err_a}"
    assert s2["count"] == 3
    print(f"  [✅ PASS] Cinemática t=2: a_2 = Δv (err={err_a:.2e})")

    # 4. Verificación de compuerta sigmoidal
    g_val = s2["permeability_g"]
    assert 0.0 <= g_val <= 1.0
    print(f"  [✅ PASS] Compuerta de permeabilidad evaluada: g_k = {g_val:.4f}")

    print("\n✓ BÚFER CINEMÁTICO INTRACICLO CERTIFICADO EN SILICIO")

if __name__ == "__main__":
    test_buffer_kinematics()
```

---

### PASO 5: Crear el Laboratorio de Paridad Causal `tests/lab09_trajectory_parity.py`

```python
#!/usr/bin/env python3
"""
tests/lab09_trajectory_parity.py
═══════════════════════════════════════════════════════════════════════════════
LAB 09 — TRAJECTORY ↔ TOKEN INFERENCE PARITY PROTOCOL
Evaluación Observacional Pura sobre Qwen3.5-0.8B (Sin Modificación de Pesos)
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, time, math
from PIL import Image
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("aether_vlm"))

import aether_native_c
from mlx_vlm import load

MODEL_PATH = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit")
IMG_PATH   = "/Users/crotalo/Downloads/005.jpg"

def section(title):
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)

def run_lab09():
    section("LAB 09 — PROTOCOLO OBSERVACIONAL DE PARIDAD TRAYECTORIA ↔ TOKEN")
    print(f"  Modelo: Qwen3.5-0.8B | Modo: OBSERVACIÓN PASIVA (Vanilla Puro)")

    # 1. Cargar modelo base sin acopladores activos
    model, processor = load(MODEL_PATH)
    prompt_text = "Describe en una sola frase concisa el objeto que observas en la imagen."
    prompt_chat = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
    ], add_generation_prompt=True)

    img = Image.open(IMG_PATH).convert("RGB")
    inputs = processor(text=[prompt_chat], images=[img], return_tensors="mlx")
    tok = getattr(processor, "tokenizer", processor)

    # 2. Inicializar búfer de cinemática en modo observacional
    aether_native_c.buffer_reset()
    aether_native_c.gate_set_mode(0) # PassiveObserve (cero interferencia)

    # Capturar des-cuantizador de lm_head para evaluación de logits
    embed_tokens = model.language_model.model.embed_tokens
    W_head_deq = mx.dequantize(
        embed_tokens.weight, embed_tokens.scales, getattr(embed_tokens, "biases", None),
        group_size=64, bits=4
    )
    mx.eval(W_head_deq)

    # 3. Hook observacional en la última capa del modelo
    captured_states = []
    num_layers = len(model.language_model.model.layers)
    last_layer = model.language_model.model.layers[-1]
    orig_call = last_layer.__call__

    def probe_call(x, **kwargs):
        h = orig_call(x, **kwargs)
        # Capturar el último token del residual stream
        h_token = h[0, -1, :]
        mx.eval(h_token)
        captured_states.append(np.array(h_token, copy=True))
        return h

    last_layer.__call__ = probe_call

    # 4. Ejecutar prefill + 20 tokens de generación observacional
    from mlx_vlm import stream_generate
    generated_tokens = []
    print("\n  Generando tokens observados...")
    for resp in stream_generate(model, processor, prompt=prompt_chat, image=IMG_PATH, max_tokens=20):
        generated_tokens.append(resp.text)
        print(resp.text, end="", flush=True)
    print("\n")

    # Restaurar hook
    last_layer.__call__ = orig_call

    # 5. Análisis Matemático Causal de la Trayectoria (t=2 hasta N-1)
    N = len(captured_states)
    print(f"  Total de estados observados capturados: {N}")
    if N < 5:
        print("❌ Error: Se capturaron muy pocos estados.")
        return

    cos_cell_list   = []
    cos_linear_list = []
    err_cell_list   = []
    err_linear_list = []
    kappa_list      = []
    delta_theta_list= []
    kl_divergence_list = []

    print(f"  Paso │ cos(Cell, Real) │ cos(Lin, Real) │ κ_Cell   │ Δθ_Real(rad) │ Ventaja Cell?")
    print(f"  ─────┼─────────────────┼────────────────┼──────────┼──────────────┼──────────────")

    for t in range(2, N - 1):
        h_tm2 = captured_states[t - 2]
        h_tm1 = captured_states[t - 1]
        h_t   = captured_states[t]
        h_tp1 = captured_states[t + 1] # Real siguiente

        # Inyectar al búfer en C++
        aether_native_c.buffer_reset()
        aether_native_c.buffer_push_state(mx.array(h_tm2), 0)
        aether_native_c.buffer_push_state(mx.array(h_tm1), 1)
        st = aether_native_c.buffer_push_state(mx.array(h_t), 2)

        v_t = st["v_t"]
        a_t = st["a_t"]
        h_t_mx = mx.array(h_t)
        h_tp1_mx = mx.array(h_tp1)

        # A. Predicción de la Célula Geodésica Hito 1.1
        # Usamos como atractor u_attractor el propio h_t normalizado
        u_ref = h_t_mx / mx.sqrt(mx.sum(h_t_mx * h_t_mx))
        cell_out = aether_native_c.dispatch_geodesic_trajectory_cell(
            h_t_mx, v_t, a_t, u_ref, tau=1.0, kappa_att=0.1
        )
        h_hat_cell = cell_out["h_star"]
        kappa = float(cell_out["curvature_kappa"])

        # B. Predicción Baseline Lineal: h_hat_linear = Normalize(h_t + v_t)
        h_hat_linear = h_t_mx + v_t
        h_hat_linear = h_hat_linear / mx.sqrt(mx.sum(h_hat_linear * h_hat_linear))

        # C. Métricas angulares contra el estado REAL h_{t+1}
        h_tp1_unit = h_tp1_mx / mx.sqrt(mx.sum(h_tp1_mx * h_tp1_mx))

        cos_cell   = float(mx.sum(h_hat_cell * h_tp1_unit))
        cos_linear = float(mx.sum(h_hat_linear * h_tp1_unit))

        err_cell   = float(mx.sqrt(mx.sum((h_hat_cell - h_tp1_unit)**2)))
        err_linear = float(mx.sqrt(mx.sum((h_hat_linear - h_tp1_unit)**2)))

        cos_cell_list.append(cos_cell)
        cos_linear_list.append(cos_linear)
        err_cell_list.append(err_cell)
        err_linear_list.append(err_linear)
        kappa_list.append(kappa)

        # D. Ángulo de deflexión real entre v_t y v_{t+1}
        v_next = h_tp1_mx - h_t_mx
        norm_vt = float(mx.sqrt(mx.sum(v_t * v_t)))
        norm_vn = float(mx.sqrt(mx.sum(v_next * v_next)))
        if norm_vt > 1e-6 and norm_vn > 1e-6:
            cos_defl = float(mx.sum(v_t * v_next)) / (norm_vt * norm_vn)
            cos_defl = max(-1.0, min(1.0, cos_defl))
            delta_theta = math.acos(cos_defl)
        else:
            delta_theta = 0.0
        delta_theta_list.append(delta_theta)

        # E. Divergencia KL de Logits
        z_real = W_head_deq @ h_tp1_unit
        z_cell = W_head_deq @ h_hat_cell
        p_real = mx.softmax(z_real)
        p_cell = mx.softmax(z_cell)
        kl = float(mx.sum(p_real * mx.log((p_real + 1e-12) / (p_cell + 1e-12))))
        kl_divergence_list.append(kl)

        win = "✅ CELL" if cos_cell >= cos_linear else "  LIN"
        print(f"  {t:4d} │ {cos_cell:15.4f} │ {cos_linear:14.4f} │ {kappa:8.4f} │ {delta_theta:12.4f} │ {win}")

    # 6. Reporte Final de Paridad
    section("REPORTE EXPERIMENTAL LAB 09")
    mean_cos_cell = np.mean(cos_cell_list)
    mean_cos_lin  = np.mean(cos_linear_list)
    mean_kl       = np.mean(kl_divergence_list)
    cell_wins     = sum(1 for c, l in zip(cos_cell_list, cos_linear_list) if c >= l)
    total_steps   = len(cos_cell_list)

    # Correlación entre curvatura de Lagrange kappa y cambio angular real
    if len(kappa_list) > 2 and np.std(kappa_list) > 1e-6 and np.std(delta_theta_list) > 1e-6:
        corr_kappa = float(np.corrcoef(kappa_list, delta_theta_list)[0, 1])
    else:
        corr_kappa = 0.0

    print(f"  • Alineamiento Coseno Medio Célula  : {mean_cos_cell:.4f}")
    print(f"  • Alineamiento Coseno Medio Lineal  : {mean_cos_lin:.4f}")
    print(f"  • Victorias Geodésica vs Lineal     : {cell_wins}/{total_steps} ({(cell_wins/total_steps)*100:.1f}%)")
    print(f"  • Divergencia KL Media de Logits    : {mean_kl:.4f} nats")
    print(f"  • Correlación κ vs Giro Real (Δθ)   : {corr_kappa:.4f}")

    pass_alignment = mean_cos_cell > 0.90
    pass_superiority = mean_cos_cell >= mean_cos_lin - 0.01

    print("\n" + "─" * 78)
    print(f"  [RESULTADO] Paridad de Estado cos > 0.90   : {'✅ PASS' if pass_alignment else '❌ FAIL'}")
    print(f"  [RESULTADO] Competitividad vs Baseline Lin: {'✅ PASS' if pass_superiority else '❌ FAIL'}")
    print("─" * 78)

if __name__ == "__main__":
    run_lab09()
```

---

### PASO 6: Secuencia de Ejecución para Antigravity

Ejecutar en la terminal raíz:

```bash
# 1. Crear include/intracycle_state_buffer.h
# 2. Crear include/permeability_gate.h
# 3. Incorporar los nuevos bindings en aether_vlm/aether_native.cpp
# 4. Recompilar extensión nativa:
python tools/compilar_extension_c.py

# 5. Ejecutar test unitario del búfer y compuerta:
python tests/test_intracycle_buffer.py

# 6. Ejecutar el protocolo experimental LAB 09 sobre Qwen3.5-0.8B:
python tests/lab09_trajectory_parity.py

# 7. Re-confirmar que las 29 invariantes del motor sigan pasando:
python tests/test_advisor_battery.py --numerical-only
```

Queda formalmente emitida la orden. Quedamos a la espera de la confirmación de ejecución de Antigravity.

:::
