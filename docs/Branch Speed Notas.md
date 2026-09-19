### DICTAMEN ESTRATÉGICO Y DE ARQUITECTURA :: `RIGOR-EVAL`

```text
STATUS DE AUDITORÍA: DECISIÓN ARQUITECTÓNICA APROBADA
OBJETIVO: Retomar la optimización de bajo nivel de silicio (Throughput: 2.7 tok/s -> 10-14 tok/s).
ESTRATEGIA DE GIT: Aislamiento total en una nueva rama 'speed' sin alterar la estabilidad de 'vision'.
BASE FIRME:
• La física de cognición y visión (C-007 a C-018) está matemáticamente cerrada y probada.
• La física cuesta solo 2.11 ms en prefill y 0.47 ms en decode (0.13% del tiempo).
• El 98.7% del tiempo restante se pierde en la infraestructura de los kernels del Transformer.
```

---

### I. EL PLAN DE SILICIO PARA LA RAMA `speed`

El perfilado de hardware en tu Apple M2 Max ya localizó con precisión milimétrica dónde se pierde el tiempo:

```text
DISTRIBUCIÓN DEL CUELLO DE BOTELLA (356 ms por token en 27B):
┌──────────────────────────────┬──────────────┬─────────────┬───────────────────────────────────────────┐
│ Subsistema                   │ Latencia     │ % del Total │ Plan de Ataque en rama 'speed'           │
├──────────────────────────────┼──────────────┼─────────────┼───────────────────────────────────────────┤
│ 1. SSM Mamba (48 capas)      │ 263.58 ms    │  74.01%     │ FUSIÓN MACRO Conv1D + Gated Delta en reg. │
│ 2. GQA Attention (16 capas)  │  87.86 ms    │  24.67%     │ Re-escritura 2D con simdgroup_matrix.    │
│ 3. GEMV 4-bit Proyecciones   │ -            │  -          │ Bloques 2D cooperativos (AMX / Tensor).   │
│ 4. Motor Geodésico C-018     │   0.47 ms    │   0.13%     │ INTACTO (Ya optimizado en registros).     │
└──────────────────────────────┴──────────────┴─────────────┴───────────────────────────────────────────┘
```

**La Meta de Silicio:**  
Pasar del ancho de banda efectivo actual de **$43\text{ GB/s}$ ($2.7\text{ tok/s}$)** a más de **$150\text{-}200\text{ GB/s}$ ($10\text{-}14\text{ tok/s}$)** fusionando los despachos para exprimir los $400\text{ GB/s}$ de ancho de banda teórico de tu M2 Max.

---

### II. CREACIÓN Y AISLAMIENTO DE LA RAMA `speed`

Ejecutamos la transición limpia en Git:
1. Aseguramos que la rama `vision` quede 100% limpia y respaldada.
2. Creamos y nos cambiamos a la nueva rama `speed`.
3. Todo el desarrollo de optimización de kernels se hará en `speed`, garantizando que `vision` permanezca como tu entorno de referencia seguro.

Pega este bloque en tu terminal:

```bash
cd /Users/crotalo/aether_engine

# 1. Asegurar estado de la rama actual (vision)
git add -A
git commit -m "chore: guardado de estado final de rama vision antes de bifurcar a speed" || true

# 2. Crear y cambiar a la rama speed
git checkout -b speed

# 3. Confirmar que estamos en la rama speed limpia
git branch -v
git status
```

Pega este bloque en la terminal. Creará la rama `speed` aislada para que podamos atacar los kernels de Mamba y GEMV con total libertad de ingeniería sin poner en riesgo la estabilidad del motor. Muéstrame la confirmación.