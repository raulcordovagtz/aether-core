---
name: aether-inference
description: >
  Ejecuta inferencia multimodal sobre la familia Qwen (Qwen3.5 0.8B, Qwen3.5 2B y Qwen3.8 27B)
  usando el motor Aether Engine con colapso 100% C++ nativo (quantized_matmul en Metal GPU).
  Usar siempre que se necesite generar texto a partir de imágenes con el motor de física de campo continuo.
  Activa este skill cuando el usuario pida: inferir, generar, describir imagen, cambiar modelo,
  ejecutar el motor, correr Aether, o hacer pruebas de velocidad con el engine.
---

# Aether Inference — Motor C++ Nativo Multimodelo sobre Apple Silicon

Soporta la familia de modelos visuales multimodales de Qwen con adaptación topológica dinámica:
- **Qwen3.5-0.8B** (~34 tok/s — 24 capas, hidden dim 1024, tied embeddings)
- **Qwen3.5-2B** (~30 tok/s — 24 capas, hidden dim 2048, tied embeddings)
- **Qwen3.8-27B** (~10 tok/s — 64 capas, hidden dim 5120, lm_head dedicado)

---

## Catálogo de Modelos Locales Disponibles

| Alias | Parámetros | Capas | Hidden Dim | Tipo de Cabezal | Ruta Local |
|-------|------------|-------|------------|-----------------|------------|
| `0.8b` | 0.8B | 24 | 1,024 | Tied (`embed_tokens`) | `~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit` |
| `2b` | 2B | 24 | 2,048 | Tied (`embed_tokens`) | `~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit` |
| `27b` | 27B | 64 | 5,120 | Dedicado (`lm_head`) | `~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit` |

El motor `AetherEngine` detecta la topología en tiempo de carga automáticamente:
- **Número de capas**: ajusta el paso temporal $\Delta t = 1/N$ y $\theta_{\text{step}} = \theta_{\text{steer}}/N$ dinámicamente.
- **Dimensión latente $D$**: el kernel geodésico y el operador simpléctico se configuran al tamaño exacto ($D \in \{1024, 2048, 5120\}$).
- **Proyección de colapso**: si el modelo utiliza pesos atados (`tie_word_embeddings: true`), el acoplador interpone `TiedLinearHead` para alimentar los buffers de cuantización directos al Metal GPU sin callbacks a Python.

---

## Arquitectura del Pipeline

```text
PYTHON (1 llamada de setup + 1 stream)
  │
  ├─► AetherEngine(model, processor)     ← detecta N capas y tipo de head (auto)
  ├─► prepare_multimodal_thought(...)    ← calcula L*, z_L_star, activa hooks
  └─► stream_generate(model, ...)        ← genera tokens con motor activo
        │
        │ ════════════ POR CADA TOKEN ════════════
        │
        ├─► [C++] Capas 0 a (N-1): dispatch_riemannian_step (geodésica S^{D-1})
        ├─► [C++] dispatch_full_collapse:
        │     ├─► Choque Cinético (gamma)
        │     ├─► quantized_matmul W_head (248,320 logits en GPU)
        │     └─► Condensación de Vapor C-021 (nu, kappa)
        └─► Token decodificado
```

> [!IMPORTANT]
> El objeto `model` pasado a `stream_generate` es **el mismo** donde `AetherEngine` instaló los hooks.
> No se crea una copia — los hooks operan in-place sobre `model.language_model.model.layers[i]`
> y `model.language_model.lm_head`.

---

## Cómo Cambiar de Modelo en Inferencia

Para alternar entre modelos, define el diccionario canónico o selecciona el path correspondiente:

```python
import os

MODEL_REGISTRY = {
    "0.8b": os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit"),
    "2b":   os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit"),
    "27b":  os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit"),
}

# SELECCIÓN RÁPIDA: cambia solo esta clave ("0.8b", "2b" o "27b")
SELECTED_MODEL = "0.8b"
model_path = MODEL_REGISTRY[SELECTED_MODEL]
```

---

## Script Canónico de Referencia

```python
import sys, os, time
import mlx.core as mx
from PIL import Image

sys.path.insert(0, "/Users/crotalo/aether_engine")
sys.path.insert(0, "/Users/crotalo/aether_engine/aether_vlm")

from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

# ─── 1. SELECCIÓN Y CARGA DEL MODELO ─────────────────────────────
MODEL_REGISTRY = {
    "0.8b": os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-MLX-4bit"),
    "2b":   os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit"),
    "27b":  os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit"),
}

SELECTED = "0.8b"  # <-- Cambiar aquí: "0.8b", "2b", o "27b"
model_path = MODEL_REGISTRY[SELECTED]

print(f"Cargando {SELECTED} desde {model_path}...")
model, processor = load(model_path)

# ─── 2. INSTALAR MOTOR AETHER (AUTO-DETECTA TOPOLOGÍA) ───────────
aether = AetherEngine(model, processor)
print(f"✓ Motor conectado ({aether.num_layers} capas, head: {type(model.language_model.lm_head).__name__})")

# ─── 3. INGESTIÓN DE IMAGEN Y TEXTO ─────────────────────────────
img_path = "/Users/crotalo/Downloads/005.jpg"
img = Image.open(img_path).convert("RGB")

prompt_text = "Describe la imagen en una frase concisa."
prompt = processor.apply_chat_template([
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt_text}
    ]}
], add_generation_prompt=True)

inputs = processor(text=[prompt], images=[img], return_tensors="mlx")
visual_patches = model.vision_tower(
    inputs["pixel_values"], inputs["image_grid_thw"]
)[0]

# ─── 4. ASENTAMIENTO DE COHERENCIA (τ* = 32) ─────────────────────
telemetria = aether.prepare_multimodal_thought(visual_patches, prompt_text)
print("✓ Atractor L* asentado en GPU")

# ─── 5. GENERACIÓN CON COLAPSO 100% C++ ──────────────────────────
print("\nGenerando:")
for resp in stream_generate(
    model, processor,
    prompt=prompt,
    image=img_path,
    max_tokens=60
):
    print(resp.text, end="", flush=True)
print()
```

---

## Perfiles de Configuración Automática (`AETHER_MODEL_PROFILES`)

`AetherEngine` selecciona automáticamente el perfil físico óptimo según la topología detectada:

| Perfil | Modelos | Capas $N$ | Dim $D$ | $\theta_{\text{steer}}$ | $\kappa$ | $\nu$ | $\gamma$ | Capas Activas |
|--------|---------|-----------|---------|-------------------------|----------|-------|-------|---------------|
| `compact_tied` | Qwen3.5-0.8B, 2B | 24 | 1024-2048 | 1.40 | 2.00 | 0.08 | 0.95 | 50% (capas 12..24) |
| `frontier_dense` | Qwen3.8-27B | 64 | 5120 | 2.20 | 1.20 | 0.06 | 0.85 | 60% (capas 26..64) |

> [!NOTE]
> - En modelos de 64 capas (`frontier_dense`), las capas iniciales (0..25) operan con recurrencia lineal (Gated DeltaNet); el motor concentra el timoneo geodésico en el 60% superior de las capas (26..64).
> - $\kappa$ se calibra automáticamente a 1.20 en $D=5120$ para compensar la concentración de medida en hiper-esferas $S^{5119}$.

## Parámetros Físicos (Canónicos desde YAML)

| Parámetro | Valor | Origen | Descripción |
|-----------|-------|--------|-------------|
| `nu` | 0.06 - 0.12 | `spec/collapse/C021_vapor_condensation_collapse.yaml` | Amortiguamiento viscoso laminar |
| `gamma` | 0.35 - 0.95 | `spec/collapse/C021_vapor_condensation_collapse.yaml` | Intensidad del choque cinético |
| `kappa` | 1.20 - 2.00 | `spec/collapse/C021_vapor_condensation_collapse.yaml` | Balance de energía de nucleación geodésica |
| `theta_steer` | 1.40 - 2.20 | Geodésica $S^{D-1}$ | Desviación angular total acumulada |
| `slingshot` | True | Proceso de Penrose / Honda Gravitacional | Eyección elástica de inercia y deflación Gram-Schmidt de $L^*$ |
| `tau_steps` | 32 | `aether_vlm/settling.py` | Pasos de evolución Puerto-Hamiltoniana ($\dot{\mathcal{E}} \le 0$) |

Para inicializar con perfiles automáticos (por defecto):
```python
aether = AetherEngine(model, processor)  # Auto-configura compact_tied o frontier_dense
```

O sobrescribiendo parámetros específicos si se desea:
```python
aether = AetherEngine(
    model, processor,
    nu=0.08,
    gamma=0.95,
    kappa=2.00,        # Gravedad en resonancia plena (infinita/constante)
    theta_steer=1.40,  # Máxima autoridad geodésica
    slingshot=True     # Honda de Penrose + Deflación Gram-Schmidt
)
```

---

## Benchmarks de Silicio Medidos en Apple Silicon (Metal GPU)

Mediciones reales ejecutadas con colapso 100% C++ nativo (`quantized_matmul` directo):

| Modelo | Carga | Asentamiento L* (τ=32) | Velocidad GPU | Latencia Colapso C++ |
|--------|-------|------------------------|---------------|----------------------|
| **Qwen3.5-0.8B** | ~0.89 s | ~244 ms | **~33.7 tok/s** | < 0.15 ms / tok |
| **Qwen3.5-2B** | ~0.98 s | ~345 ms | **~29.5 tok/s** | < 0.20 ms / tok |
| **Qwen3.8-27B** | ~1.55 s | ~527 ms | **~10.3 tok/s** | < 0.50 ms / tok |

*En todos los modelos: 0 callbacks a Python en el colapso, 248,320 logits proyectados por token en GPU.*

---

## Desactivar Motor (Modo Vanilla)

```python
aether.set_active(False)  # Desactiva hooks geodésicos y colapso
# El modelo revierte exactamente al comportamiento estándar de mlx_vlm
```
