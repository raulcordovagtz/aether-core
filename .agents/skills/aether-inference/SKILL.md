---
name: aether-inference
description: >
  Ejecuta inferencia multimodal sobre Qwen 27B usando el motor Aether Engine con
  colapso 100% C++ nativo (quantized_matmul en Metal GPU). Usar siempre que se
  necesite generar texto a partir de imágenes con el motor de física de campo continuo.
  Activa este skill cuando el usuario pida: inferir, generar, describir imagen,
  ejecutar el motor, correr Aether, o hacer pruebas de velocidad con el engine.
---

# Aether Inference — Motor C++ Nativo sobre Apple Silicon

## Arquitectura del Pipeline

```text
PYTHON (1 llamada de setup + 1 stream)
  │
  ├─► AetherEngine(model, processor)     ← instala hooks en las 64 capas + lm_head
  ├─► prepare_multimodal_thought(...)    ← calcula L*, z_L_star, activa hooks
  └─► stream_generate(model, ...)        ← genera tokens con motor activo
        │
        │ ════════════ POR CADA TOKEN ════════════
        │
        ├─► [C++] Capa  0-63: dispatch_riemannian_step (geodésica S^{D-1})
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

## Requisitos

- **Modelo local**: `~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit`
- **Módulo compilado**: `aether_vlm/aether_native_c.cpython-313-darwin.so`
- **Dependencias**: `mlx`, `mlx_vlm`, `nanobind`, `Pillow`

## Script de Referencia Canónico

```python
import sys, os, time
import mlx.core as mx
from PIL import Image

sys.path.insert(0, "/Users/crotalo/aether_engine")
from mlx_vlm import load, stream_generate
from aether_vlm import AetherEngine

# ─── 1. CARGA ───────────────────────────────────────────────────
model_path = os.path.expanduser(
    "~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit"
)
model, processor = load(model_path)

# ─── 2. INSTALAR MOTOR AETHER ──────────────────────────────────
aether = AetherEngine(model, processor)

# ─── 3. INGESTIÓN DE IMAGEN ────────────────────────────────────
img_path = "/ruta/a/imagen.jpg"
img = Image.open(img_path).convert("RGB")

prompt = processor.apply_chat_template([
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe la imagen en una frase."}
    ]}
], add_generation_prompt=True)

inputs = processor(text=[prompt], images=[img], return_tensors="mlx")
visual_patches = model.vision_tower(
    inputs["pixel_values"], inputs["image_grid_thw"]
)[0]

# ─── 4. ASENTAMIENTO DE COHERENCIA (τ* = 32) ───────────────────
aether.prepare_multimodal_thought(
    visual_patches,
    "Describe la imagen en una frase."
)

# ─── 5. GENERACIÓN CON COLAPSO C++ ─────────────────────────────
for resp in stream_generate(
    model, processor,
    prompt=prompt,
    image=img_path,
    max_tokens=100
):
    print(resp.text, end="", flush=True)
print()
```

## Parámetros Físicos (Canónicos desde YAML)

| Parámetro | Valor | Origen |
|-----------|-------|--------|
| `nu` (viscosidad) | 0.12 | `spec/collapse/C021_vapor_condensation_collapse.yaml` |
| `gamma` (choque) | 0.35 | `spec/collapse/C021_vapor_condensation_collapse.yaml` |
| `kappa` (nucleación) | 0.15 | `spec/collapse/C021_vapor_condensation_collapse.yaml` |
| `theta_steer` | 0.35 | Paso geodésico por capa |
| `tau_steps` | 32 | Iteraciones de asentamiento profundo |

## Parámetros Ajustables

Para cambiar los parámetros del motor al crear la instancia:

```python
aether = AetherEngine(
    model, processor,
    nu=0.12,        # Amortiguamiento viscoso (0 = sin viscosidad)
    gamma=0.35,     # Intensidad del choque cinético
    kappa=0.15,     # Fuerza de nucleación
    theta_steer=0.35  # Magnitud del paso geodésico por capa
)
```

## Recompilación

Si se modifica `spec/collapse/C021_vapor_condensation_collapse.yaml` o el transpilador:

```bash
cd /Users/crotalo/aether_engine
python3 tools/transpilar_aether_native_aot.py   # Regenera .cpp desde YAML
python3 tools/compilar_extension_c.py            # Compila .so con clang++ C++20
```

> [!CAUTION]
> **Nunca editar `aether_vlm/aether_native.cpp` a mano.** Es generado por el transpilador.
> Toda modificación pasa por el YAML y `tools/transpilar_aether_native_aot.py`.

## Archivos Clave

| Archivo | Rol |
|---------|-----|
| [`aether_vlm/coupler.py`](file:///Users/crotalo/aether_engine/aether_vlm/coupler.py) | Orquestador Python: hooks + extracción de tensores de cuantización |
| [`aether_vlm/aether_native.cpp`](file:///Users/crotalo/aether_engine/aether_vlm/aether_native.cpp) | C++ generado: geodésica + quantized_matmul + condensación |
| [`aether_vlm/aether_native_c.cpython-313-darwin.so`](file:///Users/crotalo/aether_engine/aether_vlm/aether_native_c.cpython-313-darwin.so) | Binario compilado |
| [`aether_vlm/settling.py`](file:///Users/crotalo/aether_engine/aether_vlm/settling.py) | Asentamiento profundo (Deep Thought τ*) |
| [`tools/transpilar_aether_native_aot.py`](file:///Users/crotalo/aether_engine/tools/transpilar_aether_native_aot.py) | Transpilador YAML → C++ |
| [`tools/compilar_extension_c.py`](file:///Users/crotalo/aether_engine/tools/compilar_extension_c.py) | Script de compilación clang++ |
| [`spec/collapse/C021_vapor_condensation_collapse.yaml`](file:///Users/crotalo/aether_engine/spec/collapse/C021_vapor_condensation_collapse.yaml) | Contrato SSOT de parámetros físicos |

## Métricas de Referencia (Qwen 27B 4-bit, Apple Silicon)

| Métrica | Valor |
|---------|-------|
| Carga del modelo | ~1.72 s |
| Asentamiento L* (τ=32) | ~545 ms |
| Velocidad sostenida | **10.29 tok/s** |
| Vocab proyectado en C++ | 248,320 logits |
| Callbacks a Python en colapso | **0** |

## Desactivar Motor (modo vanilla)

```python
aether.set_active(False)  # Todas las capas vuelven a vanilla
# stream_generate ahora opera como mlx_vlm estándar
```
