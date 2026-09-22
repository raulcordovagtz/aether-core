#!/usr/bin/env python3
"""
pack_aether_dna.py
==================
Genera el volcado consolidado del ADN de Aether Engine:
fuente limpio, kernels Metal, headers UMA, acople MLX y suites de certificación.
"""

import os
import subprocess
from datetime import datetime

ROOT_DIR = "/Users/crotalo/aether_engine"
OUTPUT_FILE = os.path.join(ROOT_DIR, "aether_core_dna.txt")

# Lista priorizada y quirúrgica de los archivos que definen el motor real
CORE_FILES = [
    # ── 1. C++ NATIVO Y KERNELS DE METAL ──
    "aether_vlm/aether_native.cpp",
    "metal/aether_c008_cognitive_engine.metal",
    "metal/aether_c018_riemannian_engine.metal",
    "metal/aether_geodesic_engine.metal",
    "include/field_inference_engine.h",
    "include/c_field_state.h",
    "include/safetensors_uma.h",
    "include/phase_branch.h",

    # ── 2. RUNTIME DE INFERENCIA CONTINUA (MLX) ──
    "aether_vlm/__init__.py",
    "aether_vlm/coupler.py",
    "aether_vlm/kernels.py",
    "aether_vlm/settling.py",

    # ── 3. SEMILLA CELULAR (PROTOTIPO HARNESS) ──
    "harness/include/cell_harness.h",
    "harness/metal/cell_alu.metal",
    "harness/src/test_harness_runner.mm",

    # ── 4. TRANSPILADORES Y HERRAMIENTAS DE COMPILACIÓN ──
    "tools/transpilar_aether_native_aot.py",
    "tools/compilar_extension_c.py",

    # ── 5. BATERÍAS DE CERTIFICACIÓN Y REPORTE ──
    "tests/test_advisor_battery.py",
    "tests/run_macbook_battery.py",
    ".agents/skills/aether-inference/SKILL.md"
]

def get_git_info():
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT_DIR).decode().strip()
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR).decode().strip()
        return f"Branch: [{branch}] | Commit: [{commit}]"
    except Exception:
        return "Git info no disponible"

def main():
    print(f"[*] Empaquetando ADN de Aether Engine desde: {ROOT_DIR}")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        # Cabecera General
        out.write("=" * 90 + "\n")
        out.write(f"AETHER ENGINE — CORE ARCHITECTURE DNA DUMP\n")
        out.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"Estado Git: {get_git_info()}\n")
        out.write(f"Ubicación raíz: {ROOT_DIR}\n")
        out.write("=" * 90 + "\n\n")

        # Índice / Árbol simplificado
        out.write("### ÍNDICE DE ARCHIVOS INCLUIDOS EN EL VOLCADO ###\n")
        for idx, rel_path in enumerate(CORE_FILES, 1):
            abs_path = os.path.join(ROOT_DIR, rel_path)
            exists = "✓" if os.path.exists(abs_path) else "✗ [NO ENCONTRADO]"
            out.write(f"  {idx:02d}. {exists} {rel_path}\n")
        out.write("\n" + "=" * 90 + "\n\n")

        # Volcado en cascada con títulos de ruta absoluta
        for rel_path in CORE_FILES:
            abs_path = os.path.join(ROOT_DIR, rel_path)
            if not os.path.exists(abs_path):
                print(f"  [!] Saltando (no existe): {rel_path}")
                continue

            print(f"  -> Integrando: {rel_path}")
            out.write("\n" + "#" * 90 + "\n")
            out.write(f"### INICIO DE ARCHIVO: {abs_path}\n")
            out.write(f"### RUTA RELATIVA: {rel_path}\n")
            out.write("#" * 90 + "\n\n")

            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    out.write(f.read())
            except Exception as e:
                out.write(f"\n[ERROR AL LEER ARCHIVO: {e}]\n")

            out.write(f"\n\n### FIN DE ARCHIVO: {abs_path}\n")
            out.write("#" * 90 + "\n\n")

    print("\n" + "=" * 60)
    print(f"✓ Volcado completado con éxito.")
    print(f"✓ Archivo generado: {OUTPUT_FILE}")
    print(f"✓ Tamaño: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
    print("=" * 60)

if __name__ == "__main__":
    main()
