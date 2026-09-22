#!/usr/bin/env python3
"""
pack_aether_dna.py (Edición Tejido Celular & Enrutador Cinemático)
=================================================================
Genera el volcado consolidado del ADN soberano de Aether Engine:
núcleo continuo, células autónomas (C1 y C2), búferes UMA, kernels Metal,
enlace conformal, enrutador de Fact Band y suites causales (LAB 09, 10 y 11).
"""

import os
import subprocess
from datetime import datetime

ROOT_DIR = "/Users/crotalo/aether_engine"
OUTPUT_FILE = os.path.join(ROOT_DIR, "aether_core_dna.txt")

# Lista priorizada y quirúrgica de los módulos que definen el sistema completo
CORE_FILES = [
    # ── 1. ESPECIFICACIONES FORMALES (SSOT) ──
    "spec/C07_dirac_eml_spinor.yaml",
    "spec/C08_attractor_potential.yaml",
    "spec/C13_boolean_attention_algebra.yaml",
    "spec/C16_hybrid_harness_coupling.yaml",
    "spec/branching/C014_phase_branch.yaml",
    "spec/collapse/C021_vapor_condensation_collapse.yaml",
    "spec/control/C020_variational_geodesic_control.yaml",
    "spec/coupling/C018_riemannian_confinement.yaml",
    "spec/simulation/C019_virtual_trajectory_prediction.yaml",

    # ── 2. CABECERAS C++20: INFRAESTRUCTURA UMA Y TEJIDO CELULAR ──
    "include/field_invariants.h",
    "include/field_inference_engine.h",
    "include/c_field_state.h",
    "include/safetensors_uma.h",
    "include/phase_branch.h",
    # Módulos del nuevo Paradigma Celular:
    "include/geodesic_trajectory_cell.h",      # Célula 1 (Proyectiva / Radar / r)
    "include/intracycle_state_buffer.h",       # Búfer UMA 3-Slot Zero-Copy
    "include/permeability_gate.h",             # Compuerta Dual (Modo 0 vs 1)
    "include/conformal_coupling_junction.h",   # Unión Conformal Inter-Modular
    "include/hilbert_memory_cell.h",           # Célula 2 (Memoria Markoviana 10 KB)
    "include/fact_band_router.h",              # Enrutador Cinemático de Cresta l*

    # ── 3. KERNELS NATIVOS DE METAL GPU ──
    "metal/aether_c008_cognitive_engine.metal",
    "metal/aether_c018_riemannian_engine.metal",
    "metal/aether_geodesic_engine.metal",
    "metal/c_field_qwen38_engine.metal",
    "metal/geodesic_trajectory_cell.metal",    # Kernel de Reducción en 2 Fases (Célula 1)
    "metal/hilbert_memory_cell.metal",         # Kernel Álgebra Booleana de Hilbert (Célula 2)
    "metal/fact_band_router.metal",            # Kernel Enrutador de Fact Band en GPU

    # ── 4. KERNEL C++ PURO Y PUENTE NATIVO NANOBIND ──
    "aether_vlm/aether_native.cpp",            # Dispatcher C++ completo sin callbacks
    "tools/compilar_extension_c.py",
    "tools/transpilar_aether_native_aot.py",

    # ── 5. RUNTIME DE INFERENCIA CONTINUA (MLX) ──
    "aether_vlm/__init__.py",
    "aether_vlm/coupler.py",
    "aether_vlm/kernels.py",
    "aether_vlm/settling.py",

    # ── 6. COPILOTO SIMBÓLICO (ALU DETERMINISTA) ──
    "harness/include/cell_harness.h",
    "harness/metal/cell_alu.metal",
    "harness/src/test_harness_runner.mm",

    # ── 7. BATERÍAS DE CERTIFICACIÓN Y LABORATORIOS CAUSALES ──
    "tests/test_advisor_battery.py",           # Batería base (27/27 tests de invariantes)
    "tests/test_geodesic_trajectory_cell.py",  # Paridad Metal Célula 1 (5 Gates)
    "tests/test_intracycle_buffer.py",         # Verificación de cinemática v_t, a_t
    "tests/test_conformal_coupling.py",        # Batería geométrica de la unión conformal
    "tests/test_hilbert_memory_cell.py",       # Suite de memoria de Hilbert
    "tests/test_intercell_coupling.py",        # Acoplamiento C1 -> C2
    "tests/test_fact_band_router.py",          # Aislamiento del router de cresta
    "tests/lab09_trajectory_parity.py",        # LAB 09 (Persistencia vs Balística)
    "tests/lab10_active_coupling_qwen.py",     # LAB 10 (Vanilla == Passive == Active-0)
    "tests/lab11_fact_band_routing.py",        # LAB 11 (Matriz causal de 8 controles y cresta L=21)
    "tests/run_macbook_battery.py",
    "tests/infer_35b.py",
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
    print(f"[*] Empaquetando ADN Soberano de Aether Engine desde: {ROOT_DIR}")
    
    found_count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        # Cabecera General
        out.write("=" * 90 + "\n")
        out.write("AETHER ENGINE — CORE ARCHITECTURE DNA (CELLULAR FABRIC & KINEMATIC ROUTER)\n")
        out.write(f"Fecha de Consolidación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"Estado Git: {get_git_info()}\n")
        out.write(f"Ubicación raíz: {ROOT_DIR}\n")
        out.write("=" * 90 + "\n\n")

        # Índice / Árbol simplificado
        out.write("### ÍNDICE DE MÓDULOS ACTIVOS DEL MOTOR ###\n")
        for idx, rel_path in enumerate(CORE_FILES, 1):
            abs_path = os.path.join(ROOT_DIR, rel_path)
            if os.path.exists(abs_path):
                size = os.path.getsize(abs_path)
                out.write(f"  {idx:02d}. ✓ {rel_path} ({size:,} bytes)\n")
                found_count += 1
            else:
                out.write(f"  {idx:02d}. ✗ [NO ENCONTRADO] {rel_path}\n")
        out.write(f"\nTotal de módulos catalogados: {len(CORE_FILES)} | Encontrados: {found_count}\n")
        out.write("=" * 90 + "\n\n")

        # Volcado en cascada con títulos de ruta absoluta
        for idx, rel_path in enumerate(CORE_FILES, 1):
            abs_path = os.path.join(ROOT_DIR, rel_path)
            if not os.path.exists(abs_path):
                continue

            print(f"  [{idx:02d}/{len(CORE_FILES)}] Integrando: {rel_path}")
            out.write("\n" + "#" * 90 + "\n")
            out.write(f"### ARCHIVO [{idx:02d}/{len(CORE_FILES)}]: {rel_path}\n")
            out.write(f"### RUTA ABSOLUTA: {abs_path}\n")
            out.write("#" * 90 + "\n\n")

            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    out.write(f.read())
            except Exception as e:
                out.write(f"\n[ERROR AL LEER ARCHIVO: {e}]\n")

            out.write(f"\n\n### FIN DE ARCHIVO: {rel_path}\n")
            out.write("#" * 90 + "\n\n")

    print("\n" + "=" * 60)
    print("✓ Volcado celular completado con éxito.")
    print(f"✓ Archivo generado: {OUTPUT_FILE}")
    print(f"✓ Módulos integrados: {found_count} de {len(CORE_FILES)}")
    print(f"✓ Tamaño total: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
    print("=" * 60)

if __name__ == "__main__":
    main()
