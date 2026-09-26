#!/usr/bin/env python3
"""
pack_aether_dna.py — VOLCADO CONSOLIDADO DEL ADN SOBERANO DE AETHER ENGINE
==========================================================================
Filtrado estricto contra .gitignore y árbol real del sistema.
Excluye: core_vlm, mlx_cxx_headers, nanobind_mlx, backups, binarios y cachés.
Incluye: Contratos SSOT, C++20, Metal GPU, Motor Markoviano y Suites Causales.
"""

import os
import subprocess
from datetime import datetime

ROOT_DIR = "/Users/crotalo/aether_engine"
OUTPUT_FILE = os.path.join(ROOT_DIR, "aether_core_dna.txt")

CORE_FILES = [
    # ── 1. DOCUMENTACIÓN OFICIAL DE PRODUCCIÓN (docs/Harness/) ──
    "docs/Harness/Motor_Markoviano_y_Predictor_Tetrapolar.md",
    "docs/Harness/PREDICTOR GEODÉSICO/Implementación.md",
    "docs/Harness/Encima intersticial/Nota de herramienta de insercion.md",
    "docs/Harness/Encima intersticial/Pla de acciona.md",
    "docs/Harness/Encima intersticial/UniversalTensorSolverEnzymeSolver.md",
    "docs/Harness/MarkovMemoryCell/Hito 2.2.md",

    # ── 2. ESPECIFICACIONES FORMALES SSOT (spec/) ──
    "spec/C07_dirac_eml_spinor.yaml",
    "spec/C08_attractor_potential.yaml",
    "spec/C13_boolean_attention_algebra.yaml",
    "spec/C16_hybrid_harness_coupling.yaml",
    "spec/C22_tetrapolar_predictor_cell.yaml",
    "spec/branching/C014_phase_branch.yaml",
    "spec/collapse/C021_vapor_condensation_collapse.yaml",
    "spec/control/C020_variational_geodesic_control.yaml",
    "spec/coupling/C018_riemannian_confinement.yaml",
    "spec/simulation/C019_virtual_trajectory_prediction.yaml",

    # ── 3. CABECERAS C++20 EN UMA (include/) ──
    "include/field_invariants.h",
    "include/field_inference_engine.h",
    "include/c_field_state.h",
    "include/safetensors_uma.h",
    "include/phase_branch.h",
    "include/hilbert_memory_cell.h",           # Célula 2 (Memoria 10 KB)
    "include/fact_band_router.h",              # Enrutador de Cresta Fact Band
    "include/tetrapolar_predictor_cell.h",     # Predictor Geodésico Tetrapolar (C-022)

    # ── 4. SHADERS NATIVOS METAL GPU (metal/) ──
    "metal/aether_c008_cognitive_engine.metal",
    "metal/aether_c018_riemannian_engine.metal",
    "metal/aether_geodesic_engine.metal",
    "metal/c_field_qwen38_engine.metal",
    "metal/hilbert_memory_cell.metal",
    "metal/fact_band_router.metal",
    "metal/tetrapolar_predictor_cell.metal",
    "metal/tetrapolar_extractor.metal",

    # ── 5. RUNTIME NATIVO Y DE INFERENCIA (aether_vlm/) ──
    "aether_vlm/__init__.py",
    "aether_vlm/aether_native.cpp",
    "aether_vlm/coupler.py",
    "aether_vlm/kernels.py",
    "aether_vlm/settling.py",

    # ── 6. COPILOTO DETERMINISTA EN SILICIO (harness/) ──
    "harness/include/cell_harness.h",
    "harness/metal/cell_alu.metal",
    "harness/src/test_harness_runner.mm",

    # ── 7. HERRAMIENTAS Y MOTORES MARKOVIANOS (tools/) ──
    "tools/__init__.py",
    "tools/compilar_extension_c.py",
    "tools/transpilar_aether_native_aot.py",
    "tools/universal_constraint_enzyme.py",
    "tools/inspeccionar_volcado.py",
    "tools/construir_almacen_markoviano.py",
    "tools/consultar_almacen_markoviano.py",
    "tools/markov_context_engine.py",

    # ── 8. BATERÍAS BASE Y CERTIFICACIÓN UNITARIA (tests/) ──
    "tests/test_advisor_battery.py",
    "tests/test_hilbert_memory_cell.py",
    "tests/test_fact_band_router.py",
    "tests/test_tetrapolar_predictor.py",
    "tests/test_tetrapolar_extractor.py",
    "tests/test_tetrapolar_real_inference.py",
    "tests/test_universal_constraint_algebra.py",
    "tests/test_analyst_challenge.py",
    "tests/run_macbook_battery.py",
    "tests/infer_35b.py",

    # ── 9. SUITE DE LABORATORIOS CAUSALES CONSOLIDADOS (tests/) ──
    "tests/lab11_fact_band_routing.py",
    "tests/lab13_r3_purified_freeze.py",
    "tests/lab14_neuro_symbolic_closed_loop.py",
    "tests/lab15_r2_causal_cross_matrix.py",
    "tests/lab16_r4_disjoint_calibration.py",
    "tests/lab18_closed_loop_neuro_symbolic.py",
    "tests/lab20_r3_purified_identifiability.py",
    "tests/lab23_symbolic_manifold_separation.py",
    "tests/lab24_r2_held_out_alpha_decoding.py",
    "tests/lab27_shared_vs_domain_decomposition.py",
    "tests/lab28_r2_lodo_leak_free.py",

    # ── 10. HABILIDADES DEL AGENTE ──
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
    missing_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("=" * 90 + "\n")
        out.write("AETHER ENGINE — CORE ARCHITECTURE DNA (SOVEREIGN CELLULAR FABRIC & MARKOV ENGINE)\n")
        out.write(f"Fecha de Consolidación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"Estado Git: {get_git_info()}\n")
        out.write(f"Ubicación raíz: {ROOT_DIR}\n")
        out.write("=" * 90 + "\n\n")

        # Índice de comprobación de integridad
        out.write("### ÍNDICE DE MÓDULOS ACTIVOS DEL MOTOR ###\n")
        for idx, rel_path in enumerate(CORE_FILES, 1):
            abs_path = os.path.join(ROOT_DIR, rel_path)
            if os.path.exists(abs_path):
                size = os.path.getsize(abs_path)
                out.write(f"  {idx:02d}. ✓ {rel_path} ({size:,} bytes)\n")
                found_count += 1
            else:
                out.write(f"  {idx:02d}. ✗ [NO ENCONTRADO] {rel_path}\n")
                missing_count += 1
        out.write(f"\nTotal de módulos catalogados: {len(CORE_FILES)} | Encontrados: {found_count} | Faltantes: {missing_count}\n")
        out.write("=" * 90 + "\n\n")

        # Volcado de código fuente completo
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
    print("✓ Volcado consolidado a conciencia completado.")
    print(f"✓ Archivo generado: {OUTPUT_FILE}")
    print(f"✓ Módulos integrados: {found_count} de {len(CORE_FILES)}")
    if missing_count > 0:
        print(f"⚠️ Advertencia: {missing_count} archivos no encontrados.")
    else:
        print("✓ Coincidencia: 100% de archivos encontrados.")
    print(f"✓ Tamaño total: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
    print("=" * 60)

if __name__ == "__main__":
    main()
