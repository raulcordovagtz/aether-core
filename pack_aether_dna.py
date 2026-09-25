#!/usr/bin/env python3
"""
pack_aether_dna.py (Edición Tejido Celular & Enrutador Cinemático + Suites Causales)
===================================================================================
Genera el volcado consolidado del ADN soberano de Aether Engine:
núcleo continuo, células autónomas (C1 y C2), búferes UMA, kernels Metal,
enlace conformal, enrutador de Fact Band, herramientas de álgebra universal
y laboratorios de causalidad e interpretabilidad (LAB 09 a LAB 29).
"""

import os
import subprocess
from datetime import datetime

ROOT_DIR = "/Users/crotalo/aether_engine"
OUTPUT_FILE = os.path.join(ROOT_DIR, "aether_core_dna.txt")

# Lista priorizada y quirúrgica de los módulos que definen el sistema completo
CORE_FILES = [
    # ── 0. DOCUMENTACIÓN Y ESPECIFICACIÓN DE INSERCIÓN ──
    "docs/Harness/Nota de herramienta de insercion.md",
    "docs/Harness/UniversalTensorSolverEnzymeSolver.md",

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
    # Módulos del Paradigma Celular:
    "include/hilbert_memory_cell.h",           # Célula 2 (Memoria Markoviana 10 KB)
    "include/fact_band_router.h",              # Enrutador Cinemático de Cresta l*

    # ── 3. KERNELS NATIVOS DE METAL GPU ──
    "metal/aether_c008_cognitive_engine.metal",
    "metal/aether_c018_riemannian_engine.metal",
    "metal/aether_geodesic_engine.metal",
    "metal/c_field_qwen38_engine.metal",
    "metal/hilbert_memory_cell.metal",         # Kernel Álgebra Booleana de Hilbert (Célula 2)
    "metal/fact_band_router.metal",            # Kernel Enrutador de Fact Band en GPU

    # ── 4. KERNEL C++ PURO, HERRAMIENTAS Y ENZIMAS SIMBÓLICAS ──
    "aether_vlm/aether_native.cpp",            # Dispatcher C++ completo sin callbacks
    "tools/__init__.py",
    "tools/universal_constraint_enzyme.py",
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

    # ── 7. BATERÍAS BASE Y CERTIFICACIÓN CINEMÁTICA ──
    "tests/test_advisor_battery.py",           # Batería base (27/27 tests de invariantes)
    "tests/test_hilbert_memory_cell.py",       # Suite de memoria de Hilbert
    "tests/test_fact_band_router.py",          # Aislamiento del router de cresta
    "tests/run_macbook_battery.py",
    "tests/infer_35b.py",

    # ── 8. SUITE DE LABORATORIOS CAUSALES: PRIMERA ETAPA ──
    "tests/lab09_trajectory_parity.py",        # LAB 09 (Persistencia vs Balística)
    "tests/lab11_fact_band_routing.py",        # LAB 11 (Matriz causal de 8 controles y cresta L=21)

    # ── 9. SUITE DE LABORATORIOS CAUSALES: INTERVENCIÓN Y RESIDUAL (LAB 12 - LAB 16) ──
    "tests/lab12_causal_activation_patching.py",
    "tests/lab12_cross_patching_and_digits.py",
    "tests/lab12_formal_matrix_4x4.py",
    "tests/lab12_uca_active_inoculation.py",
    "tests/lab13_causal_state_swap.py",
    "tests/lab13_r2_multid_freeze_test.py",
    "tests/lab13_r3_purified_freeze.py",
    "tests/lab14_neuro_symbolic_closed_loop.py",
    "tests/lab15_cross_instance_symbolic_reconstruction.py",
    "tests/lab15_r2_causal_cross_matrix.py",
    "tests/lab16_abstract_residual_intervention.py",
    "tests/lab16_r2_gain_curve_and_affine.py",
    "tests/lab16_r3_functional_specificity_matrix.py",
    "tests/lab16_r4_disjoint_calibration.py",

    # ── 10. SUITE DE LABORATORIOS CAUSALES: LAZO CERRADO Y RANGO (LAB 17 - LAB 20) ──
    "tests/lab17_full_closed_loop_autonomous.py",
    "tests/lab17_r2_symbolic_specificity.py",
    "tests/lab17_r3_full_inversion_matrix.py",
    "tests/lab17_r4_rank_expansion.py",
    "tests/lab17_r5_rank_sweep_and_grassmann.py",
    "tests/lab18_closed_loop_neuro_symbolic.py",
    "tests/lab19_paraphrase_generalization_closure.py",
    "tests/lab20_continuous_decoding_and_closure.py",
    "tests/lab20_r2_causal_convergence_fail_closed.py",
    "tests/lab20_r3_purified_identifiability.py",
    "tests/lab20_r3_symbolic_identifiability_sweep.py",

    # ── 11. SUITE DE LABORATORIOS CAUSALES: IDENTIFICABILIDAD Y TRANSFERENCIA (LAB 21 - LAB 28) ──
    "tests/lab21_neural_identification_symbolic_filter.py",
    "tests/lab22_neural_identification_causal_loop.py",
    "tests/lab23_symbolic_manifold_separation.py",
    "tests/lab24_scalar_manifold_decoding.py",
    "tests/lab24_r2_held_out_alpha_decoding.py",
    "tests/lab25_coordinate_transfer_across_contexts.py",
    "tests/lab26_cross_domain_coordinate_transfer.py",
    "tests/lab27_shared_vs_domain_decomposition.py",
    "tests/lab28_leave_one_domain_out.py",
    "tests/lab28_r2_lodo_leak_free.py",

    # ── 12. SUITE DE LABORATORIOS CAUSALES: PUZZLES EPISTÉMICOS KRIPKE (LAB 29) ──
    "tests/lab29_epistemic_puzzle_kripke.py",
    "tests/lab29_r2_epistemic_non_lexical.py",
    "tests/lab29_r3_structural_kripke_state.py",
    "tests/lab29_r4_epistemic_rank_expansion.py",
    "tests/lab29_r5_overdetermined_kripke_sweep.py",
    "tests/lab29_r6_functional_classes_competition.py",
    "tests/lab29_r7_causal_subspace_isolation.py",
    "tests/lab29_r8_direct_causal_optimization.py",
    "tests/lab29_r9_true_causal_jacobian.py",

    # ── 13. PRUEBAS DE ÁLGEBRA UNIVERSAL Y PUZZLES COMPLEJOS ──
    "tests/test_universal_constraint_algebra.py",
    "tests/test_analyst_challenge.py",
    "tests/test_hats_puzzle.py",
    "tests/test_hats_single_shot.py",
    "tests/test_hats_neurosymbolic_scaffold.py",
    "tests/test_hats_uca_inoculated.py",

    # ── 14. AGENTES Y HABILIDADES ──
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
        out.write("AETHER ENGINE — CORE ARCHITECTURE DNA (CELLULAR FABRIC & CAUSAL SUITE)\n")
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
    print("✓ Volcado consolidado completado con éxito.")
    print(f"✓ Archivo generado: {OUTPUT_FILE}")
    print(f"✓ Módulos integrados: {found_count} de {len(CORE_FILES)}")
    print(f"✓ Tamaño total: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
    print("=" * 60)

if __name__ == "__main__":
    main()