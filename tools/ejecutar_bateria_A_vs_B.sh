#!/bin/bash

echo "================================================================================="
echo " 🔬 BATERÍA DE AUDITORÍA EXPERIMENTAL: CONFIGURACIÓN A vs CONFIGURACIÓN B"
echo "================================================================================="

# ─── 1. CORRIDA CONFIGURACIÓN A (C = 0) ──────────────────────────────────────
echo -e "\n▶ FASE 1: Transpilando y Ejecutando CONFIGURACIÓN A (Desacoplada C=0)..."
python3 tools/transpilar_c007_aot.py spec/experiments/config_A_decoupled.yaml

xcrun -sdk macosx metal -c metal/aether_c007_spinor_integrator.metal -o metal/aether_c007_spinor_integrator.air
xcrun -sdk macosx metallib metal/aether_c007_spinor_integrator.air -o metal/aether_c007_spinor_integrator.metallib
rm -f metal/aether_c007_spinor_integrator.air

clang++ -O3 -std=c++17 -Iinclude src/main_aether.mm -o bin/aether_engine -framework Metal -framework Foundation

echo "• Ejecutando 200 tokens en Configuración A..."
./bin/aether_engine 200 > salida_EXP_A.txt 2>&1
echo "✓ Configuración A completada (salida_EXP_A.txt)."

# ─── 2. CORRIDA CONFIGURACIÓN B (C = 1/sqrt(D)) ──────────────────────────────
echo -e "\n▶ FASE 2: Transpilando y Ejecutando CONFIGURACIÓN B (Acoplada C=1/√D)..."
python3 tools/transpilar_c007_aot.py spec/experiments/config_B_coupled.yaml

xcrun -sdk macosx metal -c metal/aether_c007_spinor_integrator.metal -o metal/aether_c007_spinor_integrator.air
xcrun -sdk macosx metallib metal/aether_c007_spinor_integrator.air -o metal/aether_c007_spinor_integrator.metallib
rm -f metal/aether_c007_spinor_integrator.air

clang++ -O3 -std=c++17 -Iinclude src/main_aether.mm -o bin/aether_engine -framework Metal -framework Foundation

echo "• Ejecutando 200 tokens en Configuración B..."
./bin/aether_engine 200 > salida_EXP_B.txt 2>&1
echo "✓ Configuración B completada (salida_EXP_B.txt)."

echo -e "\n================================================================================="
echo " 📊 COMPARATIVA DE RESULTADOS (EXTRACTOS GENERADOS):"
echo "================================================================================="
echo -e "\n--- SALIDA CONFIGURACIÓN A (C = 0, Desacoplada) ---"
head -n 25 salida_EXP_A.txt

echo -e "\n--- SALIDA CONFIGURACIÓN B (C = 1/√D, Acoplamiento Activo) ---"
head -n 25 salida_EXP_B.txt
