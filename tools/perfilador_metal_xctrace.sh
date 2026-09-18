#!/bin/bash
echo "================================================================================="
echo " 🔬 PERFILADOR DE HARDWARE: METAL SYSTEM TRACE CON XCTRACE"
echo "================================================================================="

# Verificar si xctrace está disponible en el CLI de Xcode
if ! command -v xctrace &> /dev/null; then
    echo "❌ xctrace no está en el PATH estándar. Usando temporizador de alta resolución."
    exit 1
fi

echo "• Grabando traza de GPU de 10 segundos en bin/aether_engine_fast..."
xctrace record --template 'Metal System Trace' \
               --output perfil_aether_gpu.trace \
               --time-limit 10s \
               --launch -- ./bin/aether_engine_fast 50

echo "✓ Traza grabada en: perfil_aether_gpu.trace"
echo "  Puedes abrirla en macOS con: open perfil_aether_gpu.trace"
echo "================================================================================="
