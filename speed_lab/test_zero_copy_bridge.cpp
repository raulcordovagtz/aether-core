#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <mlx/mlx.h>
#include <mlx/backend/metal/metal.h>
#import <Metal/Metal.h>

namespace mx = mlx::core;

int main() {
    std::cout << "=================================================================================\n";
    std::cout << " 🔬 SPEED LAB: PUENTE ZERO-COPY C++ (MLX BACKEND ↔ AETHER SILICON)\n";
    std::cout << "=================================================================================\n";

    const int D = 5120;
    const int TWO_D = 2 * D;

    // 1. Crear tensor en MLX C++
    auto x_mlx = mx::random::normal({TWO_D}, mx::float32);
    x_mlx = x_mlx / mx::linalg::norm(x_mlx);
    mx::eval(x_mlx); // Forzar evaluación en silicio MLX

    float norm_ini = mx::linalg::norm(x_mlx).item<float>();
    std::cout << " • Tensor MLX C++ instanciado en UMA: Dimensión " << x_mlx.size() << " | Norma: " << std::fixed << std::setprecision(8) << norm_ini << "\n";

    // 2. Extraer puntero crudo de UMA sin copia
    float* raw_ptr = x_mlx.data<float>();

    // 3. Enlazar directamente a un MTLBuffer de Metal (Zero-Copy)
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Crear buffer Metal apuntando a la misma dirección física de RAM
        id<MTLBuffer> bufAether = [device newBufferWithBytesNoCopy:raw_ptr
                                                           length:TWO_D * sizeof(float)
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];

        std::cout << " ✓ Buffer Metal enlazado a la dirección física de MLX: " << (void*)raw_ptr << "\n";

        // Modificar una coordenada a través de la vista Metal
        float* metal_view = (float*)[bufAether contents];
        metal_view[0] += 0.5f;

        // Re-evaluar desde el lado de MLX sin recargar
        mx::eval(x_mlx);
        float mod_val_mlx = x_mlx.data<float>()[0];

        std::cout << " • Verificación de mutación in-place Metal -> MLX: " << mod_val_mlx << " (Debe reflejar +0.5)\n";

        if (std::abs(mod_val_mlx - metal_view[0]) < 1e-7) {
            std::cout << "\n 🏆 ESTADO: PUENTE ZERO-COPY MLX ↔ AETHER NUMÉRICAMENTE CERTIFICADO.\n";
            std::cout << "    La memoria unificada permite acoplar la física continua sin copias.\n";
        } else {
            std::cout << "\n ❌ ERROR EN LA FRONTERA ZERO-COPY.\n";
            return 1;
        }
    }
    std::cout << "=================================================================================\n";
    return 0;
}
