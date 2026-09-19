#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "cell_harness.h"
#include <iomanip>
#include <iostream>

int main() {
    std::cout << "=================================================================================\n";
    std::cout << " 🛠️ HARNESS DE CELDAS DETERMINISTAS SOBERANAS (APPLE M2 MAX)\n";
    std::cout << "=================================================================================\n";

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];

        CellHarness harness(dev, q);

        double t_us = 0.0;
        // Test 1: Operación XOR exacta (Paridad estricta fuera del alcance de perceptrones lineales)
        uint32_t res_xor = harness.execute_cell_op(0b1100, 0b1010, 2, t_us);
        std::cout << " • Test 1: 0b1100 XOR 0b1010 = " << res_xor << " (Esperado: " << (0b1100 ^ 0b1010) << ") | Latencia: " << std::fixed << std::setprecision(2) << t_us << " µs\n";

        // Test 2: Algoritmo de Euclides (GCD exacto en registros)
        uint32_t res_gcd = harness.execute_cell_op(1071, 462, 3, t_us);
        std::cout << " • Test 2: GCD(1071, 462) = " << res_gcd << " (Esperado: 21) | Latencia: " << t_us << " µs\n";

        // Test 3: Multiplicación Aritmética Cerrada
        uint32_t res_mul = harness.execute_cell_op(47, 63, 5, t_us);
        std::cout << " • Test 3: 47 * 63 = " << res_mul << " (Esperado: 2961) | Latencia: " << t_us << " µs\n";

        std::cout << "---------------------------------------------------------------------------------\n";
        if (res_xor == 0b0110 && res_gcd == 21 && res_mul == 2961) {
            std::cout << " 🏆 ESTADO DEL HARNESS: CERTIFICADO EN SILICIO (Respuestas exactas en microsegundos).\n";
        } else {
            std::cout << " ❌ ERROR EN VERIFICACIÓN DE CELDAS DETERMINISTAS.\n";
            return 1;
        }
        std::cout << "=================================================================================\n";
    }
    return 0;
}
