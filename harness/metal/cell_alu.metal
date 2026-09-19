#include <metal_stdlib>
using namespace metal;

// Código de operación simbólica
enum CellOp : uint {
    OP_AND   = 0,
    OP_OR    = 1,
    OP_XOR   = 2,
    OP_GCD   = 3,
    OP_ADD   = 4,
    OP_MUL   = 5
};

kernel void execute_symbolic_cell(
    device const uint* args_a       [[buffer(0)]],
    device const uint* args_b       [[buffer(1)]],
    device uint* results            [[buffer(2)]],
    constant uint& op_code          [[buffer(3)]],
    uint tid                        [[thread_position_in_grid]])
{
    uint a = args_a[tid];
    uint b = args_b[tid];
    uint res = 0;

    switch (op_code) {
        case OP_AND: res = a & b; break;
        case OP_OR:  res = a | b; break;
        case OP_XOR: res = a ^ b; break; // Resuelve la limitación de Minsky-Papert
        case OP_ADD: res = a + b; break;
        case OP_MUL: res = a * b; break;
        case OP_GCD: {
            while (b != 0) {
                uint t = b;
                b = a % b;
                a = t;
            }
            res = a;
            break;
        }
        default: res = 0; break;
    }
    results[tid] = res;
}
