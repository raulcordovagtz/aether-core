#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

int main() {
    std::cout << "=================================================================================\n";
    std::cout << " 🔬 TEST COMPARATIVO: BUCLE DINÁMICO vs UNROLL SIMD DIRECTO EN SWIGLU (13,824 filas)\n";
    std::cout << "=================================================================================\n";

    const uint32_t D = 5120;
    const uint32_t I_Dim = 13824;
    const size_t bytes_mat = I_Dim * (D / 2); // 35.38 MB por matriz

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSError* err = nil;
        NSURL* libURL = [NSURL fileURLWithPath:@"metal/test_swiglu_unroll.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];
        id<MTLComputePipelineState> psoLoop   = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"swiglu_dynamic_loop"] error:&err];
        id<MTLComputePipelineState> psoUnroll = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"swiglu_unrolled_fma"] error:&err];

        id<MTLBuffer> bufZ = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> gw = [device newBufferWithLength:bytes_mat options:MTLResourceStorageModeShared];
        id<MTLBuffer> gs = [device newBufferWithLength:I_Dim * (D / 64) * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> gb = [device newBufferWithLength:I_Dim * (D / 64) * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> uw = [device newBufferWithLength:bytes_mat options:MTLResourceStorageModeShared];
        id<MTLBuffer> us = [device newBufferWithLength:I_Dim * (D / 64) * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ub = [device newBufferWithLength:I_Dim * (D / 64) * sizeof(uint16_t) options:MTLResourceStorageModeShared];

        id<MTLBuffer> outLoop   = [device newBufferWithLength:I_Dim * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outUnroll = [device newBufferWithLength:I_Dim * sizeof(float) options:MTLResourceStorageModeShared];

        float* z_ptr = (float*)[bufZ contents];
        for (uint32_t i = 0; i < D; ++i) z_ptr[i] = 1.0f;
        uint32_t* g_ptr = (uint32_t*)[gw contents];
        uint32_t* u_ptr = (uint32_t*)[uw contents];
        for (size_t i = 0; i < bytes_mat / 4; ++i) { g_ptr[i] = 0x2468ACE0; u_ptr[i] = 0x13579BDF; }
        uint16_t* s_ptr = (uint16_t*)[gs contents];
        for (size_t i = 0; i < I_Dim * (D / 64); ++i) s_ptr[i] = 0x3C00;

        uint32_t num_tg = (I_Dim + 3) / 4;
        const int ITERS = 50;

        // 1. Con bucle dinámico
        auto t0_l = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:psoLoop];
            [enc setBuffer:bufZ offset:0 atIndex:0];
            [enc setBuffer:gw offset:0 atIndex:1];
            [enc setBuffer:gs offset:0 atIndex:2];
            [enc setBuffer:gb offset:0 atIndex:3];
            [enc setBuffer:uw offset:0 atIndex:4];
            [enc setBuffer:us offset:0 atIndex:5];
            [enc setBuffer:ub offset:0 atIndex:6];
            [enc setBuffer:outLoop offset:0 atIndex:7];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:8];
            [enc setBytes:&I_Dim length:sizeof(uint32_t) atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        auto t1_l = std::chrono::high_resolution_clock::now();
        double lat_loop = std::chrono::duration<double, std::micro>(t1_l - t0_l).count() / ITERS;

        // 2. Con Unroll SIMD FMA
        auto t0_u = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:psoUnroll];
            [enc setBuffer:bufZ offset:0 atIndex:0];
            [enc setBuffer:gw offset:0 atIndex:1];
            [enc setBuffer:gs offset:0 atIndex:2];
            [enc setBuffer:gb offset:0 atIndex:3];
            [enc setBuffer:uw offset:0 atIndex:4];
            [enc setBuffer:us offset:0 atIndex:5];
            [enc setBuffer:ub offset:0 atIndex:6];
            [enc setBuffer:outUnroll offset:0 atIndex:7];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:8];
            [enc setBytes:&I_Dim length:sizeof(uint32_t) atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        auto t1_u = std::chrono::high_resolution_clock::now();
        double lat_unroll = std::chrono::duration<double, std::micro>(t1_u - t0_u).count() / ITERS;

        // Verificación numérica
        float* p_l = (float*)[outLoop contents];
        float* p_u = (float*)[outUnroll contents];
        double max_diff = 0.0;
        for (uint32_t i = 0; i < I_Dim; ++i) {
            double d = std::abs(double(p_l[i]) - double(p_u[i]));
            if (d > max_diff) max_diff = d;
        }

        std::cout << " • SwiGLU con bucle dinámico (for b_idx) : " << std::fixed << std::setprecision(2) << lat_loop << " µs\n";
        std::cout << " • SwiGLU con Unroll SIMD FMA directo     : " << lat_unroll << " µs\n";
        std::cout << " • Factor de Aceleración en MLP          : " << (lat_loop / lat_unroll) << "x\n";
        std::cout << " • Discrepancia Máxima Numérica          : " << std::scientific << max_diff << "\n";

        if (max_diff == 0.0) {
            std::cout << " 🏆 ESTADO: BIT_EXACT_VALIDATED (Aceleración de instrucciones sin pérdida de precisión).\n";
        } else {
            std::cout << " ❌ ESTADO: FAILED_EQUIVALENCE.\n";
            return 1;
        }
        std::cout << "=================================================================================\n";
    }
    return 0;
}
