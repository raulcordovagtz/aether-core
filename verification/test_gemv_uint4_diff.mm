#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

int main() {
    std::cout << "=================================================================================\n";
    std::cout << " 🔬 BENCHMARK COMPARATIVO: GLOBAL DRAM vs CACHÉ SRAM COMPARTIDA (M2 Max)\n";
    std::cout << "=================================================================================\n";

    const uint32_t D_In = 5120;
    const uint32_t D_Out = 5120;
    const size_t bytes_weights = D_Out * (D_In / 2); // 13.1 MB

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSError* err = nil;
        NSURL* libURL = [NSURL fileURLWithPath:@"metal/test_gemv_uint4.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];
        id<MTLComputePipelineState> psoScalar = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemv_scalar_u32"] error:&err];
        id<MTLComputePipelineState> psoSRAM   = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemv_sram_cached"] error:&err];

        id<MTLBuffer> bufZ = [device newBufferWithLength:D_In * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufW = [device newBufferWithLength:bytes_weights options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufS = [device newBufferWithLength:D_Out * (D_In / 64) * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithLength:D_Out * (D_In / 64) * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outScalar = [device newBufferWithLength:D_Out * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outSRAM   = [device newBufferWithLength:D_Out * sizeof(float) options:MTLResourceStorageModeShared];

        float* z_ptr = (float*)[bufZ contents];
        for (uint32_t i = 0; i < D_In; ++i) z_ptr[i] = 1.0f;
        uint32_t* w_ptr = (uint32_t*)[bufW contents];
        for (size_t i = 0; i < bytes_weights / 4; ++i) w_ptr[i] = 0x2468ACE0;
        uint16_t* s_ptr = (uint16_t*)[bufS contents];
        for (size_t i = 0; i < D_Out * (D_In / 64); ++i) s_ptr[i] = 0x3C00;

        uint32_t num_tg = (D_Out + 3) / 4;
        const int ITERS = 100;

        // 1. Escalar DRAM Global
        auto t0_s = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:psoScalar];
            [enc setBuffer:bufZ offset:0 atIndex:0];
            [enc setBuffer:bufW offset:0 atIndex:1];
            [enc setBuffer:bufS offset:0 atIndex:2];
            [enc setBuffer:bufB offset:0 atIndex:3];
            [enc setBuffer:outScalar offset:0 atIndex:4];
            [enc setBytes:&D_In length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&D_Out length:sizeof(uint32_t) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        auto t1_s = std::chrono::high_resolution_clock::now();
        double lat_scalar = std::chrono::duration<double, std::micro>(t1_s - t0_s).count() / ITERS;

        // 2. Vectorial + SRAM Cached
        auto t0_sr = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:psoSRAM];
            [enc setBuffer:bufZ offset:0 atIndex:0];
            [enc setBuffer:bufW offset:0 atIndex:1];
            [enc setBuffer:bufS offset:0 atIndex:2];
            [enc setBuffer:bufB offset:0 atIndex:3];
            [enc setBuffer:outSRAM offset:0 atIndex:4];
            [enc setBytes:&D_In length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&D_Out length:sizeof(uint32_t) atIndex:6];
            [enc setThreadgroupMemoryLength:5120 * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        auto t1_sr = std::chrono::high_resolution_clock::now();
        double lat_sram = std::chrono::duration<double, std::micro>(t1_sr - t0_sr).count() / ITERS;

        // Verificación numérica
        float* ptr_s = (float*)[outScalar contents];
        float* ptr_sr = (float*)[outSRAM contents];
        double max_diff = 0.0;
        for (uint32_t i = 0; i < D_Out; ++i) {
            double d = std::abs(double(ptr_s[i]) - double(ptr_sr[i]));
            if (d > max_diff) max_diff = d;
        }

        double bw_scalar = (bytes_weights / 1e9) / (lat_scalar / 1e6);
        double bw_sram   = (bytes_weights / 1e9) / (lat_sram / 1e6);

        std::cout << " • DRAM Global (uint32_t sin caché)  : " << std::fixed << std::setprecision(2) << lat_scalar << " µs (Ancho de Banda: " << bw_scalar << " GB/s)\n";
        std::cout << " • SRAM Cached (uint4 + 20KB en SRAM): " << lat_sram << " µs (Ancho de Banda: " << bw_sram << " GB/s)\n";
        std::cout << " • Factor de Aceleración             : " << (lat_scalar / lat_sram) << "x\n";
        std::cout << " • Discrepancia Máxima Numérica      : " << std::scientific << max_diff << "\n";

        if (max_diff == 0.0) {
            std::cout << " 🏆 ESTADO: BIT_EXACT_VALIDATED (Equivalencia matemática exacta).\n";
        } else {
            std::cout << " ❌ ESTADO: FAILED_EQUIVALENCE.\n";
            return 1;
        }
        std::cout << "=================================================================================\n";
    }
    return 0;
}
