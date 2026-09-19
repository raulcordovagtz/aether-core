#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

int main() {
    std::cout << "=================================================================================\n";
    std::cout << " 🔬 TEST DIFERENCIAL: 2 KERNELS SEPARADOS vs 1 KERNEL FUSIONADO CONV+GATED\n";
    std::cout << "=================================================================================\n";

    const uint32_t QKV_DIM = 10240;
    const uint32_t Z_DIM = 6144;
    const uint32_t BA_DIM = 48;
    const uint32_t NUM_HEADS = 48;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSError* err = nil;
        NSURL* libBaseURL = [NSURL fileURLWithPath:@"metal/c_field_qwen38_engine.metallib"];
        id<MTLLibrary> libBase = [device newLibraryWithURL:libBaseURL error:&err];
        id<MTLFunction> fnConv = [libBase newFunctionWithName:@"ssm_conv1d_step_qwen38"];
        id<MTLFunction> fnGated = [libBase newFunctionWithName:@"ssm_gated_delta_update_qwen38"];
        id<MTLComputePipelineState> psoConv = [device newComputePipelineStateWithFunction:fnConv error:&err];
        id<MTLComputePipelineState> psoGated = [device newComputePipelineStateWithFunction:fnGated error:&err];

        NSURL* libFusedURL = [NSURL fileURLWithPath:@"metal/aether_fused_conv_gated.metallib"];
        id<MTLLibrary> libFused = [device newLibraryWithURL:libFusedURL error:&err];
        id<MTLFunction> fnFused = [libFused newFunctionWithName:@"ssm_conv_gated_delta_fused"];
        id<MTLComputePipelineState> psoFused = [device newComputePipelineStateWithFunction:fnFused error:&err];

        // Crear buffers
        id<MTLBuffer> bufQKVIn = [device newBufferWithLength:QKV_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufConvStateSep = [device newBufferWithLength:3 * QKV_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufConvStateFused = [device newBufferWithLength:3 * QKV_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufConvW = [device newBufferWithLength:QKV_DIM * 4 * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufQKVConv = [device newBufferWithLength:QKV_DIM * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufZProj = [device newBufferWithLength:Z_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithLength:BA_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufA = [device newBufferWithLength:BA_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufALog = [device newBufferWithLength:BA_DIM * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDtBias = [device newBufferWithLength:BA_DIM * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufNormW = [device newBufferWithLength:128 * sizeof(uint16_t) options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufSSep = [device newBufferWithLength:NUM_HEADS * 128 * 128 * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufSFused = [device newBufferWithLength:NUM_HEADS * 128 * 128 * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLBuffer> outSep = [device newBufferWithLength:Z_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outFused = [device newBufferWithLength:Z_DIM * sizeof(float) options:MTLResourceStorageModeShared];

        // Inicializar datos sintéticos
        float* qin = (float*)[bufQKVIn contents];
        for (uint32_t i = 0; i < QKV_DIM; ++i) qin[i] = float(i % 50) * 0.02f;
        float* zin = (float*)[bufZProj contents];
        for (uint32_t i = 0; i < Z_DIM; ++i) zin[i] = 1.0f;
        float* bin = (float*)[bufB contents];
        for (uint32_t i = 0; i < BA_DIM; ++i) bin[i] = 0.5f;
        float* ain = (float*)[bufA contents];
        for (uint32_t i = 0; i < BA_DIM; ++i) ain[i] = 0.5f;

        uint16_t* cw = (uint16_t*)[bufConvW contents];
        for (uint32_t i = 0; i < QKV_DIM * 4; ++i) cw[i] = 0x3C00; // 1.0 en FP16
        uint16_t* nw = (uint16_t*)[bufNormW contents];
        for (uint32_t i = 0; i < 128; ++i) nw[i] = 0x3C00;

        uint32_t state_head = 0;
        const int ITERS = 100;

        // 1. Método Separado (2 kernels)
        auto t0_sep = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc1 = [cmd computeCommandEncoder];
            [enc1 setComputePipelineState:psoConv];
            [enc1 setBuffer:bufQKVIn offset:0 atIndex:0];
            [enc1 setBuffer:bufConvStateSep offset:0 atIndex:1];
            [enc1 setBuffer:bufConvW offset:0 atIndex:2];
            [enc1 setBuffer:bufQKVConv offset:0 atIndex:3];
            [enc1 setBytes:&QKV_DIM length:sizeof(uint32_t) atIndex:4];
            [enc1 setBytes:&state_head length:sizeof(uint32_t) atIndex:5];
            [enc1 dispatchThreads:MTLSizeMake(QKV_DIM, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc1 endEncoding];

            id<MTLComputeCommandEncoder> enc2 = [cmd computeCommandEncoder];
            [enc2 setComputePipelineState:psoGated];
            [enc2 setBuffer:bufQKVConv offset:0 atIndex:0];
            [enc2 setBuffer:bufZProj offset:0 atIndex:1];
            [enc2 setBuffer:bufB offset:0 atIndex:2];
            [enc2 setBuffer:bufA offset:0 atIndex:3];
            [enc2 setBuffer:bufALog offset:0 atIndex:4];
            [enc2 setBuffer:bufDtBias offset:0 atIndex:5];
            [enc2 setBuffer:bufNormW offset:0 atIndex:6];
            [enc2 setBuffer:bufSSep offset:0 atIndex:7];
            [enc2 setBuffer:outSep offset:0 atIndex:8];
            [enc2 dispatchThreadgroups:MTLSizeMake(48, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc2 endEncoding];

            [cmd commit];
            [cmd waitUntilCompleted];
        }
        auto t1_sep = std::chrono::high_resolution_clock::now();
        double lat_sep = std::chrono::duration<double, std::micro>(t1_sep - t0_sep).count() / ITERS;

        // 2. Método Fusionado (1 kernel)
        auto t0_fused = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < ITERS; ++it) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:psoFused];
            [enc setBuffer:bufQKVIn offset:0 atIndex:0];
            [enc setBuffer:bufConvStateFused offset:0 atIndex:1];
            [enc setBuffer:bufConvW offset:0 atIndex:2];
            [enc setBytes:&state_head length:sizeof(uint32_t) atIndex:3];
            [enc setBuffer:bufZProj offset:0 atIndex:4];
            [enc setBuffer:bufB offset:0 atIndex:5];
            [enc setBuffer:bufA offset:0 atIndex:6];
            [enc setBuffer:bufALog offset:0 atIndex:7];
            [enc setBuffer:bufDtBias offset:0 atIndex:8];
            [enc setBuffer:bufNormW offset:0 atIndex:9];
            [enc setBuffer:bufSFused offset:0 atIndex:10];
            [enc setBuffer:outFused offset:0 atIndex:11];
            [enc dispatchThreadgroups:MTLSizeMake(48, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];

            [cmd commit];
            [cmd waitUntilCompleted];
        }
        auto t1_fused = std::chrono::high_resolution_clock::now();
        double lat_fused = std::chrono::duration<double, std::micro>(t1_fused - t0_fused).count() / ITERS;

        // 3. Comparación Bit-a-Bit de Salida
        float* p_sep = (float*)[outSep contents];
        float* p_fus = (float*)[outFused contents];
        double max_diff = 0.0;
        for (uint32_t i = 0; i < Z_DIM; ++i) {
            double d = std::abs(double(p_sep[i]) - double(p_fus[i]));
            if (d > max_diff) max_diff = d;
        }

        std::cout << " • Latencia 2 Kernels Separados : " << std::fixed << std::setprecision(2) << lat_sep << " µs\n";
        std::cout << " • Latencia 1 Kernel Fusionado  : " << lat_fused << " µs\n";
        std::cout << " • Aceleración Local            : " << (lat_sep / lat_fused) << "x\n";
        std::cout << " • Discrepancia Máxima out_state: " << std::scientific << max_diff << "\n";

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
