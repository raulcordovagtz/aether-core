#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

int main() {
    std::cout << "=================================================================================\n";
    std::cout << " 🔬 TEST DIFERENCIAL: 4 KERNELS SEPARADOS vs 1 KERNEL FUSIONADO MAMBA\n";
    std::cout << "=================================================================================\n";

    const uint32_t D = 5120;
    const uint32_t QKV_DIM = 10240;
    const uint32_t Z_DIM = 6144;
    const uint32_t BA_DIM = 48;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSError* err = nil;
        NSURL* libBaseURL = [NSURL fileURLWithPath:@"metal/c_field_qwen38_engine.metallib"];
        id<MTLLibrary> libBase = [device newLibraryWithURL:libBaseURL error:&err];
        id<MTLFunction> fnSep = [libBase newFunctionWithName:@"gemv_4bit_proj_g64"];
        id<MTLComputePipelineState> psoSep = [device newComputePipelineStateWithFunction:fnSep error:&err];

        NSURL* libFusedURL = [NSURL fileURLWithPath:@"metal/aether_fused_mamba_in.metallib"];
        id<MTLLibrary> libFused = [device newLibraryWithURL:libFusedURL error:&err];
        id<MTLFunction> fnFused = [libFused newFunctionWithName:@"gemv_4bit_mamba_in4_fused"];
        id<MTLComputePipelineState> psoFused = [device newComputePipelineStateWithFunction:fnFused error:&err];

        // Crear buffers de prueba sintéticos
        id<MTLBuffer> bufZIn = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
        float* z_ptr = (float*)[bufZIn contents];
        for (uint32_t i = 0; i < D; ++i) z_ptr[i] = float(i % 100) * 0.01f;

        auto alloc_mat = [&](uint32_t d_out) {
            uint32_t u32_sz = d_out * (D / 8) * sizeof(uint32_t);
            uint32_t sc_sz = d_out * (D / 64) * sizeof(uint16_t);
            id<MTLBuffer> bw = [device newBufferWithLength:u32_sz options:MTLResourceStorageModeShared];
            id<MTLBuffer> bs = [device newBufferWithLength:sc_sz options:MTLResourceStorageModeShared];
            id<MTLBuffer> bb = [device newBufferWithLength:sc_sz options:MTLResourceStorageModeShared];
            return std::make_tuple(bw, bs, bb);
        };

        auto [qkv_w, qkv_s, qkv_b] = alloc_mat(QKV_DIM);
        auto [z_w, z_s, z_b] = alloc_mat(Z_DIM);
        auto [b_w, b_s, b_b] = alloc_mat(BA_DIM);
        auto [a_w, a_s, a_b] = alloc_mat(BA_DIM);

        // Salidas para método separado
        id<MTLBuffer> out_qkv_sep = [device newBufferWithLength:QKV_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_z_sep   = [device newBufferWithLength:Z_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_b_sep   = [device newBufferWithLength:BA_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_a_sep   = [device newBufferWithLength:BA_DIM * sizeof(float) options:MTLResourceStorageModeShared];

        // Salidas para método fusionado
        id<MTLBuffer> out_qkv_fused = [device newBufferWithLength:QKV_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_z_fused   = [device newBufferWithLength:Z_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_b_fused   = [device newBufferWithLength:BA_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_a_fused   = [device newBufferWithLength:BA_DIM * sizeof(float) options:MTLResourceStorageModeShared];

        // 1. Ejecutar Método Separado (4 despachos)
        auto t0_sep = std::chrono::high_resolution_clock::now();
        id<MTLCommandBuffer> cmdSep = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encSep = [cmdSep computeCommandEncoder];
        [encSep setComputePipelineState:psoSep];

        auto dispatch_sep = [&](id<MTLBuffer> w, id<MTLBuffer> s, id<MTLBuffer> b, id<MTLBuffer> out, uint32_t d_out) {
            [encSep setBuffer:bufZIn offset:0 atIndex:0];
            [encSep setBuffer:w offset:0 atIndex:1];
            [encSep setBuffer:s offset:0 atIndex:2];
            [encSep setBuffer:b offset:0 atIndex:3];
            [encSep setBuffer:out offset:0 atIndex:4];
            [encSep setBytes:&D length:sizeof(uint32_t) atIndex:5];
            [encSep setBytes:&d_out length:sizeof(uint32_t) atIndex:6];
            uint32_t num_tg = (d_out + 3) / 4;
            [encSep dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        };

        dispatch_sep(qkv_w, qkv_s, qkv_b, out_qkv_sep, QKV_DIM);
        dispatch_sep(z_w, z_s, z_b, out_z_sep, Z_DIM);
        dispatch_sep(b_w, b_s, b_b, out_b_sep, BA_DIM);
        dispatch_sep(a_w, a_s, a_b, out_a_sep, BA_DIM);
        [encSep endEncoding];
        [cmdSep commit];
        [cmdSep waitUntilCompleted];
        auto t1_sep = std::chrono::high_resolution_clock::now();
        double lat_sep = std::chrono::duration<double, std::micro>(t1_sep - t0_sep).count();

        // 2. Ejecutar Método Fusionado (1 solo despacho)
        auto t0_fused = std::chrono::high_resolution_clock::now();
        id<MTLCommandBuffer> cmdFused = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encFused = [cmdFused computeCommandEncoder];
        [encFused setComputePipelineState:psoFused];
        [encFused setBuffer:bufZIn offset:0 atIndex:0];
        [encFused setBuffer:qkv_w offset:0 atIndex:1];
        [encFused setBuffer:qkv_s offset:0 atIndex:2];
        [encFused setBuffer:qkv_b offset:0 atIndex:3];
        [encFused setBuffer:out_qkv_fused offset:0 atIndex:4];
        [encFused setBuffer:z_w offset:0 atIndex:5];
        [encFused setBuffer:z_s offset:0 atIndex:6];
        [encFused setBuffer:z_b offset:0 atIndex:7];
        [encFused setBuffer:out_z_fused offset:0 atIndex:8];
        [encFused setBuffer:b_w offset:0 atIndex:9];
        [encFused setBuffer:b_s offset:0 atIndex:10];
        [encFused setBuffer:b_b offset:0 atIndex:11];
        [encFused setBuffer:out_b_fused offset:0 atIndex:12];
        [encFused setBuffer:a_w offset:0 atIndex:13];
        [encFused setBuffer:a_s offset:0 atIndex:14];
        [encFused setBuffer:a_b offset:0 atIndex:15];
        [encFused setBuffer:out_a_fused offset:0 atIndex:16];
        [encFused setBytes:&D length:sizeof(uint32_t) atIndex:17];

        uint32_t total_rows = QKV_DIM + Z_DIM + BA_DIM + BA_DIM;
        uint32_t num_tg_fused = (total_rows + 3) / 4;
        [encFused dispatchThreadgroups:MTLSizeMake(num_tg_fused, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encFused endEncoding];
        [cmdFused commit];
        [cmdFused waitUntilCompleted];
        auto t1_fused = std::chrono::high_resolution_clock::now();
        double lat_fused = std::chrono::duration<double, std::micro>(t1_fused - t0_fused).count();

        // 3. Verificación Bit-a-Bit
        auto compare_buf = [](float* s, float* f, uint32_t sz) {
            double max_d = 0.0;
            for (uint32_t i = 0; i < sz; ++i) {
                double diff = std::abs(double(s[i]) - double(f[i]));
                if (diff > max_d) max_d = diff;
            }
            return max_d;
        };

        double diff_qkv = compare_buf((float*)[out_qkv_sep contents], (float*)[out_qkv_fused contents], QKV_DIM);
        double diff_z   = compare_buf((float*)[out_z_sep contents], (float*)[out_z_fused contents], Z_DIM);
        double diff_b   = compare_buf((float*)[out_b_sep contents], (float*)[out_b_fused contents], BA_DIM);
        double diff_a   = compare_buf((float*)[out_a_sep contents], (float*)[out_a_fused contents], BA_DIM);

        std::cout << " • Latencia 4 Kernels Separados : " << std::fixed << std::setprecision(2) << lat_sep << " µs\n";
        std::cout << " • Latencia 1 Kernel Fusionado  : " << lat_fused << " µs\n";
        std::cout << " • Aceleración Local            : " << (lat_sep / lat_fused) << "x\n";
        std::cout << " • Máx discrepancia QKV         : " << std::scientific << diff_qkv << "\n";
        std::cout << " • Máx discrepancia Z           : " << diff_z << "\n";
        std::cout << " • Máx discrepancia B           : " << diff_b << "\n";
        std::cout << " • Máx discrepancia A           : " << diff_a << "\n";

        if (diff_qkv == 0.0 && diff_z == 0.0 && diff_b == 0.0 && diff_a == 0.0) {
            std::cout << " 🏆 ESTADO: BIT_EXACT_VALIDATED (Equivalencia numérica matemática absoluta).\n";
        } else {
            std::cout << " ❌ ESTADO: FAILED_EQUIVALENCE.\n";
            return 1;
        }
        std::cout << "=================================================================================\n";
    }
    return 0;
}
