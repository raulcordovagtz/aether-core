#pragma once
#import <Metal/Metal.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>

class CellHarness {
private:
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> psoALU;

public:
    CellHarness(id<MTLDevice> dev, id<MTLCommandQueue> q) : device(dev), queue(q) {
        NSError* err = nil;
        NSURL* libURL = [NSURL fileURLWithPath:@"harness/metal/cell_alu.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];
        if (!lib) {
            std::cerr << "❌ Error cargando biblioteca de celdas: " << [[err localizedDescription] UTF8String] << "\n";
            return;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"execute_symbolic_cell"];
        psoALU = [device newComputePipelineStateWithFunction:fn error:&err];
    }

    uint32_t execute_cell_op(uint32_t a, uint32_t b, uint32_t op_code, double& elapsed_us) {
        id<MTLBuffer> bufA = [device newBufferWithBytes:&a length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithBytes:&b length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufRes = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        auto t0 = std::chrono::high_resolution_clock::now();
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:psoALU];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufRes offset:0 atIndex:2];
        [enc setBytes:&op_code length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        auto t1 = std::chrono::high_resolution_clock::now();

        elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return *(static_cast<uint32_t*>([bufRes contents]));
    }
};
