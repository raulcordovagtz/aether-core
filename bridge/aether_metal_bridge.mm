#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

struct AsyncMetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> psoDirect = nil;
    bool initialized = false;
};

static AsyncMetalContext ctx;

extern "C" {

int aether_metal_init(const char* metallib_path) {
    @autoreleasepool {
        if (ctx.initialized) return 0;

        ctx.device = MTLCreateSystemDefaultDevice();
        if (!ctx.device) {
            std::cerr << "❌ [METAL] Dispositivo no disponible.\n";
            return -1;
        }
        ctx.queue = [ctx.device newCommandQueue];

        NSError* err = nil;
        NSString* pathStr = [NSString stringWithUTF8String:metallib_path];
        NSURL* libURL = [NSURL fileURLWithPath:pathStr];
        id<MTLLibrary> lib = [ctx.device newLibraryWithURL:libURL error:&err];
        if (!lib) {
            std::cerr << "❌ [METAL] Error cargando metallib: " << [[err localizedDescription] UTF8String] << "\n";
            return -1;
        }

        id<MTLFunction> fn = [lib newFunctionWithName:@"c018_riemannian_direct_step"];
        ctx.psoDirect = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (!ctx.psoDirect) {
            std::cerr << "❌ [METAL] Error compilando pipeline.\n";
            return -1;
        }

        ctx.initialized = true;
        return 0;
    }
}

int aether_metal_async_step(
    float* h_io_ptr,
    const float* u_target_ptr,
    uint32_t D,
    float dt,
    float kappa_0,
    float beta_steer,
    float M_diss,
    float nu_diff
) {
    @autoreleasepool {
        if (!ctx.initialized) return -1;

        size_t bytes = D * sizeof(float);

        // Mapeo Zero-Copy instantáneo en registros UMA
        id<MTLBuffer> bufH = [ctx.device newBufferWithBytesNoCopy:h_io_ptr
                                                           length:bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];

        id<MTLBuffer> bufU = [ctx.device newBufferWithBytesNoCopy:(void*)u_target_ptr
                                                           length:bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];

        id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx.psoDirect];
        [enc setBuffer:bufH offset:0 atIndex:0];
        [enc setBuffer:bufU offset:0 atIndex:1];
        [enc setBytes:&D length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&dt length:sizeof(float) atIndex:3];
        [enc setBytes:&kappa_0 length:sizeof(float) atIndex:4];
        [enc setBytes:&beta_steer length:sizeof(float) atIndex:5];
        [enc setBytes:&M_diss length:sizeof(float) atIndex:6];
        [enc setBytes:&nu_diff length:sizeof(float) atIndex:7];

        NSUInteger tgSize = 256;
        NSUInteger numGroups = (D + tgSize - 1) / tgSize;
        [enc dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];

        // DESPACHO ASÍNCRONO EN COLA DE HARDWARE (Cero parada de CPU)
        [cmd commit];

        return 0;
    }
}

} // extern "C"
