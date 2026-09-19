// ═════════════════════════════════════════════════════════════════════════════
// 🌌 AETHER-VL :: C-008 DISPATCH MODULE (AUTO-GENERATED FROM YAML SPEC)
// ═════════════════════════════════════════════════════════════════════════════
#pragma once
#import <Metal/Metal.h>
#include <cmath>
#include <iostream>

struct C008EngineConfig {
    static constexpr uint32_t D = 5120;
    static constexpr uint32_t R = 32;
    static constexpr uint32_t PREFILL_STEPS = 32;
    static constexpr uint32_t DECODE_STEPS  = 1;
    static constexpr float DT = 0.0156250000f;
    static constexpr float HALF_DT = 0.0078125000f;
    static constexpr float TAU_RELAX = 24.000000f;
};

static inline void aether_c008_step_dispatch(
    id<MTLCommandBuffer> cmd,
    id<MTLComputePipelineState> psoRed,
    id<MTLComputePipelineState> psoStep,
    id<MTLBuffer> bufPhi,
    id<MTLBuffer> bufPhiMid,
    id<MTLBuffer> bufRProj,
    id<MTLBuffer> bufUc, id<MTLBuffer> bufVc,
    id<MTLBuffer> bufUs, id<MTLBuffer> bufVs,
    id<MTLBuffer> bufUl, id<MTLBuffer> bufVl,
    float dt_val, float lie_decay_val)
{
    // Subpaso 1: Reducción
    id<MTLComputeCommandEncoder> enc1 = [cmd computeCommandEncoder];
    [enc1 setComputePipelineState:psoRed];
    [enc1 setBuffer:bufPhi offset:0 atIndex:0];
    [enc1 setBuffer:bufUc offset:0 atIndex:1];
    [enc1 setBuffer:bufVc offset:0 atIndex:2];
    [enc1 setBuffer:bufUs offset:0 atIndex:3];
    [enc1 setBuffer:bufVs offset:0 atIndex:4];
    [enc1 setBuffer:bufUl offset:0 atIndex:5];
    [enc1 setBuffer:bufVl offset:0 atIndex:6];
    [enc1 setBuffer:bufRProj offset:0 atIndex:7];
    [enc1 setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
    [enc1 dispatchThreadgroups:MTLSizeMake(C008EngineConfig::R, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc1 endEncoding];

    // Subpaso 2: Avance
    id<MTLComputeCommandEncoder> enc2 = [cmd computeCommandEncoder];
    [enc2 setComputePipelineState:psoStep];
    [enc2 setBuffer:bufPhi offset:0 atIndex:0];
    [enc2 setBuffer:bufPhi offset:0 atIndex:1];
    [enc2 setBuffer:bufPhiMid offset:0 atIndex:2];
    [enc2 setBuffer:bufUc offset:0 atIndex:3];
    [enc2 setBuffer:bufVc offset:0 atIndex:4];
    [enc2 setBuffer:bufUs offset:0 atIndex:5];
    [enc2 setBuffer:bufVs offset:0 atIndex:6];
    [enc2 setBuffer:bufUl offset:0 atIndex:7];
    [enc2 setBuffer:bufVl offset:0 atIndex:8];
    [enc2 setBuffer:bufRProj offset:0 atIndex:9];
    float half_dt = dt_val * 0.5f;
    [enc2 setBytes:&half_dt length:sizeof(float) atIndex:10];
    [enc2 setBytes:&lie_decay_val length:sizeof(float) atIndex:11];
    [enc2 setBuffer:bufVs offset:0 atIndex:12]; // u_vis
    [enc2 setBuffer:bufVl offset:0 atIndex:13]; // u_txt
    [enc2 dispatchThreads:MTLSizeMake(C008EngineConfig::D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc2 endEncoding];

    // Subpaso 3: Reducción Mid
    id<MTLComputeCommandEncoder> enc3 = [cmd computeCommandEncoder];
    [enc3 setComputePipelineState:psoRed];
    [enc3 setBuffer:bufPhiMid offset:0 atIndex:0];
    [enc3 setBuffer:bufUc offset:0 atIndex:1];
    [enc3 setBuffer:bufVc offset:0 atIndex:2];
    [enc3 setBuffer:bufUs offset:0 atIndex:3];
    [enc3 setBuffer:bufVs offset:0 atIndex:4];
    [enc3 setBuffer:bufUl offset:0 atIndex:5];
    [enc3 setBuffer:bufVl offset:0 atIndex:6];
    [enc3 setBuffer:bufRProj offset:0 atIndex:7];
    [enc3 setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
    [enc3 dispatchThreadgroups:MTLSizeMake(C008EngineConfig::R, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc3 endEncoding];

    // Subpaso 4: Avance Final
    id<MTLComputeCommandEncoder> enc4 = [cmd computeCommandEncoder];
    [enc4 setComputePipelineState:psoStep];
    [enc4 setBuffer:bufPhiMid offset:0 atIndex:0];
    [enc4 setBuffer:bufPhi offset:0 atIndex:1];
    [enc4 setBuffer:bufPhi offset:0 atIndex:2];
    [enc4 setBuffer:bufUc offset:0 atIndex:3];
    [enc4 setBuffer:bufVc offset:0 atIndex:4];
    [enc4 setBuffer:bufUs offset:0 atIndex:5];
    [enc4 setBuffer:bufVs offset:0 atIndex:6];
    [enc4 setBuffer:bufUl offset:0 atIndex:7];
    [enc4 setBuffer:bufVl offset:0 atIndex:8];
    [enc4 setBuffer:bufRProj offset:0 atIndex:9];
    [enc4 setBytes:&dt_val length:sizeof(float) atIndex:10];
    [enc4 setBytes:&lie_decay_val length:sizeof(float) atIndex:11];
    [enc4 setBuffer:bufVs offset:0 atIndex:12];
    [enc4 setBuffer:bufVl offset:0 atIndex:13];
    [enc4 dispatchThreads:MTLSizeMake(C008EngineConfig::D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc4 endEncoding];
}
