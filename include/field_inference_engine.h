#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "field_invariants.h"
#include <iostream>
#include <vector>
#include <chrono>

namespace c_field {

struct BestCandidate {
    float score;
    uint32_t token_id;
};

class FieldInferenceEngine {
public:
    FieldInferenceEngine(const ModelTopology& topo) : topo_(topo) {
        invariants_ = ManifoldInvariants::compute(topo_);
        init_metal();
    }

    void set_visual_condition(const std::vector<float>& patches, uint32_t num_patches) {
        num_patches_ = num_patches;
        size_t bytes = num_patches * topo_.visual_dim * sizeof(float);
        buf_visual_patches_ = [device_ newBufferWithBytes:patches.data() length:bytes options:MTLResourceStorageModeShared];

        // Aplicar deformación métrica de fondo
        id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_visual_];
        [enc setBuffer:buf_visual_patches_ offset:0 atIndex:0];
        [enc setBuffer:buf_z_curr_ offset:0 atIndex:1];
        [enc setBytes:&gpu_inv_ length:sizeof(gpu_inv_) atIndex:2];
        [enc setBytes:&num_patches_ length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&topo_.visual_dim length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&topo_.hidden_dim length:sizeof(uint32_t) atIndex:5];

        [enc dispatchThreads:MTLSizeMake(topo_.hidden_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    uint32_t step_inference(uint32_t history_len) {
        // 1. Atención Geodésica
        id<MTLCommandBuffer> cmd1 = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc1 = [cmd1 computeCommandEncoder];
        [enc1 setComputePipelineState:pso_geodesic_];
        [enc1 setBuffer:buf_z_curr_ offset:0 atIndex:0];
        [enc1 setBuffer:buf_k_hist_ offset:0 atIndex:1];
        [enc1 setBuffer:buf_v_hist_ offset:0 atIndex:2];
        [enc1 setBuffer:buf_z_acc_ offset:0 atIndex:3];
        [enc1 setBytes:&gpu_inv_ length:sizeof(gpu_inv_) atIndex:4];
        [enc1 setBytes:&topo_.hidden_dim length:sizeof(uint32_t) atIndex:5];
        [enc1 setBytes:&history_len length:sizeof(uint32_t) atIndex:6];

        [enc1 dispatchThreads:MTLSizeMake(topo_.hidden_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc1 endEncoding];
        [cmd1 commit];
        [cmd1 waitUntilCompleted];

        // 2. MoE Derivativo por Subespacios
        id<MTLCommandBuffer> cmd2 = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc2 = [cmd2 computeCommandEncoder];
        [enc2 setComputePipelineState:pso_moe_];
        [enc2 setBuffer:buf_z_acc_ offset:0 atIndex:0];
        [enc2 setBuffer:buf_z_prev_ offset:0 atIndex:1];
        [enc2 setBuffer:buf_moe_w_ offset:0 atIndex:2];
        [enc2 setBuffer:buf_z_curr_ offset:0 atIndex:3];
        [enc2 setBytes:&gpu_inv_ length:sizeof(gpu_inv_) atIndex:4];
        [enc2 setBytes:&topo_.hidden_dim length:sizeof(uint32_t) atIndex:5];

        [enc2 dispatchThreads:MTLSizeMake(topo_.hidden_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc2 endEncoding];
        [cmd2 commit];
        [cmd2 waitUntilCompleted];

        // 3. Colapso al Horizonte de Gibbs
        const uint32_t THREADS = 256;
        const uint32_t num_groups = (topo_.vocab_size + THREADS - 1) / THREADS;

        id<MTLCommandBuffer> cmd3 = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc3 = [cmd3 computeCommandEncoder];
        [enc3 setComputePipelineState:pso_gibbs_];
        [enc3 setBuffer:buf_z_curr_ offset:0 atIndex:0];
        [enc3 setBuffer:buf_vocab_w_ offset:0 atIndex:1];
        [enc3 setBuffer:buf_block_results_ offset:0 atIndex:2];
        [enc3 setBytes:&gpu_inv_ length:sizeof(gpu_inv_) atIndex:3];
        [enc3 setBytes:&topo_.hidden_dim length:sizeof(uint32_t) atIndex:4];
        [enc3 setBytes:&topo_.vocab_size length:sizeof(uint32_t) atIndex:5];
        [enc3 setThreadgroupMemoryLength:THREADS * sizeof(BestCandidate) atIndex:0];

        [enc3 dispatchThreads:MTLSizeMake(num_groups * THREADS, 1, 1) threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
        [enc3 endEncoding];
        [cmd3 commit];
        [cmd3 waitUntilCompleted];

        // Reducción final en CPU (< 1 microsegundo)
        BestCandidate* blocks = (BestCandidate*)[buf_block_results_ contents];
        float best_score = -1e30f;
        uint32_t winning_token = 0;
        for (uint32_t g = 0; g < num_groups; ++g) {
            if (blocks[g].score > best_score) {
                best_score = blocks[g].score;
                winning_token = blocks[g].token_id;
            }
        }

        return winning_token;
    }

    id<MTLBuffer> get_vocab_buffer() { return buf_vocab_w_; }
    id<MTLBuffer> get_z_curr_buffer() { return buf_z_curr_; }
    id<MTLBuffer> get_z_prev_buffer() { return buf_z_prev_; }
    id<MTLBuffer> get_moe_buffer() { return buf_moe_w_; }
    id<MTLBuffer> get_k_hist_buffer() { return buf_k_hist_; }
    id<MTLBuffer> get_v_hist_buffer() { return buf_v_hist_; }

private:
    ModelTopology topo_;
    ManifoldInvariants invariants_;

    struct FieldInvariantsGPU {
        float beta_inv_temp;
        float fermi_mu;
        float r_manifold;
        float fermat_scale;
        float mach_epsilon;
        float vacuum_threshold;
    } gpu_inv_;

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> pso_visual_;
    id<MTLComputePipelineState> pso_geodesic_;
    id<MTLComputePipelineState> pso_moe_;
    id<MTLComputePipelineState> pso_gibbs_;

    id<MTLBuffer> buf_z_curr_;
    id<MTLBuffer> buf_z_prev_;
    id<MTLBuffer> buf_z_acc_;
    id<MTLBuffer> buf_k_hist_;
    id<MTLBuffer> buf_v_hist_;
    id<MTLBuffer> buf_moe_w_;
    id<MTLBuffer> buf_vocab_w_;
    id<MTLBuffer> buf_block_results_;
    id<MTLBuffer> buf_visual_patches_;
    uint32_t num_patches_ = 0;

    void init_metal() {
        device_ = MTLCreateSystemDefaultDevice();
        queue_ = [device_ newCommandQueue];

        NSError* error = nil;
        NSURL* libURL = [NSURL fileURLWithPath:@"metal/c_field_unified.metallib"];
        id<MTLLibrary> lib = [device_ newLibraryWithURL:libURL error:&error];
        if (!lib) throw std::runtime_error([[error localizedDescription] UTF8String]);

        pso_visual_ = [device_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"apply_visual_boundary_metric"] error:&error];
        pso_geodesic_ = [device_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"geodesic_causal_step"] error:&error];
        pso_moe_ = [device_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"derivativo_moe_step"] error:&error];
        pso_gibbs_ = [device_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gibbs_collapse_vectorized"] error:&error];

        gpu_inv_ = {
            invariants_.beta_inverse_temp,
            invariants_.fermi_potential_mu,
            invariants_.r_manifold,
            invariants_.fermat_scale,
            invariants_.mach_epsilon,
            1.0f / float(topo_.hidden_dim)
        };

        const size_t D_bytes = topo_.hidden_dim * sizeof(float);
        const size_t max_hist = 512;
        const uint32_t THREADS = 256;
        const uint32_t num_groups = (topo_.vocab_size + THREADS - 1) / THREADS;

        buf_z_curr_ = [device_ newBufferWithLength:D_bytes options:MTLResourceStorageModeShared];
        buf_z_prev_ = [device_ newBufferWithLength:D_bytes options:MTLResourceStorageModeShared];
        buf_z_acc_ = [device_ newBufferWithLength:D_bytes options:MTLResourceStorageModeShared];
        buf_moe_w_ = [device_ newBufferWithLength:D_bytes options:MTLResourceStorageModeShared];
        buf_k_hist_ = [device_ newBufferWithLength:max_hist * D_bytes options:MTLResourceStorageModeShared];
        buf_v_hist_ = [device_ newBufferWithLength:max_hist * D_bytes options:MTLResourceStorageModeShared];
        buf_vocab_w_ = [device_ newBufferWithLength:(size_t)topo_.vocab_size * D_bytes options:MTLResourceStorageModeShared];
        buf_block_results_ = [device_ newBufferWithLength:num_groups * sizeof(BestCandidate) options:MTLResourceStorageModeShared];
    }
};

} // namespace c_field
