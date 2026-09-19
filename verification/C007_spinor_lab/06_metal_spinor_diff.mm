#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

const uint32_t D = 5120;
const uint32_t R = 32;
const uint32_t TWO_D = 2 * D;

void step_cpu_fp32(
    std::vector<float>& Phi,
    const std::vector<float>& Uc, const std::vector<float>& Vc,
    const std::vector<float>& Us, const std::vector<float>& Vs,
    const std::vector<float>& Ul, const std::vector<float>& Vl,
    uint32_t steps, float d_tau, float coupling_scale, float alpha_eml, float lambda_diss)
{
    std::vector<float> k1(TWO_D), k2(TWO_D), phi_mid(TWO_D);
    std::vector<float> Vs_S(R), Us_S(R), Vl_L(R), Ul_L(R), Vc_L(R), Uc_S(R);

    auto eval_flow = [&](const std::vector<float>& P, std::vector<float>& dP) {
        const float* S = P.data();
        const float* L = P.data() + D;

        std::fill(Vs_S.begin(), Vs_S.end(), 0.0f);
        std::fill(Us_S.begin(), Us_S.end(), 0.0f);
        std::fill(Vl_L.begin(), Vl_L.end(), 0.0f);
        std::fill(Ul_L.begin(), Ul_L.end(), 0.0f);
        std::fill(Vc_L.begin(), Vc_L.end(), 0.0f);
        std::fill(Uc_S.begin(), Uc_S.end(), 0.0f);

        for (uint32_t r = 0; r < R; ++r) {
            for (uint32_t i = 0; i < D; ++i) {
                Vs_S[r] += Vs[r * D + i] * S[i];
                Us_S[r] += Us[r * D + i] * S[i];
                Vl_L[r] += Vl[r * D + i] * L[i];
                Ul_L[r] += Ul[r * D + i] * L[i];
                Vc_L[r] += Vc[r * D + i] * L[i];
                Uc_S[r] += Uc[r * D + i] * S[i];
            }
        }

        std::vector<float> F_eml(TWO_D);
        float dot_F_P = 0.0f, norm_sq = 0.0f;
        for (uint32_t i = 0; i < TWO_D; ++i) {
            float x = P[i];
            float safe_x = std::max(-20.0f, std::min(20.0f, -x));
            F_eml[i] = x / (1.0f + std::exp(safe_x));
            dot_F_P += F_eml[i] * x;
            norm_sq += x * x;
        }
        float proj = (norm_sq > 1e-12f) ? (dot_F_P / norm_sq) : 0.0f;

        for (uint32_t i = 0; i < D; ++i) {
            float As_S = 0.0f, Al_L = 0.0f, C_L = 0.0f, neg_CT_S = 0.0f;
            for (uint32_t r = 0; r < R; ++r) {
                As_S     += Us[r * D + i] * Vs_S[r] - Vs[r * D + i] * Us_S[r];
                Al_L     += Ul[r * D + i] * Vl_L[r] - Vl[r * D + i] * Ul_L[r];
                C_L      += Uc[r * D + i] * Vc_L[r];
                neg_CT_S -= Vc[r * D + i] * Uc_S[r];
            }
            float dS = (As_S + coupling_scale * C_L) + alpha_eml * (F_eml[i] - proj * S[i]) - lambda_diss * S[i];
            float dL = (coupling_scale * neg_CT_S + Al_L) + alpha_eml * (F_eml[i + D] - proj * L[i]) - lambda_diss * L[i];
            dP[i] = dS;
            dP[i + D] = dL;
        }
    };

    for (uint32_t s = 0; s < steps; ++s) {
        eval_flow(Phi, k1);
        for (uint32_t i = 0; i < TWO_D; ++i) phi_mid[i] = Phi[i] + 0.5f * d_tau * k1[i];
        eval_flow(phi_mid, k2);
        for (uint32_t i = 0; i < TWO_D; ++i) Phi[i] = Phi[i] + d_tau * k2[i];
    }
}

int main() {
    std::cout << "=================================================================================\n";
    std::cout << " 🔬 VALIDACIÓN DIFERENCIAL C++ FP32 ↔ METAL FP32 (CON PROYECTOR PI_PERP)\n";
    std::cout << "=================================================================================\n";

    std::mt19937_64 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(float(D)));

    std::vector<float> Uc(R * D), Vc(R * D);
    std::vector<float> Us(R * D), Vs(R * D);
    std::vector<float> Ul(R * D), Vl(R * D);
    std::vector<float> Phi_0(TWO_D);

    for (auto& x : Uc) x = dist(rng);
    for (auto& x : Vc) x = dist(rng);
    for (auto& x : Us) x = dist(rng);
    for (auto& x : Vs) x = dist(rng);
    for (auto& x : Ul) x = dist(rng);
    for (auto& x : Vl) x = dist(rng);
    for (auto& x : Phi_0) x = dist(rng) * std::sqrt(float(D));

    float n_S = 0.0f, n_L = 0.0f;
    for (uint32_t i = 0; i < D; ++i) n_S += Phi_0[i] * Phi_0[i];
    for (uint32_t i = D; i < TWO_D; ++i) n_L += Phi_0[i] * Phi_0[i];
    n_S = std::sqrt(n_S); n_L = std::sqrt(n_L);
    for (uint32_t i = 0; i < D; ++i) Phi_0[i] /= n_S;
    for (uint32_t i = D; i < TWO_D; ++i) Phi_0[i] /= n_L;

    uint32_t steps = 64;
    float d_tau = 0.05f;
    float half_dt = 0.5f * d_tau;
    float coupling_scale = 0.05f;
    float alpha_eml = 0.02f;
    float lambda_diss = 0.001f;

    // Ejecutar referencia CPU sobre los mismos datos
    std::vector<float> Phi_cpu = Phi_0;
    step_cpu_fp32(Phi_cpu, Uc, Vc, Us, Vs, Ul, Vl, steps, d_tau, coupling_scale, alpha_eml, lambda_diss);

    double norm_cpu = 0.0;
    for (float x : Phi_cpu) norm_cpu += double(x) * double(x);

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSError* err = nil;
        NSURL* libURL = [NSURL fileURLWithPath:@"metal/aether_c007_spinor_integrator.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&err];
        id<MTLFunction> fnRed = [lib newFunctionWithName:@"c007_project_reductions"];
        id<MTLFunction> fnStep = [lib newFunctionWithName:@"c007_reconstruct_and_step"];
        id<MTLComputePipelineState> psoRed = [device newComputePipelineStateWithFunction:fnRed error:&err];
        id<MTLComputePipelineState> psoStep = [device newComputePipelineStateWithFunction:fnStep error:&err];

        id<MTLBuffer> bufPhi = [device newBufferWithBytes:Phi_0.data() length:TWO_D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPhiMid = [device newBufferWithLength:TWO_D * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bufRProj = [device newBufferWithLength:(6 * R + 2) * sizeof(float) options:MTLResourceStorageModePrivate];

        id<MTLBuffer> bufUc = [device newBufferWithBytes:Uc.data() length:R * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufVc = [device newBufferWithBytes:Vc.data() length:R * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufUs = [device newBufferWithBytes:Us.data() length:R * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufVs = [device newBufferWithBytes:Vs.data() length:R * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufUl = [device newBufferWithBytes:Ul.data() length:R * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufVl = [device newBufferWithBytes:Vl.data() length:R * D * sizeof(float) options:MTLResourceStorageModeShared];

        auto t0 = std::chrono::high_resolution_clock::now();
        id<MTLCommandBuffer> cmd = [queue commandBuffer];

        for (uint32_t s = 0; s < steps; ++s) {
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
            [enc1 dispatchThreadgroups:MTLSizeMake(R, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc1 endEncoding];

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
            [enc2 setBytes:&half_dt length:sizeof(float) atIndex:10];
            [enc2 setBytes:&coupling_scale length:sizeof(float) atIndex:11];
            [enc2 setBytes:&alpha_eml length:sizeof(float) atIndex:12];
            [enc2 setBytes:&lambda_diss length:sizeof(float) atIndex:13];
            [enc2 dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc2 endEncoding];

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
            [enc3 dispatchThreadgroups:MTLSizeMake(R, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc3 endEncoding];

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
            [enc4 setBytes:&d_tau length:sizeof(float) atIndex:10];
            [enc4 setBytes:&coupling_scale length:sizeof(float) atIndex:11];
            [enc4 setBytes:&alpha_eml length:sizeof(float) atIndex:12];
            [enc4 setBytes:&lambda_diss length:sizeof(float) atIndex:13];
            [enc4 dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc4 endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_gpu = std::chrono::duration<double, std::milli>(t1 - t0).count();
        float* out_gpu = static_cast<float*>([bufPhi contents]);

        double max_diff = 0.0;
        double norm_gpu = 0.0;
        for (uint32_t i = 0; i < TWO_D; ++i) {
            double diff = std::abs(double(out_gpu[i]) - double(Phi_cpu[i]));
            if (diff > max_diff) max_diff = diff;
            norm_gpu += double(out_gpu[i]) * double(out_gpu[i]);
        }

        std::cout << " • Tiempo GPU (64 pasos en Apple M2 Max): " << std::fixed << std::setprecision(2) << elapsed_gpu << " ms\n";
        std::cout << " • Norma cuadrática CPU (FP32) : " << std::setprecision(8) << norm_cpu << "\n";
        std::cout << " • Norma cuadrática GPU (Metal): " << norm_gpu << "\n";
        std::cout << " • Discrepancia de norma |GPU - CPU|: " << std::scientific << std::setprecision(4) << std::abs(norm_gpu - norm_cpu) << "\n";
        std::cout << " • Error máximo punto a punto ||GPU - CPU||_∞: " << max_diff << "\n";

        if (max_diff < 5e-4) {
            std::cout << " 🏆 ESTADO: NUMERICALLY_VALIDATED (Convergencia estricta C++ ↔ Metal).\n";
        } else {
            std::cout << " ❌ ESTADO: FAILED_TOLERANCE.\n";
            return 1;
        }
        std::cout << "=================================================================================\n";
    }
    return 0;
}
