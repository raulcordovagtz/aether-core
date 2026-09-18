#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

int main() {
    @autoreleasepool {
        std::cout << "═════════════════════════════════════════════════════════════════\n";
        std::cout << "🔬 ARNÉS DIFERENCIAL (C++ FP64 ↔ METAL FP32) AUTO-CALIBRADO\n";
        std::cout << "═════════════════════════════════════════════════════════════════\n";

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "❌ Error: Dispositivo Metal no disponible.\n";
            return 1;
        }

        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_geodesic_engine.metallib"] error:&err];
        if (!lib) {
            std::cerr << "❌ Error cargando metallib: " << [[err localizedDescription] UTF8String] << "\n";
            return 1;
        }

        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"aether_continuous_steering_step"] error:&err];
        if (!pso) {
            std::cerr << "❌ Error creando pipeline: " << [[err localizedDescription] UTF8String] << "\n";
            return 1;
        }

        const uint32_t D = 5120;
        const float dt = 0.01f;

        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, 1.0);

        std::vector<double> z_cpu(D), v_cpu(D), u_o(D), u_t(D), u_a(D), u_eos(D);
        std::vector<float> z_gpu(D), v_gpu(D), u_o_f(D), u_t_f(D), u_a_f(D), u_eos_f(D);

        auto normalize = [](std::vector<double>& vec) {
            double s = 0.0;
            for (double x : vec) s += x * x;
            double inv = 1.0 / std::sqrt(s + 1e-12);
            for (double& x : vec) x *= inv;
        };

        for (uint32_t i = 0; i < D; ++i) {
            z_cpu[i] = dist(rng);
            v_cpu[i] = dist(rng) * 0.1;
            u_o[i]   = dist(rng);
            u_t[i]   = dist(rng);
            u_a[i]   = dist(rng);
            u_eos[i] = dist(rng);
        }
        normalize(z_cpu);
        normalize(u_o);
        normalize(u_t);
        normalize(u_a);
        normalize(u_eos);

        for (uint32_t i = 0; i < D; ++i) {
            z_gpu[i]   = (float)z_cpu[i];
            v_gpu[i]   = (float)v_cpu[i];
            u_o_f[i]   = (float)u_o[i];
            u_t_f[i]   = (float)u_t[i];
            u_a_f[i]   = (float)u_a[i];
            u_eos_f[i] = (float)u_eos[i];
        }

        double dot_ot_cpu = 0.0, dot_ta_cpu = 0.0;
        for (uint32_t i = 0; i < D; ++i) {
            dot_ot_cpu += u_o[i] * u_t[i];
            dot_ta_cpu += u_t[i] * u_a[i];
        }
        float dot_ot_f = (float)dot_ot_cpu;
        float dot_ta_f = (float)dot_ta_cpu;

        // ─── 1. SIMULACIÓN C++ FP64 DE REFERENCIA ANALÍTICA ─────────────────────
        double dot_o = 0.0, dot_t = 0.0, dot_a = 0.0, dot_eos = 0.0, sq_z = 0.0, sq_v = 0.0;
        for (uint32_t i = 0; i < D; ++i) {
            dot_o   += z_cpu[i] * u_o[i];
            dot_t   += z_cpu[i] * u_t[i];
            dot_a   += z_cpu[i] * u_a[i];
            dot_eos += z_cpu[i] * u_eos[i];
            sq_z    += z_cpu[i] * z_cpu[i];
            sq_v    += v_cpu[i] * v_cpu[i];
        }

        double inv_sq_z       = 1.0 / (sq_z + 1e-6);
        double omega_sq       = sq_v * inv_sq_z;
        double omega          = std::sqrt(omega_sq);
        double sigma_bivector = std::sqrt(std::max(0.0, 1.0 - dot_ot_cpu * dot_ot_cpu));
        double omega_bivector = std::sqrt(std::max(0.0, 1.0 - dot_ta_cpu * dot_ta_cpu));

        double safe_dot_o   = std::max(0.0, dot_o);
        double safe_dot_t   = std::max(0.0, dot_t);
        double safe_dot_eos = std::max(0.0, dot_eos);

        std::vector<double> z_cpu_next(D), v_cpu_next(D);
        for (uint32_t i = 0; i < D; ++i) {
            double z = z_cpu[i];
            double v = v_cpu[i];
            double f_gyro = omega * sigma_bivector * (safe_dot_t * u_o[i] - safe_dot_o * u_t[i]);
            double f_dial = omega * omega_bivector * (safe_dot_t * u_a[i] - dot_a * u_t[i]);
            double u_eos_perp = u_eos[i] - (dot_eos * inv_sq_z) * z;
            double f_pozo = omega_sq * safe_dot_eos * u_eos_perp;
            double f_visc = -omega * v;
            double centripetal = omega_sq * z;

            double a_total = f_gyro + f_dial + f_pozo + f_visc - centripetal;
            double v_nxt = v + dt * a_total;
            double z_nxt = z + dt * v_nxt;

            v_cpu_next[i] = v_nxt;
            z_cpu_next[i] = z_nxt;
        }

        // ─── 2. EJECUCIÓN METAL GPU FP32 ─────────────────────────────────────────
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> bufZ = [device newBufferWithBytes:z_gpu.data() length:D*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV = [device newBufferWithBytes:v_gpu.data() length:D*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufO = [device newBufferWithBytes:u_o_f.data() length:D*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufT = [device newBufferWithBytes:u_t_f.data() length:D*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufA = [device newBufferWithBytes:u_a_f.data() length:D*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufEOS = [device newBufferWithBytes:u_eos_f.data() length:D*sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufZ offset:0 atIndex:0];
        [enc setBuffer:bufV offset:0 atIndex:1];
        [enc setBuffer:bufO offset:0 atIndex:2];
        [enc setBuffer:bufT offset:0 atIndex:3];
        [enc setBuffer:bufA offset:0 atIndex:4];
        [enc setBuffer:bufEOS offset:0 atIndex:5];
        [enc setBytes:&dt length:sizeof(float) atIndex:6];
        [enc setBytes:&dot_ot_f length:sizeof(float) atIndex:7];
        [enc setBytes:&dot_ta_f length:sizeof(float) atIndex:8];
        [enc setBytes:&D length:sizeof(uint32_t) atIndex:9];

        for (int i = 0; i < 6; ++i) {
            [enc setThreadgroupMemoryLength:512 * sizeof(float) atIndex:i];
        }
        [enc dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // ─── 3. ANÁLISIS DIFERENCIAL RESIDUAL ───────────────────────────────────
        float* z_res = (float*)[bufZ contents];
        float* v_res = (float*)[bufV contents];

        double max_err_z = 0.0, max_rel_z = 0.0;
        double max_err_v = 0.0, max_rel_v = 0.0;
        for (uint32_t i = 0; i < D; ++i) {
            double err_z = std::abs((double)z_res[i] - z_cpu_next[i]);
            double rel_z = err_z / (std::abs(z_cpu_next[i]) + 1e-7);
            if (err_z > max_err_z) max_err_z = err_z;
            if (rel_z > max_rel_z) max_rel_z = rel_z;

            double err_v = std::abs((double)v_res[i] - v_cpu_next[i]);
            double rel_v = err_v / (std::abs(v_cpu_next[i]) + 1e-7);
            if (err_v > max_err_v) max_err_v = err_v;
            if (rel_v > max_rel_v) max_rel_v = rel_v;
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "• Error Absoluto Máximo (Posición Z): " << max_err_z << "\n";
        std::cout << "• Error Relativo Máximo (Posición Z): " << max_rel_z << "\n";
        std::cout << "• Error Absoluto Máximo (Velocidad V): " << max_err_v << "\n";
        std::cout << "• Error Relativo Máximo (Velocidad V): " << max_rel_v << "\n";

        if (max_rel_z < 1e-4 && max_rel_v < 1e-4) {
            std::cout << "\n✅ CALIFICACIÓN DIFERENCIAL AUTO-CALIBRADA: APROBADA.\n";
            return 0;
        } else {
            std::cerr << "\n❌ ERROR: Discrepancia diferencial supera el umbral permitido.\n";
            return 2;
        }
    }
}
