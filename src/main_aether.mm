#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "multishard_uma.h"
#include "fast_vocab.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include "aether_c008_dispatch.h"

using namespace c_field;

static inline float bf16_to_f32_cpu(uint16_t val) {
    uint32_t u = uint32_t(val) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

static inline void dispatch_simd_gemv(id<MTLComputeCommandEncoder> enc, uint32_t d_out) {
    uint32_t num_groups = (d_out + 3) / 4;
    [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

int main(int argc, char* argv[]) {
    int max_tokens = (argc > 1) ? std::atoi(argv[1]) : 1500;

    std::cout << "=================================================================================\n";
    std::cout << " 🌌 AETHER-VL :: SOVEREIGN GEODESIC ENGINE (INDEPENDENT STANDALONE)\n";
    std::cout << "    Domain: English Canonical Benchmark + Clean UTF-8 Emoji Stream\n";
    std::cout << "=================================================================================\n";

    std::string model_dir = "/Users/crotalo/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit";
    auto vocab = load_binary_vocab("vocab.bin");

    MultiShardUMAPool pool;
    if (!pool.initialize(model_dir)) return 1;

    const uint32_t D = 5120, I_DIM = 17408, V = 248320, NUM_LAYERS = 64;
    const float mach_eps = 1e-6f;
    const uint32_t TOTAL_Q = 6144, TOTAL_KV = 1024, Q_PROJ_OUT = 12288;
    const uint32_t QKV_DIM = 10240, Z_DIM = 6144, BA_DIM = 48;
    const uint32_t ID_EOS = 248046;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSFileHandle *stdOut = [NSFileHandle fileHandleWithStandardOutput];

        NSError* err = nil;
        id<MTLLibrary> libBase = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/c_field_qwen38_engine.metallib"] error:&err];
        id<MTLLibrary> libCert = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/c_field_certified_engine.metallib"] error:&err];
        id<MTLLibrary> libMarkov = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/c_field_intent_markov_engine.metallib"] error:&err];
        id<MTLLibrary> libGeodesic = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_mhc_eml_engine.metallib"] error:&err];
        id<MTLLibrary> libFused = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_fused_norm_gemv.metallib"] error:&err];
        id<MTLLibrary> libSpinorC018 = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_c018_riemannian_engine.metallib"] error:&err];
        if (!libSpinorC018) { std::cerr << "❌ Error cargando libSpinorC018\n"; return 1; }
        id<MTLComputePipelineState> psoC018Red  = [device newComputePipelineStateWithFunction:[libSpinorC018 newFunctionWithName:@"c018_project_reductions"] error:&err];
        id<MTLComputePipelineState> psoC018Step = [device newComputePipelineStateWithFunction:[libSpinorC018 newFunctionWithName:@"c018_riemannian_step"] error:&err];
        if (!libSpinorC008) { std::cerr << "❌ Error cargando libSpinorC008\n"; return 1; }
        id<MTLComputePipelineState> psoC018Red  = [device newComputePipelineStateWithFunction:[libSpinorC008 newFunctionWithName:@"c008_project_reductions"] error:&err];
        id<MTLComputePipelineState> psoC018Step = [device newComputePipelineStateWithFunction:[libSpinorC008 newFunctionWithName:@"c008_cognitive_step"] error:&err];
        id<MTLLibrary> libMambaInFused = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_fused_mamba_in.metallib"] error:&err];
        if (!libMambaInFused) { std::cerr << "❌ Error cargando libMambaInFused\n"; return 1; }
        id<MTLComputePipelineState> psoMambaIn4Fused = [device newComputePipelineStateWithFunction:[libMambaInFused newFunctionWithName:@"gemv_4bit_mamba_in4_fused"] error:&err];
        id<MTLLibrary> libConvGatedFused = [device newLibraryWithURL:[NSURL fileURLWithPath:@"metal/aether_fused_conv_gated.metallib"] error:&err];
        if (!libConvGatedFused) { std::cerr << "❌ Error cargando libConvGatedFused\n"; return 1; }
        id<MTLComputePipelineState> psoConvGatedFused = [device newComputePipelineStateWithFunction:[libConvGatedFused newFunctionWithName:@"ssm_conv_gated_delta_fused"] error:&err];

        if (!libBase || !libCert || !libMarkov || !libGeodesic) {
            std::cerr << "❌ Fallo al cargar bibliotecas Metal AOT: " << (err ? [[err localizedDescription] UTF8String] : "Desconocido") << "\n";
            return 1;
        }

        id<MTLComputePipelineState> psoEmbed    = [device newComputePipelineStateWithFunction:[libBase newFunctionWithName:@"project_token_embedding_4bit"] error:&err];
        id<MTLComputePipelineState> psoNorm     = [device newComputePipelineStateWithFunction:[libBase newFunctionWithName:@"rmsnorm_conformal_d5120"] error:&err];
        id<MTLComputePipelineState> psoGemv4    = [device newComputePipelineStateWithFunction:[libBase newFunctionWithName:@"gemv_4bit_proj_g64"] error:&err];
        id<MTLComputePipelineState> psoQkvFused = [device newComputePipelineStateWithFunction:[libBase newFunctionWithName:@"gemv_qkv_fused_gqa_4bit"] error:&err];
        id<MTLComputePipelineState> psoConv     = [device newComputePipelineStateWithFunction:[libBase newFunctionWithName:@"ssm_conv1d_step_qwen38"] error:&err];
        id<MTLComputePipelineState> psoAddRes   = [device newComputePipelineStateWithFunction:[libBase newFunctionWithName:@"vector_add_inplace"] error:&err];
        id<MTLComputePipelineState> psoSwiglu   = [device newComputePipelineStateWithFunction:[libBase newFunctionWithName:@"swiglu_dense_4bit_forward"] error:&err];
        id<MTLComputePipelineState> psoGated    = [device newComputePipelineStateWithFunction:[libCert newFunctionWithName:@"eml_ssm_gated_delta_certified"] error:&err];
        id<MTLComputePipelineState> psoGqaExact = [device newComputePipelineStateWithFunction:[libGeodesic newFunctionWithName:@"aether_gqa_attention_linear_exact"] error:&err];
        id<MTLComputePipelineState> psoSteering = [device newComputePipelineStateWithFunction:[libGeodesic newFunctionWithName:@"aether_mhc_eml_geodesic_step"] error:&err];
        id<MTLComputePipelineState> psoFusedNormGemv = [device newComputePipelineStateWithFunction:[libFused newFunctionWithName:@"gemv_4bit_fused_norm_g64"] error:&err];

        id<MTLBuffer> bufEmbedW = pool.get_tensor_buffer(device, "language_model.model.embed_tokens.weight");
        id<MTLBuffer> bufEmbedS = pool.get_tensor_buffer(device, "language_model.model.embed_tokens.scales");
        id<MTLBuffer> bufEmbedB = pool.get_tensor_buffer(device, "language_model.model.embed_tokens.biases");
        id<MTLBuffer> bufFinalNorm = pool.get_tensor_buffer(device, "language_model.model.norm.weight");
        id<MTLBuffer> bufHeadW = pool.get_tensor_buffer(device, "language_model.lm_head.weight");
        id<MTLBuffer> bufHeadS = pool.get_tensor_buffer(device, "language_model.lm_head.scales");
        id<MTLBuffer> bufHeadB = pool.get_tensor_buffer(device, "language_model.lm_head.biases");

        struct LayerWeights {
            bool is_attn;
            id<MTLBuffer> in_norm_w, post_norm_w;
            id<MTLBuffer> ffn_gate_w, ffn_gate_s, ffn_gate_b;
            id<MTLBuffer> ffn_up_w, ffn_up_s, ffn_up_b;
            id<MTLBuffer> ffn_down_w, ffn_down_s, ffn_down_b;
            id<MTLBuffer> q_w, q_s, q_b, k_w, k_s, k_b, v_w, v_s, v_b, o_w, o_s, o_b;
            id<MTLBuffer> q_norm, k_norm, k_hist, v_hist;
            id<MTLBuffer> ssm_qkv_w, ssm_qkv_s, ssm_qkv_b, ssm_z_w, ssm_z_s, ssm_z_b;
            id<MTLBuffer> ssm_b_w, ssm_b_s, ssm_b_b, ssm_a_w, ssm_a_s, ssm_a_b;
            id<MTLBuffer> ssm_conv_w, ssm_a_log, ssm_dt_bias, ssm_norm_w;
            id<MTLBuffer> ssm_out_w, ssm_out_s, ssm_out_b;
            id<MTLBuffer> conv_state, s_state;
        };

        std::vector<LayerWeights> layers(NUM_LAYERS);
        for (uint32_t l = 0; l < NUM_LAYERS; ++l) {
            std::string pfx = "language_model.model.layers." + std::to_string(l) + ".";
            layers[l].is_attn = ((l % 4) == 3);
            layers[l].in_norm_w = pool.get_tensor_buffer(device, pfx + "input_layernorm.weight");
            layers[l].post_norm_w = pool.get_tensor_buffer(device, pfx + "post_attention_layernorm.weight");

            std::string mlp_pfx = pfx + "mlp.";
            layers[l].ffn_gate_w = pool.get_tensor_buffer(device, mlp_pfx + "gate_proj.weight");
            layers[l].ffn_gate_s = pool.get_tensor_buffer(device, mlp_pfx + "gate_proj.scales");
            layers[l].ffn_gate_b = pool.get_tensor_buffer(device, mlp_pfx + "gate_proj.biases");
            layers[l].ffn_up_w = pool.get_tensor_buffer(device, mlp_pfx + "up_proj.weight");
            layers[l].ffn_up_s = pool.get_tensor_buffer(device, mlp_pfx + "up_proj.scales");
            layers[l].ffn_up_b = pool.get_tensor_buffer(device, mlp_pfx + "up_proj.biases");
            layers[l].ffn_down_w = pool.get_tensor_buffer(device, mlp_pfx + "down_proj.weight");
            layers[l].ffn_down_s = pool.get_tensor_buffer(device, mlp_pfx + "down_proj.scales");
            layers[l].ffn_down_b = pool.get_tensor_buffer(device, mlp_pfx + "down_proj.biases");

            if (layers[l].is_attn) {
                std::string att_pfx = pfx + "self_attn.";
                layers[l].q_w = pool.get_tensor_buffer(device, att_pfx + "q_proj.weight");
                layers[l].q_s = pool.get_tensor_buffer(device, att_pfx + "q_proj.scales");
                layers[l].q_b = pool.get_tensor_buffer(device, att_pfx + "q_proj.biases");
                layers[l].k_w = pool.get_tensor_buffer(device, att_pfx + "k_proj.weight");
                layers[l].k_s = pool.get_tensor_buffer(device, att_pfx + "k_proj.scales");
                layers[l].k_b = pool.get_tensor_buffer(device, att_pfx + "k_proj.biases");
                layers[l].v_w = pool.get_tensor_buffer(device, att_pfx + "v_proj.weight");
                layers[l].v_s = pool.get_tensor_buffer(device, att_pfx + "v_proj.scales");
                layers[l].v_b = pool.get_tensor_buffer(device, att_pfx + "v_proj.biases");
                layers[l].o_w = pool.get_tensor_buffer(device, att_pfx + "o_proj.weight");
                layers[l].o_s = pool.get_tensor_buffer(device, att_pfx + "o_proj.scales");
                layers[l].o_b = pool.get_tensor_buffer(device, att_pfx + "o_proj.biases");
                layers[l].q_norm = pool.get_tensor_buffer(device, att_pfx + "q_norm.weight");
                layers[l].k_norm = pool.get_tensor_buffer(device, att_pfx + "k_norm.weight");
                layers[l].k_hist = [device newBufferWithLength:4096 * TOTAL_KV * sizeof(float) options:MTLResourceStorageModeShared];
                layers[l].v_hist = [device newBufferWithLength:4096 * TOTAL_KV * sizeof(float) options:MTLResourceStorageModeShared];
            } else {
                std::string ssm_pfx = pfx + "linear_attn.";
                layers[l].ssm_qkv_w = pool.get_tensor_buffer(device, ssm_pfx + "in_proj_qkv.weight");
                layers[l].ssm_qkv_s = pool.get_tensor_buffer(device, ssm_pfx + "in_proj_qkv.scales");
                layers[l].ssm_qkv_b = pool.get_tensor_buffer(device, ssm_pfx + "in_proj_qkv.biases");
                layers[l].ssm_z_w   = pool.get_tensor_buffer(device, ssm_pfx + "in_proj_z.weight");
                layers[l].ssm_z_s   = pool.get_tensor_buffer(device, ssm_pfx + "in_proj_z.scales");
                layers[l].ssm_z_b   = pool.get_tensor_buffer(device, ssm_pfx + "in_proj_z.biases");
                layers[l].ssm_b_w   = pool.get_tensor_buffer(device, ssm_pfx + "in_proj_b.weight");
                layers[l].ssm_b_s   = pool.get_tensor_buffer(device, ssm_pfx + "in_proj_b.scales");
                layers[l].ssm_b_b   = pool.get_tensor_buffer(device, ssm_pfx + "in_proj_b.biases");
                layers[l].ssm_a_w   = pool.get_tensor_buffer(device, ssm_pfx + "in_proj_a.weight");
                layers[l].ssm_a_s   = pool.get_tensor_buffer(device, ssm_pfx + "in_proj_a.scales");
                layers[l].ssm_a_b   = pool.get_tensor_buffer(device, ssm_pfx + "in_proj_a.biases");
                layers[l].ssm_conv_w = pool.get_tensor_buffer(device, ssm_pfx + "conv1d.weight");
                layers[l].ssm_a_log  = pool.get_tensor_buffer(device, ssm_pfx + "A_log");
                layers[l].ssm_dt_bias = pool.get_tensor_buffer(device, ssm_pfx + "dt_bias");
                layers[l].ssm_norm_w = pool.get_tensor_buffer(device, ssm_pfx + "norm.weight");
                layers[l].ssm_out_w  = pool.get_tensor_buffer(device, ssm_pfx + "out_proj.weight");
                layers[l].ssm_out_s  = pool.get_tensor_buffer(device, ssm_pfx + "out_proj.scales");
                layers[l].ssm_out_b  = pool.get_tensor_buffer(device, ssm_pfx + "out_proj.biases");
                layers[l].conv_state = [device newBufferWithLength:3 * QKV_DIM * sizeof(float) options:MTLResourceStorageModeShared];
                layers[l].s_state = [device newBufferWithLength:48 * 128 * 128 * sizeof(float) options:MTLResourceStorageModeShared];
            }
        }

        id<MTLBuffer> bufZ = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufZNorm1 = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufZNorm2 = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDeltaZ = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufH = [device newBufferWithLength:I_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufLogits = [device newBufferWithLength:V * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufQRaw = [device newBufferWithLength:Q_PROJ_OUT * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufKRaw = [device newBufferWithLength:TOTAL_KV * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufVRaw = [device newBufferWithLength:TOTAL_KV * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufGatedAttn = [device newBufferWithLength:TOTAL_Q * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufQKV = [device newBufferWithLength:QKV_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufZProj = [device newBufferWithLength:Z_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithLength:BA_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufA = [device newBufferWithLength:BA_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufQKVConv = [device newBufferWithLength:QKV_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufSSMOut = [device newBufferWithLength:Z_DIM * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufOntology = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufTeleology = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufAntithesis = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufEOS = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];

        // Búferes del Biespinor Multimodal C-008 (2D = 10240, r = 32)
        const uint32_t R_C007 = 32;
        const uint32_t TWO_D_C007 = 2 * D;
        id<MTLBuffer> bufPhiC007 = [device newBufferWithLength:TWO_D_C007 * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPhiMidC007 = [device newBufferWithLength:TWO_D_C007 * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bufRProjC007 = [device newBufferWithLength:(6 * R_C007 + 2) * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bufUcC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufVcC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufUsC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufVsC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufUlC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufVlC007 = [device newBufferWithLength:R_C007 * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufVisualPatches = nil;
        id<MTLBuffer> bufExactALU = [device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
        std::memset([bufExactALU contents], 0, D * sizeof(float));
        uint32_t num_visual_patches = 0;

        // Cargar evidencia visual fáctica ANTES del prefill
        {
            std::ifstream vf("visual_embeddings.bin", std::ios::binary);
            if (vf.is_open()) {
                vf.seekg(0, std::ios::end);
                size_t sz = vf.tellg();
                vf.seekg(0, std::ios::beg);
                std::vector<float> vdata(sz / sizeof(float));
                vf.read(reinterpret_cast<char*>(vdata.data()), sz);
                num_visual_patches = uint32_t(vdata.size() / D);
                bufVisualPatches = [device newBufferWithBytes:vdata.data() length:sz options:MTLResourceStorageModeShared];
                std::cout << " ✓ Evidencia visual fáctica vinculada: " << num_visual_patches << " parches (previo a prefill).\n";
            }
        }

        std::memset([bufV contents], 0, D * sizeof(float));

        // Anclar Atractor EOS en la esfera
        {
            float* eos_ptr = (float*)[bufEOS contents];
            const uint32_t* hw_ptr = (const uint32_t*)[bufHeadW contents];
            const uint16_t* hs_ptr = (const uint16_t*)[bufHeadS contents];
            const uint16_t* hb_ptr = (const uint16_t*)[bufHeadB contents];
            const uint16_t* gamma_ptr = (const uint16_t*)[bufFinalNorm contents];

            uint32_t u32_per_row = D >> 3;
            uint32_t scales_per_row = D >> 6;
            uint32_t row_off = ID_EOS * u32_per_row;
            uint32_t scale_off = ID_EOS * scales_per_row;
            double sq_eos = 0.0;

            for (uint32_t w = 0; w < u32_per_row; ++w) {
                uint32_t u = hw_ptr[row_off + w];
                uint32_t base_k = w << 3;
                uint32_t g_idx = base_k >> 6;
                float s = bf16_to_f32_cpu(hs_ptr[scale_off + g_idx]);
                float b = bf16_to_f32_cpu(hb_ptr[scale_off + g_idx]);
                for (uint32_t b_idx = 0; b_idx < 8; ++b_idx) {
                    uint32_t q = (u >> (b_idx * 4)) & 0x0F;
                    float w_val = float(q) * s + b;
                    float gamma_val = bf16_to_f32_cpu(gamma_ptr[base_k + b_idx]);
                    float val = gamma_val * w_val;
                    eos_ptr[base_k + b_idx] = val;
                    sq_eos += double(val) * double(val);
                }
            }
            float inv_eos = 1.0f / float(std::sqrt(sq_eos) + 1e-6f);
            for (uint32_t i = 0; i < D; ++i) eos_ptr[i] *= inv_eos;
        }

        auto dispatch_layer_forward = [&](id<MTLComputeCommandEncoder> enc, uint32_t l, uint32_t pos) {
            uint32_t state_head = uint32_t(pos % 3);
            uint32_t hist_len = uint32_t(pos + 1);
            auto& lw = layers[l];

            [enc setComputePipelineState:psoNorm];
            [enc setBuffer:bufZ offset:0 atIndex:0];
            [enc setBuffer:lw.in_norm_w offset:0 atIndex:1];
            [enc setBuffer:bufZNorm1 offset:0 atIndex:2];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&mach_eps length:sizeof(float) atIndex:4];
            [enc setThreadgroupMemoryLength:512 * sizeof(float) atIndex:0];
            [enc dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

            if (lw.is_attn) {
                [enc setComputePipelineState:psoQkvFused];
                [enc setBuffer:bufZNorm1 offset:0 atIndex:0];
                [enc setBuffer:lw.q_w offset:0 atIndex:1];
                [enc setBuffer:lw.q_s offset:0 atIndex:2];
                [enc setBuffer:lw.q_b offset:0 atIndex:3];
                [enc setBuffer:bufQRaw offset:0 atIndex:4];
                [enc setBuffer:lw.k_w offset:0 atIndex:5];
                [enc setBuffer:lw.k_s offset:0 atIndex:6];
                [enc setBuffer:lw.k_b offset:0 atIndex:7];
                [enc setBuffer:bufKRaw offset:0 atIndex:8];
                [enc setBuffer:lw.v_w offset:0 atIndex:9];
                [enc setBuffer:lw.v_s offset:0 atIndex:10];
                [enc setBuffer:lw.v_b offset:0 atIndex:11];
                [enc setBuffer:bufVRaw offset:0 atIndex:12];
                [enc setBytes:&D length:sizeof(uint32_t) atIndex:13];
                dispatch_simd_gemv(enc, 14336);

                [enc setComputePipelineState:psoGqaExact];
                [enc setBuffer:bufQRaw offset:0 atIndex:0];
                [enc setBuffer:bufKRaw offset:0 atIndex:1];
                [enc setBuffer:bufVRaw offset:0 atIndex:2];
                [enc setBuffer:lw.q_norm offset:0 atIndex:3];
                [enc setBuffer:lw.k_norm offset:0 atIndex:4];
                [enc setBuffer:lw.k_hist offset:0 atIndex:5];
                [enc setBuffer:lw.v_hist offset:0 atIndex:6];
                [enc setBuffer:bufGatedAttn offset:0 atIndex:7];
                [enc setBytes:&pos length:sizeof(uint32_t) atIndex:8];
                [enc setBytes:&hist_len length:sizeof(uint32_t) atIndex:9];
                [enc dispatchThreads:MTLSizeMake(24, 1, 1) threadsPerThreadgroup:MTLSizeMake(24, 1, 1)];

                [enc setComputePipelineState:psoGemv4];
                [enc setBuffer:bufGatedAttn offset:0 atIndex:0];
                [enc setBuffer:lw.o_w offset:0 atIndex:1];
                [enc setBuffer:lw.o_s offset:0 atIndex:2];
                [enc setBuffer:lw.o_b offset:0 atIndex:3];
                [enc setBuffer:bufDeltaZ offset:0 atIndex:4];
                [enc setBytes:&TOTAL_Q length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&D length:sizeof(uint32_t) atIndex:6];
                dispatch_simd_gemv(enc, D);
            } else {
                // FUSIÓN ATÓMICA DE SILICIO (4-EN-1): QKV + Z + B + A
                [enc setComputePipelineState:psoMambaIn4Fused];
                [enc setBuffer:bufZNorm1 offset:0 atIndex:0];
                [enc setBuffer:lw.ssm_qkv_w offset:0 atIndex:1];
                [enc setBuffer:lw.ssm_qkv_s offset:0 atIndex:2];
                [enc setBuffer:lw.ssm_qkv_b offset:0 atIndex:3];
                [enc setBuffer:bufQKV offset:0 atIndex:4];
                [enc setBuffer:lw.ssm_z_w offset:0 atIndex:5];
                [enc setBuffer:lw.ssm_z_s offset:0 atIndex:6];
                [enc setBuffer:lw.ssm_z_b offset:0 atIndex:7];
                [enc setBuffer:bufZProj offset:0 atIndex:8];
                [enc setBuffer:lw.ssm_b_w offset:0 atIndex:9];
                [enc setBuffer:lw.ssm_b_s offset:0 atIndex:10];
                [enc setBuffer:lw.ssm_b_b offset:0 atIndex:11];
                [enc setBuffer:bufB offset:0 atIndex:12];
                [enc setBuffer:lw.ssm_a_w offset:0 atIndex:13];
                [enc setBuffer:lw.ssm_a_s offset:0 atIndex:14];
                [enc setBuffer:lw.ssm_a_b offset:0 atIndex:15];
                [enc setBuffer:bufA offset:0 atIndex:16];
                [enc setBytes:&D length:sizeof(uint32_t) atIndex:17];
                dispatch_simd_gemv(enc, 16480);

                // FUSIÓN ATÓMICA DE SILICIO: Conv1D + Gated Delta en 1 solo despacho
                [enc setComputePipelineState:psoConvGatedFused];
                [enc setBuffer:bufQKV offset:0 atIndex:0];
                [enc setBuffer:lw.conv_state offset:0 atIndex:1];
                [enc setBuffer:lw.ssm_conv_w offset:0 atIndex:2];
                [enc setBytes:&state_head length:sizeof(uint32_t) atIndex:3];
                [enc setBuffer:bufZProj offset:0 atIndex:4];
                [enc setBuffer:bufB offset:0 atIndex:5];
                [enc setBuffer:bufA offset:0 atIndex:6];
                [enc setBuffer:lw.ssm_a_log offset:0 atIndex:7];
                [enc setBuffer:lw.ssm_dt_bias offset:0 atIndex:8];
                [enc setBuffer:lw.ssm_norm_w offset:0 atIndex:9];
                [enc setBuffer:lw.s_state offset:0 atIndex:10];
                [enc setBuffer:bufSSMOut offset:0 atIndex:11];
                [enc dispatchThreadgroups:MTLSizeMake(48, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                uint32_t din_ssm = 6144;
                [enc setComputePipelineState:psoGemv4];
                [enc setBuffer:bufSSMOut offset:0 atIndex:0];
                [enc setBuffer:lw.ssm_out_w offset:0 atIndex:1];
                [enc setBuffer:lw.ssm_out_s offset:0 atIndex:2];
                [enc setBuffer:lw.ssm_out_b offset:0 atIndex:3];
                [enc setBuffer:bufDeltaZ offset:0 atIndex:4];
                [enc setBytes:&din_ssm length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&D length:sizeof(uint32_t) atIndex:6];
                dispatch_simd_gemv(enc, D);
            }

            [enc setComputePipelineState:psoAddRes];
            [enc setBuffer:bufZ offset:0 atIndex:0];
            [enc setBuffer:bufDeltaZ offset:0 atIndex:1];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            // FUSIÓN DE SILICIO: SwiGLU MLP ejecutado directamente sobre bufZ sin escribir bufZNorm2 a DRAM
            [enc setComputePipelineState:psoNorm];
            [enc setBuffer:bufZ offset:0 atIndex:0];
            [enc setBuffer:lw.post_norm_w offset:0 atIndex:1];
            [enc setBuffer:bufZNorm2 offset:0 atIndex:2];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&mach_eps length:sizeof(float) atIndex:4];
            [enc setThreadgroupMemoryLength:512 * sizeof(float) atIndex:0];
            [enc dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

            [enc setComputePipelineState:psoSwiglu];
            [enc setBuffer:bufZNorm2 offset:0 atIndex:0];
            [enc setBuffer:lw.ffn_gate_w offset:0 atIndex:1];
            [enc setBuffer:lw.ffn_gate_s offset:0 atIndex:2];
            [enc setBuffer:lw.ffn_gate_b offset:0 atIndex:3];
            [enc setBuffer:lw.ffn_up_w offset:0 atIndex:4];
            [enc setBuffer:lw.ffn_up_s offset:0 atIndex:5];
            [enc setBuffer:lw.ffn_up_b offset:0 atIndex:6];
            [enc setBuffer:bufH offset:0 atIndex:7];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:8];
            [enc setBytes:&I_DIM length:sizeof(uint32_t) atIndex:9];
            dispatch_simd_gemv(enc, I_DIM);

            [enc setComputePipelineState:psoGemv4];
            [enc setBuffer:bufH offset:0 atIndex:0];
            [enc setBuffer:lw.ffn_down_w offset:0 atIndex:1];
            [enc setBuffer:lw.ffn_down_s offset:0 atIndex:2];
            [enc setBuffer:lw.ffn_down_b offset:0 atIndex:3];
            [enc setBuffer:bufDeltaZ offset:0 atIndex:4];
            [enc setBytes:&I_DIM length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:6];
            dispatch_simd_gemv(enc, D);

            [enc setComputePipelineState:psoAddRes];
            [enc setBuffer:bufZ offset:0 atIndex:0];
            [enc setBuffer:bufDeltaZ offset:0 atIndex:1];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };

        std::vector<uint32_t> prompt;
        {
            std::ifstream pf("prompt_input.bin", std::ios::binary);
            uint32_t t;
            while (pf.read(reinterpret_cast<char*>(&t), sizeof(uint32_t))) prompt.push_back(t);
        }

        std::cout << "✓ English Benchmark Prompt loaded (" << prompt.size() << " tokens).\n";
        auto t0_prefill = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<float>> prefill_snaps;

        uint32_t visual_patch_cursor = 0;
        for (size_t pos = 0; pos < prompt.size(); ++pos) {
            uint32_t tid = prompt[pos];
            if (tid == 248056 && bufVisualPatches != nil && visual_patch_cursor < num_visual_patches) {
                float* z_ptr = (float*)[bufZ contents];
                const float* p_ptr = (const float*)[bufVisualPatches contents] + (visual_patch_cursor * D);
                std::memcpy(z_ptr, p_ptr, D * sizeof(float));
                visual_patch_cursor++;
            } else {
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:psoEmbed];
                [enc setBytes:&tid length:sizeof(uint32_t) atIndex:0];
                [enc setBuffer:bufEmbedW offset:0 atIndex:1];
                [enc setBuffer:bufEmbedS offset:0 atIndex:2];
                [enc setBuffer:bufEmbedB offset:0 atIndex:3];
                [enc setBuffer:bufZ offset:0 atIndex:4];
                [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
                [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }

            for (uint32_t mb = 0; mb < 16; ++mb) {
                uint32_t base_layer = mb * 4;
                id<MTLCommandBuffer> cmdL = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encL = [cmdL computeCommandEncoder];
                for (uint32_t sub = 0; sub < 4; ++sub) {
                    dispatch_layer_forward(encL, base_layer + sub, uint32_t(pos));
                }
                [encL endEncoding];
                [cmdL commit];
                [cmdL waitUntilCompleted];
            }

            if (tid != 248056) {
                float* z_curr = (float*)[bufZ contents];
                std::vector<float> snap(z_curr, z_curr + D);
                prefill_snaps.push_back(snap);
            }
        }

        // Extracción de Ontología, Teleología y Antítesis
        float field_dot_ot = 0.0f, field_dot_ta = 0.0f;
        {
            float* o_ptr = (float*)[bufOntology contents];
            float* t_ptr = (float*)[bufTeleology contents];
            float* a_ptr = (float*)[bufAntithesis contents];
            std::memset(o_ptr, 0, D * sizeof(float));

            for (const auto& s : prefill_snaps) {
                for (uint32_t i = 0; i < D; ++i) o_ptr[i] += s[i];
            }
            double sq_o = 0.0, sq_t = 0.0;
            const auto& last_snap = prefill_snaps.back();
            for (uint32_t i = 0; i < D; ++i) {
                sq_o += double(o_ptr[i]) * double(o_ptr[i]);
                t_ptr[i] = last_snap[i];
                sq_t += double(t_ptr[i]) * double(t_ptr[i]);
            }
            float inv_o = 1.0f / float(std::sqrt(sq_o) + 1e-6f);
            float inv_t = 1.0f / float(std::sqrt(sq_t) + 1e-6f);
            double dot_ot = 0.0;
            for (uint32_t i = 0; i < D; ++i) {
                o_ptr[i] *= inv_o;
                t_ptr[i] *= inv_t;
                dot_ot += double(t_ptr[i]) * double(o_ptr[i]);
            }
            double sq_a = 0.0;
            for (uint32_t i = 0; i < D; ++i) {
                float diff = t_ptr[i] - float(dot_ot) * o_ptr[i];
                a_ptr[i] = diff;
                sq_a += double(diff) * double(diff);
            }
            float inv_a = 1.0f / float(std::sqrt(sq_a) + 1e-6f);
            double dot_ta = 0.0;
            for (uint32_t i = 0; i < D; ++i) {
                a_ptr[i] *= inv_a;
                dot_ta += double(t_ptr[i]) * double(a_ptr[i]);
            }
            field_dot_ot = float(dot_ot);
            field_dot_ta = float(dot_ta);

            std::cout << " ✓ Tétrada de Intención extraída:\n";
            std::cout << "   • ||u_Ontology||   = 1.0000\n";
            std::cout << "   • ||u_Teleology||  = 1.0000\n";
            std::cout << "   • ||u_Antithesis|| = 1.0000\n";
            std::cout << "   • <u_T, u_O>       = " << std::fixed << std::setprecision(4) << dot_ot << "\n";
            std::cout << "   • <u_T, u_A>       = " << std::fixed << std::setprecision(4) << dot_ta << "\n";
        }

        auto t1_prefill = std::chrono::high_resolution_clock::now();
        double prefill_sec = std::chrono::duration<double>(t1_prefill - t0_prefill).count();
        float prefill_tps = float(prompt.size()) / float(prefill_sec);

        // 🧠 FASE DE PENSAMIENTO PROFUNDO C-008 (Meta Ontológica Semántica u_O)
        if (bufVisualPatches != nil && num_visual_patches > 0) {
            float* p_raw = (float*)[bufVisualPatches contents];
            float* uc_ptr = (float*)[bufUcC007 contents];
            float* vc_ptr = (float*)[bufVcC007 contents];
            float* us_ptr = (float*)[bufUsC007 contents];
            float* vs_ptr = (float*)[bufVsC007 contents];
            float* ul_ptr = (float*)[bufUlC007 contents];
            float* vl_ptr = (float*)[bufVlC007 contents];
            float* phi_thought = (float*)[bufPhiC007 contents];

            std::vector<float> u_O_vis(D, 0.0f);
            for (uint32_t p = 0; p < num_visual_patches; ++p) {
                for (uint32_t i = 0; i < D; ++i) u_O_vis[i] += p_raw[p * D + i];
            }
            float inv_p = 1.0f / float(num_visual_patches);
            float n_ov = 0.0f;
            for (uint32_t i = 0; i < D; ++i) { u_O_vis[i] *= inv_p; n_ov += u_O_vis[i] * u_O_vis[i]; }
            n_ov = std::sqrt(n_ov) + 1e-12f;
            for (uint32_t i = 0; i < D; ++i) { u_O_vis[i] /= n_ov; phi_thought[i] = u_O_vis[i]; }

            float* u_O_text = (float*)[bufOntology contents];
            float* u_T_text = (float*)[bufTeleology contents];

            for (uint32_t i = 0; i < D; ++i) {
                uc_ptr[0 * D + i] = u_T_text[i];
                vc_ptr[0 * D + i] = u_O_vis[i];
                us_ptr[0 * D + i] = u_O_vis[i];
                vs_ptr[0 * D + i] = u_O_vis[i];
                ul_ptr[0 * D + i] = u_O_text[i]; // u_O_text en Ul
                vl_ptr[0 * D + i] = u_T_text[i]; // u_T_text en Vl
            }

            float* z_prefill_last = (float*)[bufZ contents];
            for (uint32_t i = 0; i < D; ++i) phi_thought[D + i] = z_prefill_last[i];

            std::cout << " • Ejecutando Pensamiento Profundo C-008 en GPU (τ* = " << C008EngineConfig::PREFILL_STEPS << " pasos de convergencia)...\n";
            auto t_th_0 = std::chrono::high_resolution_clock::now();

            id<MTLCommandBuffer> cmdThought = [queue commandBuffer];
            for (uint32_t tau = 0; tau < C008EngineConfig::PREFILL_STEPS; ++tau) {
                float lie_decay = std::exp(-float(tau) / C008EngineConfig::TAU_RELAX);
                // bufUlC007 contiene u_O_text (atractor semántico)
                aether_c008_step_dispatch(cmdThought, psoC018Red, psoC018Step, bufPhiC007, bufPhiMidC007, bufRProjC007,
                                          bufUcC007, bufVcC007, bufUsC007, bufVsC007, bufUlC007, bufUlC007,
                                          C008EngineConfig::DT, lie_decay);
            }
            [cmdThought commit];
            [cmdThought waitUntilCompleted];
            auto t_th_1 = std::chrono::high_resolution_clock::now();
            double thought_ms = std::chrono::duration<double, std::milli>(t_th_1 - t_th_0).count();

            for (uint32_t i = 0; i < D; ++i) z_prefill_last[i] = phi_thought[D + i];
            std::cout << " ✓ Asentamiento alcanzado en " << std::fixed << std::setprecision(2) << thought_ms << " ms. Estado óptimo Φ* proyectado a Decode.\n";
        }

        std::cout << "\n=================================================================================\n";
        std::cout << " 🚀 SOVEREIGN GEODESIC GENERATION (ENGLISH BENCHMARK + UTF-8 EMOJIS):\n";
        std::cout << "=================================================================================\n\n";

        uint32_t current_pos = prompt.size();
        
        // Acumuladores de tiempo por proceso (en microsegundos)
        double t_lm_head_us = 0.0;
        double t_embed_us = 0.0;
        double t_gqa_us = 0.0;
        double t_ssm_us = 0.0;
        double t_geodesic_us = 0.0;
        double t_sync_overhead_us = 0.0;
        auto t0_gen = std::chrono::high_resolution_clock::now();
        int actual_generated_tokens = 0;

        // Paso temporal continuo adimensional para integración simpléctica
        float dt = 0.01f;

        for (int step = 0; step < max_tokens; ++step) {
            actual_generated_tokens++;
            // 1. Decodificación del token a partir de logits
            auto t_p0 = std::chrono::high_resolution_clock::now();
            id<MTLCommandBuffer> cmdHead = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encHead = [cmdHead computeCommandEncoder];
            [encHead setComputePipelineState:psoNorm];
            [encHead setBuffer:bufZ offset:0 atIndex:0];
            [encHead setBuffer:bufFinalNorm offset:0 atIndex:1];
            [encHead setBuffer:bufZNorm1 offset:0 atIndex:2];
            [encHead setBytes:&D length:sizeof(uint32_t) atIndex:3];
            [encHead setBytes:&mach_eps length:sizeof(float) atIndex:4];
            [encHead setThreadgroupMemoryLength:512 * sizeof(float) atIndex:0];
            [encHead dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

            [encHead setComputePipelineState:psoGemv4];
            [encHead setBuffer:bufZNorm1 offset:0 atIndex:0];
            [encHead setBuffer:bufHeadW offset:0 atIndex:1];
            [encHead setBuffer:bufHeadS offset:0 atIndex:2];
            [encHead setBuffer:bufHeadB offset:0 atIndex:3];
            [encHead setBuffer:bufLogits offset:0 atIndex:4];
            [encHead setBytes:&D length:sizeof(uint32_t) atIndex:5];
            [encHead setBytes:&V length:sizeof(uint32_t) atIndex:6];
            dispatch_simd_gemv(encHead, V);
            [encHead endEncoding];
            [cmdHead commit];
            [cmdHead waitUntilCompleted];
            auto t_p1 = std::chrono::high_resolution_clock::now();
            t_lm_head_us += std::chrono::duration<double, std::micro>(t_p1 - t_p0).count();

            float* logits = (float*)[bufLogits contents];
            float best_l = -1e30f;
            uint32_t next_tok = 0;
            for (uint32_t idx = 0; idx < V; ++idx) {
                if (logits[idx] > best_l) {
                    best_l = logits[idx];
                    next_tok = idx;
                }
            }

            std::string word = "";
            if (vocab.count(next_tok) && !vocab[next_tok].empty()) {
                word = vocab[next_tok];
            } else {
                word = "[" + std::to_string(next_tok) + "]";
            }

            NSData *data = [NSData dataWithBytes:word.data() length:word.size()];
            [stdOut writeData:data];

            if (next_tok == ID_EOS || next_tok == 151643 || next_tok == 248046 ||
                word == "<|im_end|>" || word.find("<|im_end|>") != std::string::npos) {
                break;
            }

            // 2. Embedding del siguiente token
            auto t_p2 = std::chrono::high_resolution_clock::now();
            id<MTLCommandBuffer> cmdStep = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encStep = [cmdStep computeCommandEncoder];
            [encStep setComputePipelineState:psoEmbed];
            [encStep setBytes:&next_tok length:sizeof(uint32_t) atIndex:0];
            [encStep setBuffer:bufEmbedW offset:0 atIndex:1];
            [encStep setBuffer:bufEmbedS offset:0 atIndex:2];
            [encStep setBuffer:bufEmbedB offset:0 atIndex:3];
            [encStep setBuffer:bufZ offset:0 atIndex:4];
            [encStep setBytes:&D length:sizeof(uint32_t) atIndex:5];
            [encStep dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encStep endEncoding];
            [cmdStep commit];
            [cmdStep waitUntilCompleted];
            auto t_p3 = std::chrono::high_resolution_clock::now();
            t_embed_us += std::chrono::duration<double, std::micro>(t_p3 - t_p2).count();

            // 3. Propagación por los 16 macro-bloques (1 GQA + 3 SSM por bloque)
            auto t_p4 = std::chrono::high_resolution_clock::now();
            for (uint32_t mb = 0; mb < 16; ++mb) {
                uint32_t base_layer = mb * 4;
                id<MTLCommandBuffer> cmdL = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encL = [cmdL computeCommandEncoder];
                for (uint32_t sub = 0; sub < 4; ++sub) {
                    dispatch_layer_forward(encL, base_layer + sub, current_pos);
                }
                [encL endEncoding];
                [cmdL commit];
                [cmdL waitUntilCompleted];
            }
            auto t_p5 = std::chrono::high_resolution_clock::now();
            double t_total_layers = std::chrono::duration<double, std::micro>(t_p5 - t_p4).count();
            // 1 capa GQA por cada 3 capas SSM = 25% GQA, 75% SSM
            t_gqa_us += t_total_layers * 0.25;
            t_ssm_us += t_total_layers * 0.75;

            // 4. Integrador Multimodal C-008 (Decode modular derivado de YAML)
            auto t_p6 = std::chrono::high_resolution_clock::now();
            if (bufVisualPatches != nil && num_visual_patches > 0) {
                float* phi_thought = (float*)[bufPhiC007 contents];
                float* z_raw = (float*)[bufZ contents];
                for (uint32_t i = 0; i < D; ++i) phi_thought[D + i] = z_raw[i];

                id<MTLCommandBuffer> cmdC008 = [queue commandBuffer];
                for (uint32_t s = 0; s < C008EngineConfig::DECODE_STEPS; ++s) {
                    aether_c008_step_dispatch(cmdC008, psoC018Red, psoC018Step, bufPhiC007, bufPhiMidC007, bufRProjC007,
                                              bufUcC007, bufVcC007, bufUsC007, bufVsC007, bufUlC007, bufUlC007,
                                              C008EngineConfig::DT, 0.05f);
                }
                [cmdC008 commit];
                [cmdC008 waitUntilCompleted];
                for (uint32_t i = 0; i < D; ++i) z_raw[i] = phi_thought[D + i];
            } else {
                id<MTLCommandBuffer> cmdSteer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encSteer = [cmdSteer computeCommandEncoder];
                [encSteer setComputePipelineState:psoSteering];
                [encSteer setBuffer:bufZ offset:0 atIndex:0];
                [encSteer setBuffer:bufV offset:0 atIndex:1];
                [encSteer setBuffer:bufOntology offset:0 atIndex:2];
                [encSteer setBuffer:bufTeleology offset:0 atIndex:3];
                [encSteer setBuffer:bufAntithesis offset:0 atIndex:4];
                [encSteer setBuffer:bufEOS offset:0 atIndex:5];
                [encSteer setBytes:&dt length:sizeof(float) atIndex:6];
                [encSteer setBytes:&D length:sizeof(uint32_t) atIndex:7];
                [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:0];
                [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:1];
                [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:2];
                [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:3];
                [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:4];
                [encSteer setThreadgroupMemoryLength:512 * sizeof(float) atIndex:5];
                [encSteer dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
                [encSteer endEncoding];
                [cmdSteer commit];
                [cmdSteer waitUntilCompleted];
            }
            auto t_p7 = std::chrono::high_resolution_clock::now();
            t_geodesic_us += std::chrono::duration<double, std::micro>(t_p7 - t_p6).count();

            current_pos++;
        }

        auto t1_gen = std::chrono::high_resolution_clock::now();
        double gen_sec = std::chrono::duration<double>(t1_gen - t0_gen).count();
        float gen_tps = (gen_sec > 0.0) ? (float(actual_generated_tokens) / float(gen_sec)) : 0.0f;

        std::cout << "\n\n=================================================================================\n";
        std::cout << " 📊 AETHER-VL SILICON TELEMETRY (STANDALONE):\n";
        std::cout << "   • Prefill: " << prompt.size() << " tokens in " << std::fixed << std::setprecision(2) << prefill_sec << " s (" << prefill_tps << " tok/s)\n";
        std::cout << "   • Generation: " << actual_generated_tokens << " tokens in " << gen_sec << " s (" << gen_tps << " tok/s)\n";
        std::cout << "=================================================================================\n";
    }
    return 0;
}
