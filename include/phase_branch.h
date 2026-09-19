#pragma once
#import <Metal/Metal.h>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>

struct BranchTriggerConfig {
    float energy_threshold = 0.85f;
    float gradient_threshold = 1.20f;
    float task_resonance_threshold = 0.65f;
    uint32_t max_sub_steps = 64;
    float beta_coupling = 1.5f;
    std::string prompt_injection = "";
    std::string target_engine = "AETHER-C008-COGNITIVE";
};

class PhaseBranchManager {
private:
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    BranchTriggerConfig config;
    bool branch_active = false;

public:
    PhaseBranchManager(id<MTLDevice> dev, id<MTLCommandQueue> q) : device(dev), queue(q) {}

    void load_config_simple(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return;
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("energy_threshold:") != std::string::npos) {
                config.energy_threshold = std::stof(line.substr(line.find(":") + 1));
            } else if (line.find("coupling_strength_beta:") != std::string::npos) {
                config.beta_coupling = std::stof(line.substr(line.find(":") + 1));
            } else if (line.find("max_sub_steps:") != std::string::npos) {
                config.max_sub_steps = std::stoi(line.substr(line.find(":") + 1));
            }
        }
        std::cout << " ✓ Gestor de Bifurcación C-014 configurado desde " << path << "\n";
    }

    // Evaluación en tiempo real de las condiciones de disparo
    bool evaluate_trigger(float current_energy, float current_grad_norm, float resonance) {
        if (current_energy > config.energy_threshold || 
            current_grad_norm > config.gradient_threshold || 
            resonance > config.task_resonance_threshold) {
            return true;
        }
        return false;
    }

    // Clonación de estado latente Phi en GPU (Cero copia a CPU, pura UMA)
    id<MTLBuffer> spawn_clone_buffer(id<MTLBuffer> parent_phi, size_t size_bytes) {
        id<MTLBuffer> clone_phi = [device newBufferWithLength:size_bytes options:MTLResourceStorageModePrivate];
        
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:parent_phi sourceOffset:0 toBuffer:clone_phi destinationOffset:0 size:size_bytes];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted]; // Copia interna instantánea en la GPU

        std::cout << " ⚡ [FORK] Sub-agente interno clonado en silicio (Memoria Aislada UMA).\n";
        return clone_phi;
    }

    float get_beta() const { return config.beta_coupling; }
};
