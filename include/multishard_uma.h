#pragma once
#include "safetensors_uma.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <memory>
#include <vector>

namespace c_field {

class MultiShardUMAPool {
public:
    MultiShardUMAPool() {}

    bool initialize(const std::string& model_dir) {
        model_dir_ = model_dir;
        std::string index_path = model_dir + "/model.safetensors.index.json";

        std::ifstream file(index_path);
        if (!file.is_open()) {
            std::cerr << "❌ No se pudo abrir index.json en: " << index_path << "\n";
            return false;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string json = buffer.str();

        // Extraer bloque weight_map
        size_t wm_pos = json.find("\"weight_map\"");
        if (wm_pos == std::string::npos) return false;

        size_t open_brace = json.find('{', wm_pos);
        if (open_brace == std::string::npos) return false;
        size_t close_brace = json.find('}', open_brace);
        if (close_brace == std::string::npos) return false;

        std::string wm_str = json.substr(open_brace + 1, close_brace - open_brace - 1);
        size_t pos = 0;
        while (pos < wm_str.size()) {
            size_t k_start = wm_str.find('"', pos);
            if (k_start == std::string::npos) break;
            size_t k_end = wm_str.find('"', k_start + 1);
            if (k_end == std::string::npos) break;
            std::string tensor_name = wm_str.substr(k_start + 1, k_end - k_start - 1);

            size_t colon = wm_str.find(':', k_end);
            if (colon == std::string::npos) break;

            size_t v_start = wm_str.find('"', colon);
            if (v_start == std::string::npos) break;
            size_t v_end = wm_str.find('"', v_start + 1);
            if (v_end == std::string::npos) break;
            std::string shard_name = wm_str.substr(v_start + 1, v_end - v_start - 1);

            weight_map_[tensor_name] = shard_name;
            pos = v_end + 1;
        }

        std::cout << " ✓ Total de tensores catalogados: " << weight_map_.size() << "\n";
        return true;
    }

    id<MTLBuffer> get_tensor_buffer(id<MTLDevice> device, const std::string& tensor_name) {
        auto it = weight_map_.find(tensor_name);
        if (it == weight_map_.end()) return nil;

        std::string shard_name = it->second;

        // Cargar fragmento bajo demanda si no está en memoria virtual
        if (pools_.find(shard_name) == pools_.end()) {
            std::string full_path = model_dir_ + "/" + shard_name;
            auto p = std::make_unique<SafetensorsUMAPool>();
            if (!p->load_file(full_path)) {
                std::cerr << "❌ Error al mapear: " << shard_name << "\n";
                return nil;
            }
            std::cout << "   [UMA mmap] Fragmento montado: " << shard_name 
                      << " (" << (p->file_size() / (1024.0 * 1024.0)) << " MB)\n";
            pools_[shard_name] = std::move(p);
        }

        return pools_[shard_name]->get_metal_buffer(device, tensor_name);
    }

    size_t total_mapped_bytes() const {
        size_t total = 0;
        for (const auto& [name, pool] : pools_) {
            total += pool->file_size();
        }
        return total;
    }

private:
    std::string model_dir_;
    std::unordered_map<std::string, std::string> weight_map_;
    std::unordered_map<std::string, std::unique_ptr<SafetensorsUMAPool>> pools_;
};

} // namespace c_field
