#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

#import <Metal/Metal.h>

namespace c_field {

struct TensorDescriptor {
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t start_offset;
    uint64_t end_offset;
};

class SafetensorsUMAPool {
public:
    SafetensorsUMAPool() : fd_(-1), file_size_(0), mapped_data_(nullptr), header_size_(0) {}

    ~SafetensorsUMAPool() {
        close_file();
    }

    bool load_file(const std::string& filepath) {
        close_file();
        fd_ = open(filepath.c_str(), O_RDONLY);
        if (fd_ < 0) {
            std::cerr << "❌ No se pudo abrir: " << filepath << "\n";
            return false;
        }

        struct stat sb;
        if (fstat(fd_, &sb) == -1) return false;
        file_size_ = sb.st_size;

        mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (mapped_data_ == MAP_FAILED) {
            mapped_data_ = nullptr;
            return false;
        }

        header_size_ = *reinterpret_cast<uint64_t*>(mapped_data_);
        const char* header_json = reinterpret_cast<const char*>(mapped_data_) + 8;
        
        parse_header(std::string(header_json, header_size_));
        return true;
    }

    // Retorna MTLBuffer garantizando alineación de hardware estricta (16 bytes)
    id<MTLBuffer> get_metal_buffer(id<MTLDevice> device, const std::string& tensor_name) {
        auto it = registry_.find(tensor_name);
        if (it == registry_.end()) return nil;

        const TensorDescriptor& desc = it->second;
        size_t length = desc.end_offset - desc.start_offset;
        uint8_t* tensor_address = reinterpret_cast<uint8_t*>(mapped_data_) + 8 + header_size_ + desc.start_offset;

        uintptr_t addr = reinterpret_cast<uintptr_t>(tensor_address);

        // Si la dirección física está perfectamente alineada a 16 bytes: Zero-Copy instantáneo
        if ((addr % 16) == 0) {
            return [device newBufferWithBytesNoCopy:tensor_address
                                            length:length
                                           options:MTLResourceStorageModeShared
                                       deallocator:^(void* pointer, NSUInteger len) {}];
        }

        // Si el shard de origen tiene cabecera desalineada: Crear buffer alineado en UMA y copiar
        id<MTLBuffer> aligned_buf = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
        std::memcpy([aligned_buf contents], tensor_address, length);
        return aligned_buf;
    }

    const std::unordered_map<std::string, TensorDescriptor>& tensors() const {
        return registry_;
    }

    size_t file_size() const { return file_size_; }

private:
    int fd_;
    size_t file_size_;
    void* mapped_data_;
    uint64_t header_size_;
    std::unordered_map<std::string, TensorDescriptor> registry_;

    void close_file() {
        if (mapped_data_) {
            munmap(mapped_data_, file_size_);
            mapped_data_ = nullptr;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
        registry_.clear();
    }

    void parse_header(const std::string& json) {
        size_t pos = 0;
        while ((pos = json.find("\"", pos)) != std::string::npos) {
            size_t key_end = json.find("\"", pos + 1);
            if (key_end == std::string::npos) break;
            std::string key = json.substr(pos + 1, key_end - pos - 1);
            
            if (key == "__metadata__") {
                pos = json.find("}", key_end);
                continue;
            }

            size_t block_start = json.find("{", key_end);
            size_t block_end = json.find("}", block_start);
            if (block_start == std::string::npos || block_end == std::string::npos) break;
            
            std::string block = json.substr(block_start, block_end - block_start + 1);
            TensorDescriptor desc;
            
            size_t off_pos = block.find("data_offsets");
            if (off_pos != std::string::npos) {
                size_t arr_start = block.find("[", off_pos);
                size_t comma = block.find(",", arr_start);
                size_t arr_end = block.find("]", comma);
                if (arr_start != std::string::npos && comma != std::string::npos && arr_end != std::string::npos) {
                    desc.start_offset = std::stoull(block.substr(arr_start + 1, comma - arr_start - 1));
                    desc.end_offset = std::stoull(block.substr(comma + 1, arr_end - comma - 1));
                }
            }

            registry_[key] = desc;
            pos = block_end + 1;
        }
    }
};

} // namespace c_field
