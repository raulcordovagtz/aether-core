#pragma once
#include <unordered_map>
#include <string>
#include <fstream>
#include <vector>
#include <cstdint>

namespace c_field {

inline std::unordered_map<uint32_t, std::string> load_binary_vocab(const std::string& path) {
    std::unordered_map<uint32_t, std::string> vocab;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return vocab;

    uint32_t total = 0;
    f.read(reinterpret_cast<char*>(&total), sizeof(uint32_t));
    vocab.reserve(total);

    for (uint32_t i = 0; i < total; ++i) {
        uint32_t id = 0, len = 0;
        f.read(reinterpret_cast<char*>(&id), sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(&len), sizeof(uint32_t));

        if (len > 0) {
            std::string word(len, '\0');
            f.read(&word[0], len);
            vocab[id] = word;
        } else {
            vocab[id] = "";
        }
    }
    return vocab;
}

}
