#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <map>

#ifndef __GGUF__H_
#define __GGUF__H_

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint32_t ggml_type;
    uint64_t offset;
};

class GGUF{
    public:
        // read_binary is now its OWN template method, callable with any primitive
        template<typename T>
        T read_binary(std::ifstream& file) {
            T value;
            file.read(reinterpret_cast<char*>(&value), sizeof(T));
            return value;
        }

        std::string read_string(std::ifstream& file) {
            // Now correctly reads a uint64_t length prefix, always
            uint64_t length = read_binary<uint64_t>(file);
            std::string str(length, '\0');
            file.read(&str[0], length);
            return str;
        }

        void parse_gguf(const std::string& filepath) {
            std::ifstream file(filepath, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open GGUF file: " << filepath << "\n";
                return;
            }

            // 1. HEADER — types are fixed by the GGUF spec
            char magic[4];
            file.read(magic, 4);
            if (std::string(magic, 4) != "GGUF") {
                std::cerr << "Invalid magic number. Not a GGUF file.\n";
                return;
            }

            uint32_t version      = read_binary<uint32_t>(file);
            uint64_t tensor_count = read_binary<uint64_t>(file);
            uint64_t kv_count     = read_binary<uint64_t>(file);

            std::cout << "GGUF Version: " << version << "\n";
            std::cout << "Tensors: " << tensor_count << " | KV Pairs: " << kv_count << "\n\n";

            // 2. METADATA KEY-VALUE PAIRS
            uint32_t alignment = 32;

            for (uint64_t i = 0; i < kv_count; ++i) {
                std::string key  = read_string(file);
                uint32_t val_type = read_binary<uint32_t>(file);

                if (val_type == 4) {       // UINT32
                    uint32_t val = read_binary<uint32_t>(file);
                    if (key == "general.alignment") alignment = val;

                } else if (val_type == 8) { // STRING
                    std::string val = read_string(file);

                } else {
                    std::cerr << "Warning: Unhandled metadata type "
                            << val_type << " for key: " << key << "\n";
                    return;
                }
            }

            // 3. TENSOR INFO
            std::vector<TensorInfo> tensors(tensor_count);
            for (uint64_t i = 0; i < tensor_count; ++i) {
                tensors[i].name = read_string(file);

                uint32_t n_dims = read_binary<uint32_t>(file);
                tensors[i].dimensions.resize(n_dims);

                for (uint32_t d = 0; d < n_dims; ++d) {
                    tensors[i].dimensions[d] = read_binary<uint64_t>(file);
                }

                tensors[i].ggml_type = read_binary<uint32_t>(file);
                tensors[i].offset    = read_binary<uint64_t>(file);
            }

            // 4. ALIGNMENT PADDING
            uint64_t current_pos     = file.tellg();
            uint64_t padding         = (alignment - (current_pos % alignment)) % alignment;
            uint64_t data_start_offset = current_pos + padding;

            std::cout << "\nBinary tensor data starts at byte offset: "
                    << data_start_offset << "\n";

            // 5. NEXT: use mmap() from data_start_offset
    }
};

#endif // !__GGUF__H_