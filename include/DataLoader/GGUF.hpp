#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <map>

#include "../Types/types.hpp"

template<typename T>
class Matrix;

#ifndef __GGUF__H_
#define __GGUF__H_

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint32_t ggml_type;
    uint64_t offset;
};

// GGUF value types
enum GGUFValueType : uint32_t {
    UINT8   = 0,  INT8    = 1,
    UINT16  = 2,  INT16   = 3,
    UINT32  = 4,  INT32   = 5,
    FLOAT32 = 6,  BOOL    = 7,
    STRING  = 8,  ARRAY   = 9,
    UINT64  = 10, INT64   = 11,
    FLOAT64 = 12
};



// GGML quantization types in Phi/SmolLM GGUF files
enum GGMLType : uint32_t {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 8,
    Q6_K = 14,
    Q4_K = 12,
    Q5_K = 13,
};

// Add ggml type 6

class GGUF{
    public:
        // read_binary is now its OWN template method, callable with any primitive
        std::ifstream file;
        std::vector<TensorInfo> tensors;
        uint64_t data_start_offset = 0;

        template<typename T>
        T read_binary(std::ifstream& file) {
            T value;
            file.read(reinterpret_cast<char*>(&value), sizeof(T));
            return value;
        }

        // Q8_0 → FP8<E,M>
        template<int E, int M>
        std::vector<FP8<E,M>> dequant_q8_to_fp8(std::ifstream& file, size_t n_elem)
        {
            std::vector<FP8<E,M>> out(n_elem);
            size_t n_blocks = n_elem / 32;

            for (size_t b = 0; b < n_blocks; ++b) {
                float scale;
                file.read((char*)&scale, sizeof(float));

                for (int j = 0; j < 32; ++j) {
                    int8_t q;
                    file.read((char*)&q, 1);
                    // dequantize: multiply int8 by scale, re-encode into FP8
                    out[b*32 + j] = FP8<E,M>(scale * (float)q);
                }
            }
            return out;
        }

        // Q4_0 → FP4<E,M>
        template<unsigned E, unsigned M>
        std::vector<FP4<E,M>> dequant_q4_to_fp4(std::ifstream& file, size_t n_elem)
        {
            std::vector<FP4<E,M>> out(n_elem);
            size_t n_blocks = n_elem / 32;

            for (size_t b = 0; b < n_blocks; ++b) {
                // Q4_0 scale is float16
                uint16_t scale_bits;
                file.read((char*)&scale_bits, sizeof(uint16_t));
                float scale = fp16_to_float(scale_bits);

                // 32 values packed as 16 bytes, 2 nibbles per byte
                // values are offset by 8 (range -8..7 stored as 0..15)
                for (int j = 0; j < 16; ++j) {
                    uint8_t byte;
                    file.read((char*)&byte, 1);

                    int8_t lo = (int8_t)((byte & 0x0F) - 8);
                    int8_t hi = (int8_t)((byte >> 4)   - 8);

                    out[b*32 + j*2    ] = FP4<E,M>(scale * (float)lo);
                    out[b*32 + j*2 + 1] = FP4<E,M>(scale * (float)hi);
                }
            }
            return out;
        }

        std::string read_string(std::ifstream& file) {
            // Reads a uint64_t length prefix, always
            uint64_t length = read_binary<uint64_t>(file);
            std::string str(length, '\0');
            file.read(&str[0], length);
            return str;
        }

        void parse_gguf(const std::string& filepath) 
        {
            file.open(filepath, std::ios::binary);
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
                std::string key   = read_string(file);
                uint32_t val_type = read_binary<uint32_t>(file);

                if (key == "general.alignment" && val_type == 4)
                    alignment = read_binary<uint32_t>(file);
                else
                    skip_kv_value(file, val_type);  // skip everything else correctly
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
            this->tensors = tensors;
            this->data_start_offset = data_start_offset;

            std::cout << "\nBinary tensor data starts at byte offset: "
                    << data_start_offset << "\n";

            // 5. NEXT: use mmap() from data_start_offset
                // void* map = mmap(nullptr, file_size - data_start_offset,
                //     PROT_READ, MAP_PRIVATE, fd, data_start_offset);
        }

        static inline float fp16_to_float(uint16_t h) {
            uint32_t sign     = (h >> 15) & 0x1;
            uint32_t exp      = (h >> 10) & 0x1F;
            uint32_t mant     = h & 0x3FF;

            if (exp == 0)    // subnormal
                return (sign ? -1.f : 1.f) * std::ldexp((float)mant, -24);
            if (exp == 31)   // inf / nan
                return mant ? NAN : (sign ? -INFINITY : INFINITY);

            uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            float result;
            std::memcpy(&result, &f, 4);
            return result;
        }

        void skip_kv_value(std::ifstream& file, uint32_t val_type) {
            switch (val_type) {
                case 0: case 7:          read_binary<uint8_t>(file);   break;
                case 1:                  read_binary<int8_t>(file);    break;
                case 2:                  read_binary<uint16_t>(file);  break;
                case 3:                  read_binary<int16_t>(file);   break;
                case 4:                  read_binary<uint32_t>(file);  break;
                case 5:                  read_binary<int32_t>(file);   break;
                case 6:                  read_binary<float>(file);     break;
                case 8:                  read_string(file);            break;
                case 10:                 read_binary<uint64_t>(file);  break;
                case 11:                 read_binary<int64_t>(file);   break;
                case 12:                 read_binary<double>(file);    break;
                case 9: {                // ARRAY — recursive
                    uint32_t elem_type  = read_binary<uint32_t>(file);
                    uint64_t arr_len    = read_binary<uint64_t>(file);
                    for (uint64_t k = 0; k < arr_len; ++k)
                        skip_kv_value(file, elem_type);
                    break;
                }
                default:
                    throw std::runtime_error("Unknown GGUF value type: " 
                                            + std::to_string(val_type));
            }
        }

        template<typename T>
        Matrix<T> load_tensor(std::ifstream& file,
                            const TensorInfo& info,
                            uint64_t data_start)
        {
            size_t n = 1;
            shape_t shape;
            for (int d = info.dimensions.size()-1; d >= 0; --d) {
                shape.push_back(info.dimensions[d]);
                n *= info.dimensions[d];
            }

            file.seekg(data_start + info.offset);
            std::vector<T> data(n);

            if (info.ggml_type == 0) {          // F32
                std::vector<float> raw(n);
                file.read((char*)raw.data(), n * sizeof(float));
                for (size_t i = 0; i < n; ++i) data[i] = (T)raw[i];

            } else if (info.ggml_type == 1) {   // F16
                std::vector<uint16_t> raw(n);
                file.read((char*)raw.data(), n * sizeof(uint16_t));
                for (size_t i = 0; i < n; ++i)
                    data[i] = (T)fp16_to_float(raw[i]);

            } else if (info.ggml_type == 8) {   // Q8_0
                if constexpr (is_fp8<T>::value) {
                    // re-encode into your FP8 type directly
                    auto dq = dequant_q8_to_fp8<T::exp_bits, T::mant_bits>(file, n);
                    for (size_t i = 0; i < n; ++i) data[i] = dq[i];
                } else {
                    // fallback: dequant to float
                    size_t n_blocks = n / 32;
                    for (size_t b = 0; b < n_blocks; ++b) {
                        float scale;
                        file.read((char*)&scale, sizeof(float));
                        for (int j = 0; j < 32; ++j) {
                            int8_t q; file.read((char*)&q, 1);
                            data[b*32+j] = (T)(scale * (float)q);
                        }
                    }
                }

            } else if (info.ggml_type == 2) {   // Q4_0
                if constexpr (is_fp4<T>::value) {
                    auto dq = dequant_q4_to_fp4<T::exp_bits, T::mant_bits>(file, n);
                    for (size_t i = 0; i < n; ++i) data[i] = dq[i];
                } else {
                    size_t n_blocks = n / 32;
                    for (size_t b = 0; b < n_blocks; ++b) {
                        uint16_t scale_bits;
                        file.read((char*)&scale_bits, 2);
                        float scale = fp16_to_float(scale_bits);
                        for (int j = 0; j < 16; ++j) {
                            uint8_t byte; file.read((char*)&byte, 1);
                            data[b*32+j*2  ] = (T)(scale * (float)((int8_t)((byte&0x0F)-8)));
                            data[b*32+j*2+1] = (T)(scale * (float)((int8_t)((byte>>4) -8)));
                        }
                    }
                }
            } else {
                throw std::runtime_error("Unsupported GGML type: " 
                                        + std::to_string(info.ggml_type));
            }

            return Matrix<T>(data, shape);
        }
};

#endif // !__GGUF__H_


