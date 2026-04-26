#ifndef __SERIALIZE_HPP
#define __SERIALIZE_HPP

#include "../Types/types.hpp"

#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdint>

/**
 * Binary Model Serializer
 *
 * File format layout (little-endian):
 * ┌────────────────────────────────────────────┐
 * │  Magic bytes: "TNSF"  (4 bytes)            │
 * │  Version:             (uint32_t)           │
 * │  Number of tensors:   (uint64_t)           │
 * ├────────────────────────────────────────────┤
 * │  For each tensor:                          │
 * │    Rank (number of dims): (uint32_t)       │
 * │    Each dim size:         (int64_t × rank) │
 * │    Number of elements:    (uint64_t)       │
 * │    Raw data:              (T × n_elements) │
 * └────────────────────────────────────────────┘
 */

static constexpr char     MAGIC[4]   = {'T','N','S','F'};
static constexpr uint32_t VERSION     = 1;

template<typename T>
class Serializer {

private:

    // -----------------------------------------------------------------------
    // Write helpers
    // -----------------------------------------------------------------------
    template<typename V>
    void write_val(std::ofstream& f, const V& v) {
        f.write(reinterpret_cast<const char*>(&v), sizeof(V));
    }

    void write_bytes(std::ofstream& f, const void* ptr, size_t n) {
        f.write(reinterpret_cast<const char*>(ptr), n);
    }

    // -----------------------------------------------------------------------
    // Read helpers
    // -----------------------------------------------------------------------
    template<typename V>
    V read_val(std::ifstream& f) {
        V v;
        f.read(reinterpret_cast<char*>(&v), sizeof(V));
        if (!f) throw std::runtime_error("Unexpected end of file while reading.");
        return v;
    }

    void read_bytes(std::ifstream& f, void* ptr, size_t n) {
        f.read(reinterpret_cast<char*>(ptr), n);
        if (!f) throw std::runtime_error("Unexpected end of file while reading bytes.");
    }

public:

    // -----------------------------------------------------------------------
    // Save  — serializes all tensors in the model to a binary file
    // -----------------------------------------------------------------------
    void save(const std::vector<Tensor_t<T>>& model, const std::string& path) {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file.is_open())
            throw std::runtime_error("Could not open file for writing: " + path);

        // Header
        write_bytes(file, MAGIC, 4);
        write_val<uint32_t>(file, VERSION);
        write_val<uint64_t>(file, static_cast<uint64_t>(model.size()));

        // Tensors
        for (const auto& tensor : model) {
            const shape_t& shape = tensor->data.shape;

            uint32_t rank = static_cast<uint32_t>(shape.size());
            write_val<uint32_t>(file, rank);

            for (auto dim : shape)
                write_val<int64_t>(file, static_cast<int64_t>(dim));

            uint64_t n_elem = static_cast<uint64_t>(tensor->data.data.size());
            write_val<uint64_t>(file, n_elem);

            write_bytes(file, tensor->data.data.data(), n_elem * sizeof(T));
        }

        file.close();
        if (!file)
            throw std::runtime_error("Error occurred while writing file: " + path);

        std::cout << "[Serializer] Saved " << model.size()
                  << " tensors to " << path << "\n";
    }

    // -----------------------------------------------------------------------
    // Load  — deserializes tensors from a binary file
    // -----------------------------------------------------------------------
    std::vector<Tensor_t<T>> load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Could not open file for reading: " + path);

        // Validate magic bytes
        char magic[4];
        read_bytes(file, magic, 4);
        if (std::memcmp(magic, MAGIC, 4) != 0)
            throw std::runtime_error("Invalid file format: bad magic bytes in " + path);

        // Validate version
        uint32_t version = read_val<uint32_t>(file);
        if (version != VERSION)
            throw std::runtime_error("Unsupported serializer version: " + std::to_string(version));

        uint64_t n_tensors = read_val<uint64_t>(file);
        std::vector<Tensor_t<T>> model;
        model.reserve(static_cast<size_t>(n_tensors));

        for (uint64_t i = 0; i < n_tensors; i++) {
            uint32_t rank = read_val<uint32_t>(file);

            shape_t shape(rank);
            for (uint32_t d = 0; d < rank; d++)
                shape[d] = static_cast<long>(read_val<int64_t>(file));

            uint64_t n_elem = read_val<uint64_t>(file);
            std::vector<T> raw(static_cast<size_t>(n_elem));
            read_bytes(file, raw.data(), n_elem * sizeof(T));

            Matrix<T> mat(raw, shape);
            model.push_back(std::make_shared<Tensor<T>>(mat));
        }

        std::cout << "[Serializer] Loaded " << model.size()
                  << " tensors from " << path << "\n";
        return model;
    }
};

#endif