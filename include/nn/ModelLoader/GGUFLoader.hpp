#pragma once
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <iostream>
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor.hpp"
#include "DataLoader/GGUF.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// GGUFLoader
//
// Weight shape contract
// ─────────────────────────────────────────────────────────────────────────────
// GGUF stores dimensions with the fastest-varying (last PyTorch) dim first.
// PyTorch weight [out, in]  →  GGUF info.dimensions = [in, out]
// load_tensor() reverses    →  Matrix shape {out, in}
//
// New Linear stores weight as {out, in} and computes x @ weight.T
// Therefore: load_raw() already delivers the correct {out, in} shape.
//   • No .transpose() is ever needed for a Linear weight.
//   • Embeddings [vocab, D] → load_raw → {vocab, D} — no transpose.
//   • Fused QKV [3D, D]    → load_raw → {3D, D}    — slice_row per head.
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
class GGUFLoader {
    protected:
        GGUF gguf;
        std::map<std::string, size_t> tensor_index;

        void build_index() {
            tensor_index.clear();
            for (size_t i = 0; i < gguf.tensors.size(); i++)
                tensor_index[gguf.tensors[i].name] = i;
        }

        // ── raw loading ──────────────────────────────────────────────────────
        // Returns Matrix with shape as delivered by load_tensor() (reversed
        // from GGUF info.dimensions), which equals the original PyTorch shape.

        Matrix<T> load_raw(const std::string& name, bool required = true) {
            auto it = tensor_index.find(name);
            if (it == tensor_index.end()) {
                if (required)
                    std::cerr << "[GGUFLoader] MISSING: " << name << "\n";
                return Matrix<T>();
            }
            const TensorInfo& info = gguf.tensors[it->second];
            std::cout << "[GGUFLoader] " << name << "  [";
            for (size_t d = 0; d < info.dimensions.size(); d++) {
                std::cout << info.dimensions[d];
                if (d + 1 < info.dimensions.size()) std::cout << "x";
            }
            std::cout << "]  ggml_type=" << info.ggml_type << "\n";
            return gguf.load_tensor<T>(gguf.file, info, gguf.data_start_offset);
        }

        // ── copy helper ──────────────────────────────────────────────────────

        void copy_into(Tensor_t<T> dst, const Matrix<T>& src,
                       const std::string& debug_name) {
            if (src.get_size() == 0) return;
            if (src.get_size() != dst->val.get_size()) {
                std::cerr << "[GGUFLoader] Size mismatch '" << debug_name
                          << "': GGUF=" << src.get_size()
                          << " model=" << dst->val.get_size() << " → skipped\n";
                return;
            }
            dst->val.copy_from(src);
            dst->shape = src.shape;
        }

        // ── lifecycle ────────────────────────────────────────────────────────

        void open(const std::string& path) {
            std::cout << "[GGUFLoader] Parsing: " << path << "\n";
            gguf.parse_gguf(path);
            build_index();
        }

    public:
        std::vector<std::string> get_metadata_array(const std::string& key) const {
            return gguf.get_array(key);
        }

        std::string get_metadata_string(const std::string& key) const {
            return gguf.get_string(key);
        }

        void inspect(const std::string& path) {
            gguf.parse_gguf(path);
            std::cout << "\n=== GGUF Inventory: " << path
                      << "  (" << gguf.tensors.size() << " tensors) ===\n";
            for (const auto& t : gguf.tensors) {
                std::cout << "  " << t.name << "  [";
                for (size_t d = 0; d < t.dimensions.size(); d++) {
                    std::cout << t.dimensions[d];
                    if (d + 1 < t.dimensions.size()) std::cout << "x";
                }
                std::cout << "]  type=" << t.ggml_type << "\n";
            }
            std::cout << "===\n\n";
        }

        template<typename ModelT>
        void report_params(const ModelT& model) const {
            size_t n = 0;
            for (auto& p : model.parameters()) n += p->val.get_size();
            std::cout << "[GGUFLoader] Total parameters: " << n << "\n";
        }

        virtual ~GGUFLoader() = default;
};