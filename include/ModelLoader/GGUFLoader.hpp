
#pragma once
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <iostream>

#include "../DataStructures/Matrix.hpp"
#include "../DataStructures/Tensor.hpp"
#include "../DataLoader/GGUF.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// GGUFLoader
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
class GGUFLoader {
    protected:
        GGUF gguf;

        // ── index ────────────────────────────────────────────────────────────────

        std::map<std::string, size_t> tensor_index;

        void build_index() {
            tensor_index.clear();
            for (size_t i = 0; i < gguf.tensors.size(); i++)
                tensor_index[gguf.tensors[i].name] = i;
        }

        // ── raw loading ──────────────────────────────────────────────────────────

        /**
         * Load a tensor by name, dequantising to T automatically.
         * If required=true  and the name is absent → logs an error, returns empty.
         * If required=false and the name is absent → returns empty silently.
         */
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

        /**
         * load_raw + transpose (for 2-D weight matrices stored [out, in] in GGUF).
         * 1-D vectors are returned as-is.
         */
        Matrix<T> load_raw_t(const std::string& name, bool required = true) {
            Matrix<T> w = load_raw(name, required);
            if (w.get_size() == 0) return w;
            if (w.shape.size() == 2) return w.transpose();
            return w; // 1-D: no transpose
        }

        // ── copy helpers ─────────────────────────────────────────────────────────

        /**
         * Copy src into dst->val with a size guard.
         * Logs a warning and skips if sizes do not match.
         */
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

        // ── lifecycle ────────────────────────────────────────────────────────────

        /**
         * Open and parse the GGUF file; build the name→index map.
         * Call this first from every load_model() implementation.
         */
        void open(const std::string& path) {
            std::cout << "[GGUFLoader] Parsing: " << path << "\n";
            gguf.parse_gguf(path);
            build_index();
        }

    public:

        // ── utility ──────────────────────────────────────────────────────────────

        /**
         * Print every tensor in the file: name, shape, ggml_type.
         * Useful for mapping a new model architecture by inspection.
         */
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

        /**
         * Count and print total parameter count for any model whose Module
         * exposes a parameters() method.
         */
        template<typename ModelT>
        void report_params(const ModelT& model) const {
            size_t n = 0;
            for (auto& p : model.parameters()) n += p->val.get_size();
            std::cout << "[GGUFLoader] Total parameters: " << n << "\n";
        }

        virtual ~GGUFLoader() = default;
};


