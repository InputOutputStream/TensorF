#ifndef __QUANTIZER_HPP__
#define __QUANTIZER_HPP__

#include "../Types/types.hpp"
#include "../DataStructures/Matrix.hpp"
#include "../DataStructures/Tensor.hpp"
#include "../Modules/Module.hpp"

#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <type_traits>

// ─────────────────────────────────────────────────────────────────────────────
// quant:: — block-wise quantization between a full-precision compute type
// (float/double — whatever T your Trainer actually trains in) and a narrow
// software float storage type (FP8<E,M> / FP4<E,M> from fp8.hpp / fp4.hpp).
//
// Convention matches this project's own GGUF Q4_0/Q8_0 *reader* (GGUF.hpp):
// every contiguous run of `group` elements (default 32, same as GGUF) shares
// one float absmax scale.
//
//     encode:  code  = QT(value / scale)
//     decode:  value = float(code) * scale
//
// scale is picked so the block's largest-magnitude value lands exactly on
// QT's largest finite representable magnitude (symmetric absmax — the same
// scheme GPTQ / AWQ / GGUF / the MX formats all use; QT itself supplies the
// mantissa/exponent split instead of a fixed-point integer).
//
// Why this file exists instead of just casting Matrix<float> -> Matrix<FP4>:
// a plain `std::vector<FP4<E,M>>` is *not* 4-bit-per-element — each FP4 is
// still its own 1-byte struct. Two things make FP4 genuinely smaller than
// FP8 here:
//   1. The per-block scale (this is what gives a 4-bit float usable dynamic
//      range at all — raw E2M1 alone covers maybe [0.5, 3] in magnitude).
//   2. Real bit-packing on save (two 4-bit codes per byte) — done in
//      save_quantized()/load_quantized() below, not in the in-memory
//      QuantTensor (which stays one struct per element for easy arithmetic).
// ─────────────────────────────────────────────────────────────────────────────

namespace quant {

// Largest finite magnitude representable by QT (FP4<E,M> or FP8<E,M>).
// Both types reserve the top exponent field for inf/nan and decode finite
// values as (1 + mantissa/mant_scale) * 2^(exp_field - bias), so the largest
// finite exponent field is (max_exp - 1) and largest mantissa field is
// (mant_scale - 1) — this holds for both your FP4 and FP8 encode/decode.
template<typename QT>
inline float qmax_value() {
    return (1.0f + float(QT::mant_scale - 1) / float(QT::mant_scale))
         * std::pow(2.0f, float((QT::max_exp - 1) - QT::bias));
}

template<typename QT>
struct QuantTensor {
    std::vector<QT>    codes;     // one QT value per original element
    std::vector<float> scales;    // one scale per block of `group` elements
    shape_t            shape;     // original tensor shape (for decode reshape)
    size_t             group = 32;
};

// ── Encode: Matrix<SrcT> (e.g. float) -> QuantTensor<QT> (e.g. fp8_e4m3) ────
template<typename QT, typename SrcT>
QuantTensor<QT> quantize(const Matrix<SrcT>& m, size_t group = 32) {
    QuantTensor<QT> q;
    q.shape = m.shape;
    q.group = group;

    size_t n = m.data.size();
    size_t n_blocks = (n + group - 1) / group;
    q.codes.resize(n);
    q.scales.resize(n_blocks);

    const float qmax = qmax_value<QT>();

    for (size_t b = 0; b < n_blocks; ++b) {
        size_t start = b * group;
        size_t end   = std::min(start + group, n);

        float amax = 0.0f;
        for (size_t i = start; i < end; ++i)
            amax = std::max(amax, std::fabs(static_cast<float>(m.data[i])));

        float scale = (amax > 0.0f) ? (amax / qmax) : 1.0f;
        q.scales[b] = scale;

        for (size_t i = start; i < end; ++i)
            q.codes[i] = QT(static_cast<float>(m.data[i]) / scale);
    }
    return q;
}

// ── Decode: QuantTensor<QT> -> Matrix<DstT> (e.g. back into float/T) ────────
template<typename QT, typename DstT = float>
Matrix<DstT> dequantize(const QuantTensor<QT>& q) {
    size_t n = q.codes.size();
    std::vector<DstT> out(n);

    size_t n_blocks = q.scales.size();
    for (size_t b = 0; b < n_blocks; ++b) {
        size_t start = b * q.group;
        size_t end   = std::min(start + q.group, n);
        float scale  = q.scales[b];
        for (size_t i = start; i < end; ++i)
            out[i] = static_cast<DstT>(static_cast<float>(q.codes[i]) * scale);
    }
    return Matrix<DstT>(out, q.shape);
}

// ── Binary checkpoint format ─────────────────────────────────────────────────
//
//  "QNTF" | version:u32 | format_tag:u8 (0=dense 1B/elem, 1=packed FP4 nibbles)
//  | group:u64 | n_tensors:u64
//  per tensor:
//    rank:u32 | dims:i64×rank | n_elem:u64 | n_blocks:u64
//    | scales:f32×n_blocks | codes (packed per format_tag)

namespace detail {
    inline constexpr char     QNTF_MAGIC[4] = {'Q', 'N', 'T', 'F'};
    inline constexpr uint32_t QNTF_VERSION  = 1;

    template<typename QT>
    void write_codes(std::ofstream& f, const std::vector<QT>& codes) {
        if constexpr (is_fp4<QT>::value) {
            size_t n = codes.size();
            std::vector<uint8_t> packed((n + 1) / 2, 0);
            for (size_t i = 0; i < n; ++i) {
                uint8_t nib = codes[i].bits & 0x0F;
                if (i % 2 == 0) packed[i / 2] |= nib;
                else            packed[i / 2] |= (nib << 4);
            }
            f.write(reinterpret_cast<const char*>(packed.data()), packed.size());
        } else {
            f.write(reinterpret_cast<const char*>(codes.data()), codes.size() * sizeof(QT));
        }
    }

    template<typename QT>
    void read_codes(std::ifstream& f, std::vector<QT>& codes, size_t n_elem) {
        codes.resize(n_elem);
        if constexpr (is_fp4<QT>::value) {
            std::vector<uint8_t> packed((n_elem + 1) / 2);
            f.read(reinterpret_cast<char*>(packed.data()), packed.size());
            for (size_t i = 0; i < n_elem; ++i) {
                uint8_t byte = packed[i / 2];
                uint8_t nib  = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                codes[i].bits = nib;
            }
        } else {
            f.read(reinterpret_cast<char*>(codes.data()), n_elem * sizeof(QT));
        }
        if (!f) throw std::runtime_error("Quantizer: unexpected EOF reading codes");
    }
} // namespace detail

/// Quantize and save every tensor in `params` (e.g. model.parameters()).
template<typename QT, typename SrcT>
void save_quantized(const std::vector<Tensor_t<SrcT>>& params,
                     const std::string& path, size_t group = 32) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open())
        throw std::runtime_error("Quantizer::save_quantized: cannot open " + path);

    f.write(detail::QNTF_MAGIC, 4);
    uint32_t version = detail::QNTF_VERSION;
    f.write(reinterpret_cast<const char*>(&version), sizeof(version));
    uint8_t format_tag = is_fp4<QT>::value ? 1 : 0;
    f.write(reinterpret_cast<const char*>(&format_tag), sizeof(format_tag));
    uint64_t group64 = group;
    f.write(reinterpret_cast<const char*>(&group64), sizeof(group64));
    uint64_t n_tensors = params.size();
    f.write(reinterpret_cast<const char*>(&n_tensors), sizeof(n_tensors));

    size_t orig_bytes = 0, packed_bytes = 0;

    for (auto& p : params) {
        auto qt = quantize<QT, SrcT>(p->val, group);

        uint32_t rank = static_cast<uint32_t>(qt.shape.size());
        f.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
        for (auto d : qt.shape) {
            int64_t dd = static_cast<int64_t>(d);
            f.write(reinterpret_cast<const char*>(&dd), sizeof(dd));
        }

        uint64_t n_elem   = qt.codes.size();
        uint64_t n_blocks = qt.scales.size();
        f.write(reinterpret_cast<const char*>(&n_elem),   sizeof(n_elem));
        f.write(reinterpret_cast<const char*>(&n_blocks), sizeof(n_blocks));
        f.write(reinterpret_cast<const char*>(qt.scales.data()), n_blocks * sizeof(float));
        detail::write_codes<QT>(f, qt.codes);

        orig_bytes   += n_elem * sizeof(SrcT);
        packed_bytes += (is_fp4<QT>::value ? (n_elem + 1) / 2 : n_elem * sizeof(QT))
                      + n_blocks * sizeof(float);
    }

    if (!f) throw std::runtime_error("Quantizer::save_quantized: write error on " + path);

    std::cout << "[Quantizer] Saved " << params.size() << " tensors to " << path
              << "  (" << (orig_bytes / (1024.0 * 1024.0)) << " MB -> "
              << (packed_bytes / (1024.0 * 1024.0)) << " MB)\n";
}

/// Load a quantized checkpoint, dequantizing every tensor back to DstT
/// (typically your model's compute type T).
template<typename QT, typename DstT = float>
std::vector<Matrix<DstT>> load_quantized(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Quantizer::load_quantized: cannot open " + path);

    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, detail::QNTF_MAGIC, 4) != 0)
        throw std::runtime_error("Quantizer::load_quantized: bad magic in " + path);

    uint32_t version;    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != detail::QNTF_VERSION)
        throw std::runtime_error("Quantizer::load_quantized: unsupported version "
            + std::to_string(version) + " in " + path);
    uint8_t  format_tag; f.read(reinterpret_cast<char*>(&format_tag), sizeof(format_tag));
    uint8_t  expected    = is_fp4<QT>::value ? 1 : 0;
    if (format_tag != expected)
        throw std::runtime_error(
            "Quantizer::load_quantized: format mismatch in " + path +
            " — saved with a different QT (fp4 vs fp8) than requested for loading");

    uint64_t group;     f.read(reinterpret_cast<char*>(&group), sizeof(group));
    uint64_t n_tensors; f.read(reinterpret_cast<char*>(&n_tensors), sizeof(n_tensors));

    std::vector<Matrix<DstT>> out;
    out.reserve(n_tensors);

    for (uint64_t t = 0; t < n_tensors; ++t) {
        uint32_t rank; f.read(reinterpret_cast<char*>(&rank), sizeof(rank));
        shape_t shape(rank);
        for (uint32_t d = 0; d < rank; ++d) {
            int64_t dd; f.read(reinterpret_cast<char*>(&dd), sizeof(dd));
            shape[d] = static_cast<size_t>(dd);
        }

        uint64_t n_elem, n_blocks;
        f.read(reinterpret_cast<char*>(&n_elem),   sizeof(n_elem));
        f.read(reinterpret_cast<char*>(&n_blocks), sizeof(n_blocks));

        QuantTensor<QT> qt;
        qt.shape = shape;
        qt.group = static_cast<size_t>(group);
        qt.scales.resize(n_blocks);
        f.read(reinterpret_cast<char*>(qt.scales.data()), n_blocks * sizeof(float));
        detail::read_codes<QT>(f, qt.codes, static_cast<size_t>(n_elem));

        if (!f) throw std::runtime_error("Quantizer::load_quantized: truncated file " + path);

        out.push_back(dequantize<QT, DstT>(qt));
    }

    std::cout << "[Quantizer] Loaded " << out.size() << " tensors from " << path << "\n";
    return out;
}

// ── Module<T> bridges — no edits to Module.hpp required ──────────────────
//
// The two calls client.cpp / server.cpp actually use:
//
//   quant::save_module<fp8_e4m3>(student, path)       // PTQ checkpoint, ~4x smaller on disk
//   quant::load_module_into<fp8_e4m3>(student, path)  // dequantizes straight back into the
//                                                      // live model — training/distillation
//                                                      // can continue immediately, no separate
//                                                      // "quantized model" type to manage.
//
// Works for ANY Module<T> — GPT<T>, GPTLoRA<T>, even a bare LoRALinear<T> —
// because it only touches the public parameters()/Tensor_t<T> surface, never
// the model's internals.

template<typename QT, typename T>
void save_module(Module<T>& m, const std::string& path, size_t group = 32) {
    save_quantized<QT, T>(m.parameters(), path, group);
}

template<typename QT, typename T>
void load_module_into(Module<T>& m, const std::string& path) {
    auto mats   = load_quantized<QT, T>(path);   // dequantized Matrix<T>, one per tensor
    auto params = m.parameters();
    if (mats.size() != params.size())
        throw std::runtime_error(
            "Quantizer::load_module_into: tensor count mismatch — checkpoint has "
            + std::to_string(mats.size()) + " tensors, model has "
            + std::to_string(params.size()) + " (wrong checkpoint for this model?)");
    for (size_t i = 0; i < params.size(); ++i)
        params[i]->val.copy_from(mats[i]);
}

} // namespace quant

#endif // __QUANTIZER_HPP__