#pragma once
// TfgFormat.hpp — "TFG1": compact binary graph-dataset format (little-endian, std only).
//
// Why: OGB ships gzipped CSV (ogbn-products = 61.9M edges, 2.4M x 100 features). Parsing that on
// every run is slow and memory hungry; scripts/download_datasets.sh converts it ONCE (tools/ogb2tfg)
// into this file, which loads by streaming straight into the Graph without a temporary copy.
//
//   header (64 B): "TFG1" | u32 version(1) | u64 N | u64 F | u32 C | u32 flags(0)
//                  | u64 E (directed edge entries, as given) | u64 n_train | u64 n_val | u64 n_test
//   body        : i32 labels[N]            (-1 = unlabeled)
//                 u32 train[n_train], val[n_val], test[n_test]     (official split, node ids)
//                 f32 features[N*F]
//                 u32 edges[2*E]           (src,dst pairs; the loader symmetrises + dedupes via Graph::build())
//   footer      : "END1"                   (detects truncated files)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace tf { namespace fed { namespace graph {

struct TfgHeader {
    uint64_t N = 0, F = 0, E = 0, n_train = 0, n_val = 0, n_test = 0; uint32_t C = 0;
    static constexpr size_t BYTES = 64;
    void pack(uint8_t* b) const {
        std::memset(b, 0, BYTES); std::memcpy(b, "TFG1", 4);
        auto p32 = [&](size_t o, uint32_t v) { for (int i = 0; i < 4; i++) b[o + i] = uint8_t(v >> (8 * i)); };
        auto p64 = [&](size_t o, uint64_t v) { for (int i = 0; i < 8; i++) b[o + i] = uint8_t(v >> (8 * i)); };
        p32(4, 1); p64(8, N); p64(16, F); p32(24, C); p32(28, 0); p64(32, E); p64(40, n_train); p64(48, n_val); p64(56, n_test);
    }
    static TfgHeader unpack(const uint8_t* b) {
        if (std::memcmp(b, "TFG1", 4) != 0) throw std::runtime_error("not a TFG1 file (bad magic)");
        auto g32 = [&](size_t o) { uint32_t v = 0; for (int i = 0; i < 4; i++) v |= uint32_t(b[o + i]) << (8 * i); return v; };
        auto g64 = [&](size_t o) { uint64_t v = 0; for (int i = 0; i < 8; i++) v |= uint64_t(b[o + i]) << (8 * i); return v; };
        if (g32(4) != 1) throw std::runtime_error("unsupported TFG version " + std::to_string(g32(4)));
        TfgHeader h; h.N = g64(8); h.F = g64(16); h.C = g32(24); h.E = g64(32); h.n_train = g64(40); h.n_val = g64(48); h.n_test = g64(56);
        if (h.N == 0 || h.F == 0 || h.C == 0 || h.N > 0xFFFFFFFFull) throw std::runtime_error("corrupt TFG header");
        return h;
    }
};

inline bool host_le() { const uint16_t x = 1; uint8_t b; std::memcpy(&b, &x, 1); return b == 1; }

inline TfgHeader tfg_read_header(const std::string& path) {
    if (!host_le()) throw std::runtime_error("TFG1 requires a little-endian host");
    FILE* f = std::fopen(path.c_str(), "rb"); if (!f) throw std::runtime_error("cannot open " + path);
    uint8_t b[TfgHeader::BYTES]; size_t n = std::fread(b, 1, sizeof b, f); std::fclose(f);
    if (n != sizeof b) throw std::runtime_error("truncated TFG header: " + path);
    return TfgHeader::unpack(b);
}

}}} // namespace tf::fed::graph
