#pragma once
#include "PartitionerPrelude.hpp"   // must come before DataStructures/Graph.hpp (KaMinPar name clash)
// FedGraphData.hpp — one loader for every dataset the distributed layer can use.
//
//   data=Datasets/cora          LINQS  (cora.content + cora.cites)     split: Planetoid-style (per_class/n_val/n_test)
//   data=Datasets/ogbn-arxiv    TFG1   (graph.tfg, from download_datasets.sh)  split: the OFFICIAL OGB split
//   data=path/to/file.tfg       TFG1 directly
// Options: split=official|planetoid (TFG defaults to official), row_normalize=0|1 (TFG default 0),
//          per_class n_val n_test seed (planetoid split).
// Unlabeled nodes (-1 in the file) are mapped to class 0 and never appear in any split mask.

#include "TfgFormat.hpp"
#include "../FedTask.hpp"

#include "DataStructures/Graph.hpp"
#include "GraphLoader.hpp"

#include <algorithm>
#include <memory>
#include <sys/stat.h>
#include <vector>

namespace tf { namespace fed { namespace graph {

inline void linqs_paths(const TaskOptions& o, std::string& content, std::string& cites) {
    std::string dir = o.str("data", "Datasets/cora");
    while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
    std::string base = dir.substr(dir.find_last_of('/') == std::string::npos ? 0 : dir.find_last_of('/') + 1);
    content = o.str("content", dir + "/" + base + ".content");
    cites   = o.str("cites",   dir + "/" + base + ".cites");
}

struct FedGraphData {
    Graph<float> graph; size_t num_classes = 0; Split split; bool official_split = false; std::string source;
    explicit FedGraphData(Graph<float>&& g) : graph(std::move(g)) {}
};

inline bool path_is_file(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode); }

/// Locate the dataset described by the options: returns {"tfg", file} or {"linqs", dir}.
inline std::pair<std::string, std::string> locate_dataset(const TaskOptions& o) {
    std::string d = o.str("data", "Datasets/cora");
    while (d.size() > 1 && d.back() == '/') d.pop_back();
    if (d.size() > 4 && d.substr(d.size() - 4) == ".tfg") return {"tfg", d};
    if (path_is_file(d + "/graph.tfg")) return {"tfg", d + "/graph.tfg"};
    return {"linqs", d};
}

/// Dimensions without loading the graph (TFG: header only).
inline void dataset_dims(const TaskOptions& o, size_t& F, size_t& C, size_t* N = nullptr) {
    auto loc = locate_dataset(o);
    if (loc.first == "tfg") { auto h = tfg_read_header(loc.second); F = h.F; C = h.C; if (N) *N = h.N; return; }
    std::string cp, ep; linqs_paths(o, cp, ep);
    auto ds = load_linqs<float>(cp, ep, true); F = ds.graph.feature_dim(); C = ds.num_classes(); if (N) *N = ds.graph.num_nodes();
}

inline FedGraphData load_tfg(const std::string& path, bool row_normalize = false) {
    TfgHeader h = tfg_read_header(path);
    FILE* f = std::fopen(path.c_str(), "rb"); if (!f) throw std::runtime_error("cannot open " + path);
    struct Closer { FILE* f; ~Closer() { if (f) std::fclose(f); } } closer{f};
    std::fseek(f, long(TfgHeader::BYTES), SEEK_SET);
    auto rd = [&](void* dst, size_t bytes) { if (std::fread(dst, 1, bytes, f) != bytes) throw std::runtime_error("truncated TFG file: " + path); };
    const size_t N = h.N, F = h.F;
    std::vector<int32_t> lab(N); rd(lab.data(), N * 4);
    std::vector<uint32_t> tr(h.n_train), va(h.n_val), te(h.n_test);
    if (!tr.empty()) rd(tr.data(), tr.size() * 4); if (!va.empty()) rd(va.data(), va.size() * 4); if (!te.empty()) rd(te.data(), te.size() * 4);
    Graph<float> g(F, N);
    std::vector<float> row(F * 4096);                                     // features streamed in blocks of 4096 rows
    for (size_t i = 0; i < N; i += 4096) {
        size_t n = std::min<size_t>(4096, N - i); rd(row.data(), n * F * 4);
        for (size_t r = 0; r < n; r++) {
            float* x = &row[r * F];
            if (row_normalize) { double s = 0; for (size_t k = 0; k < F; k++) s += x[k]; if (s != 0) for (size_t k = 0; k < F; k++) x[k] = float(x[k] / s); }
            g.add_node(x, F, lab[i + r] < 0 ? 0 : lab[i + r]);
        }
    }
    std::vector<uint32_t> eb(2 * 1000000);                                // edges streamed in blocks of 1M pairs
    for (uint64_t i = 0; i < h.E; i += 1000000) {
        size_t n = size_t(std::min<uint64_t>(1000000, h.E - i)); rd(eb.data(), n * 8);
        for (size_t k = 0; k < n; k++) g.add_edge(static_cast<node_t>(eb[2 * k]), static_cast<node_t>(eb[2 * k + 1]));
    }
    char foot[4]; rd(foot, 4); if (std::memcmp(foot, "END1", 4) != 0) throw std::runtime_error("TFG footer missing (corrupt file): " + path);
    g.build();
    FedGraphData d(std::move(g)); d.num_classes = h.C; d.official_split = true; d.source = path;
    d.split.train.assign(N, 0); d.split.val.assign(N, 0); d.split.test.assign(N, 0);
    auto mark = [&](const std::vector<uint32_t>& idx, std::vector<char>& m, size_t& cnt) { for (auto v : idx) if (v < N && lab[v] >= 0 && !m[v]) { m[v] = 1; cnt++; } };
    mark(tr, d.split.train, d.split.n_train); mark(va, d.split.val, d.split.n_val); mark(te, d.split.test, d.split.n_test);
    return d;
}

inline FedGraphData load_fed_graph(const TaskOptions& o) {
    auto loc = locate_dataset(o);
    if (loc.first == "tfg") {
        FedGraphData d = load_tfg(loc.second, o.b("row_normalize", false));
        if (o.str("split", "official") == "planetoid") {
            d.split = make_split(d.graph.labels(), d.num_classes, size_t(o.i("per_class", 20)), size_t(o.i("n_val", 500)), size_t(o.i("n_test", 1000)), unsigned(o.i("seed", 42)));
            d.official_split = false;
        }
        return d;
    }
    std::string cp, ep; linqs_paths(o, cp, ep);
    GraphDataset<float> ds = load_linqs<float>(cp, ep, o.b("row_normalize", true));
    size_t C = ds.num_classes();
    FedGraphData d(std::move(ds.graph)); d.num_classes = C; d.source = cp;
    d.split = make_split(d.graph.labels(), C, size_t(o.i("per_class", 20)), size_t(o.i("n_val", 500)), size_t(o.i("n_test", 1000)), unsigned(o.i("seed", 42)));
    return d;
}

}}} // namespace tf::fed::graph
