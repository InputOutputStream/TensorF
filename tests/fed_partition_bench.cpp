// fed_partition_bench — run EVERY compiled-in partitioner backend on the same graph and device scores,
// and print one comparison table, so backends can be tested one by one.
//   ./fed_partition_bench --data Datasets/cora --scores 95,80,40,15
//   ./fed_partition_bench --data Datasets/ogbn-arxiv --scores 95,80,60,40,30,15 --only metis,scotch
//   ./fed_partition_bench --synthetic 200000,10 --scores 90,60,30       (planted-community random graph, no files)
// Metrics: edge cut and communication volume (lower is better), imbalance = max_i weight_i / target_i
// (1.00 = exactly the capacity-proportional target), wall time. Vertex weights = 1 + degree.
#include "net/Mobile/graph/FedGraphData.hpp"
#include "net/Mobile/graph/WeightedPartitioner.hpp"
#include <chrono>
#include <cstdlib>
#include <random>
#include <sstream>
using namespace tf::fed; using namespace tf::fed::graph;

static std::vector<std::string> split(const std::string& s, char c) { std::vector<std::string> v; std::stringstream ss(s); std::string t; while (std::getline(ss, t, c)) if (!t.empty()) v.push_back(t); return v; }

int main(int argc, char** argv) {
    TaskOptions o; std::vector<float> scores{95, 80, 40, 15}; std::vector<std::string> only; size_t sn = 0, sdeg = 8; int reps = 1; double eps = 0.03, hi = 70; int threads = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i]; auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--data") o.kv["data"] = next();
        else if (a == "--scores") { scores.clear(); for (auto& t : split(next(), ',')) scores.push_back(std::stof(t)); }
        else if (a == "--only") only = split(next(), ',');
        else if (a == "--synthetic") { auto p = split(next(), ','); sn = std::stoull(p[0]); if (p.size() > 1) sdeg = std::stoull(p[1]); }
        else if (a == "--reps") reps = std::atoi(next().c_str());
        else if (a == "--eps") eps = std::atof(next().c_str());
        else if (a == "--hi") hi = std::atof(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else { std::fprintf(stderr, "usage: %s [--data D | --synthetic N,deg] [--scores a,b,c] [--only metis,scotch,...] [--reps R] [--eps 0.03] [--hi 70] [--threads T]\n", argv[0]); return 2; }
    }
    std::unique_ptr<FedGraphData> d;
    try {
        if (sn) {   // planted-community graph: edges mostly inside one of 64 communities
            Graph<float> g(4, sn); std::mt19937 rng(1); const size_t Cm = 64; float f[4] = {1, 0, 0, 0};
            for (size_t i = 0; i < sn; i++) g.add_node(f, 4, int(i % Cm));
            std::uniform_int_distribution<size_t> any(0, sn - 1), mem(0, sn / Cm - 1);
            for (size_t u = 0; u < sn; u++) for (size_t k = 0; k < sdeg / 2; k++) { size_t v = (rng() % 10 < 8) ? (mem(rng) * Cm + u % Cm) % sn : any(rng); if (v != u) g.add_edge(static_cast<node_t>(u), static_cast<node_t>(v)); }
            g.build(); d = std::make_unique<FedGraphData>(std::move(g)); d->num_classes = Cm;
        } else d = std::make_unique<FedGraphData>(load_fed_graph(o));
    } catch (const std::exception& e) { std::fprintf(stderr, "cannot load graph: %s\n", e.what()); return 1; }
    const CSR<float>& adj = d->graph.adjacency(); const size_t K = scores.size();
    std::vector<DeviceCaps> dev(K); for (size_t i = 0; i < K; i++) { dev[i].id = "d" + std::to_string(i); dev[i].score = scores[i]; }
    CapacityPolicy pol; pol.hi = float(hi); auto frac = target_fractions(dev, pol); auto vw = degree_weights(adj);
    std::printf("graph: %zu nodes, %zu adjacency entries | %zu devices, targets:", d->graph.num_nodes(), adj.col_idx.size(), K);
    for (double f : frac) std::printf(" %.1f%%", 100 * f); std::printf("\n");
    std::printf("compiled-in backends:"); for (Backend b : available_backends()) std::printf(" %s", backend_name(b)); std::printf("\n\n");
    std::printf("%-10s %8s %10s %9s %10s   %s\n", "backend", "cut", "comm.vol", "imbalance", "time(ms)", "nodes per device");
    for (Backend b : available_backends()) {
        if (!only.empty() && std::find(only.begin(), only.end(), backend_name(b)) == only.end()) continue;
        try {
            double best_ms = 1e30; std::vector<int> part;
            for (int r = 0; r < reps; r++) { auto t = std::chrono::steady_clock::now(); part = partition_with(b, adj, vw, frac, eps, 42 + r, false, threads);
                best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count()); }
            auto s = evaluate(adj, part, vw, frac);
            std::printf("%-10s %7.2f%% %10zu %9.3f %10.1f   ", backend_name(b), 100 * s.cut_fraction, s.comm_volume, s.max_imbalance, best_ms);
            for (auto n : s.nodes) std::printf("%zu ", n); std::printf("\n");
        } catch (const std::exception& e) { std::printf("%-10s FAILED: %s\n", backend_name(b), e.what()); }
    }
    return 0;
}
