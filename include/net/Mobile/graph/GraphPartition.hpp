#pragma once
#include "PartitionerPrelude.hpp"   // must come before DataStructures/Graph.hpp (KaMinPar name clash)
// GraphPartition.hpp — split one graph across devices for federated graph learning.
// Independent of the transport. Each device keeps the subgraph induced by its nodes;
// edges between devices are LOST locally (that information gap is inherent to
// federated GNNs; cut_fraction() tells you how much you are losing).
//
//   Random : nodes assigned uniformly            -> worst case, high edge cut, ~IID labels
//   Bfs    : balanced multi-source region growing -> community-like, low edge cut, non-IID labels

#include "DataStructures/Graph.hpp"

#include <algorithm>
#include <deque>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace tf { namespace fed { namespace graph {

enum class PartitionMode { Random, Bfs };

inline PartitionMode parse_partition(const std::string& s) {
    if (s == "random") return PartitionMode::Random;
    if (s == "bfs" || s.empty()) return PartitionMode::Bfs;
    throw std::invalid_argument("partition must be 'random' or 'bfs', got '" + s + "'");
}

/// part[i] in [0,K) for every node.
template <typename T>
std::vector<int> partition_nodes(const CSR<T>& adj, size_t K, PartitionMode mode, unsigned seed) {
    const size_t N = adj.n;
    if (K == 0) throw std::invalid_argument("partition_nodes: K must be >= 1");
    std::vector<int> part(N, -1);
    std::mt19937 rng(seed);
    std::vector<size_t> order(N); std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    if (K == 1) { std::fill(part.begin(), part.end(), 0); return part; }
    if (mode == PartitionMode::Random) { for (size_t i = 0; i < N; i++) part[order[i]] = int(i % K); return part; }

    const size_t cap = (N + K - 1) / K;
    std::vector<size_t> size(K, 0); std::vector<std::deque<size_t>> q(K);
    size_t next_seed = 0, assigned = 0;
    auto claim = [&](size_t v, size_t p) { part[v] = int(p); size[p]++; assigned++; q[p].push_back(v); };
    auto new_seed = [&](size_t p) { while (next_seed < N && part[order[next_seed]] >= 0) next_seed++; if (next_seed < N) claim(order[next_seed], p); };
    for (size_t p = 0; p < K; p++) new_seed(p);
    while (assigned < N) {
        bool progressed = false;
        for (size_t p = 0; p < K; p++) {
            if (size[p] >= cap) continue;
            if (q[p].empty()) { new_seed(p); progressed = true; continue; }   // region exhausted (disconnected graph): jump
            size_t u = q[p].front(); q[p].pop_front(); progressed = true;
            for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1] && size[p] < cap; e++) {
                size_t v = adj.col_idx[e];
                if (part[v] < 0) claim(v, p);
            }
        }
        if (!progressed) break;
    }
    for (size_t v = 0; v < N; v++) if (part[v] < 0) { size_t p = std::min_element(size.begin(), size.end()) - size.begin(); part[v] = int(p); size[p]++; }
    return part;
}

/// Fraction of (non self-loop) adjacency entries whose endpoints are on different devices.
template <typename T>
double cut_fraction(const CSR<T>& adj, const std::vector<int>& part) {
    size_t cut = 0, tot = 0;
    for (size_t u = 0; u < adj.n; u++)
        for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1]; e++) {
            size_t v = adj.col_idx[e]; if (v == u) continue;
            tot++; cut += part[u] != part[v];
        }
    return tot ? double(cut) / double(tot) : 0.0;
}

template <typename T>
struct LocalGraph {
    Graph<T> g;                          // built (normalised adjacency ready)
    std::vector<size_t> global_id;       // local node -> node id in the full graph
};

/// Subgraph induced by the nodes of partition `p` (features + labels + inner edges).
template <typename T>
LocalGraph<T> induced_subgraph(const Graph<T>& full, const std::vector<int>& part, int p) {
    LocalGraph<T> out{Graph<T>(full.feature_dim()), {}};
    const size_t F = full.feature_dim();
    std::vector<long long> local(full.num_nodes(), -1);
    for (size_t i = 0; i < full.num_nodes(); i++)
        if (part[i] == p) {
            local[i] = static_cast<long long>(out.global_id.size());
            out.global_id.push_back(i);
            out.g.add_node(&full.features()[i * F], F, full.labels()[i]);
        }
    const CSR<T>& adj = full.adjacency();
    for (size_t u = 0; u < full.num_nodes(); u++) {
        if (local[u] < 0) continue;
        for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1]; e++) {
            size_t v = adj.col_idx[e];
            if (v != u && local[v] >= 0) out.g.add_edge(static_cast<node_t>(local[u]), static_cast<node_t>(local[v]));
        }
    }
    out.g.build();
    return out;
}

}}} // namespace tf::fed::graph
