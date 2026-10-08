#pragma once
// Graph.hpp — plain data container + graph construction. No autograd.
//
// J6: connect_by_similarity() spawns std::thread workers.
//     Link with -pthread (POSIX) when using this header.
//
// Lifecycle:
//   1. add_node(...) / add_edge(...)            -> raw, mutable
//   2. connect_by_similarity(threshold, ...)    -> optional, adds edges from features
//   3. build()                                  -> symmetrize, dedupe, self-loops,
//                                                  D^-1/2 (A+I) D^-1/2  as CSR
//   4. adjacency(), features(), labels()        -> read-only views for the model

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "core/Types/scalar.hpp"   // scalar::to_acc for low-precision element types

using node_t = uint32_t;

struct Edge {
    node_t src;
    node_t dst;
};

template <typename T>
class CSR {
    public:
        size_t n = 0;
        std::vector<size_t> row_ptr;
        std::vector<node_t> col_idx;
        std::vector<T>      values;
        bool symmetric = false;
        size_t nnz() const { return col_idx.size(); }
};

template <typename T>
class Graph {
public:
    explicit Graph(size_t feature_dim, size_t reserve_nodes = 0) : F_(feature_dim) {
        if (feature_dim == 0) throw std::invalid_argument("Graph: feature_dim must be > 0");
        features_.reserve(reserve_nodes * F_);
        labels_.reserve(reserve_nodes);
    }

    node_t add_node(const T* feat, size_t len, int label = -1) {
        if (len != F_) throw std::invalid_argument("Graph::add_node: wrong feature length");
        features_.insert(features_.end(), feat, feat + len);
        labels_.push_back(label);
        built_ = false;
        return static_cast<node_t>(labels_.size() - 1);
    }
    node_t add_node(const std::vector<T>& feat, int label = -1) {
        return add_node(feat.data(), feat.size(), label);
    }

    void add_edge(node_t u, node_t v) {
        if (u >= num_nodes() || v >= num_nodes())
            throw std::out_of_range("Graph::add_edge: node id out of range");
        edges_.push_back({u, v});
        built_ = false;
    }

    // J6: cosine similarity now accumulates in double, so norms/dots are
    // computed identically for float / double / FP8 / FP4 / float16.
    size_t connect_by_similarity(T threshold, size_t num_threads = 0, size_t block_size = 128,
                                 size_t max_edges = SIZE_MAX) {
        const size_t N = num_nodes();
        if (N < 2) return 0;
        if (block_size == 0) block_size = 128;
        if (num_threads == 0) num_threads = std::max<size_t>(1, std::thread::hardware_concurrency());

        std::vector<T> unit(features_.size());
        std::vector<char> usable(N, 1);
        for (size_t i = 0; i < N; i++) {
            const T* x = &features_[i * F_];
            double s = 0;
            for (size_t k = 0; k < F_; k++) {
                double v = static_cast<double>(scalar::to_acc<T>(x[k]));
                s += v * v;
            }
            if (s <= 0) { usable[i] = 0; continue; }
            const T inv = T(1.0 / std::sqrt(s));
            for (size_t k = 0; k < F_; k++)
                unit[i * F_ + k] = T(x[k] * inv);
        }

        const double thr = static_cast<double>(scalar::to_acc<T>(threshold));   // explicit FP8/FP4 -> float -> double
        const size_t n_blocks = (N + block_size - 1) / block_size;
        std::atomic<size_t> next_block{0};
        std::atomic<size_t> found{0};
        std::vector<std::vector<Edge>> local(num_threads);

        auto worker = [&](size_t tid) {
            auto& out = local[tid];
            for (;;) {
                size_t b = next_block.fetch_add(1);
                if (b >= n_blocks || found.load(std::memory_order_relaxed) > max_edges) return;
                size_t i0 = b * block_size, i1 = std::min(N, i0 + block_size);
                for (size_t i = i0; i < i1; i++) {
                    if (!usable[i]) continue;
                    const T* xi = &unit[i * F_];
                    for (size_t j = i + 1; j < N; j++) {
                        if (!usable[j]) continue;
                        const T* xj = &unit[j * F_];
                        double dot = 0;
                        for (size_t k = 0; k < F_; k++)
                            dot += static_cast<double>(scalar::to_acc<T>(xi[k]))
                                 * static_cast<double>(scalar::to_acc<T>(xj[k]));
                        if (dot >= thr) {
                            out.push_back({static_cast<node_t>(i), static_cast<node_t>(j)});
                            found.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            }
        };

        std::vector<std::thread> pool;
        for (size_t t = 1; t < num_threads; t++) pool.emplace_back(worker, t);
        worker(0);
        for (auto& th : pool) th.join();

        size_t added = 0;
        for (auto& v : local) added += v.size();
        if (added > max_edges)
            throw std::runtime_error("Graph::connect_by_similarity: more than max_edges edges; raise the threshold");
        for (auto& v : local) edges_.insert(edges_.end(), v.begin(), v.end());
        if (added) built_ = false;
        return added;
    }

    void build() {
        const size_t N = num_nodes();
        std::vector<std::pair<node_t, node_t>> e;
        e.reserve(edges_.size() * 2 + N);
        for (const Edge& ed : edges_) {
            e.emplace_back(ed.src, ed.dst);
            if (ed.src != ed.dst) e.emplace_back(ed.dst, ed.src);
        }
        for (size_t i = 0; i < N; i++) e.emplace_back(node_t(i), node_t(i));
        std::sort(e.begin(), e.end());
        e.erase(std::unique(e.begin(), e.end()), e.end());

        auto a = std::make_shared<CSR<T>>();
        a->n = N;
        a->symmetric = true;
        a->row_ptr.assign(N + 1, 0);
        a->col_idx.resize(e.size());
        a->values.resize(e.size());
        for (size_t k = 0; k < e.size(); k++) {
            a->row_ptr[e[k].first + 1]++;
            a->col_idx[k] = e[k].second;
        }
        for (size_t i = 0; i < N; i++) a->row_ptr[i + 1] += a->row_ptr[i];

        std::vector<T> inv_sqrt_deg(N);
        for (size_t i = 0; i < N; i++)
            inv_sqrt_deg[i] = T(1.0 / std::sqrt(double(a->row_ptr[i + 1] - a->row_ptr[i])));
        for (size_t i = 0; i < N; i++)
            for (size_t k = a->row_ptr[i]; k < a->row_ptr[i + 1]; k++)
                a->values[k] = inv_sqrt_deg[i] * inv_sqrt_deg[a->col_idx[k]];
        adj_ = std::move(a);
        built_ = true;
    }

    size_t num_nodes()    const { return labels_.size(); }
    size_t feature_dim()  const { return F_; }
    size_t num_raw_edges() const { return edges_.size(); }
    const std::vector<T>&   features() const { return features_; }
    const std::vector<int>& labels()   const { return labels_; }
    const CSR<T>& adjacency() const { return *adjacency_shared(); }
    std::shared_ptr<const CSR<T>> adjacency_shared() const {
        if (!built_) throw std::logic_error("Graph: call build() before adjacency()");
        return adj_;
    }
    std::vector<T> dense_adjacency() const {
        const CSR<T>& a = adjacency();
        std::vector<T> d(a.n * a.n, T(0));
        for (size_t i = 0; i < a.n; i++)
            for (size_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; k++) d[i * a.n + a.col_idx[k]] = a.values[k];
        return d;
    }

private:
    size_t F_;
    std::vector<T>   features_;
    std::vector<int> labels_;
    std::vector<Edge> edges_;
    std::shared_ptr<const CSR<T>> adj_;
    bool built_ = false;
};