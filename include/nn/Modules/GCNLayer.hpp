#ifndef __TENSORF_GCNLAYER_HPP_
#define __TENSORF_GCNLAYER_HPP_

// One graph-convolution layer:  out = A_hat * (X * W) + b
// A_hat comes from Graph::build() (already normalised, with self loops).
// Two forward() overloads:
//   * dense  A_hat tensor (N x N)      - simple, O(N^2) memory, fine for a few thousand nodes
//   * sparse CSR  A_hat (SpMMOperation) - O(nnz) memory and time, use this for real graphs

#include "Types/types.hpp"
#include "DataStructures/Tensor.hpp"
#include "Module.hpp"
#include "core/DataStructures/Graph.hpp"
#include <cmath>

// Helpers that bridge Graph (plain data) -> TensorF tensors.
template <typename T>
Tensor_t<T> features_tensor(const Graph<T>& g) {
    auto t = make_tensor<T>(Matrix<T>(g.features(), shape_t{g.num_nodes(), g.feature_dim()}));
    t->requires_grad = false;
    return t;
}

template <typename T>
Tensor_t<T> dense_adjacency_tensor(const Graph<T>& g) {
    auto t = make_tensor<T>(Matrix<T>(g.dense_adjacency(), shape_t{g.num_nodes(), g.num_nodes()}));
    t->requires_grad = false;                                  // A_hat is a constant
    return t;
}

// One-hot targets (N x C). Unlabeled nodes (label < 0) get an all-zero row, so they add
// nothing to cross_entropy. `train_mask[i] == false` hides a labeled node as well.
template <typename T>
Tensor_t<T> onehot_targets(const Graph<T>& g, size_t num_classes, const std::vector<char>& train_mask) {
    std::vector<T> y(g.num_nodes() * num_classes, T(0));
    for (size_t i = 0; i < g.num_nodes(); i++)
        if (train_mask[i] && g.labels()[i] >= 0) y[i * num_classes + g.labels()[i]] = T(1);
    auto t = make_tensor<T>(Matrix<T>(y, shape_t{g.num_nodes(), num_classes}));
    t->requires_grad = false;
    return t;
}

template <typename T>
class GCNLayer : public Module<T> {
public:
    Tensor_t<T> weight;     // [in, out]
    Tensor_t<T> bias;       // [out]
    bool use_bias;

    GCNLayer(size_t in_features, size_t out_features, bool use_bias = true, std::optional<unsigned int> seed = std::nullopt) : use_bias(use_bias) {
        T limit = std::sqrt(T(6) / T(in_features + out_features));       // Glorot
        weight = make_tensor<T>(Matrix<T>::randu(-limit, limit, {in_features, out_features}, seed));
        this->register_parameter(weight);
        bias = make_tensor<T>(Matrix<T>::zeros({out_features}));
        if (use_bias) this->register_parameter(bias);
    }

    // adj: N x N (constant), x: N x in  ->  N x out
    Tensor_t<T> forward(Tensor_t<T> adj, Tensor_t<T> x) {
        Tensor_t<T> h = adj->matmul(x->matmul(weight));    // multiply X*W first: N*in*out, cheaper than A*X
        return use_bias ? h + bias : h;
    }

    // adj: CSR from Graph::adjacency_shared(); x: N x in  ->  N x out
    Tensor_t<T> forward(std::shared_ptr<const CSR<T>> adj, Tensor_t<T> x, size_t num_threads = 1) {
        Tensor_t<T> h = spmm<T>(std::move(adj), x->matmul(weight), num_threads);
        return use_bias ? h + bias : h;
    }
};

#endif