// Synthetic test: two feature clusters -> graph built by cosine similarity -> 2-layer GCN.
// Only 3 labeled nodes per class are used for training; accuracy is measured on ALL nodes.
#include "core/DataStructures/Graph.hpp"
#include "nn/Modules/GCNLayer.hpp"
#include "nn/Modules/Optimizer.hpp"
#include "core/Types/types.hpp"
#include <iostream>
#include <random>

int main() {
    const size_t PER = 30, F = 8, C = 2, N = PER * C;
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.f, 0.35f);

    Graph<float> g(F, N);
    for (size_t c = 0; c < C; c++)
        for (size_t i = 0; i < PER; i++) {
            std::vector<float> x(F);
            for (size_t k = 0; k < F; k++) x[k] = noise(rng) + ((k % C == c) ? 1.5f : 0.f);
            g.add_node(x, int(c));
        }

    size_t added = g.connect_by_similarity(0.80f, /*threads=*/4, /*block=*/16);
    g.build();
    std::cout << "nodes=" << g.num_nodes() << " similarity edges=" << added
              << " csr nnz=" << g.adjacency().nnz() << "\n";

    // sanity: row sums of normalised adjacency are finite and positive; matrix is symmetric
    auto dense = g.dense_adjacency();
    double asym = 0;
    for (size_t i = 0; i < N; i++) for (size_t j = 0; j < N; j++) asym = std::max(asym, (double)std::abs(dense[i*N+j] - dense[j*N+i]));
    std::cout << "max |A - A^T| = " << asym << "\n";

    std::vector<char> train(N, 0);
    for (size_t c = 0; c < C; c++) for (size_t i = 0; i < 3; i++) train[c * PER + i] = 1;

    auto X = features_tensor(g);
    auto A = dense_adjacency_tensor(g);
    auto Y = onehot_targets(g, C, train);

    GCNLayer<float> l1(F, 8), l2(8, C);
    std::vector<Tensor_t<float>> params;
    for (auto p : l1.parameters()) params.push_back(p);
    for (auto p : l2.parameters()) params.push_back(p);
    Optimizer<float> opt(params, 0.05f, ADAM, true);

    for (int epoch = 0; epoch <= 150; epoch++) {
        opt.zero_grad();
        auto h   = l1.forward(A, X)->relu();
        auto out = l2.forward(A, h)->softmax();
        auto loss = Tensor<float>::cross_entropy(Y, out);
        loss->backward(make_tensor<float>(1.0f));
        opt.step();
        if (epoch % 30 == 0) {
            size_t ok = 0;
            for (size_t i = 0; i < N; i++) {
                float a = out->val.data[i*C], b = out->val.data[i*C+1];
                ok += ((b > a) ? 1 : 0) == (size_t)g.labels()[i];
            }
            std::cout << "epoch " << epoch << " loss " << loss->val.data[0] << " acc(all) " << 100.0 * ok / N << "%\n";
        }
        loss->reset_graph();
    }
}