// Train a 2-layer GCN on Cora (Kipf & Welling recipe) with TensorF.
//   coratrain [cora.content] [cora.cites] [epochs] [seed]
// Defaults: Datasets/cora/cora.content Datasets/cora/cora.cites 200 42

#include "DataStructures/Graph.hpp"
#include "GraphLoader.hpp"
#include "Modules/GCNLayer.hpp"
#include "Modules/Optimizer.hpp"

#include <chrono>
#include <iostream>
#include <random>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// Inverted dropout: multiply by a constant 0 / 1/(1-p) mask 
static Tensor_t<float> dropout(Tensor_t<float> x, float p, std::mt19937& rng) {
    std::vector<float> m(x->val.get_size());
    std::bernoulli_distribution keep(1.0 - p);
    const float scale = 1.0f / (1.0f - p);
    for (auto& v : m) v = keep(rng) ? scale : 0.0f;
    auto M = make_tensor<float>(Matrix<float>(m, x->val.shape));
    M->requires_grad = false;
    return x * M;
}

// Mean cross-entropy over the labelled (training) nodes only. Rows of Y that are all zero add nothing.
static Tensor_t<float> masked_cross_entropy(Tensor_t<float> Y, Tensor_t<float> prob, size_t n_labelled) {
    auto safe = prob + make_tensor<float>(1e-8f);
    return -(Y * safe->ln())->sum() / make_tensor<float>((float)n_labelled);
}

static double accuracy(const Tensor_t<float>& logits, const std::vector<int>& labels, const std::vector<char>& mask) {
    const size_t C = logits->val.shape[1];
    size_t ok = 0, n = 0;
    for (size_t i = 0; i < labels.size(); i++) {
        if (!mask[i]) continue;
        const float* r = &logits->val.data[i * C];
        size_t best = std::max_element(r, r + C) - r;
        ok += (int)best == labels[i];
        n++;
    }
    return n ? 100.0 * ok / n : 0.0;
}

int main(int argc, char** argv) {
    std::string content = argc > 1 ? argv[1] : "Datasets/cora/cora.content";
    std::string cites   = argc > 2 ? argv[2] : "Datasets/cora/cora.cites";
    int epochs          = argc > 3 ? std::atoi(argv[3]) : 200;
    unsigned seed       = argc > 4 ? (unsigned)std::atoi(argv[4]) : 42;
    const size_t HIDDEN = 16;
    const float LR = 0.01f;
    const float DROPOUT = 0.5f;
    const float WEIGHT_DECAY = 5e-4f;

    auto t0 = Clock::now();
    auto ds = load_linqs<float>(content, cites, /*row_normalize=*/true);
    const auto& g = ds.graph;
    const size_t N = g.num_nodes(), F = g.feature_dim(), C = ds.num_classes();
    std::cout << "loaded " << N << " nodes, " << F << " features, " << C << " classes, "
              << g.num_raw_edges() << " raw edges (" << ds.skipped_edges << " skipped), csr nnz "
              << g.adjacency().nnz() << "  [" << ms_since(t0) << " ms]\n";

    Split sp = make_split(g.labels(), C, 20, 500, 1000, seed);
    std::cout << "split: train " << sp.n_train << " / val " << sp.n_val << " / test " << sp.n_test << "\n";

    auto X   = features_tensor(g);
    auto adj = g.adjacency_shared();
    auto Y   = onehot_targets(g, C, sp.train);

    std::mt19937 rng(seed);
    GCNLayer<float> l1(F, HIDDEN, false), l2(HIDDEN, C, false);
    std::vector<Tensor_t<float>> params;
    for (auto p : l1.parameters()) params.push_back(p);
    for (auto p : l2.parameters()) params.push_back(p);
    Optimizer<float> opt(params, LR, ADAM, true);

    double best_val = -1, test_at_best = 0, train_ms = 0;
    int best_epoch = 0;
    double last_val = 0, last_test = 0;
    for (int epoch = 1; epoch <= epochs; epoch++) {
        auto te = Clock::now();
        // ---- train step (dropout on input features and on hidden layer) ----
        opt.zero_grad();
        auto h    = l1.forward(adj, dropout(X, DROPOUT, rng))->relu();
        auto out  = l2.forward(adj, dropout(h, DROPOUT, rng))->softmax();
        auto loss = masked_cross_entropy(Y, out, sp.n_train);

        // L2 penalty on the first layer only, as in the paper: wd * sum(W^2) / 2
        auto pen  = (l1.weight * l1.weight)->sum() * make_tensor<float>(WEIGHT_DECAY / 2);
        auto total = loss + pen;
        total->backward(make_tensor<float>(1.0f));
        opt.step();
        float loss_v = loss->val.data[0];
        total->reset_graph();
        train_ms += ms_since(te);

        // ---- evaluation (no dropout) ----
        auto ev = l2.forward(adj, l1.forward(adj, X)->relu());
        double tr = accuracy(ev, g.labels(), sp.train);
        double va = accuracy(ev, g.labels(), sp.val);
        double ts = accuracy(ev, g.labels(), sp.test);

        ev->reset_graph();
        last_val = va; 
        last_test = ts;
        if (va > best_val) 
        { 
            best_val = va; 
            test_at_best = ts; 
            best_epoch = epoch; 
        }
        if (epoch == 1 || epoch % 20 == 0)
            std::cout << "epoch " << epoch << "  loss " << loss_v << "  train " << tr << "%  val " << va
                      << "%  (" << ms_since(te) << " ms/epoch)\n";
    }
    std::cout << "\nfinal epoch:        val " << last_val << "%  test " << last_test << "%\n"
              << "best val (epoch " << best_epoch << "): val " << best_val << "%  test " << test_at_best << "%\n"
              << "avg train step: " << train_ms / epochs << " ms\n";
    return 0;
}