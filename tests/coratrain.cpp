// Train a 2-layer GCN on Cora (Kipf & Welling recipe) with TensorF.
//   coratrain [x.content] [x.cites] [epochs] [seed] [patience]
// Defaults: Datasets/cora/cora.content Datasets/cora/cora.cites 200 42 10
// Early-stopping / model selection use VALIDATION LOSS (Kipf & Welling, ICLR 2017, §4),
// not validation accuracy — accuracy plateaus for 15-20 epochs on Cora while the loss
// is still falling, so stopping on accuracy truncates training at ~50% val.

#include "DataStructures/Graph.hpp"
#include "GraphLoader.hpp"
#include "Modules/GCNLayer.hpp"
#include "Modules/Optimizer.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

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
    int patience        = argc > 5 ? std::atoi(argv[5]) : 10;   // epochs without val-loss improvement before stopping; 0 = never stop
    const size_t HIDDEN = 16;
    const float LR = 0.01f;
    const float DROPOUT = 0.5f;
    const float WEIGHT_DECAY = 5e-4f;
    mxd::rng(seed);

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
    X->requires_grad = false;

    // Integer labels as a constant tensor (class indices, no grad).
    std::vector<float> labels_f(g.labels().begin(), g.labels().end());
    auto labels_t = make_tensor<float>(labels_f);
    labels_t->requires_grad = false;

    // Training mask as a constant tensor (1.0 on train nodes, 0.0 elsewhere).
    std::vector<float> mask_f(N, 0.0f);
    for (size_t i = 0; i < N; ++i) if (sp.train[i]) mask_f[i] = 1.0f;
    auto mask_t = make_tensor<float>(mask_f);
    mask_t->requires_grad = false;

    // Validation mask (needed for the val-loss evaluation).
    std::vector<float> vmask_f(N, 0.0f);
    for (size_t i = 0; i < N; ++i) if (sp.val[i]) vmask_f[i] = 1.0f;
    auto val_mask_t = make_tensor<float>(vmask_f);
    val_mask_t->requires_grad = false;

    std::mt19937 rng(seed + 1);
    GCNLayer<float> l1(F, HIDDEN, false), l2(HIDDEN, C, false, seed+2);
    std::vector<Tensor_t<float>> params;
    for (auto p : l1.parameters()) params.push_back(p);
    for (auto p : l2.parameters()) params.push_back(p);
    Optimizer<float> opt(params, LR, ADAM, true);

    // Best-model bookkeeping — selected by val LOSS (K&W).
    double best_val_loss     = std::numeric_limits<double>::infinity();
    double val_acc_at_best   = 0.0;
    double test_acc_at_best  = 0.0;
    int    best_epoch        = 0;
    int    counter           = 0;      // epochs since best val loss
    double train_ms          = 0.0;
    int    epoch_run         = 0;
    double last_val          = 0.0;
    double last_test         = 0.0;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        auto te = Clock::now();
        epoch_run = epoch;

        // ---- train step (dropout on input features and hidden layer) ----
        opt.zero_grad();
        auto h          = l1.forward(adj, X->dropout(DROPOUT, rng))->relu();
        auto out_logits = l2.forward(adj, h->dropout(DROPOUT, rng));    // no softmax
        auto loss       = Tensor<float>::masked_cross_entropy_from_logits(out_logits, labels_t, mask_t, sp.n_train);

        // L2 penalty on the first layer only, as in the paper: wd * sum(W^2) / 2
        auto pen   = (l1.weight * l1.weight)->sum() * make_tensor<float>(WEIGHT_DECAY / 2);
        auto total = loss + pen;
        total->backward(make_tensor<float>(1.0f));
        opt.step();
        float loss_v = loss->val.data[0];
        total->reset_graph();
        train_ms += ms_since(te);

        // ---- evaluation (no dropout, no autograd) ----
        Tensor_t<float> ev;
        float val_loss_v;
        {
            mxd::NoGrad guard;
            ev = l2.forward(adj, l1.forward(adj, X)->relu());
            auto vl = Tensor<float>::masked_cross_entropy_from_logits(
                ev, labels_t, val_mask_t, sp.n_val);
            val_loss_v = vl->val.data[0];
        }

        double tr = accuracy(ev, g.labels(), sp.train);
        double va = accuracy(ev, g.labels(), sp.val);
        double ts = accuracy(ev, g.labels(), sp.test);

        last_val  = va;
        last_test = ts;

        // Selection / early stopping on val LOSS, not val accuracy.
        if (val_loss_v < best_val_loss) {
            best_val_loss    = val_loss_v;
            val_acc_at_best  = va;
            test_acc_at_best = ts;
            best_epoch       = epoch;
            counter          = 0;
        } else {
            counter++;
        }

        if (epoch == 1 || epoch % 10 == 0)
            std::cout << "epoch " << epoch
                      << "  train_loss " << loss_v
                      << "  val_loss "   << val_loss_v
                      << "  train " << tr << "%  val " << va << "%  test " << ts << "%"
                      << "  (" << ms_since(te) << " ms/epoch)\n";

        if (patience > 0 && counter >= patience) {
            std::cout << "\nstopped at epoch " << epoch
                      << "  (no val-loss improvement for " << patience << " epochs)\n";
            break;
        }
    }

    std::cout << "\nfinal epoch:        val " << last_val << "%  test " << last_test << "%\n"
              // Keep this literal prefix "best val (epoch ...): val X% test Y%"
              // — scripts/run_gcn_benchmarks.sh parses it.
              << "best val (epoch " << best_epoch << "): val "
              << val_acc_at_best << "%  test " << test_acc_at_best << "%"
              << "  [selected by val loss " << best_val_loss << "]\n"
              << "avg train step: " << train_ms / epoch_run << " ms\n";
    return 0;
}