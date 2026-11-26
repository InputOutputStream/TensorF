#pragma once
#include "PartitionerPrelude.hpp"   // must come before DataStructures/Graph.hpp (KaMinPar name clash)
// NodeClassifierTask.hpp — federated node classification for ANY graph model.
//
// The task knows how to: load a graph, give this device its partition, run local
// full-batch training, expose the model's parameters as a flat vector. The *model*
// is a tiny policy class, so adding GraphSAGE / GAT / RGCN / ... means writing one
// struct and one registry line — the transport, server and client binaries do not change:
//
//   struct Model {
//       static const char* name();                                   // "gcn", "sage", ...
//       Model(size_t F, size_t C, const TaskOptions& o, unsigned seed);
//       Tensor_t<float> forward(std::shared_ptr<const CSR<float>> adj, Tensor_t<float> x,
//                               bool training, std::mt19937& rng);   // -> logits [N, C]
//       std::vector<Tensor_t<float>> parameters();                   // what gets averaged
//       Tensor_t<float> regularizer(float weight_decay);             // L2 term (or a 0 tensor)
//   };
//
// Data: `data=<dir>` with <dir>/<name>.content + .cites (LINQS: Cora, Citeseer, ...). In this
// simulation every device loads the full graph and keeps its own partition (`part=i parts=K`);
// a real phone would ship only its local subgraph. `eval_full=1` additionally scores the global
// model on the full graph's test nodes — a simulation-only metric no real device could compute.

#include "../FedTask.hpp"
#include "../FedStrategies.hpp"
#include "../TensorParams.hpp"
#include "GraphPartition.hpp"
#include "FedGraphData.hpp"

#include "DataStructures/Graph.hpp"
#include "GraphLoader.hpp"
#include "Modules/GCNLayer.hpp"
#include "Modules/Optimizer.hpp"

#include <algorithm>
#include <cstdio>
#include <chrono>
#include <mutex>
#include <random>
#include <thread>

namespace tf { namespace fed { namespace graph {

// ── GCN (Kipf & Welling), 2 layers ------------------------------------------------
struct GCN2 {
    static const char* name() { return "gcn"; }
    GCNLayer<float> l1, l2; float dropout;
    GCN2(size_t F, size_t C, const TaskOptions& o, unsigned seed)
        : l1(F, size_t(o.i("hidden", 16)), o.b("bias", false), seed + 1),
          l2(size_t(o.i("hidden", 16)), C, o.b("bias", false), seed + 2),
          dropout(float(o.f("dropout", 0.5))) {}
    Tensor_t<float> forward(std::shared_ptr<const CSR<float>> adj, Tensor_t<float> x, bool training, std::mt19937& rng) {
        if (!training) return l2.forward(adj, l1.forward(adj, x)->relu());
        auto h = l1.forward(adj, x->dropout(dropout, rng))->relu();
        return l2.forward(adj, h->dropout(dropout, rng));
    }
    std::vector<Tensor_t<float>> parameters() {
        std::vector<Tensor_t<float>> p; for (auto& t : l1.parameters()) p.push_back(t); for (auto& t : l2.parameters()) p.push_back(t); return p;
    }
    Tensor_t<float> regularizer(float wd) { return (l1.weight * l1.weight)->sum() * make_tensor<float>(wd / 2); }   // first layer only, as in the paper
};

// ── helpers ---------------------------------------------------------------------
inline double accuracy(const Tensor_t<float>& logits, const std::vector<int>& labels, const std::vector<char>& mask) {
    const size_t C = logits->val.shape[1]; size_t ok = 0, n = 0;
    for (size_t i = 0; i < labels.size(); i++) {
        if (!mask[i]) continue;
        const float* r = &logits->val.data[i * C];
        ok += int(std::max_element(r, r + C) - r) == labels[i]; n++;
    }
    return n ? 100.0 * double(ok) / double(n) : 0.0;
}

template <class Model>
class NodeClassifierTask : public ClientTask {
public:
    explicit NodeClassifierTask(const TaskOptions& o) : o_(o), rng_(unsigned(o.i("seed", 42)) + 7) {
        const unsigned seed = unsigned(o.i("seed", 42));
        shard_mode_ = o.str("source", "partition") == "shard";
        size_t F = 0;
        if (shard_mode_) {
            // No dataset on the device: the layout is fixed by features/classes, the data arrives as a shard.
            F = size_t(o.i("features", 0)); C_ = size_t(o.i("classes", 0));
            if (F == 0 || C_ == 0) throw std::invalid_argument("source=shard needs --opt features=F --opt classes=C (they fix the parameter layout)");
        } else {
            ds_ = std::make_unique<FedGraphData>(load_fed_graph(o));      // LINQS (Cora, ...) or TFG1 (ogbn-*)
            const auto& full = ds_->graph; F = full.feature_dim(); C_ = ds_->num_classes;
            split_ = ds_->split;
            const size_t K = size_t(o.i("parts", 1)); const int p = int(o.i("part", 0));
            if (p < 0 || size_t(p) >= K) throw std::invalid_argument("part must be in [0, parts)");
            part_ = partition_nodes(full.adjacency(), K, parse_partition(o.str("partition", "bfs")), seed);
            cut_ = cut_fraction(full.adjacency(), part_);
            LocalGraph<float> lg = induced_subgraph(full, part_, p);
            std::vector<uint32_t> flags(lg.global_id.size());
            for (size_t i = 0; i < flags.size(); i++) { size_t gi = lg.global_id[i]; flags[i] = (split_.train[gi] ? 1u : 0u) | (split_.val[gi] ? 2u : 0u); }
            init_local(std::move(lg.g), flags);
            if (o.b("eval_full", false)) { Xfull_ = features_tensor(full); adjfull_ = full.adjacency_shared(); }
        }
        F_ = F;
        model_ = std::make_unique<Model>(F, C_, o, seed);
        pv_ = std::make_unique<ParamView<float>>(model_->parameters());
        opt_ = std::make_unique<Optimizer<float>>(model_->parameters(), float(o.f("lr", 0.01)), ADAM, true);
        local_epochs_ = int(o.i("local_epochs", 5)); wd_ = float(o.f("weight_decay", 5e-4)); mu_ = float(o.f("prox_mu", 0.0));
        speed_ = std::max(0.01, std::min(1.0, o.f("speed", 1.0)));      // simulation: <1 = slower device (time dilation)
        tag_ = "[" + o.str("id", "part" + std::to_string(o.i("part", 0))) + "]";
    }

    TaskSpec spec() const override { TaskSpec s; s.algo = Model::name(); s.param_count = pv_->total(); s.fingerprint = pv_->fingerprint(Model::name()); return s; }

    bool wants_shard() const override { return shard_mode_ && !have_shard_; }
    void on_shard(Shard&& sh) override {
        if (!shard_mode_) return;
        if (size_t(sh.i("features", -1)) != F_ || size_t(sh.i("classes", -1)) != C_) throw std::runtime_error("shard does not match the configured model (features/classes)");
        const Blob *x = sh.blob("x"), *y = sh.blob("y"), *fl = sh.blob("flags"), *ed = sh.blob("edges");
        if (!x || !y || !fl || !ed) throw std::runtime_error("shard is missing blobs (x,y,flags,edges)");
        const size_t N = y->u.size();
        if (x->f.size() != N * F_ || fl->u.size() != N || ed->u.size() % 2) throw std::runtime_error("inconsistent shard blob sizes");
        Graph<float> g(F_);
        for (size_t i = 0; i < N; i++) { if (y->u[i] >= C_) throw std::runtime_error("shard label out of range"); g.add_node(&x->f[i * F_], F_, int(y->u[i])); }
        for (size_t k = 0; k + 1 < ed->u.size(); k += 2) {
            if (ed->u[k] >= N || ed->u[k + 1] >= N) throw std::runtime_error("shard edge out of range");
            g.add_edge(static_cast<node_t>(ed->u[k]), static_cast<node_t>(ed->u[k + 1]));
        }
        g.build();
        init_local(std::move(g), fl->u);
        have_shard_ = true;
        if (o_.b("verbose", true)) { std::printf("%s shard [%s]: %zu nodes (%lld owned), %llu edges, %zu train / %zu val nodes%s\n", tag_.c_str(), sh.str("mode", "?").c_str(), N, sh.i("owned", 0), (unsigned long long)sh.i("edges", 0), n_train_, n_val_, sh.str("reason").empty() ? "" : (" [downgraded: " + sh.str("reason") + "]").c_str()); std::fflush(stdout); }
    }

    uint64_t train(const RoundInfo& r) override {
        if (n_train_ == 0 || !lgraph_) return 0;
        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> anchor; if (mu_ > 0) anchor = pv_->flatten();
        double loss_v = 0;
        for (int e = 0; e < local_epochs_; e++) {
            opt_->zero_grad();
            auto logits = model_->forward(adj_, X_, true, rng_);
            auto loss = Tensor<float>::masked_cross_entropy_from_logits(logits, labels_t_, train_t_, n_train_);
            auto total = loss + model_->regularizer(wd_);
            total->backward(make_tensor<float>(1.0f));
            opt_->step();
            loss_v = loss->val.data[0];
            total->reset_graph();
            if (mu_ > 0) {                                  // FedProx: pull toward the received global model
                auto cur = pv_->flatten(); fedprox_pull(cur.data(), anchor.data(), cur.size(), mu_ * float(o_.f("lr", 0.01)));
                pv_->assign(cur);
            }
        }
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (speed_ < 1.0) std::this_thread::sleep_for(std::chrono::duration<double>(el * (1.0 / speed_ - 1.0)));   // simulated slow device
        last_train_s_ = el / speed_;
        if (o_.b("verbose", true)) { std::printf("%s round %llu: %d local epochs on %zu train nodes, loss %.4f, %.2fs\n", tag_.c_str(), (unsigned long long)(r.base_round + 1), local_epochs_, n_train_, loss_v, last_train_s_); std::fflush(stdout); }
        return n_train_;
    }
    void get_params(uint64_t off, size_t len, float* out) override { pv_->get(off, len, out); }
    void set_params(uint64_t off, const float* d, size_t len) override { pv_->set(off, d, len); }
    void on_global(uint64_t round) override {
        last_round_ = round;
        mxd::NoGrad guard;
        if (lgraph_ && n_val_ > 0) {
            auto ev = model_->forward(adj_, X_, false, rng_);
            last_local_val_ = accuracy(ev, lgraph_->labels(), val_mask_);
        }
        if (Xfull_) {
            auto ev = model_->forward(adjfull_, Xfull_, false, rng_);
            last_test_ = accuracy(ev, ds_->graph.labels(), split_.test);
            std::printf("%s global round %llu: local val %.1f%%  full-graph test %.1f%%\n", tag_.c_str(), (unsigned long long)round, last_local_val_, last_test_);
        } else if (o_.b("verbose", true)) std::printf("%s global round %llu: local val %.1f%%\n", tag_.c_str(), (unsigned long long)round, last_local_val_);
        std::fflush(stdout);
    }

    // simulation / baselines -----------------------------------------------------
    void run_local(int epochs) { int keep = local_epochs_; local_epochs_ = epochs; RoundInfo r; train(r); local_epochs_ = keep; }
    double eval_full_test() { if (!Xfull_) { Xfull_ = features_tensor(ds_->graph); adjfull_ = ds_->graph.adjacency_shared(); }
        mxd::NoGrad g; auto ev = model_->forward(adjfull_, Xfull_, false, rng_); return accuracy(ev, ds_->graph.labels(), split_.test); }
    size_t n_train() const { return n_train_; }
    size_t n_nodes() const { return lgraph_ ? lgraph_->num_nodes() : 0; }
    size_t n_edges_local() const { return lgraph_ ? lgraph_->adjacency().col_idx.size() : 0; }
    const Graph<float>* local_graph() const { return lgraph_.get(); }
    double cut() const { return cut_; }
    double last_test() const { return last_test_; }
    double last_local_val() const { return last_local_val_; }
    uint64_t last_round() const { return last_round_; }
    bool have_shard() const { return have_shard_; }

private:
    void init_local(Graph<float>&& g, const std::vector<uint32_t>& flags) {
        lgraph_ = std::make_unique<Graph<float>>(std::move(g));
        const size_t N = lgraph_->num_nodes();
        std::vector<float> lab(N), tm(N, 0.f), vm(N, 0.f); val_mask_.assign(N, 0); n_train_ = n_val_ = 0;
        for (size_t i = 0; i < N; i++) {
            lab[i] = float(lgraph_->labels()[i]);
            if (flags[i] & 1u) { tm[i] = 1.f; n_train_++; }
            if (flags[i] & 2u) { vm[i] = 1.f; n_val_++; val_mask_[i] = 1; }
        }
        X_ = features_tensor(*lgraph_); adj_ = lgraph_->adjacency_shared();
        labels_t_ = make_tensor<float>(lab); labels_t_->requires_grad = false;
        train_t_ = make_tensor<float>(tm);   train_t_->requires_grad = false;
        val_t_ = make_tensor<float>(vm);     val_t_->requires_grad = false;
    }

    TaskOptions o_; std::mt19937 rng_; bool shard_mode_ = false, have_shard_ = false;
    std::unique_ptr<FedGraphData> ds_; Split split_; size_t C_ = 0, F_ = 0;
    std::vector<int> part_; double cut_ = 0;
    std::unique_ptr<Graph<float>> lgraph_;
    size_t n_train_ = 0, n_val_ = 0; std::vector<char> val_mask_;
    Tensor_t<float> X_, labels_t_, train_t_, val_t_, Xfull_;
    std::shared_ptr<const CSR<float>> adj_, adjfull_;
    std::unique_ptr<Model> model_; std::unique_ptr<ParamView<float>> pv_; std::unique_ptr<Optimizer<float>> opt_;
    int local_epochs_ = 5; float wd_ = 5e-4f, mu_ = 0.f; double speed_ = 1.0, last_train_s_ = 0; std::string tag_;
    double last_test_ = 0, last_local_val_ = 0; uint64_t last_round_ = 0;
};

/// Server side for a node classifier: spec + initial parameters from the model shape alone
/// (features/classes given directly, or read from `data=` for convenience in demos).
template <class Model>
ServerTaskInfo node_classifier_server(const TaskOptions& o) {
    size_t F = size_t(o.i("features", 0)), C = size_t(o.i("classes", 0));
    if (F == 0 || C == 0) dataset_dims(o, F, C);        // TFG: header only; LINQS: loads the (small) graph
    Model m(F, C, o, unsigned(o.i("seed", 42)));
    ParamView<float> pv(m.parameters());
    ServerTaskInfo t; t.spec.algo = Model::name(); t.spec.param_count = pv.total(); t.spec.fingerprint = pv.fingerprint(Model::name());
    t.initial = pv.flatten();
    if (o.b("server_eval", false)) {      // evaluate each new global model on the full graph's test nodes (the server owns the data)
        struct Ctx { std::unique_ptr<FedGraphData> ds; Split split; std::unique_ptr<Model> m; std::unique_ptr<ParamView<float>> pv;
                     Tensor_t<float> X; std::shared_ptr<const CSR<float>> adj; std::mt19937 rng{1}; std::mutex mu; };
        auto c = std::make_shared<Ctx>();
        c->ds = std::make_unique<FedGraphData>(load_fed_graph(o)); c->split = c->ds->split;
        c->m = std::make_unique<Model>(F, C, o, unsigned(o.i("seed", 42))); c->pv = std::make_unique<ParamView<float>>(c->m->parameters());
        c->X = features_tensor(c->ds->graph); c->adj = c->ds->graph.adjacency_shared();
        t.eval = [c](const std::vector<float>& g, uint64_t round) {
            std::lock_guard<std::mutex> l(c->mu); c->pv->assign(g); mxd::NoGrad ng;
            auto ev = c->m->forward(c->adj, c->X, false, c->rng);
            return J().kv("type", "fed_eval").kv("round", round).kv("val", accuracy(ev, c->ds->graph.labels(), c->split.val))
                       .kv("test", accuracy(ev, c->ds->graph.labels(), c->split.test)).str();
        };
    }
    return t;
}

template <class Model>
TaskFactory node_classifier_factory(const std::string& help) {
    TaskFactory f;
    f.server = [](const TaskOptions& o) { return node_classifier_server<Model>(o); };
    f.client = [](const TaskOptions& o) { return std::unique_ptr<ClientTask>(new NodeClassifierTask<Model>(o)); };
    f.help = help;
    return f;
}

}}} // namespace tf::fed::graph
