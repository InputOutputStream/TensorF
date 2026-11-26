#pragma once
// BuiltinTasks.hpp — algorithms shipped with TensorF's distributed layer.
// Call register_builtin_tasks() once from main(). Adding an algorithm = one line here.
//
//   synthetic : averages a flat vector (transport smoke test; no data, no model)
//   gcn       : 2-layer GCN node classification on a partitioned LINQS graph (Cora, Citeseer, ...)
//   (planned) sage, saint, gat, rgcn, gnn: implement a Model policy (see NodeClassifierTask.hpp)
//             and register node_classifier_factory<YourModel>("...") — nothing else changes.

#include "FedTask.hpp"
#include "graph/NodeClassifierTask.hpp"
#include "graph/GraphShardProvider.hpp"

#include <thread>

namespace tf { namespace fed {

class SyntheticTask : public ClientTask {
    uint64_t n_; std::vector<float> p_; float target_; int train_ms_; std::string id_;
public:
    explicit SyntheticTask(const TaskOptions& o)
        : n_(uint64_t(o.i("params", 100000))), p_(n_, 0.f), train_ms_(int(o.i("train_ms", 200))), id_(o.str("id", "synthetic")) {
        target_ = 1.f + float(std::hash<std::string>{}(id_) % 100) / 100.f;
    }
    TaskSpec spec() const override { TaskSpec s; s.algo = "synthetic"; s.param_count = n_; s.fingerprint = 0x5EED; return s; }
    uint64_t train(const RoundInfo& r) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(train_ms_));
        for (auto& x : p_) x += 0.5f * (target_ - x);
        std::printf("[%s] trained from global round %llu (target %.2f)\n", id_.c_str(), (unsigned long long)r.base_round, target_); std::fflush(stdout);
        return 100;
    }
    void get_params(uint64_t o, size_t n, float* out) override { std::memcpy(out, &p_[o], n * 4); }
    void set_params(uint64_t o, const float* d, size_t n) override { std::memcpy(&p_[o], d, n * 4); }
    float first() const { return p_.empty() ? 0.f : p_[0]; }
};

inline void register_builtin_tasks() {
    auto& r = TaskRegistry::get();
    TaskFactory syn;
    syn.server = [](const TaskOptions& o) { ServerTaskInfo t; t.spec.algo = "synthetic"; t.spec.param_count = uint64_t(o.i("params", 100000)); t.spec.fingerprint = 0x5EED; return t; };
    syn.client = [](const TaskOptions& o) { return std::unique_ptr<ClientTask>(new SyntheticTask(o)); };
    syn.help = "flat-vector averaging (opts: params, train_ms)";
    r.add("synthetic", syn);
    TaskFactory gcn = graph::node_classifier_factory<graph::GCN2>(
        "2-layer GCN node classification (opts: data=Datasets/cora hidden=16 lr local_epochs=5 prox_mu; simulation: part parts partition eval_full; "
        "shards: server shards=1 partitioner=auto|metis|builtin policy=capacity|equal server_eval=1, client source=shard features= classes=)");
    auto gcn_server = gcn.server;
    gcn.server = [gcn_server](const TaskOptions& o) {
        ServerTaskInfo t = gcn_server(o);
        if (o.b("shards", false)) t.shards = std::make_shared<graph::GraphShardProvider>(o);
        return t;
    };
    r.add("gcn", gcn);
}

}} // namespace tf::fed
