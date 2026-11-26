#pragma once
// FedTask.hpp — what a distributable algorithm looks like to the transport.
//
// The transport (FedWire / FedCoordinator / MobileClient) only ever sees:
//   * a flat vector of floats of fixed length (`TaskSpec::param_count`),
//   * an algorithm name + fingerprint (so GCN and GraphSAGE can never be mixed up),
//   * three client callbacks (train / get_params / set_params).
// It knows nothing about LLMs, graphs, tensors or optimizers. Anything that can be
// expressed as "each device improves a vector locally, the server combines the
// vectors" is distributable: GCN, GraphSAGE, GAT, RGCN, an MLP, LoRA adapters, ...
//
// To add an algorithm you write ONE ClientTask (+ a server-side spec) and register
// it; no change to the server, the client binaries or the protocol:
//
//     TaskRegistry::get().add("sage", { server_fn, client_fn, "GraphSAGE node classification" });
//
// Which "distribution method" runs is chosen separately, by data (FedAvg weights,
// FedAvgM/FedAdam server optimizers in FedStrategies.hpp, FedProx pull on the client).

#include "MobileClient.hpp"
#include "FedCoordinator.hpp"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tf { namespace fed {

struct TaskSpec {
    std::string algo;                 // "gcn", "sage", "gat", "rgcn", "synthetic", ...
    uint64_t    param_count = 0;      // length of the exchanged flat vector
    uint32_t    fingerprint = 0;      // hash of algo + parameter layout (see TensorParams.hpp)
};

/// String key/value options for a task (CLI `--opt k=v`, dashboard JSON, ...).
struct TaskOptions {
    std::map<std::string, std::string> kv;
    bool has(const std::string& k) const { return kv.count(k) > 0; }
    std::string str(const std::string& k, const std::string& def = "") const { auto it = kv.find(k); return it == kv.end() ? def : it->second; }
    long long   i(const std::string& k, long long def = 0) const { auto it = kv.find(k); return it == kv.end() ? def : std::stoll(it->second); }
    double      f(const std::string& k, double def = 0) const { auto it = kv.find(k); return it == kv.end() ? def : std::stod(it->second); }
    bool        b(const std::string& k, bool def = false) const { auto it = kv.find(k); if (it == kv.end()) return def; return it->second == "1" || it->second == "true" || it->second == "yes"; }
    /// parse "key=value"; returns false on malformed input
    bool parse(const std::string& s) { auto p = s.find('='); if (p == std::string::npos || p == 0) return false; kv[s.substr(0, p)] = s.substr(p + 1); return true; }
};

struct RoundInfo { uint64_t base_round = 0; };   // the global round the local model currently corresponds to

/// Client side of a distributed algorithm.
class ClientTask {
public:
    virtual ~ClientTask() = default;
    virtual TaskSpec spec() const = 0;
    /// Improve the local model starting from the current parameters. Return the number
    /// of samples used (it is the FedAvg weight); 0 = nothing to contribute this round.
    virtual uint64_t train(const RoundInfo& r) = 0;
    virtual void get_params(uint64_t off, size_t len, float* out) = 0;
    virtual void set_params(uint64_t off, const float* data, size_t len) = 0;
    /// A complete global model was just applied (evaluate, log, checkpoint, ...).
    virtual void on_global(uint64_t /*round*/) {}
    /// Participation policy (battery / thermal / network). Default: always.
    virtual bool can_participate() { return true; }
    /// Data shard support: return true until the shard has been received; on_shard() gets it.
    virtual bool wants_shard() const { return false; }
    virtual void on_shard(Shard&&) {}
};

/// Server side: what the coordinator must know, with no data and no training code.
struct ServerTaskInfo {
    TaskSpec spec;
    std::vector<float> initial;       // initial global parameters (size == spec.param_count); may be empty = zeros
    std::shared_ptr<ShardProvider> shards;   // optional: serve data shards (partitioned dataset) to devices
    /// Optional server-side evaluation of the global vector after each round (returns a JSON object string or "").
    std::function<std::string(const std::vector<float>& global, uint64_t round)> eval;
};

struct TaskFactory {
    std::function<ServerTaskInfo(const TaskOptions&)>             server;
    std::function<std::unique_ptr<ClientTask>(const TaskOptions&)> client;
    std::string help;
};

class TaskRegistry {
    std::map<std::string, TaskFactory> m_;
public:
    static TaskRegistry& get() { static TaskRegistry r; return r; }
    void add(const std::string& name, TaskFactory f) { m_[name] = std::move(f); }
    const TaskFactory& find(const std::string& name) const {
        auto it = m_.find(name);
        if (it == m_.end()) throw std::runtime_error("unknown algorithm '" + name + "' (registered: " + names() + ")");
        return it->second;
    }
    std::string names() const { std::string s; for (auto& kv : m_) { if (!s.empty()) s += ", "; s += kv.first; } return s; }
    const std::map<std::string, TaskFactory>& all() const { return m_; }
};

// ── glue -----------------------------------------------------------------------
/// Fill a coordinator config from a task's server-side spec.
inline void configure(CoordinatorConfig& c, const ServerTaskInfo& t) {
    c.algo = t.spec.algo; c.param_count = t.spec.param_count; c.model_fingerprint = t.spec.fingerprint;
    c.shards = t.shards;
}
/// Fill a client config from a task's spec.
inline void configure(ClientConfig& c, const TaskSpec& s) {
    c.algo = s.algo; c.param_count = s.param_count; c.model_fingerprint = s.fingerprint;
}
/// Hooks that forward to a ClientTask (the task must outlive the returned hooks).
inline ClientHooks hooks_for(ClientTask& t) {
    ClientHooks h;
    h.train = [&t](uint64_t base) { RoundInfo r; r.base_round = base; return t.train(r); };
    h.get_params = [&t](uint64_t o, size_t n, float* out) { t.get_params(o, n, out); };
    h.set_params = [&t](uint64_t o, const float* d, size_t n) { t.set_params(o, d, n); };
    h.can_participate = [&t] { return t.can_participate(); };
    h.on_global = [&t](uint64_t r) { t.on_global(r); };
    h.wants_shard = [&t] { return t.wants_shard(); };
    h.on_shard = [&t](Shard&& sh) { t.on_shard(std::move(sh)); };
    return h;
}

}} // namespace tf::fed
