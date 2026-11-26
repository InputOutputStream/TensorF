#pragma once
// GraphShardProvider.hpp — server-side: capacity-aware partitioning + per-device shards.
//
//   enrolled devices (id, score, memory budget, what data they want)
//     -> CapacityPolicy -> target fractions            (who should get how much)
//     -> partitioner backend (metis | kaminpar | scotch | builtin), vertex weights = 1 + degree
//     -> enforce_caps (hard memory limit per device)
//     -> per device, a shard of the chosen KIND:
//          partition        only the nodes it owns (edges to other devices' nodes are dropped)
//          halo:K           owned nodes + their K-hop neighbourhood (context only: never trained on),
//                           so cross-device edges survive; K = number of GNN layers gives exact outputs
//          full             the whole graph; the device still trains/evaluates only its OWNED nodes
//        Each device asks (client --shard ...), the server may impose one (shard_mode=...), and a request
//        that does not fit the device's memory budget is downgraded (full -> halo -> partition) with the
//        reason recorded in the shard metadata.
//
// Options (TaskOptions): data, seed, split options (see FedGraphData),
//   partitioner=auto|metis|kaminpar|scotch|builtin   policy=capacity|equal   lo hi floor   eps=0.03
//   shard_mode=auto|partition|halo[:K]|full   (auto = honour each device's request; default request = partition)
//   vertex_weight=degree|uniform   usable=0.5   objective=cut|volume (METIS)   threads=1 (KaMinPar)
//
// Shards are built on demand (not cached): sending `full` to several devices of a big graph does not keep
// several copies in the server's memory beyond the transfers in flight.

#include "../FedShard.hpp"
#include "../FedCoordinator.hpp"
#include "../FedTask.hpp"
#include "GraphPartition.hpp"
#include "WeightedPartitioner.hpp"
#include "FedGraphData.hpp"
#include "NodeClassifierTask.hpp"

#include <chrono>
#include <map>
#include <mutex>

namespace tf { namespace fed { namespace graph {

struct ShardPlan { ShardPref mode = ShardPref::Partition; int hops = 0; size_t owned = 0, nodes = 0; std::string requested, reason; };

class GraphShardProvider : public ShardProvider {
public:
    explicit GraphShardProvider(const TaskOptions& o) : o_(o) {
        ds_ = std::make_unique<FedGraphData>(load_fed_graph(o)); split_ = ds_->split;
        log_ = [](const std::string& s) { std::printf("@@TFEVT %s\n", s.c_str()); std::fflush(stdout); };
    }
    void set_event_sink(std::function<void(const std::string&)> f) override { log_ = std::move(f); }

    void prepare(const std::vector<EnrollInfo>& enrolled) override {
        const auto& g = ds_->graph; const CSR<float>& adj = g.adjacency(); const size_t K = enrolled.size(), N = g.num_nodes();
        std::vector<DeviceCaps> dev(K);
        for (size_t i = 0; i < K; i++) { dev[i].id = enrolled[i].id; dev[i].score = enrolled[i].score; dev[i].mem_budget_mb = enrolled[i].mem_budget_mb; dev[i].platform = enrolled[i].platform; }
        CapacityPolicy pol; pol.lo = float(o_.f("lo", 30)); pol.hi = float(o_.f("hi", 70)); pol.floor_score = float(o_.f("floor", 5));
        std::vector<double> frac;
        if (o_.str("policy", "capacity") == "equal") frac.assign(K, 1.0 / double(K)); else frac = target_fractions(dev, pol);
        std::vector<double> vw = o_.str("vertex_weight", "degree") == "uniform" ? std::vector<double>(N, 1.0) : degree_weights(adj);
        const double eps = o_.f("eps", 0.03); const unsigned seed = unsigned(o_.i("seed", 42));
        Backend b = resolve_backend(parse_backend(o_.str("partitioner", "auto")));
        auto t0 = std::chrono::steady_clock::now();
        std::vector<int> part = partition_with(b, adj, vw, frac, eps, seed, o_.str("objective", "cut") == "volume", int(o_.i("threads", 1)));
        const double avg_deg = double(adj.col_idx.size()) / double(std::max<size_t>(1, N));
        const double bpn = bytes_per_node(g.feature_dim(), avg_deg, false);
        auto caps = node_caps(dev, bpn, o_.f("usable", 0.5));
        enforce_caps(adj, part, caps);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        PartitionStats st = evaluate(adj, part, vw, frac);

        // per-device shard plan: what it asked for, what the server allows, what fits in its memory
        std::vector<ShardPlan> plan(K); std::vector<std::vector<uint32_t>> sets(K);
        for (size_t i = 0; i < K; i++) {
            ShardPref want; int hops;
            if (!parse_mode(o_.str("shard_mode", "auto"), want, hops)) {
                want = enrolled[i].shard_pref == 0 ? ShardPref::Partition : ShardPref(enrolled[i].shard_pref); hops = enrolled[i].halo_hops;
            }
            ShardPlan& pl = plan[i]; pl.requested = mode_name(want, hops); pl.mode = want; pl.hops = hops;
            std::vector<uint32_t> owned; for (size_t u = 0; u < N; u++) if (part[u] == int(i)) owned.push_back(uint32_t(u));
            pl.owned = owned.size();
            for (;;) {                                                           // downgrade until it fits the memory budget
                size_t n = pl.mode == ShardPref::Full ? N : (pl.mode == ShardPref::Halo ? expand(adj, owned, part, pl.hops, sets[i]) : (sets[i] = owned).size());
                pl.nodes = n;
                if (caps[i] == UINT64_MAX || n <= caps[i]) break;
                if (pl.mode == ShardPref::Full) { pl.mode = ShardPref::Halo; pl.hops = 2; }
                else if (pl.mode == ShardPref::Halo && pl.hops > 1) pl.hops--;
                else { pl.mode = ShardPref::Partition; pl.hops = 0; }
                pl.reason = "memory budget";
            }
            if (pl.mode == ShardPref::Full) sets[i].clear();                      // all nodes: no list needed
        }
        { std::lock_guard<std::mutex> l(mu_); part_ = std::move(part); plan_ = plan; sets_ = std::move(sets); ids_.clear();
          for (size_t i = 0; i < K; i++) { ids_[dev[i].id] = i; targets_[dev[i].id] = frac[i]; scores_[dev[i].id] = dev[i].score; } K_ = K; }

        std::string devs;
        for (size_t i = 0; i < K; i++) {
            if (i) devs += ',';
            devs += J().kv("id", dev[i].id).kv("score", double(dev[i].score)).kv("target", frac[i]).kv("nodes", uint64_t(st.nodes[i]))
                       .kv("weight_ratio", st.target[i] > 0 ? st.weight[i] / st.target[i] : 0.0).kv("shard", mode_name(plan[i].mode, plan[i].hops))
                       .kv("shard_nodes", uint64_t(plan[i].nodes)).kv("requested", plan[i].requested).kv("reason", plan[i].reason).str();
        }
        std::string head = J().kv("type", "fed_partition").kv("partitioner", backend_name(b)).kv("policy", o_.str("policy", "capacity"))
                              .kv("cut", st.cut_fraction).kv("volume", uint64_t(st.comm_volume)).kv("max_imbalance", st.max_imbalance).kv("ms", ms).str();
        head.pop_back();
        log_(head + ",\"devices\":[" + devs + "]}");
        last_stats_ = st; last_backend_ = b; last_ms_ = ms;
    }

    std::shared_ptr<const Shard> shard_for(const std::string& id) override {
        size_t i; ShardPlan pl; std::vector<uint32_t> set; double target; float score;
        { std::lock_guard<std::mutex> l(mu_); auto it = ids_.find(id); if (it == ids_.end()) return nullptr;
          i = it->second; pl = plan_[i]; if (pl.mode != ShardPref::Full) set = sets_[i]; target = targets_[id]; score = scores_[id]; }
        return build_shard(int(i), pl, set, target, score);
    }
    const PartitionStats& stats() const { return last_stats_; }
    Backend backend() const { return last_backend_; }
    double partition_ms() const { return last_ms_; }
    const FedGraphData& dataset() const { return *ds_; }
    ShardPlan plan_of(const std::string& id) { std::lock_guard<std::mutex> l(mu_); return plan_[ids_.at(id)]; }

private:
    static bool parse_mode(const std::string& s, ShardPref& m, int& hops) {
        if (s == "auto" || s.empty()) return false;
        if (s == "partition") { m = ShardPref::Partition; hops = 0; return true; }
        if (s == "full") { m = ShardPref::Full; hops = 0; return true; }
        if (s.rfind("halo", 0) == 0) { m = ShardPref::Halo; hops = s.size() > 5 && s[4] == ':' ? std::atoi(s.c_str() + 5) : 2; if (hops < 1) hops = 1; return true; }
        throw std::invalid_argument("shard_mode must be auto|partition|halo[:K]|full");
    }
    static std::string mode_name(ShardPref m, int hops) { return m == ShardPref::Full ? "full" : m == ShardPref::Halo ? "halo:" + std::to_string(hops) : "partition"; }

    /// owned nodes + `hops` BFS layers of neighbours (any device's nodes); returns the set size
    static size_t expand(const CSR<float>& adj, const std::vector<uint32_t>& owned, const std::vector<int>&, int hops, std::vector<uint32_t>& out) {
        std::vector<char> in(adj.n, 0); out = owned; for (auto u : owned) in[u] = 1;
        size_t lo = 0;
        for (int h = 0; h < hops; h++) {
            size_t hi = out.size();
            for (size_t k = lo; k < hi; k++) { size_t u = out[k]; for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1]; e++) { size_t v = adj.col_idx[e]; if (!in[v]) { in[v] = 1; out.push_back(uint32_t(v)); } } }
            lo = hi;
        }
        std::sort(out.begin(), out.end()); return out.size();
    }

    std::shared_ptr<const Shard> build_shard(int p, const ShardPlan& pl, std::vector<uint32_t>& set, double target, float score) const {
        const auto& g = ds_->graph; const CSR<float>& adj = g.adjacency(); const size_t F = g.feature_dim(), N = g.num_nodes();
        if (pl.mode == ShardPref::Full) { set.resize(N); for (size_t u = 0; u < N; u++) set[u] = uint32_t(u); }
        std::vector<long long> local(N, -1); for (size_t i = 0; i < set.size(); i++) local[set[i]] = static_cast<long long>(i);
        auto sh = std::make_shared<Shard>();
        Blob x; x.name = "x"; x.kind = o_.b("shard_fp16", false) ? Blob::Kind::F16 : Blob::Kind::F32; x.f.resize(set.size() * F);
        Blob y; y.name = "y"; y.kind = Blob::Kind::U32; Blob fl; fl.name = "flags"; fl.kind = Blob::Kind::U32; Blob ed; ed.name = "edges"; ed.kind = Blob::Kind::U32;
        y.u.reserve(set.size()); fl.u.reserve(set.size());
        std::vector<int> part; { std::lock_guard<std::mutex> l(mu_); part = part_; }
        size_t owned = 0;
        for (size_t i = 0; i < set.size(); i++) {
            size_t u = set[i]; const bool own = part[u] == p;
            std::memcpy(&x.f[i * F], &g.features()[u * F], F * sizeof(float));
            y.u.push_back(uint32_t(g.labels()[u]));
            fl.u.push_back(own ? ((split_.train[u] ? 1u : 0u) | (split_.val[u] ? 2u : 0u) | (split_.test[u] ? 4u : 0u) | 8u) : 0u);
            owned += own;
            for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1]; e++) { size_t v = adj.col_idx[e]; if (v > u && local[v] >= 0) { ed.u.push_back(uint32_t(i)); ed.u.push_back(uint32_t(local[v])); } }
        }
        sh->set("nodes", (long long)set.size()); sh->set("owned", (long long)owned); sh->set("features", (long long)F); sh->set("classes", (long long)ds_->num_classes);
        sh->set("edges", (long long)(ed.u.size() / 2)); sh->set("part", (long long)p); sh->set("parts", (long long)K_);
        sh->set("mode", mode_name(pl.mode, pl.hops)); sh->set("requested", pl.requested); if (!pl.reason.empty()) sh->set("reason", pl.reason);
        sh->set("score", std::to_string(score)); sh->set("target", std::to_string(target));
        sh->blobs = {std::move(x), std::move(y), std::move(fl), std::move(ed)};
        return sh;
    }

    TaskOptions o_; std::unique_ptr<FedGraphData> ds_; Split split_;
    mutable std::mutex mu_; std::vector<int> part_; std::vector<ShardPlan> plan_; std::vector<std::vector<uint32_t>> sets_;
    std::map<std::string, size_t> ids_; std::map<std::string, double> targets_; std::map<std::string, float> scores_; size_t K_ = 0;
    std::function<void(const std::string&)> log_; PartitionStats last_stats_; Backend last_backend_ = Backend::Builtin; double last_ms_ = 0;
};

}}} // namespace tf::fed::graph
