#pragma once
#include "PartitionerPrelude.hpp"   // must come before DataStructures/Graph.hpp (KaMinPar name clash)
// WeightedPartitioner.hpp — heterogeneity-aware k-way graph partitioning.
//
//   * partition_builtin(): dependency-free fallback. Target-aware multi-source region growing
//     followed by greedy boundary refinement (the label-propagation/FM-lite family used as
//     refinement in KaMinPar/KaHIP/METIS) — decent, NOT state of the art.
//   * partition_metis():   METIS 5 multilevel k-way (Karypis & Kumar 1998) through its tpwgts/ubvec API.
//                          Only compiled with -DTENSORF_WITH_METIS (link -lmetis).
//   * enforce_caps():      hard per-device limits (METIS/ubvec only give tolerances).
//   * evaluate():          sizes vs targets, edge cut, communication volume, imbalance.
//
// All take per-vertex compute weights (default 1 + degree: aggregation cost of a GNN layer is
// proportional to the number of neighbours) and per-part TARGET FRACTIONS of the total weight.

#include "DataStructures/Graph.hpp"
#include "CapacityPolicy.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>



namespace tf { namespace fed { namespace graph {

struct PartitionStats {
    std::vector<size_t> nodes;            // per part
    std::vector<double> weight, target;   // achieved / requested weight per part
    double max_imbalance = 0;             // max_i weight_i / target_i
    double cut_fraction = 0;              // cut adjacency entries / all
    size_t comm_volume = 0;               // sum over nodes of (#foreign parts among neighbours) — what a halo exchange would ship
};

template <typename T>
std::vector<double> degree_weights(const CSR<T>& adj, double cap_factor = 8.0) {
    std::vector<double> w(adj.n);
    for (size_t u = 0; u < adj.n; u++) {
        size_t d = 0; for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1]; e++) d += size_t(adj.col_idx[e]) != u;
        w[u] = 1.0 + double(d);
    }
    // METIS degrades with very large vertex-weight ratios (hubs): clamp relative to the median.
    std::vector<double> s = w; std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
    double cap = std::max(1.0, s[s.size() / 2]) * cap_factor;
    for (auto& x : w) x = std::min(x, cap);
    return w;
}

template <typename T>
PartitionStats evaluate(const CSR<T>& adj, const std::vector<int>& part, const std::vector<double>& vw, const std::vector<double>& frac) {
    const size_t K = frac.size(); PartitionStats s; s.nodes.assign(K, 0); s.weight.assign(K, 0); s.target.assign(K, 0);
    double total = std::accumulate(vw.begin(), vw.end(), 0.0);
    for (size_t k = 0; k < K; k++) s.target[k] = frac[k] * total;
    size_t cut = 0, all = 0; std::vector<int> seen(K, -1);
    for (size_t u = 0; u < adj.n; u++) {
        s.nodes[part[u]]++; s.weight[part[u]] += vw[u];
        for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1]; e++) {
            size_t v = adj.col_idx[e]; if (v == u) continue;
            all++; if (part[v] != part[u]) { cut++; if (seen[part[v]] != int(u)) { seen[part[v]] = int(u); s.comm_volume++; } }
        }
    }
    s.cut_fraction = all ? double(cut) / double(all) : 0;
    for (size_t k = 0; k < K; k++) s.max_imbalance = std::max(s.max_imbalance, s.target[k] > 0 ? s.weight[k] / s.target[k] : 0.0);
    return s;
}

/// Move boundary nodes (best cut gain first) out of parts whose NODE COUNT exceeds caps[i] into parts with room.
/// Throws if the caps cannot hold the graph.
template <typename T>
void enforce_caps(const CSR<T>& adj, std::vector<int>& part, const std::vector<uint64_t>& caps) {
    const size_t K = caps.size(); std::vector<uint64_t> cnt(K, 0);
    for (int p : part) cnt[p]++;
    double room = 0; for (auto c : caps) room += (c == UINT64_MAX ? double(adj.n) : double(c));
    if (room < double(adj.n)) throw std::runtime_error("memory caps of the enrolled devices cannot hold the graph (" + std::to_string(adj.n) + " nodes)");
    for (size_t k = 0; k < K; k++) {
        while (cnt[k] > caps[k]) {
            // best candidate: node of part k with the most neighbours in some part that has room
            long long bv = -1; int bp = -1; double bg = -1e18;
            for (size_t u = 0; u < adj.n; u++) {
                if (part[u] != int(k)) continue;
                std::vector<int> c(K, 0); int own = 0;
                for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1]; e++) { size_t v = adj.col_idx[e]; if (v == u) continue; if (part[v] == int(k)) own++; else c[part[v]]++; }
                for (size_t q = 0; q < K; q++) if (q != k && cnt[q] < caps[q] && double(c[q] - own) > bg) { bg = double(c[q] - own); bv = static_cast<long long>(u); bp = int(q); }
            }
            if (bv < 0) throw std::runtime_error("enforce_caps: no part has room");
            part[bv] = bp; cnt[k]--; cnt[bp]++;
        }
    }
}

/// Greedy boundary refinement: move a node to the neighbouring part with the best positive cut gain,
/// as long as the destination stays under ub and the source above lb (weights relative to targets).
template <typename T>
void refine_greedy(const CSR<T>& adj, std::vector<int>& part, const std::vector<double>& vw, const std::vector<double>& target,
                   double eps, int passes, unsigned seed) {
    const size_t K = target.size(), N = adj.n; std::vector<double> w(K, 0.0);
    for (size_t u = 0; u < N; u++) w[part[u]] += vw[u];
    std::mt19937 rng(seed + 13); std::vector<size_t> order(N); std::iota(order.begin(), order.end(), 0);
    std::vector<int> cnt(K);
    for (int pass = 0; pass < passes; pass++) {
        std::shuffle(order.begin(), order.end(), rng); size_t moved = 0;
        for (size_t u : order) {
            int a = part[u]; std::fill(cnt.begin(), cnt.end(), 0); bool boundary = false;
            for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1]; e++) { size_t v = adj.col_idx[e]; if (v == u) continue; cnt[part[v]]++; if (part[v] != a) boundary = true; }
            if (!boundary) continue;
            int best = a; int bgain = 0;
            for (size_t b = 0; b < K; b++) {
                if (int(b) == a || cnt[b] == 0) continue;
                int gain = cnt[b] - cnt[a];
                bool fits = w[b] + vw[u] <= target[b] * (1 + eps) && w[a] - vw[u] >= target[a] * (1 - eps);
                bool balances = gain == 0 && (w[b] + vw[u]) / target[b] < (w[a]) / target[a] * 0.98 && w[b] + vw[u] <= target[b] * (1 + eps);
                if (((gain > bgain) && fits) || (gain == 0 && balances && best == a && bgain == 0)) { best = int(b); bgain = std::max(gain, 0); }
            }
            if (best != a) { part[u] = best; w[a] -= vw[u]; w[best] += vw[u]; moved++; }
        }
        if (moved == 0) break;
    }
}

/// Dependency-free heterogeneity-aware partitioner (region growing + greedy refinement).
template <typename T>
std::vector<int> partition_builtin(const CSR<T>& adj, const std::vector<double>& vw, const std::vector<double>& frac,
                                   double eps = 0.03, unsigned seed = 42, int refine_passes = 10) {
    const size_t N = adj.n, K = frac.size();
    if (K == 0) throw std::invalid_argument("partition_builtin: K == 0");
    std::vector<int> part(N, -1); if (K == 1) { std::fill(part.begin(), part.end(), 0); return part; }
    double total = std::accumulate(vw.begin(), vw.end(), 0.0);
    std::vector<double> target(K), ub(K), w(K, 0.0); double tmin = 1e300;
    for (size_t k = 0; k < K; k++) { target[k] = frac[k] * total; tmin = std::min(tmin, target[k]); }
    double vmax = *std::max_element(vw.begin(), vw.end());
    for (size_t k = 0; k < K; k++) ub[k] = std::max(target[k] * (1 + eps), target[k] + vmax);
    std::mt19937 rng(seed); std::vector<size_t> order(N); std::iota(order.begin(), order.end(), 0); std::shuffle(order.begin(), order.end(), rng);
    std::vector<std::deque<size_t>> q(K); size_t next = 0, assigned = 0;
    auto claim = [&](size_t v, size_t p) { part[v] = int(p); w[p] += vw[v]; assigned++; q[p].push_back(v); };
    auto new_seed = [&](size_t p) { while (next < N && part[order[next]] >= 0) next++; if (next < N) claim(order[next], p); };
    for (size_t p = 0; p < K; p++) new_seed(p);
    std::vector<size_t> burst(K); for (size_t k = 0; k < K; k++) burst[k] = std::max<size_t>(1, size_t(std::lround(target[k] / tmin)));
    while (assigned < N) {
        bool progressed = false;
        for (size_t p = 0; p < K; p++) {
            if (w[p] >= target[p]) continue;
            for (size_t b = 0; b < burst[p] && w[p] < target[p]; b++) {
                if (q[p].empty()) { new_seed(p); progressed = true; if (q[p].empty()) break; continue; }
                size_t u = q[p].front(); q[p].pop_front(); progressed = true;
                for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1] && w[p] < ub[p]; e++) { size_t v = adj.col_idx[e]; if (part[v] < 0) claim(v, p); }
            }
        }
        if (!progressed) break;
    }
    for (size_t v = 0; v < N; v++) if (part[v] < 0) {          // leftovers (rounding): most deficient part among neighbours, else globally
        std::vector<int> c(K, 0); for (size_t e = adj.row_ptr[v]; e < adj.row_ptr[v + 1]; e++) { int pv = part[adj.col_idx[e]]; if (pv >= 0) c[pv]++; }
        size_t bp = 0; double bs = -1e300;
        for (size_t k = 0; k < K; k++) { double s = (target[k] - w[k]) / target[k] + 0.05 * c[k]; if (s > bs) { bs = s; bp = k; } }
        part[v] = int(bp); w[bp] += vw[v];
    }
    refine_greedy(adj, part, vw, target, eps, refine_passes, seed);
    return part;
}

#ifdef TENSORF_WITH_METIS
/// METIS multilevel k-way with heterogeneous target weights. Objective: edge cut (or communication volume).
template <typename T>
std::vector<int> partition_metis(const CSR<T>& adj, const std::vector<double>& vw, const std::vector<double>& frac,
                                 double eps = 0.03, unsigned seed = 42, bool minimize_volume = false, int ncuts = 4) {
    const size_t N = adj.n, K = frac.size();
    std::vector<int> part(N, 0); if (K == 1) return part;
    std::vector<idx_t> xadj(N + 1, 0), adjncy; adjncy.reserve(adj.col_idx.size());
    for (size_t u = 0; u < N; u++) { for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1]; e++) if (size_t(adj.col_idx[e]) != u) adjncy.push_back(idx_t(adj.col_idx[e])); xadj[u + 1] = idx_t(adjncy.size()); }
    double wmin = *std::min_element(vw.begin(), vw.end());
    std::vector<idx_t> vwgt(N); for (size_t u = 0; u < N; u++) vwgt[u] = std::max<idx_t>(1, idx_t(std::lround(vw[u] / wmin)));
    std::vector<real_t> tp(K); for (size_t k = 0; k < K; k++) tp[k] = real_t(frac[k]);
    real_t ub = real_t(1.0 + eps); idx_t nv = idx_t(N), ncon = 1, nparts = idx_t(K), objval = 0;
    idx_t opts[METIS_NOPTIONS]; METIS_SetDefaultOptions(opts);
    opts[METIS_OPTION_SEED] = idx_t(seed); opts[METIS_OPTION_NCUTS] = ncuts;
    opts[METIS_OPTION_OBJTYPE] = minimize_volume ? METIS_OBJTYPE_VOL : METIS_OBJTYPE_CUT;
    std::vector<idx_t> out(N);
    int rc = METIS_PartGraphKway(&nv, &ncon, xadj.data(), adjncy.data(), vwgt.data(), nullptr, nullptr, &nparts, tp.data(), &ub, opts, &objval, out.data());
    if (rc != METIS_OK) throw std::runtime_error("METIS_PartGraphKway failed (code " + std::to_string(rc) + ")");
    for (size_t u = 0; u < N; u++) part[u] = int(out[u]);
    return part;
}
inline constexpr bool has_metis() { return true; }
#else
inline constexpr bool has_metis() { return false; }
#endif


// ── Scotch: map the weighted graph onto a WEIGHTED COMPLETE target architecture (one vertex per device,
//    weight = its capacity). This is Scotch's native way to express heterogeneous processors. ─────────────
#ifdef TENSORF_WITH_SCOTCH
template <typename T>
std::vector<int> partition_scotch(const CSR<T>& adj, const std::vector<double>& vw, const std::vector<double>& frac,
                                  double eps = 0.03, unsigned seed = 42) {
    const size_t N = adj.n, K = frac.size(); std::vector<int> part(N, 0); if (K == 1) return part;
    std::vector<SCOTCH_Num> verttab(N + 1, 0), edgetab; edgetab.reserve(adj.col_idx.size());
    for (size_t u = 0; u < N; u++) { for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1]; e++) if (size_t(adj.col_idx[e]) != u) edgetab.push_back(SCOTCH_Num(adj.col_idx[e])); verttab[u + 1] = SCOTCH_Num(edgetab.size()); }
    double wmin = *std::min_element(vw.begin(), vw.end());
    std::vector<SCOTCH_Num> velo(N); for (size_t u = 0; u < N; u++) velo[u] = std::max<SCOTCH_Num>(1, SCOTCH_Num(std::lround(vw[u] / wmin)));
    std::vector<SCOTCH_Num> cap(K); for (size_t k = 0; k < K; k++) cap[k] = std::max<SCOTCH_Num>(1, SCOTCH_Num(std::lround(frac[k] * 10000.0)));
    std::vector<SCOTCH_Num> out(N);
    SCOTCH_randomSeed(SCOTCH_Num(seed)); SCOTCH_randomReset();
    SCOTCH_Graph g; SCOTCH_Arch a; SCOTCH_Strat st; int rc = 0;
    SCOTCH_graphInit(&g); SCOTCH_archInit(&a); SCOTCH_stratInit(&st);
    if (SCOTCH_graphBuild(&g, 0, SCOTCH_Num(N), verttab.data(), nullptr, velo.data(), nullptr, SCOTCH_Num(edgetab.size()), edgetab.data(), nullptr) != 0) rc = 1;
    else if (SCOTCH_archCmpltw(&a, SCOTCH_Num(K), cap.data()) != 0) rc = 2;
    else if (SCOTCH_stratGraphMapBuild(&st, SCOTCH_STRATQUALITY, SCOTCH_Num(K), eps) != 0) rc = 3;
    else if (SCOTCH_graphMap(&g, &a, &st, out.data()) != 0) rc = 4;
    SCOTCH_stratExit(&st); SCOTCH_archExit(&a); SCOTCH_graphExit(&g);
    if (rc) throw std::runtime_error("Scotch failed at step " + std::to_string(rc));
    for (size_t u = 0; u < N; u++) part[u] = int(out[u]);
    return part;
}
inline constexpr bool has_scotch() { return true; }
#else
inline constexpr bool has_scotch() { return false; }
#endif

// ── KaMinPar (MIT): deep multilevel, shared memory. Heterogeneous targets via relative max block weights. ──
#ifdef TENSORF_WITH_KAMINPAR
template <typename T>
std::vector<int> partition_kaminpar(const CSR<T>& adj, const std::vector<double>& vw, const std::vector<double>& frac,
                                    double eps = 0.03, unsigned seed = 42, int threads = 1) {
    using namespace kaminpar;
    const size_t N = adj.n, K = frac.size(); std::vector<int> part(N, 0); if (K == 1) return part;
    std::vector<shm::EdgeID> xadj(N + 1, 0); std::vector<shm::NodeID> adjncy; adjncy.reserve(adj.col_idx.size());
    for (size_t u = 0; u < N; u++) { for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1]; e++) if (size_t(adj.col_idx[e]) != u) adjncy.push_back(shm::NodeID(adj.col_idx[e])); xadj[u + 1] = shm::EdgeID(adjncy.size()); }
    double wmin = *std::min_element(vw.begin(), vw.end());
    std::vector<shm::NodeWeight> vwgt(N); for (size_t u = 0; u < N; u++) vwgt[u] = std::max<shm::NodeWeight>(1, shm::NodeWeight(std::lround(vw[u] / wmin)));
    KaMinPar::reseed(int(seed));
    KaMinPar p(threads, shm::create_default_context()); p.set_output_level(OutputLevel::QUIET);
    p.copy_graph(xadj, adjncy, vwgt);
    // KaMinPar only bounds blocks from above by default, which can leave the smallest device under-loaded:
    // also give it lower bounds so every device gets at least its share minus the tolerance.
    std::vector<double> rel(K), rmin(K); for (size_t k = 0; k < K; k++) { rel[k] = frac[k] * (1.0 + eps); rmin[k] = frac[k] * (1.0 - eps); }
    p.set_k(shm::BlockID(K)); p.set_relative_max_block_weights(rel); p.set_relative_min_block_weights(rmin);
    std::vector<shm::BlockID> out(N); p.compute_partition(out);
    for (size_t u = 0; u < N; u++) part[u] = int(out[u]);
    return part;
}
inline constexpr bool has_kaminpar() { return true; }
#else
inline constexpr bool has_kaminpar() { return false; }
#endif

// ── one entry point for all backends ----------------------------------------------------------------------
enum class Backend { Auto, Builtin, Metis, KaMinPar, Scotch };
inline const char* backend_name(Backend b) { switch (b) { case Backend::Auto: return "auto"; case Backend::Builtin: return "builtin"; case Backend::Metis: return "metis"; case Backend::KaMinPar: return "kaminpar"; case Backend::Scotch: return "scotch"; } return "?"; }
inline Backend parse_backend(const std::string& s) {
    for (Backend b : {Backend::Auto, Backend::Builtin, Backend::Metis, Backend::KaMinPar, Backend::Scotch}) if (s == backend_name(b)) return b;
    throw std::invalid_argument("partitioner must be one of: auto, builtin, metis, kaminpar, scotch (got '" + s + "')");
}
inline bool backend_available(Backend b) { switch (b) { case Backend::Auto: case Backend::Builtin: return true; case Backend::Metis: return has_metis(); case Backend::KaMinPar: return has_kaminpar(); case Backend::Scotch: return has_scotch(); } return false; }
inline std::vector<Backend> available_backends() { std::vector<Backend> v; for (Backend b : {Backend::Builtin, Backend::Metis, Backend::KaMinPar, Backend::Scotch}) if (backend_available(b)) v.push_back(b); return v; }
/// Auto = best measured available backend: METIS, then Scotch, then KaMinPar, then the builtin fallback.
inline Backend resolve_backend(Backend b) {
    if (b != Backend::Auto) return b;
    for (Backend c : {Backend::Metis, Backend::Scotch, Backend::KaMinPar}) if (backend_available(c)) return c;
    return Backend::Builtin;
}

template <typename T>
std::vector<int> partition_with(Backend b, const CSR<T>& adj, const std::vector<double>& vw, const std::vector<double>& frac,
                                double eps = 0.03, unsigned seed = 42, bool minimize_volume = false, int threads = 1) {
    b = resolve_backend(b);
    if (!backend_available(b)) {
        std::string have; for (Backend x : available_backends()) { if (!have.empty()) have += ", "; have += backend_name(x); }
        throw std::runtime_error(std::string("partitioner '") + backend_name(b) + "' is not compiled in this build (available: " + have + ")");
    }
    switch (b) {
        case Backend::Builtin: return partition_builtin(adj, vw, frac, eps, seed);
#ifdef TENSORF_WITH_METIS
        case Backend::Metis: return partition_metis(adj, vw, frac, eps, seed, minimize_volume);
#endif
#ifdef TENSORF_WITH_SCOTCH
        case Backend::Scotch: return partition_scotch(adj, vw, frac, eps, seed);
#endif
#ifdef TENSORF_WITH_KAMINPAR
        case Backend::KaMinPar: return partition_kaminpar(adj, vw, frac, eps, seed, threads);
#endif
        default: break;
    }
    (void)minimize_volume; (void)threads;
    throw std::logic_error("unreachable partitioner");
}

}}} // namespace tf::fed::graph
