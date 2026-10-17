// AlgorithmProfiler.hpp
//
// The swappable part of the stack. An algorithm (CNN, GNN, RNN, Markov chain,
// Transformer, particle simulation, or your own) is described as a list of
// LayerCost entries plus the tunables it is willing to have changed. Nothing in
// the cost model, the memory model or the advisor knows which algorithm it is
// looking at: changing the subject from CNN to GNN means constructing a
// different AlgorithmModel.
//
//   LayerCost            one layer/kernel invocation: flops, bytes moved, irregular
//                        gather bytes, dependent loads, sequential depth, parallel
//                        width, retained activation bytes, parameters, dispatch count
//   builders             dense, conv2d, attention, ffn, spmm_aggregate, embedding,
//                        recurrent_cell, elementwise, pairwise, markov_walk
//   AlgorithmModel       tunables (with the range the algorithm declared and whether
//                        changing it changes the maths), microbatch + accumulation +
//                        checkpointing handled once for every algorithm, memory
//                        estimate by category
//   DeviceView/CostModel prices LayerCosts against the CapabilityProfile measured
//                        by DeviceBenchmark: GEMM rate by shape, conv, recurrent,
//                        pairwise, gather bandwidth, dependent-load latency by
//                        working set, per-thread rates of big.LITTLE cores,
//                        fork-join overhead, sustained (throttled) fraction
//   AlgorithmProfiler    model + capability + memory budget: predict time, memory,
//                        fit; calibrate against a real instrumented step
//
// What the cost model assumes (and what calibrate() corrects)
//   * Low-precision storage (F16/FP8/FP4) is dequantised on the fly and computed in
//     f32: it changes bytes moved and memory, not the arithmetic rate.
//   * Layers run one after another; each layer is parallel over its `width`.
//   * Time = max(compute, regular memory) + irregular gather + dependent loads +
//     overhead. A layer that overlaps better than that is predicted slow, never fast.
//   * One global scale factor and a spread are fitted by calibrate(); the advisor
//     reports the spread as the prediction's error.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DeviceBenchmark.hpp"
#include "MemoryProfiler.hpp"

namespace tensorf::profile {

// ───────────────────────────── layer costs ─────────────────────────────

enum class Kernel { Gemm, Conv, Gather, Recurrent, Pairwise, Stream, Chase };

inline const char* to_string(Kernel k) {
    switch (k) {
        case Kernel::Gemm: return "gemm";
        case Kernel::Conv: return "conv";
        case Kernel::Gather: return "gather";
        case Kernel::Recurrent: return "recurrent";
        case Kernel::Pairwise: return "pairwise";
        case Kernel::Stream: return "stream";
        case Kernel::Chase: return "chase";
    }
    return "?";
}

struct LayerCost {
    std::string name;
    Kernel kernel = Kernel::Stream;
    double flops_fwd = 0, flops_bwd = 0;
    double bytes_fwd = 0, bytes_bwd = 0;               // regular (streaming) traffic
    double gather_bytes_fwd = 0, gather_bytes_bwd = 0; // irregular row gathers / scatters
    double interactions = 0;                           // pairwise kernel: pair evaluations
    double dep_loads = 0;                              // chase kernel: dependent loads
    double gemm_m = 0, gemm_n = 0, gemm_k = 0;         // dominant matrix shape (Gemm/Recurrent)
    double width = 1;                                  // independent work items: bounds useful threads
    double seq_depth = 1;                              // steps that cannot overlap (time steps)
    double working_set = 0;                            // bytes touched repeatedly: picks the memory level
    double dispatches = 1;                             // kernel launches (each is a fork-join when parallel)
    double params = 0;                                 // trainable (or resident) element count
    double act_bytes = 0;                              // retained for backward
    double scratch_bytes = 0;
    int repeat = 1;                                    // e.g. gradient-accumulation passes
};

namespace layers {

inline double elt_bytes(Dtype d) { return double(tensor_bytes(d, 1024)) / 1024.0; }

inline LayerCost dense(const std::string& name, double rows, double in, double out, Dtype dt, bool backward = true) {
    LayerCost c;
    c.name = name; c.kernel = Kernel::Gemm;
    const double e = elt_bytes(dt);
    c.flops_fwd = 2.0 * rows * in * out;
    c.flops_bwd = backward ? 4.0 * rows * in * out : 0.0;
    c.bytes_fwd = (rows * in + in * out + rows * out) * e;
    c.bytes_bwd = backward ? 2.0 * c.bytes_fwd : 0.0;
    c.gemm_m = rows; c.gemm_n = out; c.gemm_k = in;
    c.width = rows;
    c.working_set = (in * out + rows * (in + out)) * e;
    c.dispatches = backward ? 3 : 1;
    c.params = in * out + out;
    c.act_bytes = backward ? rows * out * e : 0.0;
    c.scratch_bytes = rows * std::max(in, out) * 4.0;
    return c;
}

inline LayerCost conv2d(const std::string& name, double rows, double cin, double cout, double k, double h, double w, double stride,
                        Dtype dt, bool backward = true) {
    LayerCost c;
    c.name = name; c.kernel = Kernel::Conv;
    const double e = elt_bytes(dt), ho = std::ceil(h / stride), wo = std::ceil(w / stride);
    c.flops_fwd = 2.0 * rows * cout * cin * k * k * ho * wo;
    c.flops_bwd = backward ? 2.0 * c.flops_fwd : 0.0;
    c.bytes_fwd = (rows * cin * h * w + cout * cin * k * k + rows * cout * ho * wo) * e;
    c.bytes_bwd = backward ? 2.0 * c.bytes_fwd : 0.0;
    c.width = rows * cout;
    c.working_set = (cin * h * w + cout * ho * wo) * e + cout * cin * k * k * e;
    c.dispatches = backward ? 3 : 1;
    c.params = cout * cin * k * k + cout;
    c.act_bytes = backward ? rows * cout * ho * wo * e : 0.0;
    c.scratch_bytes = rows * cout * ho * wo * 4.0;
    return c;
}

// Multi-head self attention over `rows` sequences: projections + score/value matmuls.
inline std::vector<LayerCost> attention(const std::string& name, double rows, double seq, double dim, double heads, Dtype dt,
                                        bool backward = true) {
    std::vector<LayerCost> out;
    out.push_back(dense(name + ".qkvo", rows * seq, dim, 4.0 * dim, dt, backward));      // 4 projections as one wide matmul
    LayerCost s;
    s.name = name + ".scores"; s.kernel = Kernel::Gemm;
    const double e = elt_bytes(dt), dh = dim / std::max(1.0, heads);
    s.flops_fwd = 4.0 * rows * seq * seq * dim;                                          // QK^T and AV
    s.flops_bwd = backward ? 2.0 * s.flops_fwd : 0.0;
    s.bytes_fwd = (2.0 * rows * seq * dim + 2.0 * rows * heads * seq * seq) * e;        // scores written and re-read
    s.bytes_bwd = backward ? 2.0 * s.bytes_fwd : 0.0;
    s.gemm_m = seq; s.gemm_n = seq; s.gemm_k = dh;
    s.width = rows * heads;
    s.working_set = (rows * heads * seq * seq) * e;
    s.dispatches = backward ? 6 : 2;
    s.act_bytes = backward ? rows * heads * seq * seq * e : 0.0;
    s.scratch_bytes = rows * heads * seq * seq * 4.0;
    out.push_back(s);
    return out;
}

inline LayerCost elementwise(const std::string& name, double n, Dtype dt, double passes = 2.0, bool backward = true) {
    LayerCost c;
    c.name = name; c.kernel = Kernel::Stream;
    const double e = elt_bytes(dt);
    c.flops_fwd = n * passes;
    c.flops_bwd = backward ? n * passes : 0.0;
    c.bytes_fwd = 2.0 * n * e * (passes / 2.0 > 1 ? passes / 2.0 : 1.0);
    c.bytes_bwd = backward ? 3.0 * n * e : 0.0;
    c.width = std::max(1.0, n / 4096.0);
    c.working_set = n * e;
    c.dispatches = backward ? 2 : 1;
    c.act_bytes = backward ? n * e : 0.0;
    return c;
}

// Neighbour aggregation: out[i] = sum_j x[j] over sampled edges (SpMM / message passing).
inline LayerCost spmm_aggregate(const std::string& name, double nodes_out, double edges, double nodes_in, double feat, Dtype dt,
                                bool backward = true) {
    LayerCost c;
    c.name = name; c.kernel = Kernel::Gather;
    const double e = elt_bytes(dt);
    c.flops_fwd = edges * feat;
    c.flops_bwd = backward ? edges * feat : 0.0;
    c.gather_bytes_fwd = edges * feat * e;
    c.gather_bytes_bwd = backward ? edges * feat * e : 0.0;             // scatter-add of the gradient
    c.bytes_fwd = nodes_out * feat * e;
    c.bytes_bwd = backward ? nodes_out * feat * e : 0.0;
    c.width = nodes_out;
    c.working_set = nodes_in * feat * e;
    c.dispatches = backward ? 2 : 1;
    c.act_bytes = backward ? nodes_out * feat * e : 0.0;
    return c;
}

inline LayerCost embedding(const std::string& name, double tokens, double dim, double vocab, Dtype dt, bool backward = true) {
    LayerCost c;
    c.name = name; c.kernel = Kernel::Gather;
    const double e = elt_bytes(dt);
    c.gather_bytes_fwd = tokens * dim * e;
    c.gather_bytes_bwd = backward ? tokens * dim * e : 0.0;
    c.flops_fwd = tokens * dim; c.flops_bwd = backward ? tokens * dim : 0.0;
    c.width = tokens;
    c.working_set = vocab * dim * e;
    c.dispatches = backward ? 2 : 1;
    c.params = vocab * dim;
    c.act_bytes = backward ? tokens * dim * e : 0.0;
    return c;
}

// Recurrent layer over `steps` time steps. gates: 1 RNN, 3 GRU, 4 LSTM.
inline LayerCost recurrent_cell(const std::string& name, double batch, double in, double hidden, double steps, double gates, Dtype dt,
                                bool backward = true) {
    LayerCost c;
    c.name = name; c.kernel = Kernel::Recurrent;
    const double e = elt_bytes(dt);
    c.flops_fwd = steps * 2.0 * batch * (in + hidden) * hidden * gates;
    c.flops_bwd = backward ? 2.0 * c.flops_fwd : 0.0;
    c.bytes_fwd = steps * ((in + hidden) * hidden * gates + batch * (in + 2.0 * hidden * gates)) * e;
    c.bytes_bwd = backward ? 2.0 * c.bytes_fwd : 0.0;
    c.gemm_m = batch; c.gemm_n = gates * hidden; c.gemm_k = in + hidden;
    c.width = batch;
    c.seq_depth = steps;
    c.working_set = ((in + hidden) * hidden * gates + batch * hidden * gates) * e;
    c.dispatches = steps * (backward ? 3.0 : 1.0);
    c.params = (in + hidden) * hidden * gates + hidden * gates;
    c.act_bytes = backward ? steps * batch * hidden * (gates + 1.0) * e : 0.0;
    c.scratch_bytes = batch * hidden * gates * 4.0;
    return c;
}

// Short-range pair forces with neighbour lists (particle simulation, forward only).
inline LayerCost pairwise(const std::string& name, double particles, double neighbors, double rebuild_every, Dtype dt) {
    LayerCost c;
    c.name = name; c.kernel = Kernel::Pairwise;
    const double e = elt_bytes(dt);
    c.interactions = particles * neighbors;
    c.flops_fwd = 20.0 * c.interactions;
    c.bytes_fwd = particles * 9.0 * e + particles * neighbors * 4.0 / std::max(1.0, rebuild_every);   // pos+vel+force, amortised list rebuild
    c.width = particles;
    c.working_set = particles * 9.0 * e;
    c.dispatches = 2;
    c.params = particles * 6.0;                                                // positions and velocities are the state
    c.scratch_bytes = particles * neighbors * 4.0;                             // neighbour list
    return c;
}

// Independent random walks over a transition table: each step is one dependent load.
inline LayerCost markov_walk(const std::string& name, double walkers, double steps, double table_bytes) {
    LayerCost c;
    c.name = name; c.kernel = Kernel::Chase;
    c.dep_loads = walkers * steps;
    c.flops_fwd = walkers * steps * 4.0;
    c.bytes_fwd = walkers * 16.0;
    c.width = walkers;
    c.seq_depth = steps;
    c.working_set = table_bytes;
    c.dispatches = 1;
    c.params = table_bytes / 4.0;
    c.scratch_bytes = walkers * 16.0;
    return c;
}

}  // namespace layers

// ───────────────────────────── algorithm interface ─────────────────────────────

using Config = std::map<std::string, double>;

inline double cfg_get(const Config& c, const std::string& k, double def) {
    auto it = c.find(k);
    return it == c.end() ? def : it->second;
}

struct Tunable {
    std::string name;
    std::vector<double> values;       // the range the algorithm declared (ascending)
    double def = 0;
    // true: changing it changes the maths (precision, fanout, truncation length, cutoff, batch size).
    // The advisor only moves these inside `values`, and only for algorithms that listed them.
    // false: pure execution choice (microbatch, checkpointing): results are unchanged.
    bool semantic = false;
    std::string note;
};

struct StepCosts {
    std::vector<LayerCost> layers;
    int64_t batch = 1;
    int64_t microbatch = 1;
    int accum = 1;
    bool checkpoint = false;
    Dtype dtype = Dtype::F32;
    MemoryEstimate memory;
    double units = 1;                 // work units finished per step
};

class AlgorithmModel {
public:
    virtual ~AlgorithmModel() = default;
    virtual std::string name() const = 0;
    virtual std::string unit_name() const = 0;                       // "samples", "target nodes", "tokens", ...
    virtual std::vector<Tunable> tunables() const = 0;
    // Layers for ONE microbatch of `mb` units. Backward costs are zero for forward-only models.
    virtual std::vector<LayerCost> microbatch_layers(const Config& c, int64_t mb) const = 0;

    virtual bool training() const { return true; }
    virtual OptimizerKind optimizer() const { return OptimizerKind::Adam; }
    virtual double units_per_step(const Config& c) const { return cfg_get(c, "batch", 1); }
    virtual uint64_t data_bytes(const Config&, int64_t) const { return 0; }
    virtual bool valid(const Config&) const { return true; }

    Config defaults() const {
        Config c;
        for (const auto& t : tunables()) c[t.name] = t.def;
        return c;
    }
    std::optional<Tunable> find_tunable(const std::string& n) const {
        for (auto& t : tunables()) if (t.name == n) return t;
        return std::nullopt;
    }

    // Applies microbatching, gradient accumulation and activation checkpointing once, for every algorithm.
    StepCosts costs(const Config& c) const {
        StepCosts sc;
        sc.dtype = Dtype(int(cfg_get(c, "dtype", double(int(Dtype::F32)))));
        sc.batch = std::max<int64_t>(1, int64_t(cfg_get(c, "batch", 1)));
        sc.microbatch = std::clamp<int64_t>(int64_t(cfg_get(c, "microbatch", double(sc.batch))), 1, sc.batch);
        sc.accum = int((sc.batch + sc.microbatch - 1) / sc.microbatch);
        sc.units = units_per_step(c);
        sc.layers = microbatch_layers(c, sc.microbatch);
        sc.checkpoint = training() && cfg_get(c, "checkpoint", 0) > 0 && sc.layers.size() >= 3;
        for (auto& l : sc.layers) {
            l.repeat = sc.accum;
            if (sc.checkpoint) { l.flops_bwd += l.flops_fwd; l.bytes_bwd += l.bytes_fwd; l.gather_bytes_bwd += l.gather_bytes_fwd; }
        }
        double params = 0, acts = 0, max_act = 0, scratch = 0;
        for (const auto& l : sc.layers) {
            params += l.params; acts += l.act_bytes; max_act = std::max(max_act, l.act_bytes); scratch = std::max(scratch, l.scratch_bytes);
        }
        MemoryEstimate& m = sc.memory;
        const uint64_t p = uint64_t(params);
        if (training()) {
            m[MemCat::Parameters] = 4 * p;                           // f32 master weights
            m[MemCat::Gradients] = 4 * p;
            m[MemCat::OptimizerState] = optimizer_state_bytes(optimizer(), p);
            const double n = double(sc.layers.size());
            m[MemCat::Activations] = uint64_t(sc.checkpoint ? acts / std::sqrt(std::max(1.0, n)) + max_act : acts);
        } else {
            m[MemCat::Parameters] = tensor_bytes(sc.dtype, p);       // inference: stored in the chosen precision
            m[MemCat::Activations] = 0;
        }
        m[MemCat::Scratch] = uint64_t(scratch);
        m[MemCat::Data] = data_bytes(c, sc.microbatch);
        return sc;
    }
};

// ───────────────────────────── device view ─────────────────────────────

struct ExecPlan {
    int threads = 1;
    bool sustained = true;            // price for a long run: divide by the sustained/peak fraction
};

// The capability profile, read the way the cost model needs it.
class DeviceView {
public:
    explicit DeviceView(const CapabilityProfile& p) : P_(p) {}

    bool complete(std::vector<std::string>* missing = nullptr) const {
        bool ok = true;
        auto need = [&](bool have, const char* what) { if (!have) { ok = false; if (missing) missing->push_back(what); } };
        need(any_gemm() > 0, "gemm");
        need(P_.conv_gflops.median > 0, "conv");
        need(P_.rnn_gflops > 0, "rnn");
        need(P_.md_ginteractions.median > 0, "md");
        need(P_.gather_gbps > 0, "gather");
        need(!P_.stream_1t.empty(), "stream");
        need(!P_.latency.empty(), "latency");
        return ok;
    }

    double gemm_gflops(double M, double N, double K) const {
        const double mn = std::min(N, K);
        std::vector<const char*> order;
        if (mn > 0 && M >= 64.0 * mn) order.push_back("f32_tall");
        const double c = std::cbrt(std::max(1.0, M * N * K));
        if (c <= 24) order.push_back("f32_16");
        else if (c <= 96) order.push_back("f32_64");
        else if (c <= 384) order.push_back("f32_256");
        else { order.push_back("f32_512"); order.push_back("f32_256"); }
        for (const char* k : {"f32_256", "f32_64", "f32_512", "f32_16", "f32_tall"}) order.push_back(k);
        for (const char* k : order) { double v = gemm_key(k); if (v > 0) return v; }
        return 0.1;                                                  // nothing measured: a deliberately pessimistic floor
    }
    double conv_gflops() const { return pos(P_.conv_gflops.median, 0.1); }
    double rnn_gflops() const { return pos(P_.rnn_gflops, 0.1); }
    double md_rate() const { return pos(P_.md_ginteractions.median, 0.01) * 1e9; }          // interactions/s
    // Gather speed was measured on one feature-table size; a smaller table caches better, a larger one worse.
    // Scale by the stream-bandwidth ratio between the two sizes (bounded: it is an estimate, not a law).
    double gather_gbps(double ws = 0) const {
        const double base = pos(P_.gather_gbps, 0.1);
        if (P_.gather_ws_bytes == 0 || ws <= 0 || P_.stream_1t.empty()) return base;
        const double f = bw_gbps(ws) / std::max(1e-9, bw_gbps(double(P_.gather_ws_bytes)));
        return base * std::clamp(f, 0.5, 3.0);
    }
    double bw_gbps(double ws) const { return interp(P_.stream_1t, ws, true); }
    double latency_ns(double ws) const { return interp(P_.latency, ws, false); }
    double fork_join_s() const { return P_.fork_join_us * 1e-6; }
    double call_s() const { return P_.call_overhead_ns * 1e-9; }

    // How much faster T threads are than one, for compute-bound work. Uses the measured per-thread
    // rates of every core running together (fastest first), so a big.LITTLE device is not
    // priced as T equal cores.
    double speedup_compute(int T) const {
        T = std::max(1, T);
        const auto& r = P_.all_cores.gflops;
        if (!r.empty() && r.front() > 0) {
            std::vector<double> s = r;
            std::sort(s.begin(), s.end(), std::greater<double>());
            const int n = std::min<int>(T, int(s.size()));
            double sum = 0;
            for (int i = 0; i < n; ++i) sum += s[size_t(i)];
            return std::max(1.0, sum / s.front());
        }
        return scale_curve(T, false);
    }
    double speedup_bw(int T) const { return scale_curve(std::max(1, T), true); }
    int max_threads() const {
        int m = 1;
        for (const auto& s : P_.scaling) m = std::max(m, s.threads);
        m = std::max<int>(m, int(P_.all_cores.gflops.size()));
        return m;
    }
    double sustained_fraction() const {
        const auto& s = P_.sustained;
        if (!s.ran || s.window_gflops.size() < 3) return 1.0;
        return std::clamp(s.fraction, 0.3, 1.0);
    }

private:
    double gemm_key(const char* k) const { auto it = P_.gemm.find(k); return (it != P_.gemm.end() && it->second.ok) ? it->second.median : 0.0; }
    double any_gemm() const { double m = 0; for (const auto& kv : P_.gemm) m = std::max(m, kv.second.median); return m; }
    static double pos(double v, double floor) { return v > 0 ? v : floor; }

    // log-linear interpolation over working-set size
    static double interp(const std::vector<LevelPoint>& pts, double ws, bool gbps) {
        if (pts.empty()) return gbps ? 1.0 : 100.0;
        auto val = [&](const LevelPoint& p) { return gbps ? p.gbps : p.ns; };
        if (ws <= double(pts.front().bytes)) return val(pts.front());
        if (ws >= double(pts.back().bytes)) return val(pts.back());
        for (size_t i = 0; i + 1 < pts.size(); ++i) {
            const double a = double(pts[i].bytes), b = double(pts[i + 1].bytes);
            if (ws >= a && ws <= b) {
                const double t = (std::log(ws) - std::log(a)) / (std::log(b) - std::log(a));
                return val(pts[i]) + t * (val(pts[i + 1]) - val(pts[i]));
            }
        }
        return val(pts.back());
    }
    double scale_curve(int T, bool bw) const {
        const auto& sc = P_.scaling;
        if (sc.empty()) return double(T);
        auto v = [&](const ScalePoint& s) { return bw ? s.gbps : s.gflops; };
        const double base = v(sc.front());
        if (base <= 0) return double(T);
        if (T <= sc.front().threads) return 1.0;
        if (T >= sc.back().threads) return std::max(1.0, v(sc.back()) / base);
        for (size_t i = 0; i + 1 < sc.size(); ++i)
            if (T >= sc[i].threads && T <= sc[i + 1].threads) {
                const double t = double(T - sc[i].threads) / double(sc[i + 1].threads - sc[i].threads);
                return std::max(1.0, (v(sc[i]) + t * (v(sc[i + 1]) - v(sc[i]))) / base);
            }
        return 1.0;
    }
    const CapabilityProfile& P_;
};

// ───────────────────────────── cost model ─────────────────────────────

struct LayerTime {
    std::string name;
    double seconds = 0;               // including repeat and sustained scaling
    double compute = 0, memory = 0, irregular = 0, overhead = 0;   // components before the max/sum
};

struct StepPrediction {
    double seconds = 0;
    double compute_s = 0, memory_s = 0, irregular_s = 0, overhead_s = 0;   // which part of each layer set its time
    double units_per_s = 0;
    double core_seconds = 0;          // busy core-seconds: a proxy for energy
    std::string bound;                // "compute" | "memory" | "irregular" | "overhead"
    bool complete = true;             // false: the profile lacked a metric the model needed (pessimistic floor used)
    std::vector<LayerTime> layers;
    std::vector<std::string> notes;
};

class CostModel {
public:
    explicit CostModel(const CapabilityProfile& P) : P_(P), dev_(P_) {}
    const DeviceView& device() const { return dev_; }

    StepPrediction predict(const StepCosts& sc, const ExecPlan& plan) const {
        StepPrediction out;
        std::vector<std::string> missing;
        if (!dev_.complete(&missing)) {
            out.complete = false;
            std::string m = "capability profile lacks:";
            for (auto& s : missing) m += " " + s;
            out.notes.push_back(m + " (pessimistic floors used)");
        }
        const double sus = plan.sustained ? dev_.sustained_fraction() : 1.0;
        double comp = 0, mem = 0, irr = 0, ovh = 0;
        for (const auto& l : sc.layers) {
            LayerTime lt;
            lt.name = l.name;
            const int Tw = int(std::max(1.0, std::min<double>(plan.threads, l.width)));
            const double su_c = dev_.speedup_compute(Tw), su_m = dev_.speedup_bw(Tw);
            const double flops = l.flops_fwd + l.flops_bwd;
            const double bytes = l.bytes_fwd + l.bytes_bwd;
            const double gath = l.gather_bytes_fwd + l.gather_bytes_bwd;
            double t_comp = 0;
            switch (l.kernel) {
                case Kernel::Gemm: t_comp = flops / (dev_.gemm_gflops(l.gemm_m, l.gemm_n, l.gemm_k) * 1e9); break;
                case Kernel::Conv: t_comp = flops / (dev_.conv_gflops() * 1e9); break;
                case Kernel::Recurrent: {
                    // one row per step is a latency-bound matvec; more rows turn it into a matmul
                    const double gem = dev_.gemm_gflops(l.gemm_m, l.gemm_n, l.gemm_k);
                    const double rate = std::min(dev_.rnn_gflops() * std::max(1.0, l.width), gem);
                    t_comp = flops / (rate * 1e9);
                    break;
                }
                case Kernel::Pairwise: t_comp = l.interactions / dev_.md_rate(); break;
                case Kernel::Gather: case Kernel::Stream: case Kernel::Chase: break;
            }
            const double t_mem = bytes / (dev_.bw_gbps(l.working_set) * 1e9);
            const double t_gather = gath / (dev_.gather_gbps(l.working_set) * 1e9);
            const double t_chase = l.dep_loads * dev_.latency_ns(l.working_set) * 1e-9;
            const double a = t_comp / su_c, b = t_mem / su_m;
            const double irregular = t_gather / su_m + t_chase / su_c;
            const double over = l.dispatches * ((Tw > 1 ? dev_.fork_join_s() : 0.0) + dev_.call_s());
            const double per = std::max(a, b) + irregular + over;
            lt.compute = a; lt.memory = b; lt.irregular = irregular; lt.overhead = over;
            lt.seconds = per * l.repeat / sus;
            out.layers.push_back(lt);
            out.seconds += lt.seconds;
            out.core_seconds += lt.seconds * Tw;
            (a >= b ? comp : mem) += std::max(a, b) * l.repeat / sus;
            irr += irregular * l.repeat / sus;
            ovh += over * l.repeat / sus;
        }
        out.compute_s = comp; out.memory_s = mem; out.irregular_s = irr; out.overhead_s = ovh;
        out.bound = "compute";
        double best = comp;
        if (mem > best) { best = mem; out.bound = "memory"; }
        if (irr > best) { best = irr; out.bound = "irregular"; }
        if (ovh > best) { out.bound = "overhead"; }
        out.units_per_s = out.seconds > 0 ? sc.units / out.seconds : 0.0;
        return out;
    }

private:
    const CapabilityProfile& P_;
    DeviceView dev_;
};

// ───────────────────────────── the profiler ─────────────────────────────

struct ProfileResult {
    Config config;
    ExecPlan plan;
    StepPrediction time;              // calibrated
    StepCosts costs;
    FitResult fit;
    double real_bytes = 0;            // tensor bytes x allocator overhead
    bool feasible = false;            // fits the memory budget and the algorithm accepts the config
    double error = 0;                 // relative spread of the model on validated runs (0 = never calibrated)
    std::vector<std::string> notes;
    std::string to_json() const;
};

struct ModelFit {
    int samples = 0;
    double scale = 1.0;               // measured / predicted (geometric mean over samples)
    double spread = 0;                // max |ratio / scale - 1|: what the advisor reports as prediction error
    bool valid = false;
};

using ProbeFn = std::function<double(const Config&, const ExecPlan&)>;   // seconds for one real step

class AlgorithmProfiler {
public:
    AlgorithmProfiler(std::shared_ptr<const AlgorithmModel> model, CapabilityProfile capability, MemoryBudget budget)
        : model_(std::move(model)), P_(std::move(capability)), budget_(std::move(budget)) {}

    const AlgorithmModel& model() const { return *model_; }
    const CapabilityProfile& capability() const { return P_; }
    const MemoryBudget& budget() const { return budget_; }
    void set_budget(MemoryBudget b) { budget_ = std::move(b); }
    double scale() const { return scale_; }
    double error() const { return error_; }
    void set_calibration(double scale, double error) { scale_ = scale; error_ = error; }

    // Uncalibrated analytic time, used by calibrate() so repeated calibration does not compound.
    StepPrediction raw_prediction(const Config& cfg, const ExecPlan& plan) const {
        CostModel cm(P_);
        return cm.predict(model_->costs(cfg), plan);
    }

    ProfileResult evaluate(const Config& cfg, const ExecPlan& plan) const {
        return evaluate_with(cfg, model_->costs(cfg), plan);
    }

    // Same, with the step costs already built (they do not depend on the thread count, so a search
    // over threads builds them once).
    ProfileResult evaluate_with(const Config& cfg, StepCosts costs, const ExecPlan& plan) const {
        ProfileResult r;
        r.config = cfg; r.plan = plan;
        r.costs = std::move(costs);
        CostModel cm(P_);
        r.time = cm.predict(r.costs, plan);
        r.time.seconds *= scale_;
        r.time.compute_s *= scale_; r.time.memory_s *= scale_; r.time.irregular_s *= scale_; r.time.overhead_s *= scale_;
        r.time.core_seconds *= scale_;
        r.time.units_per_s = r.time.seconds > 0 ? r.costs.units / r.time.seconds : 0.0;
        r.fit = check_fit(r.costs.memory, budget_);
        r.real_bytes = double(r.costs.memory.total()) * budget_.overhead_factor;
        r.error = error_;
        const bool valid = model_->valid(cfg);
        if (!valid) r.notes.push_back("configuration rejected by the algorithm");
        if (!r.fit.fits) r.notes.push_back("does not fit the memory budget");
        if (plan.threads > cm.device().max_threads()) r.notes.push_back("more threads than were measured");
        r.feasible = valid && r.fit.fits;
        for (const auto& n : r.time.notes) r.notes.push_back(n);
        return r;
    }

    struct Sample { Config config; ExecPlan plan; double measured_s = 0; };

    // Fit the model to measurements that were already taken.
    ModelFit fit_samples(const std::vector<Sample>& samples) {
        ModelFit f;
        std::vector<double> ratios;
        for (const auto& s : samples) {
            const double predicted = raw_prediction(s.config, s.plan).seconds;
            if (s.measured_s > 0 && predicted > 0 && std::isfinite(s.measured_s) && std::isfinite(predicted))
                ratios.push_back(s.measured_s / predicted);
        }
        f.samples = int(ratios.size());
        if (ratios.empty()) return f;
        double lg = 0;
        for (double r : ratios) lg += std::log(r);
        f.scale = std::exp(lg / double(ratios.size()));
        for (double r : ratios) f.spread = std::max(f.spread, std::abs(r / f.scale - 1.0));
        f.valid = true;
        scale_ = f.scale;
        error_ = f.spread;
        return f;
    }

    // Fit the model to real measurements: probe(cfg, plan) must run ONE real step and return seconds.
    ModelFit calibrate(const ProbeFn& probe, const std::vector<std::pair<Config, ExecPlan>>& runs, int repeats = 3) {
        std::vector<Sample> samples;
        for (const auto& run : runs) {
            std::vector<double> t;
            for (int i = 0; i < std::max(1, repeats); ++i) t.push_back(probe(run.first, run.second));
            std::sort(t.begin(), t.end());
            samples.push_back({run.first, run.second, t[t.size() / 2]});
        }
        return fit_samples(samples);
    }

private:
    std::shared_ptr<const AlgorithmModel> model_;
    CapabilityProfile P_;
    MemoryBudget budget_;
    double scale_ = 1.0;
    double error_ = 0.0;
};

// ───────────────────────────── algorithm plug-ins ─────────────────────────────
namespace algo {

inline std::vector<double> dtype_values(std::initializer_list<Dtype> l) {
    std::vector<double> v;
    for (auto d : l) v.push_back(double(int(d)));
    return v;
}
inline Dtype dtype_of(const Config& c) { return Dtype(int(cfg_get(c, "dtype", double(int(Dtype::F32))))); }

// Execution tunables every training algorithm shares.
inline void add_exec_tunables(std::vector<Tunable>& t, bool checkpointable) {
    t.push_back({"microbatch", {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}, 1e9, false, "units per pass; accumulation covers the rest"});
    if (checkpointable) t.push_back({"checkpoint", {0, 1}, 0, false, "recompute forward in backward to hold fewer activations"});
}

// Plain multilayer perceptron: dims = {in, hidden..., out}.
class MlpModel : public AlgorithmModel {
public:
    explicit MlpModel(std::vector<int> dims = {784, 512, 256, 10}) : dims_(std::move(dims)) {}
    std::string name() const override { return "mlp"; }
    std::string unit_name() const override { return "samples"; }
    std::vector<Tunable> tunables() const override {
        std::vector<Tunable> t;
        t.push_back({"batch", {32, 64, 128, 256, 512, 1024}, 128, true, "changes gradient noise"});
        t.push_back({"dtype", dtype_values({Dtype::F32, Dtype::F16, Dtype::FP8}), double(int(Dtype::F32)), true, "activation storage precision"});
        add_exec_tunables(t, true);
        return t;
    }
    std::vector<LayerCost> microbatch_layers(const Config& c, int64_t mb) const override {
        std::vector<LayerCost> out;
        const Dtype d = dtype_of(c);
        for (size_t i = 0; i + 1 < dims_.size(); ++i) {
            out.push_back(layers::dense("fc" + std::to_string(i), double(mb), dims_[i], dims_[i + 1], d, training()));
            if (i + 2 < dims_.size()) out.push_back(layers::elementwise("act" + std::to_string(i), double(mb) * dims_[i + 1], d, 2.0, training()));
        }
        return out;
    }
    uint64_t data_bytes(const Config&, int64_t mb) const override { return uint64_t(mb) * uint64_t(dims_.front()) * 4; }
private:
    std::vector<int> dims_;
};

// Small convolutional classifier: conv3x3 + relu + 2x downsample per stage, dense head.
class CnnModel : public AlgorithmModel {
public:
    CnnModel(int channels_in = 3, int h = 32, int w = 32, std::vector<int> widths = {32, 64, 128}, int classes = 10)
        : cin_(channels_in), h_(h), w_(w), widths_(std::move(widths)), classes_(classes) {}
    std::string name() const override { return "cnn"; }
    std::string unit_name() const override { return "images"; }
    std::vector<Tunable> tunables() const override {
        std::vector<Tunable> t;
        t.push_back({"batch", {16, 32, 64, 128, 256, 512}, 64, true, "changes gradient noise"});
        t.push_back({"dtype", dtype_values({Dtype::F32, Dtype::F16, Dtype::FP8}), double(int(Dtype::F32)), true, "activation storage precision"});
        add_exec_tunables(t, true);
        return t;
    }
    std::vector<LayerCost> microbatch_layers(const Config& c, int64_t mb) const override {
        std::vector<LayerCost> out;
        const Dtype d = dtype_of(c);
        int ch = cin_, h = h_, w = w_;
        for (size_t i = 0; i < widths_.size(); ++i) {
            out.push_back(layers::conv2d("conv" + std::to_string(i), double(mb), ch, widths_[i], 3, h, w, 1, d, training()));
            out.push_back(layers::elementwise("relu" + std::to_string(i), double(mb) * widths_[i] * h * w, d, 2.0, training()));
            ch = widths_[i]; h = (h + 1) / 2; w = (w + 1) / 2;             // pooling
        }
        out.push_back(layers::dense("head", double(mb), double(ch) * h * w, classes_, d, training()));
        return out;
    }
    uint64_t data_bytes(const Config&, int64_t mb) const override { return uint64_t(mb) * cin_ * h_ * w_ * 4; }
private:
    int cin_, h_, w_;
    std::vector<int> widths_;
    int classes_;
};

// Node classification with neighbour sampling. Unit = target node.
class GnnModel : public AlgorithmModel {
public:
    GnnModel(int feat_in = 128, int hidden = 128, int depth = 2, int classes = 16, double graph_nodes = 1e6)
        : feat_(feat_in), hidden_(hidden), depth_(depth), classes_(classes), graph_nodes_(graph_nodes) {}
    std::string name() const override { return "gnn"; }
    std::string unit_name() const override { return "target nodes"; }
    std::vector<Tunable> tunables() const override {
        std::vector<Tunable> t;
        t.push_back({"batch", {64, 128, 256, 512, 1024, 2048}, 256, true, "target nodes per step"});
        t.push_back({"fanout", {5, 10, 15, 25}, 10, true, "neighbours sampled per node per layer: changes the estimator"});
        t.push_back({"dtype", dtype_values({Dtype::F32, Dtype::F16, Dtype::FP8}), double(int(Dtype::F32)), true, "feature/activation storage precision"});
        add_exec_tunables(t, true);
        return t;
    }
    std::vector<LayerCost> microbatch_layers(const Config& c, int64_t mb) const override {
        std::vector<LayerCost> out;
        const Dtype d = dtype_of(c);
        const double f = cfg_get(c, "fanout", 10);
        // frontier sizes: s[0] = targets, s[k] = nodes needed k hops out (capped by the graph)
        std::vector<double> s{double(mb)};
        for (int k = 1; k <= depth_; ++k) s.push_back(std::min(graph_nodes_, s.back() * f));
        out.push_back(layers::embedding("feature_fetch", s.back(), feat_, graph_nodes_, d, false));
        out.back().params = 0;                                           // the feature table is data, not trained
        double in = feat_;
        for (int l = 0; l < depth_; ++l) {
            const double nout = s[size_t(depth_ - 1 - l)], nin = s[size_t(depth_ - l)];
            out.push_back(layers::spmm_aggregate("agg" + std::to_string(l), nout, nout * f, nin, in, d, training()));
            const double outdim = l + 1 == depth_ ? classes_ : hidden_;
            out.push_back(layers::dense("lin" + std::to_string(l), nout, in, outdim, d, training()));
            if (l + 1 < depth_) out.push_back(layers::elementwise("relu" + std::to_string(l), nout * outdim, d, 2.0, training()));
            in = outdim;
        }
        return out;
    }
    uint64_t data_bytes(const Config& c, int64_t mb) const override {
        const double f = cfg_get(c, "fanout", 10);
        double n = double(mb);
        for (int k = 0; k < depth_; ++k) n = std::min(graph_nodes_, n * f);
        return uint64_t(n * feat_ * 4);
    }
private:
    int feat_, hidden_, depth_, classes_;
    double graph_nodes_;
};

// Recurrent sequence model (LSTM by default). Unit = time step.
class RnnModel : public AlgorithmModel {
public:
    RnnModel(int in = 64, int hidden = 256, int gates = 4, int classes = 32) : in_(in), hidden_(hidden), gates_(gates), classes_(classes) {}
    std::string name() const override { return "rnn"; }
    std::string unit_name() const override { return "time steps"; }
    double units_per_step(const Config& c) const override { return cfg_get(c, "batch", 1) * cfg_get(c, "bptt", 32); }
    std::vector<Tunable> tunables() const override {
        std::vector<Tunable> t;
        t.push_back({"batch", {8, 16, 32, 64, 128, 256}, 32, true, "sequences per step"});
        t.push_back({"bptt", {16, 32, 64, 128}, 32, true, "truncation length: changes the gradient"});
        t.push_back({"dtype", dtype_values({Dtype::F32, Dtype::F16}), double(int(Dtype::F32)), true, "activation storage precision"});
        add_exec_tunables(t, false);
        return t;
    }
    std::vector<LayerCost> microbatch_layers(const Config& c, int64_t mb) const override {
        const Dtype d = dtype_of(c);
        const double steps = cfg_get(c, "bptt", 32);
        std::vector<LayerCost> out;
        out.push_back(layers::recurrent_cell("cell", double(mb), in_, hidden_, steps, gates_, d, training()));
        out.push_back(layers::dense("readout", double(mb) * steps, hidden_, classes_, d, training()));
        return out;
    }
    uint64_t data_bytes(const Config& c, int64_t mb) const override { return uint64_t(double(mb) * cfg_get(c, "bptt", 32) * in_ * 4); }
private:
    int in_, hidden_, gates_, classes_;
};

// Decoder-only transformer. Unit = token.
class TransformerModel : public AlgorithmModel {
public:
    TransformerModel(int dim = 256, int heads = 4, int depth = 4, int vocab = 8000, int ffn_mult = 4)
        : dim_(dim), heads_(heads), depth_(depth), vocab_(vocab), mult_(ffn_mult) {}
    std::string name() const override { return "transformer"; }
    std::string unit_name() const override { return "tokens"; }
    double units_per_step(const Config& c) const override { return cfg_get(c, "batch", 1) * cfg_get(c, "seq", 128); }
    std::vector<Tunable> tunables() const override {
        std::vector<Tunable> t;
        t.push_back({"batch", {1, 2, 4, 8, 16, 32}, 4, true, "sequences per step"});
        t.push_back({"seq", {64, 128, 256, 512}, 128, true, "context length: changes the task"});
        t.push_back({"dtype", dtype_values({Dtype::F32, Dtype::F16, Dtype::FP8}), double(int(Dtype::F32)), true, "activation storage precision"});
        add_exec_tunables(t, true);
        return t;
    }
    std::vector<LayerCost> microbatch_layers(const Config& c, int64_t mb) const override {
        const Dtype d = dtype_of(c);
        const double seq = cfg_get(c, "seq", 128), tokens = double(mb) * seq;
        std::vector<LayerCost> out;
        out.push_back(layers::embedding("embed", tokens, dim_, vocab_, d, training()));
        for (int l = 0; l < depth_; ++l) {
            const std::string p = "L" + std::to_string(l);
            out.push_back(layers::elementwise(p + ".ln1", tokens * dim_, d, 3.0, training()));
            for (auto& a : layers::attention(p + ".attn", double(mb), seq, dim_, heads_, d, training())) out.push_back(a);
            out.push_back(layers::elementwise(p + ".ln2", tokens * dim_, d, 3.0, training()));
            out.push_back(layers::dense(p + ".ffn1", tokens, dim_, double(dim_) * mult_, d, training()));
            out.push_back(layers::elementwise(p + ".gelu", tokens * dim_ * mult_, d, 2.0, training()));
            out.push_back(layers::dense(p + ".ffn2", tokens, double(dim_) * mult_, dim_, d, training()));
        }
        out.push_back(layers::dense("lm_head", tokens, dim_, vocab_, d, training()));
        return out;
    }
    uint64_t data_bytes(const Config& c, int64_t mb) const override { return uint64_t(double(mb) * cfg_get(c, "seq", 128) * 8); }
private:
    int dim_, heads_, depth_, vocab_, mult_;
};

// Independent random walks / Markov-chain sampling. Forward only. Unit = walker step.
class MarkovModel : public AlgorithmModel {
public:
    explicit MarkovModel(double states = 4e6, double branch = 4) : states_(states), branch_(branch) {}
    std::string name() const override { return "markov"; }
    std::string unit_name() const override { return "walker steps"; }
    bool training() const override { return false; }
    double units_per_step(const Config& c) const override { return cfg_get(c, "batch", 1) * cfg_get(c, "steps", 256); }
    std::vector<Tunable> tunables() const override {
        std::vector<Tunable> t;
        t.push_back({"batch", {256, 1024, 4096, 16384, 65536}, 4096, true, "parallel walkers: changes the estimator variance"});
        t.push_back({"steps", {64, 256, 1024}, 256, true, "steps per walker per round"});
        t.push_back({"microbatch", {256, 1024, 4096, 16384, 65536}, 1e9, false, "walkers per pass"});
        return t;
    }
    std::vector<LayerCost> microbatch_layers(const Config& c, int64_t mb) const override {
        return {layers::markov_walk("walk", double(mb), cfg_get(c, "steps", 256), states_ * branch_ * 4.0)};
    }
private:
    double states_, branch_;
};

// Short-range particle simulation. Forward only; the problem size is fixed, the tunables are
// the accuracy/speed knobs. Unit = particle step.
class ParticleModel : public AlgorithmModel {
public:
    explicit ParticleModel(double particles = 20000, double rebuild_every = 10) : n_(particles), rebuild_(rebuild_every) {}
    std::string name() const override { return "particles"; }
    std::string unit_name() const override { return "particle steps"; }
    bool training() const override { return false; }
    std::vector<Tunable> tunables() const override {
        std::vector<Tunable> t;
        t.push_back({"batch", {n_}, n_, false, "number of particles (problem size, not tuned)"});
        t.push_back({"neighbors", {16, 32, 64, 128}, 64, true, "neighbours per particle: sets the force cutoff, changes accuracy"});
        t.push_back({"dtype", dtype_values({Dtype::F32, Dtype::F16}), double(int(Dtype::F32)), true, "state storage precision"});
        t.push_back({"microbatch", {512, 1024, 2048, 4096, 8192, 16384, 32768}, 1e9, false, "particles per tile"});
        return t;
    }
    std::vector<LayerCost> microbatch_layers(const Config& c, int64_t mb) const override {
        return {layers::pairwise("forces", double(mb), cfg_get(c, "neighbors", 64), rebuild_, dtype_of(c))};
    }
private:
    double n_, rebuild_;
};

inline std::shared_ptr<AlgorithmModel> make(const std::string& name) {
    if (name == "mlp") return std::make_shared<MlpModel>();
    if (name == "cnn") return std::make_shared<CnnModel>();
    if (name == "gnn") return std::make_shared<GnnModel>();
    if (name == "rnn") return std::make_shared<RnnModel>();
    if (name == "transformer") return std::make_shared<TransformerModel>();
    if (name == "markov") return std::make_shared<MarkovModel>();
    if (name == "particles") return std::make_shared<ParticleModel>();
    return nullptr;
}
inline std::vector<std::string> names() { return {"mlp", "cnn", "gnn", "rnn", "transformer", "markov", "particles"}; }

}  // namespace algo

// ───────────────────────────── output ─────────────────────────────

inline std::string ProfileResult::to_json() const {
    detail::JsonOut j;
    j.begin_obj();
    j.key("config"); j.begin_obj(); for (const auto& kv : config) j.kv(kv.first, kv.second); j.end_obj();
    j.kv("threads", plan.threads); j.kv("sustained", plan.sustained);
    j.kv("feasible", feasible); j.kv("error", error);
    j.key("time"); j.begin_obj();
    j.kv("seconds", time.seconds); j.kv("units_per_s", time.units_per_s); j.kv("core_seconds", time.core_seconds);
    j.kv("bound", time.bound); j.kv("compute_s", time.compute_s); j.kv("memory_s", time.memory_s);
    j.kv("irregular_s", time.irregular_s); j.kv("overhead_s", time.overhead_s); j.kv("complete", time.complete);
    j.end_obj();
    j.key("memory"); j.begin_obj();
    for (size_t i = 0; i < kMemCats; ++i) j.kv(to_string(MemCat(i)), uint64_t(costs.memory.bytes[i]));
    j.kv("tensor_total", uint64_t(costs.memory.total())); j.kv("real_bytes", real_bytes);
    j.kv("budget", uint64_t(fit.tensor_budget_bytes)); j.kv("utilization", fit.utilization);
    j.end_obj();
    j.kv("microbatch", uint64_t(costs.microbatch)); j.kv("accum", costs.accum); j.kv("checkpoint", costs.checkpoint);
    j.key("notes"); j.begin_arr(); for (const auto& n : notes) j.str(n); j.end_arr();
    j.end_obj();
    return j.str();
}

}  // namespace tensorf::profile
