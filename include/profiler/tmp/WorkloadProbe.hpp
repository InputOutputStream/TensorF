#pragma once

/*
 * WorkloadProbe.hpp
 * =================
 * The validation loop (design doc, section 3 "Probe" and section 4 "Validation"):
 * run small REAL workloads through TensorF's own Matrix / Tensor ops, compare the
 * measured time with WorkloadModel's prediction, fit Calibration, and report how far off
 * the model still is on cases it was NOT fitted on.
 *
 *   measure_cases()    generic: takes any {workload, step()} list. Algorithm-agnostic, so a
 *                      new algorithm validates itself by supplying its own cases. Handles
 *                      time budget, thermal guard, memory guard and exceptions.
 *   reference_cases()  the built-in set that runs real framework ops:
 *                        dense      Matrix::matmul at 5 shapes (square x3, skinny, rectangular)
 *                        elementwise Matrix a+b, cache-sized and DRAM-sized
 *                        spmm       spmm() on a power-law graph, 3 sizes
 *                        recurrent  Tensor matmul/add/sigmoid chain, 2 sizes
 *                        mlp        full forward+backward through Tensor autograd, 2 sizes
 *   run_probe()        reference_cases() + measure_cases()
 *
 * Honest error: the reported accuracy is LEAVE-ONE-OUT. For each case, the correction
 * factor is fitted on the other cases of its kind and applied to it; the miss that remains
 * is the number the advisor should quote. In-sample spread would flatter the model.
 *
 * Scope limits (also written into the report so they travel with the profile):
 *   - threads: the framework has no thread pool yet (the Parallel layer is not built), so
 *     every case runs on ONE thread. Multi-thread predictions stay unvalidated and
 *     ProbeReport::threads_validated says 1.
 *   - conv2d and particle_interaction have no framework op to measure, so they are NOT
 *     covered. Their predictions keep the uncalibrated error until an op exists.
 *   - BLAS: Matrix::matmul calls whatever cblas the build links. When that is the portable
 *     shim in include/tools/cblas.h (a plain i-k-j loop), the probe is validating the model
 *     against the shim, not OpenBLAS. ProbeReport::blas records which one it was, and a
 *     calibration is only valid for the build + device it was measured on.
 *
 * Build: needs the framework on the include path as well as the profiler headers:
 *   g++ -std=c++20 -O2 -Iinclude -Iinclude/core -Iinclude/tools ... -pthread
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Types/types.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor.hpp"
#include "DataStructures/Graph.hpp"

#include "WorkloadModel.hpp"

namespace tensorf::prof::wl {

// ─── Cases and configuration ─────────────────────────────────────────────────

struct ProbeCase {
    std::string name;
    Workload    workload;                        // the analytic description of EXACTLY what step() executes
    int         kind = kOpKinds;                 // OpKind index this case calibrates; kOpKinds = composite (global)
    // Allocates its data and returns the step closure. Called only if the case is not skipped, and the
    // closure (and everything it captured) is destroyed right after measuring, so memory is freed between cases.
    std::function<std::function<void()>()> make;
};

struct ProbeConfig {
    double time_budget_s = 25.0;                 // whole probe
    double per_case_s    = 1.2;                  // timing time per case, excluding warm-up
    int    warmup        = 2;
    int    min_samples   = 5;
    int    max_samples   = 25;
    int    threads       = 1;                    // threads passed to predict(); the reference cases run single-threaded
    double memory_fraction = 0.25;               // skip a case whose estimated process memory exceeds this share of the safe budget
    double size_scale    = 1.0;                  // < 1 shrinks the reference cases for weak devices
    std::string sysroot;                         // "" = real filesystem; tests point this at a fixture
    SuiteConfig guard;                           // thermal thresholds, shared with the capability suite
    bool verbose = false;
};

struct ProbeResult {
    std::string name;
    int    kind = kOpKinds;
    bool   ok = false;                           // measured and usable for fitting
    std::string note;                            // why not, when !ok
    double predicted_s = 0;                      // uncalibrated
    double measured_s = 0;                       // median of samples
    double ratio = 0;                            // measured / predicted
    double spread = 0;                           // relative spread of the samples (noise)
    int    samples = 0;
    std::array<double, kOpKinds> kind_s{};       // uncalibrated predicted seconds by layer kind (sums to predicted_s)
    double loo_predicted_s = 0;                  // prediction after a factor fitted WITHOUT this case
    double loo_error = 0;                        // |loo_predicted - measured| / measured
};

struct KindError {
    int    kind = kOpKinds;
    int    cases = 0;
    double factor = 1.0;                         // fitted on all cases of the kind
    double loo_median_error = 0;                 // median leave-one-out miss
    double loo_max_error = 0;
    double raw_median_error = 0;                 // uncalibrated miss, for comparison: how much calibration helped
};

struct ProbeReport {
    std::vector<ProbeResult> results;
    std::vector<KindError>   kinds;
    Calibration calibration;                     // fitted on every usable case; hand this to predict()
    double overall_loo_error = Calibration::kUncalibratedError;   // median over all usable cases
    double overall_raw_error = 0;
    double global_factor = 1.0;                  // median measured/predicted over all cases: what kinds with no probe (conv ...) get
    int    usable = 0;
    int    threads_validated = 1;
    std::string blas;                            // "shim", "external" or "none"
    std::string stopped_reason;
    double time_used_s = 0;
    std::vector<std::string> not_covered;        // layer kinds with no probe: their error stays uncalibrated
};

inline const char* blas_name() {
#if defined(MATRIX_NO_BLAS)
    return "none";
#elif defined(TENSORF_CBLAS_SHIM_H)
    return "shim";
#else
    return "external";
#endif
}

// ─── Generic measurement loop ────────────────────────────────────────────────

namespace detail {

// Median of per-sample seconds, warm-up excluded; stops on count, per-case time, or the global deadline.
// At least min_samples are taken even past the deadline's per-case share, so one case is never a single noisy sample.
inline bool time_step(const std::function<void()>& step, const ProbeConfig& cfg, const Deadline& overall,
                      double& median_s, double& spread, int& n) {
    for (int i = 0; i < cfg.warmup; ++i) step();
    std::vector<double> t;
    Stopwatch sw;
    while ((int)t.size() < cfg.max_samples) {
        Stopwatch one; step(); t.push_back(one.elapsed());
        if ((int)t.size() >= cfg.min_samples && (sw.elapsed() >= cfg.per_case_s || overall.expired())) break;
    }
    n = (int)t.size();
    median_s = median(t);
    spread = rel_spread(t);
    return n > 0 && median_s > 0 && std::isfinite(median_s);
}

}  // namespace detail

// Fits calibration and computes the leave-one-out error. Exposed so it can be tested without timing anything.
inline void fit_report(ProbeReport& rep) {
    std::vector<Observation> all;
    for (const auto& r : rep.results) {
        if (!r.ok) continue;
        all.push_back({r.kind, r.predicted_s, r.measured_s});
        if (r.kind != kOpKinds) all.push_back({kOpKinds, r.predicted_s, r.measured_s});   // every case also informs the global fallback
    }
    rep.calibration = Calibration{};
    rep.calibration.fit(all);

    // Leave-one-out: rebuild the calibration without case i and apply it to case i.
    std::vector<double> loo_all, raw_all;
    for (size_t i = 0; i < rep.results.size(); ++i) {
        ProbeResult& ri = rep.results[i];
        if (!ri.ok) continue;
        std::vector<Observation> rest;
        for (size_t j = 0; j < rep.results.size(); ++j) {
            if (j == i || !rep.results[j].ok) continue;
            const auto& rj = rep.results[j];
            rest.push_back({rj.kind, rj.predicted_s, rj.measured_s});
            if (rj.kind != kOpKinds) rest.push_back({kOpKinds, rj.predicted_s, rj.measured_s});
        }
        Calibration c; c.fit(rest);
        if (ri.kind >= 0 && ri.kind < kOpKinds) {
            ri.loo_predicted_s = ri.predicted_s * c.factor_for((OpKind)ri.kind);
        } else {
            // Composite (e.g. a full training step): the advisor prices it layer by layer with the per-kind
            // factors, so that is what is held out here: sum of each kind's time times its factor.
            double t = 0;
            for (int k = 0; k < kOpKinds; ++k) t += ri.kind_s[(size_t)k] * c.factor_for((OpKind)k);
            ri.loo_predicted_s = t > 0 ? t : ri.predicted_s;
        }
        ri.loo_error = std::fabs(ri.loo_predicted_s - ri.measured_s) / ri.measured_s;
        loo_all.push_back(ri.loo_error);
        raw_all.push_back(std::fabs(ri.predicted_s - ri.measured_s) / ri.measured_s);
    }

    rep.kinds.clear();
    for (int k = 0; k <= kOpKinds; ++k) {
        std::vector<double> loo, raw; int n = 0;
        for (const auto& r : rep.results) if (r.ok && r.kind == k) { ++n; loo.push_back(r.loo_error); raw.push_back(std::fabs(r.predicted_s - r.measured_s) / r.measured_s); }
        if (!n) continue;
        KindError ke; ke.kind = k; ke.cases = n; ke.factor = rep.calibration.factor[(size_t)k];
        ke.loo_median_error = median(loo); ke.loo_max_error = *std::max_element(loo.begin(), loo.end());
        ke.raw_median_error = median(raw);
        rep.kinds.push_back(ke);
    }
    rep.global_factor = rep.calibration.calibrated(kOpKinds) ? rep.calibration.factor[(size_t)kOpKinds] : 1.0;
    rep.usable = (int)loo_all.size();
    rep.overall_loo_error = loo_all.empty() ? Calibration::kUncalibratedError : median(loo_all);
    rep.overall_raw_error = raw_all.empty() ? 0.0 : median(raw_all);

    // A kind's stored error should be the honest (leave-one-out) one when we have it, never the flattering in-sample spread.
    for (const auto& ke : rep.kinds)
        if (ke.cases >= 2) rep.calibration.error[(size_t)ke.kind] = std::max(rep.calibration.error[(size_t)ke.kind], ke.loo_median_error);
}

inline ProbeReport measure_cases(std::vector<ProbeCase> cases, const DeviceFingerprint& fp, const CapabilityVector& cv,
                                 const ProbeConfig& cfg = {}) {
    ProbeReport rep;
    rep.blas = blas_name();
    rep.threads_validated = 1;
    Stopwatch total;
    const Deadline overall(cfg.time_budget_s);

    ThermalProbe thermal(cfg.sysroot);
    const double start_c = thermal.hottest_c();
    const bool charging = fp.battery.charging();

    PredictOptions po; po.threads = std::max(1, cfg.threads);

    for (auto& c : cases) {
        ProbeResult r; r.name = c.name; r.kind = c.kind;

        if (!rep.stopped_reason.empty() || overall.expired()) {
            if (rep.stopped_reason.empty()) rep.stopped_reason = "time budget exhausted";
            r.note = "not run: " + rep.stopped_reason;
            rep.results.push_back(std::move(r)); continue;
        }
        const std::string why = thermal_verdict(thermal.read(), start_c, charging, cfg.guard);
        if (!why.empty()) {
            rep.stopped_reason = "thermal: " + why;
            r.note = "not run: " + rep.stopped_reason;
            rep.results.push_back(std::move(r)); continue;
        }

        const MemoryEstimate me = estimate_memory(c.workload, po.mem_overhead_factor);
        if (fp.memory.safe_budget_bytes > 0 && me.process_total > cfg.memory_fraction * (double)fp.memory.safe_budget_bytes) {
            r.note = "skipped: needs ~" + std::to_string((long long)(me.process_total / 1048576.0)) + " MB, over the probe's memory share";
            rep.results.push_back(std::move(r)); continue;
        }

        const Prediction p = predict(c.workload, fp, cv, po, nullptr);
        if (!p.valid) { r.note = "skipped: no prediction (capability vector incomplete)"; rep.results.push_back(std::move(r)); continue; }
        r.predicted_s = p.step_s;
        for (const auto& lp : p.layers) r.kind_s[(size_t)lp.kind] += lp.seconds;

        try {
            std::function<void()> step = c.make();
            double med = 0, spread = 0; int n = 0;
            const bool good = detail::time_step(step, cfg, overall, med, spread, n);
            step = nullptr;                                          // free the case's data before the next one
            if (!good) { r.note = "measurement unusable"; }
            else {
                r.measured_s = med; r.spread = spread; r.samples = n; r.ratio = med / p.step_s; r.ok = true;
            }
        } catch (const std::exception& e) {
            r.note = std::string("exception: ") + e.what();
        } catch (...) {
            r.note = "unknown exception";
        }
        if (cfg.verbose)
            std::printf("[probe] %-22s predicted %9.3f ms  measured %9.3f ms  x%.2f  noise %.0f%%%s%s\n", r.name.c_str(),
                        r.predicted_s * 1e3, r.measured_s * 1e3, r.ratio, r.spread * 100, r.ok ? "" : "  ", r.ok ? "" : r.note.c_str());
        rep.results.push_back(std::move(r));
    }

    fit_report(rep);
    rep.time_used_s = total.elapsed();
    return rep;
}

// ─── Reference cases: real framework ops ─────────────────────────────────────

namespace detail {

inline size_t scaled(double n, double scale, size_t mult = 8, size_t lo = 8) {
    const size_t v = (size_t)std::max<double>((double)lo, std::floor(n * scale / (double)mult + 0.5) * (double)mult);
    return v;
}

inline Matrix<float> rnd(size_t r, size_t c, float a = -0.5f, float b = 0.5f) { return Matrix<float>::randu(a, b, {r, c}); }
inline Matrix<float> rnd1(size_t n, float a = -0.5f, float b = 0.5f) { return Matrix<float>::randu(a, b, {n}); }

inline Matrix<float> seed_scalar() { return Matrix<float>(std::vector<float>{1.0f}, shape_t{1}); }

// Dense forward through Matrix::matmul (no autograd).
inline ProbeCase dense_case(const char* name, size_t M, size_t K, size_t N) {
    ProbeCase c; c.name = std::string(name) + " " + std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);
    c.kind = (int)OpKind::DENSE;
    RunSpec r; r.mode = Mode::INFER;
    Workload w; w.name = c.name; w.units_per_micro_step = (double)M;
    w.layers = {dense((double)M, (double)K, (double)N, r, "matmul", false)};
    c.workload = std::move(w);
    c.make = [=] {
        auto A = std::make_shared<Matrix<float>>(rnd(M, K));
        auto B = std::make_shared<Matrix<float>>(rnd(K, N));
        return std::function<void()>([A, B] { Matrix<float> C = A->matmul(*B); do_not_optimize(C.data[0]); });
    };
    return c;
}

// Element-wise add of two vectors (a + b allocates the result: that cost is part of what is measured).
inline ProbeCase add_case(const char* name, size_t n) {
    ProbeCase c; c.name = std::string(name) + " n=" + std::to_string(n);
    c.kind = (int)OpKind::ELEMENTWISE;
    RunSpec r; r.mode = Mode::INFER;
    Workload w; w.name = c.name; w.units_per_micro_step = (double)n;
    w.layers = {elementwise((double)n, 1.0, 3.0, r, "add")};
    c.workload = std::move(w);
    c.make = [=] {
        auto a = std::make_shared<Matrix<float>>(rnd1(n));
        auto b = std::make_shared<Matrix<float>>(rnd1(n));
        return std::function<void()>([a, b] { Matrix<float> o = *a + *b; do_not_optimize(o.data[0]); });
    };
    return c;
}

// Graph aggregation through the autograd spmm op, forward only.
inline ProbeCase spmm_case(const char* name, size_t nodes, double mean_deg, size_t feat) {
    ProbeCase c; c.name = std::string(name) + " n=" + std::to_string(nodes) + " F=" + std::to_string(feat);
    c.kind = (int)OpKind::SPMM;
    // The workload needs the REAL nnz, so build the graph once here (cheap) and reuse it in make().
    auto g = std::make_shared<kernels::CsrGraph>(kernels::make_power_law_graph(nodes, mean_deg));
    RunSpec r; r.mode = Mode::INFER;
    Workload w; w.name = c.name; w.units_per_micro_step = (double)nodes;
    w.layers = {spmm_aggregate((double)nodes, (double)g->col.size(), (double)feat, r, "spmm")};
    c.workload = std::move(w);
    c.make = [=] {
        auto A = std::make_shared<CSR<float>>();
        A->n = g->nodes;
        A->row_ptr.assign(g->rowptr.begin(), g->rowptr.end());
        A->col_idx.assign(g->col.begin(), g->col.end());
        A->values.assign(g->col.size(), 0.1f);
        std::shared_ptr<const CSR<float>> cA = A;
        auto X = make_tensor<float>(rnd(g->nodes, feat));
        X->requires_grad = false;
        return std::function<void()>([cA, X] { auto y = spmm<float>(cA, X, 1); do_not_optimize(y->val.data[0]); });
    };
    return c;
}

// h_t = sigmoid(x_t Wx + h_{t-1} Wh) unrolled over `steps`, forward only.
inline ProbeCase rnn_case(const char* name, size_t B, size_t in, size_t hid, size_t steps) {
    ProbeCase c; c.name = std::string(name) + " B=" + std::to_string(B) + " h=" + std::to_string(hid) + " T=" + std::to_string(steps);
    c.kind = (int)OpKind::RECURRENT;
    RunSpec r; r.mode = Mode::INFER;
    Workload w; w.name = c.name; w.units_per_micro_step = (double)(B * steps);
    w.layers = {recurrent_cell((double)B, (double)in, (double)hid, (double)steps, 1.0, r, "rnn")};
    c.workload = std::move(w);
    c.make = [=] {
        struct S { Tensor_t<float> Wx, Wh; std::vector<Tensor_t<float>> xs; Tensor_t<float> h0; };
        auto s = std::make_shared<S>();
        s->Wx = make_tensor<float>(rnd(in, hid, -0.1f, 0.1f));  s->Wx->requires_grad = false;
        s->Wh = make_tensor<float>(rnd(hid, hid, -0.1f, 0.1f)); s->Wh->requires_grad = false;
        s->h0 = make_tensor<float>(rnd(B, hid, -0.1f, 0.1f));   s->h0->requires_grad = false;
        for (size_t t = 0; t < steps; ++t) { auto x = make_tensor<float>(rnd(B, in)); x->requires_grad = false; s->xs.push_back(x); }
        return std::function<void()>([s, steps] {
            auto h = s->h0;
            for (size_t t = 0; t < steps; ++t) h = (s->xs[t]->matmul(s->Wx) + h->matmul(s->Wh))->sigmoid();
            do_not_optimize(h->val.data[0]);
        });
    };
    return c;
}

// Two-layer MLP, forward + backward through autograd (the training path). Composite: calibrates the global factor.
inline ProbeCase mlp_case(const char* name, size_t B, size_t I, size_t H, size_t O) {
    ProbeCase c; c.name = std::string(name) + " B=" + std::to_string(B) + " H=" + std::to_string(H);
    c.kind = kOpKinds;                                       // a composite of dense + elementwise: attributed to the global factor
    RunSpec r; r.mode = Mode::TRAIN; r.opt_states = 0;
    RunSpec first = r; first.need_input_grad = false;
    Workload w; w.name = c.name; w.units_per_micro_step = (double)B;
    w.layers = {dense((double)B, (double)I, (double)H, first, "fc1", false),
                elementwise((double)(B * H), 1.0, 2.0, r, "relu"),
                dense((double)B, (double)H, (double)O, r, "fc2", false),
                elementwise((double)(B * O), 2.0, 2.0, r, "square+sum")};
    c.workload = std::move(w);
    c.make = [=] {
        struct S { Tensor_t<float> x, w1, w2; };
        auto s = std::make_shared<S>();
        s->x  = make_tensor<float>(rnd(B, I, -1.f, 1.f)); s->x->requires_grad = false;       // inputs need no gradient
        s->w1 = make_tensor<float>(rnd(I, H, -0.05f, 0.05f));
        s->w2 = make_tensor<float>(rnd(H, O, -0.05f, 0.05f));
        return std::function<void()>([s] {
            auto y = s->x->matmul(s->w1)->relu()->matmul(s->w2);
            auto loss = (y * y)->sum();
            loss->backward(seed_scalar());
            s->w1->zero_grad(); s->w2->zero_grad();
        });
    };
    return c;
}

}  // namespace detail

inline std::vector<ProbeCase> reference_cases(const ProbeConfig& cfg = {}) {
    using namespace detail;
    const double s = cfg.size_scale > 0 ? cfg.size_scale : 1.0;
    std::vector<ProbeCase> v;
    // dense: cache-resident square, mid square, larger square, skinny (decode-like), rectangular (MLP-like)
    v.push_back(dense_case("dense_sq", scaled(128, s, 16, 16), scaled(128, s, 16, 16), scaled(128, s, 16, 16)));
    v.push_back(dense_case("dense_sq", scaled(256, s, 16, 16), scaled(256, s, 16, 16), scaled(256, s, 16, 16)));
    v.push_back(dense_case("dense_sq", scaled(512, s, 16, 16), scaled(512, s, 16, 16), scaled(512, s, 16, 16)));
    v.push_back(dense_case("dense_skinny", 4, scaled(1024, s, 16, 16), scaled(1024, s, 16, 16)));
    v.push_back(dense_case("dense_rect", scaled(64, s, 8, 8), 784, scaled(256, s, 16, 16)));
    // elementwise: cache-sized (dispatch-dominated) and DRAM-sized (bandwidth-dominated)
    v.push_back(add_case("add_small", scaled(16384, s, 1024, 1024)));
    v.push_back(add_case("add_large", scaled(4u << 20, s, 4096, 4096)));
    // spmm on a power-law graph
    v.push_back(spmm_case("spmm", scaled(5000, s, 100, 100), 8.0, 16));
    v.push_back(spmm_case("spmm", scaled(20000, s, 100, 100), 8.0, 64));
    v.push_back(spmm_case("spmm", scaled(50000, s, 100, 100), 8.0, 32));
    // recurrent chain
    v.push_back(rnn_case("rnn", 8, 32, scaled(64, s, 8, 8), 50));
    v.push_back(rnn_case("rnn", 32, 64, scaled(128, s, 8, 8), 25));
    // full training step (forward + backward) through autograd
    v.push_back(mlp_case("mlp_train", scaled(32, s, 8, 8), 784, scaled(128, s, 16, 16), 10));
    v.push_back(mlp_case("mlp_train", scaled(128, s, 8, 8), 784, scaled(256, s, 16, 16), 10));
    return v;
}

inline ProbeReport run_probe(const DeviceFingerprint& fp, const CapabilityVector& cv, const ProbeConfig& cfg = {}) {
    ProbeReport rep = measure_cases(reference_cases(cfg), fp, cv, cfg);
    rep.not_covered = {"conv2d", "attention", "norm", "embedding", "particle", "optimizer"};   // no framework op measured here
    return rep;
}

// ─── Output ──────────────────────────────────────────────────────────────────

inline void to_json(Json& j, const ProbeReport& r, const char* key = "validation") {
    j.begin_object(key);
    j.str("blas", r.blas).integer("threads_validated", r.threads_validated).integer("usable_cases", r.usable)
     .num("global_factor", r.global_factor).num("overall_loo_error", r.overall_loo_error).num("overall_raw_error", r.overall_raw_error)
     .num("time_used_s", r.time_used_s).str("stopped_reason", r.stopped_reason);
    j.begin_array("kinds");
    for (const auto& k : r.kinds)
        j.begin_object().str("kind", k.kind == kOpKinds ? "composite" : to_string((OpKind)k.kind)).integer("cases", k.cases)
         .num("factor", k.factor).num("loo_median_error", k.loo_median_error).num("loo_max_error", k.loo_max_error)
         .num("raw_median_error", k.raw_median_error).end_object();
    j.end_array();
    j.begin_array("cases");
    for (const auto& c : r.results)
        j.begin_object().str("name", c.name).boolean("ok", c.ok).num("predicted_s", c.predicted_s).num("measured_s", c.measured_s)
         .num("ratio", c.ratio).num("noise", c.spread).num("loo_error", c.loo_error).str("note", c.note).end_object();
    j.end_array();
    j.begin_array("not_covered");
    for (const auto& s : r.not_covered) { j.begin_object().str("kind", s).end_object(); }
    j.end_array();
    j.end_object();
}

inline void print_report(const ProbeReport& r) {
    std::printf("\nWorkload model validation  (blas: %s, threads validated: %d, %.1f s)\n", r.blas.c_str(), r.threads_validated, r.time_used_s);
    std::printf("  %-26s %10s %10s %7s %6s %8s\n", "case", "predicted", "measured", "ratio", "noise", "LOO err");
    for (const auto& c : r.results) {
        if (c.ok) std::printf("  %-26s %8.2f ms %8.2f ms %6.2fx %5.0f%% %7.0f%%\n", c.name.c_str(), c.predicted_s * 1e3, c.measured_s * 1e3, c.ratio, c.spread * 100, c.loo_error * 100);
        else      std::printf("  %-26s  %s\n", c.name.c_str(), c.note.c_str());
    }
    std::printf("  per kind (factor = measured/predicted; LOO = error on a held-out case):\n");
    for (const auto& k : r.kinds)
        std::printf("    %-12s cases %d  factor %.2f  raw miss %3.0f%% -> LOO miss %3.0f%% (max %.0f%%)\n",
                    k.kind == kOpKinds ? "composite" : to_string((OpKind)k.kind), k.cases, k.factor,
                    k.raw_median_error * 100, k.loo_median_error * 100, k.loo_max_error * 100);
    std::printf("    %-12s factor %.2f  (applied to kinds with no probe)\n", "fallback", r.global_factor);
    std::printf("  overall: raw median miss %.0f%%, leave-one-out median miss %.0f%%\n", r.overall_raw_error * 100, r.overall_loo_error * 100);
    if (!r.not_covered.empty()) { std::printf("  not validated (no op to measure):"); for (const auto& s : r.not_covered) std::printf(" %s", s.c_str()); std::printf("\n"); }
}

}  // namespace tensorf::prof::wl
