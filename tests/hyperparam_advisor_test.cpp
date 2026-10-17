#include <cstdio>
#include <fstream>
#include "profiler/HyperparamAdvisor.hpp"
using namespace tensorf::profile;

static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++fails; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_UNDEFINED__)
  constexpr bool kPerf = false;
#else
  constexpr bool kPerf = true;
#endif
constexpr double KB = 1024.0, MB = 1024.0 * 1024.0;
static Metric M(double v) { Metric m; m.median = m.best = v; m.windows = 5; m.ok = true; return m; }

static CapabilityProfile synth() {
    CapabilityProfile P;
    P.gemm["f32_16"] = M(2); P.gemm["f32_64"] = M(5); P.gemm["f32_256"] = M(8); P.gemm["f32_512"] = M(7);
    P.gemm["f32_tall"] = M(6); P.gemm["f64_256"] = M(4);
    P.conv_gflops = M(4); P.rnn_gflops = 3; P.md_ginteractions = M(0.5); P.gather_gbps = 4;
    P.stream_1t = {{uint64_t(32 * KB), 0, 40}, {uint64_t(1 * MB), 0, 30}, {uint64_t(32 * MB), 0, 10}, {uint64_t(256 * MB), 0, 8}};
    P.latency = {{uint64_t(32 * KB), 2, 0}, {uint64_t(1 * MB), 8, 0}, {uint64_t(32 * MB), 100, 0}, {uint64_t(256 * MB), 120, 0}};
    P.scaling = {{1, 10, 8, 1, 1}, {2, 18, 14, 0.9, 1.75}, {4, 30, 20, 0.75, 2.5}};
    P.all_cores.gflops = {10, 9, 6, 5};
    P.fork_join_us = 3; P.call_overhead_ns = 30;
    P.sustained.ran = true; P.sustained.window_gflops = {10, 10, 9, 8, 8, 8}; P.sustained.fraction = 0.8;
    return P;
}
static MemoryBudget budget(uint64_t tensor_bytes) {
    MemoryBudget b; b.tensor_budget_bytes = tensor_bytes; b.max_additional_bytes = uint64_t(double(tensor_bytes) * 1.25);
    b.overhead_factor = 1.25;
    return b;
}
constexpr uint64_t BIG = 64ull << 30;

static uint64_t mem_of(const AlgorithmModel& m, Config c) { return m.costs(c).memory.total(); }

static void t_basic() {
    auto model = algo::make("mlp");
    AlgorithmProfiler prof(model, synth(), budget(BIG));
    HyperparamAdvisor adv(prof);
    auto a = adv.advise();
    CHECK(a.feasible && a.exhaustive);
    CHECK(a.config.at("batch") == 128 && a.config.at("dtype") == 0);        // semantic tunables untouched without opt-in
    CHECK(a.config.at("microbatch") == 128 && a.config.at("checkpoint") == 0);   // ample memory: whole batch, no recompute
    CHECK(a.evaluated == 16);                                                // 8 distinct microbatches x 2 checkpoint settings
    CHECK(a.plan.threads == 4 && a.plan.sustained);                          // 128 rows split 4 ways is worth it
    CHECK(a.predicted.feasible && a.predicted.time.units_per_s > 0 && !a.alternatives.empty());
    CHECK(a.alternatives[0].units_per_s == a.predicted.time.units_per_s || std::abs(a.alternatives[0].units_per_s - a.predicted.time.units_per_s) < 1e-6 * a.predicted.time.units_per_s);
    for (size_t i = 0; i + 1 < a.alternatives.size(); ++i) CHECK(a.alternatives[i].units_per_s >= a.alternatives[i + 1].units_per_s);
    CHECK(!a.calibrated && a.error == 1.0);                                  // never calibrated: the documented analytic spread
    bool uncal = false; for (auto& n : a.notes) uncal |= n.find("uncalibrated") != std::string::npos;
    CHECK(uncal);
}

static void t_memory() {
    auto model = algo::make("mlp");
    Config d = model->defaults();
    Config mb1 = d; mb1["microbatch"] = 1; mb1["checkpoint"] = 1;
    Config mb32 = d; mb32["microbatch"] = 32;
    const uint64_t m_default = mem_of(*model, d), m32 = mem_of(*model, mb32), m1c = mem_of(*model, mb1);
    CHECK(m1c < m32 && m32 < m_default);
    // budget fits mb=32 but not the whole batch
    {
        AlgorithmProfiler prof(model, synth(), budget(m32 + 1000));
        HyperparamAdvisor adv(prof);
        auto a = adv.advise();
        CHECK(a.feasible && a.predicted.fit.fits);
        CHECK(a.config.at("microbatch") < 128 || a.config.at("checkpoint") == 1);
        CHECK(a.predicted.costs.memory.total() <= m32 + 1000);
        AlgorithmProfiler loose(model, synth(), budget(BIG));
        auto b = HyperparamAdvisor(loose).advise();
        CHECK(a.predicted.time.units_per_s <= b.predicted.time.units_per_s * (1 + 1e-12));    // a constraint never helps
    }
    // budget so tight only microbatch 1 + checkpointing fits: memory is the binding constraint, and removing it gains > 5%
    {
        AlgorithmProfiler prof(model, synth(), budget(m1c + 10));
        HyperparamAdvisor adv(prof);
        auto a = adv.advise();
        CHECK(a.feasible && a.config.at("microbatch") <= 2);
        CHECK(a.binding == "memory" && a.gain_if_relaxed.count("memory") && a.gain_if_relaxed.at("memory") > 1.05);
    }
    // below even that: infeasible, reports the smallest configuration and the shortfall
    {
        const uint64_t params_only = 4ull * (784 * 512 + 512 + 512 * 256 + 256 + 2570) * 4 / 1;     // generous lower bound on weights+grads+opt
        AlgorithmProfiler prof(model, synth(), budget(params_only / 8));
        auto a = HyperparamAdvisor(prof).advise();
        CHECK(!a.feasible && a.binding == "memory" && !a.config.empty());
        bool msg = false; for (auto& n : a.notes) msg |= n.find("no configuration fits") != std::string::npos;
        CHECK(msg);
        CHECK(a.config.at("microbatch") == 1 || a.config.at("checkpoint") == 1);        // the smallest-memory configuration
    }
    // an unconstrained search says memory does not bind
    {
        AlgorithmProfiler prof(model, synth(), budget(BIG));
        auto a = HyperparamAdvisor(prof).advise();
        CHECK(a.binding == "compute" && a.gain_if_relaxed.at("memory") < 1.0001);
    }
}

static void t_opt_in() {
    auto model = algo::make("mlp");
    AlgorithmProfiler prof(model, synth(), budget(BIG));
    HyperparamAdvisor adv(prof);
    // opted in to batch with the whole range: largest batch wins on throughput
    auto a = adv.advise({}, OptIn{{"batch", {}}});
    {   // the pick is the brute-force best over the batch range (throughput need not grow with batch: cache spill)
        double best = 0, best_b = 0;
        for (double b : {32., 64., 128., 256., 512., 1024.}) {
            Config c = model->defaults(); c["batch"] = b;
            for (int th : {1, 2, 4}) for (double mb : {1., 2., 4., 8., 16., 32., 64., 128., 256., 512., 1024.}) for (double ck : {0., 1.}) {
                Config cc = c; cc["microbatch"] = std::min(mb, b); cc["checkpoint"] = ck;
                auto r = prof.evaluate(cc, ExecPlan{th, true});
                if (r.feasible && r.time.units_per_s > best) { best = r.time.units_per_s; best_b = b; }
            }
        }
        CHECK(a.feasible && a.config.at("dtype") == 0 && a.config.at("batch") == best_b);
        CHECK(std::abs(a.predicted.time.units_per_s - best) < 1e-9 * best);
    }
    // opted in with a subset
    auto b = adv.advise({}, OptIn{{"batch", {64, 256}}});
    CHECK(b.config.at("batch") == 256 || b.config.at("batch") == 64);
    auto b64 = adv.advise({}, OptIn{{"batch", {64}}});
    CHECK(b64.config.at("batch") == 64);
    // values outside the declared range are dropped with a note; unknown names are ignored with a note
    auto c = adv.advise({}, OptIn{{"batch", {64, 99999}}, {"nonsense", {}}, {"microbatch", {}}});
    CHECK(c.config.at("batch") == 64);
    int dropped = 0, unknown = 0, exec_note = 0;
    for (auto& n : c.notes) {
        dropped += n.find("outside the declared range") != std::string::npos;
        unknown += n.find("unknown tunable") != std::string::npos;
        exec_note += n.find("does not change the maths") != std::string::npos;
    }
    CHECK(dropped == 1 && unknown == 1 && exec_note == 1);
    // opting in to precision: allowed to pick it, and only from the allowed set
    auto d = adv.advise({}, OptIn{{"dtype", {double(int(Dtype::F16))}}});
    CHECK(d.config.at("dtype") == double(int(Dtype::F16)));                    // single allowed value: that one
    // forced values win over opt-in, and an out-of-range forced value is accepted with a note
    Constraints cf; cf.fixed["batch"] = 64; cf.fixed["microbatch"] = 16;
    auto e = adv.advise(cf, OptIn{{"batch", {}}});
    CHECK(e.config.at("batch") == 64 && e.config.at("microbatch") == 16);
    Constraints odd; odd.fixed["batch"] = 100; odd.fixed["bogus"] = 1;
    auto f = adv.advise(odd);
    CHECK(f.config.at("batch") == 100);
    int range_note = 0, bogus_note = 0;
    for (auto& n : f.notes) { range_note += n.find("outside the declared range") != std::string::npos; bogus_note += n.find("unknown tunable") != std::string::npos; }
    CHECK(range_note == 1 && bogus_note == 1);
}

static void t_all_models_semantic_untouched() {
    for (const auto& name : algo::names()) {
        auto model = algo::make(name);
        AlgorithmProfiler prof(model, synth(), budget(BIG));
        auto a = HyperparamAdvisor(prof).advise();
        CHECK(a.feasible);
        for (const auto& t : model->tunables())
            if (t.semantic) CHECK(a.config.at(t.name) == t.def);                // the maths is never changed without opt-in
        auto b = HyperparamAdvisor(prof).advise({}, [&] { OptIn o; for (auto& t : model->tunables()) if (t.semantic) o[t.name] = {}; return o; }());
        CHECK(b.feasible && b.predicted.time.units_per_s >= a.predicted.time.units_per_s * (1 - 1e-12));   // more freedom never hurts
    }
}

static void t_step_cap() {
    auto model = algo::make("mlp");
    AlgorithmProfiler prof(model, synth(), budget(BIG));
    HyperparamAdvisor adv(prof);
    auto base = adv.advise();
    const double t0 = base.predicted.time.seconds;
    Constraints cap; cap.max_step_seconds = 0.4 * t0;
    // without opt-in the batch is fixed, so the cap cannot be met: infeasible, says why, and suggests the opt-in
    auto a = adv.advise(cap);
    CHECK(!a.feasible && a.binding == "step_time");
    bool msg = false; for (auto& n : a.notes) msg |= n.find("step-time cap") != std::string::npos;
    CHECK(msg);
    CHECK(a.unlock && std::find(a.unlock->opt_in.begin(), a.unlock->opt_in.end(), "batch") != a.unlock->opt_in.end());
    CHECK(a.to_json().find("\"unlock\":{") != std::string::npos);
    // opted in: a smaller batch meets the cap
    auto b = adv.advise(cap, OptIn{{"batch", {}}});
    CHECK(b.feasible && b.predicted.time.seconds <= cap.max_step_seconds * (1 + 1e-12) && b.config.at("batch") < 128);
    // a generous cap changes nothing
    Constraints loose; loose.max_step_seconds = 100 * t0;
    auto c = adv.advise(loose);
    CHECK(c.feasible && c.config == base.config && c.plan.threads == base.plan.threads);
    CHECK(c.gain_if_relaxed.at("step_time") < 1.0001);
}

static void t_threads() {
    auto mlp = algo::make("mlp");
    AlgorithmProfiler prof(mlp, synth(), budget(BIG));
    HyperparamAdvisor adv(prof);
    Constraints one; one.max_threads = 1;
    auto a = adv.advise(one);
    CHECK(a.plan.threads == 1 && a.gain_if_relaxed.at("threads") > 2.0);        // 4 threads would be ~3x
    CHECK(a.binding == "threads");
    // tolerance: with a loose tolerance fewer threads are enough
    Constraints loose; loose.thread_tolerance = 0.9;
    CHECK(adv.advise(loose).plan.threads == 2);
    Constraints tight; tight.thread_tolerance = 0.0;
    CHECK(adv.advise(tight).plan.threads == 4);
    // nothing to parallelise -> one thread
    auto rnn = algo::make("rnn");
    AlgorithmProfiler rp(rnn, synth(), budget(BIG));
    Constraints fb; fb.fixed["batch"] = 1; fb.fixed["bptt"] = 16;
    auto r = HyperparamAdvisor(rp).advise(fb);
    CHECK(r.feasible && r.plan.threads == 1);
    // the governor's cap is respected even when more threads were measured
    Constraints two; two.max_threads = 2;
    CHECK(adv.advise(two).plan.threads <= 2);
}

static void t_search_modes() {
    auto model = algo::make("transformer");
    AlgorithmProfiler prof(model, synth(), budget(BIG));
    HyperparamAdvisor adv(prof);
    OptIn all; for (auto& t : model->tunables()) if (t.semantic) all[t.name] = {};
    SearchOptions ex;                               // exhaustive
    auto full = adv.advise({}, all, ex);
    CHECK(full.exhaustive && full.evaluated > 500);
    SearchOptions ds; ds.max_grid = 100;            // forces coordinate descent
    auto desc = adv.advise({}, all, ds);
    CHECK(!desc.exhaustive && desc.feasible);
    CHECK(desc.evaluated < full.evaluated);
    CHECK(desc.predicted.time.units_per_s >= 0.9 * full.predicted.time.units_per_s);
    CHECK(desc.predicted.time.units_per_s <= full.predicted.time.units_per_s * (1 + 1e-9));
    // descent also finds feasibility when the default does not fit
    auto tm = algo::make("transformer");
    Config d = tm->defaults(); Config tiny = d; tiny["microbatch"] = 1; tiny["checkpoint"] = 1;
    AlgorithmProfiler tight(tm, synth(), budget(mem_of(*tm, tiny) + 1000));
    auto t = HyperparamAdvisor(tight).advise({}, {}, ds);
    CHECK(t.feasible && t.predicted.fit.fits);
}

static void t_unlock() {
    auto gnn = algo::make("gnn");
    AlgorithmProfiler prof(gnn, synth(), budget(BIG));
    HyperparamAdvisor adv(prof);
    auto a = adv.advise();
    CHECK(a.feasible && a.unlock);
    if (a.unlock) {
        CHECK(a.unlock->gain > 1.5 && !a.unlock->changes.empty() && !a.unlock->opt_in.empty());
        for (auto& o : a.unlock->opt_in) CHECK(a.config.at(o) == gnn->find_tunable(o)->def);       // the suggestion was NOT applied
        // applying exactly the suggested opt-ins reaches the suggested throughput
        OptIn oi; for (auto& o : a.unlock->opt_in) oi[o] = {};
        auto applied = adv.advise({}, oi);
        CHECK(applied.predicted.time.units_per_s >= a.unlock->units_per_s * 0.999);
    }
    // fully opted-in: nothing left to suggest
    OptIn all; for (auto& t : gnn->tunables()) if (t.semantic) all[t.name] = {};
    CHECK(!adv.advise({}, all).unlock);
    // unlock can be switched off
    SearchOptions off; off.unlock_hint = false;
    CHECK(!adv.advise({}, {}, off).unlock);
    // a particle simulation fixes its size; the only semantic knobs are accuracy ones
    auto pm = algo::make("particles");
    AlgorithmProfiler pp(pm, synth(), budget(BIG));
    auto p = HyperparamAdvisor(pp).advise();
    CHECK(p.feasible && p.config.at("neighbors") == 64 && p.config.at("batch") == 20000);
    CHECK(p.unlock && p.unlock->gain > 1.5);                                  // fewer neighbours would be much faster
}

static void t_trials() {
    auto model = algo::make("mlp");
    AlgorithmProfiler prof(model, synth(), budget(BIG));
    HyperparamAdvisor adv(prof);
    auto pure = adv.advise();
    const Config predicted_best = pure.config;
    // reality disagrees with the model: the predicted-best configuration is 3x slower than predicted, others 1.5x
    int calls = 0;
    ProbeFn probe = [&](const Config& c, const ExecPlan& p) {
        ++calls;
        AlgorithmProfiler ref(model, synth(), budget(BIG));
        const double raw = ref.raw_prediction(c, p).seconds;
        return raw * (c == predicted_best ? 3.0 : 1.5);
    };
    SearchOptions so; so.trial_count = 3; so.trial_repeats = 2;
    auto a = adv.advise({}, {}, so, &probe);
    CHECK(calls == 6 && a.trials_run == 3 && a.measured);
    CHECK(a.config != predicted_best);                                         // the measured winner replaced the predicted winner
    CHECK(a.calibrated && a.error > 0);                                        // fitted from the trials, spread reported
    CHECK(std::abs(a.measured_units_per_s - a.predicted.costs.units / a.measured_seconds) < 1e-6 * a.measured_units_per_s);
    // after fitting, predictions are in the measured range (scale between 1.5 and 3)
    CHECK(prof.scale() > 1.5 && prof.scale() < 3.0);
    // governor can refuse trials
    {
        AlgorithmProfiler p2(model, synth(), budget(BIG));
        int n = 0; ProbeFn pr = [&](const Config&, const ExecPlan&) { ++n; return 1.0; };
        auto b = HyperparamAdvisor(p2).advise({}, {}, so, &pr, [] { return false; });
        CHECK(n == 0 && !b.measured && b.feasible);
        bool note = false; for (auto& x : b.notes) note |= x.find("declined") != std::string::npos;
        CHECK(note);
    }
    // a run predicted to be too long is never started
    {
        AlgorithmProfiler p3(model, synth(), budget(BIG));
        int n = 0; ProbeFn pr = [&](const Config&, const ExecPlan&) { ++n; return 1.0; };
        SearchOptions s2; s2.trial_max_seconds = 1e-9;
        auto b = HyperparamAdvisor(p3).advise({}, {}, s2, &pr);
        CHECK(n == 0 && !b.measured);
        bool note = false; for (auto& x : b.notes) note |= x.find("longer than trial_max_seconds") != std::string::npos;
        CHECK(note);
    }
    // unusable probe results are ignored
    {
        AlgorithmProfiler p4(model, synth(), budget(BIG));
        ProbeFn bad = [&](const Config&, const ExecPlan&) { return std::nan(""); };
        auto b = HyperparamAdvisor(p4).advise({}, {}, so, &bad);
        CHECK(b.feasible && !b.measured && b.trials_run == 0 && p4.scale() == 1.0);
    }
    // a measured candidate over the step cap is not chosen
    {
        AlgorithmProfiler p5(model, synth(), budget(BIG));
        Constraints cap; cap.max_step_seconds = 0.2;
        ProbeFn slow = [&](const Config&, const ExecPlan&) { return 5.0; };      // reality: every step takes 5 s
        auto b = HyperparamAdvisor(p5).advise(cap, {}, so, &slow);
        CHECK(!b.measured);
        bool note = false; for (auto& x : b.notes) note |= x.find("exceeded the step-time cap") != std::string::npos;
        CHECK(note);
    }
}

static void t_determinism_json() {
    auto model = algo::make("gnn");
    OptIn all; for (auto& t : model->tunables()) if (t.semantic) all[t.name] = {};
    auto run = [&] {
        AlgorithmProfiler prof(model, synth(), budget(BIG));
        return HyperparamAdvisor(prof).advise({}, all).to_json();
    };
    auto j1 = run(), j2 = run();
    CHECK(j1 == j2);
    std::ofstream("/tmp/advice.json") << j1;
    AlgorithmProfiler prof(model, synth(), budget(BIG));
    auto a = HyperparamAdvisor(prof).advise();
    CHECK(!a.summary().empty() && a.summary().find("limited by") != std::string::npos);
}

static void t_real_device() {
    auto fp = collect();
    auto cfg = BenchConfig::quick();
    cfg.sustained_seconds = kPerf ? 1.0 : 0;
    if (!kPerf) cfg.budget_seconds *= 10;
    auto cap = run_device_benchmark(fp, cfg);
    ProcessMemory proc = read_process_memory();
    BudgetPolicy pol; pol.overhead_factor = measure_allocator_overhead([] { auto p = read_process_memory(); return p.ok ? std::optional<uint64_t>(p.rss) : std::nullopt; }).factor;
    auto budget_real = plan_budget(fp, proc, pol);
    std::printf("---- end to end on this machine: budget %llu MB of tensors ----\n", (unsigned long long)(budget_real.tensor_budget_bytes >> 20));
    for (const auto& name : algo::names()) {
        AlgorithmProfiler prof(algo::make(name), cap, budget_real);
        OptIn all; for (auto& t : prof.model().tunables()) if (t.semantic) all[t.name] = {};
        auto a = HyperparamAdvisor(prof).advise({}, all);
        std::printf("[%s] %s", name.c_str(), a.summary().c_str());
        CHECK(a.feasible && a.predicted.time.units_per_s > 0 && a.plan.threads >= 1 && a.plan.threads <= fp.usable_threads);
    }
}

int main() {
    t_basic(); t_memory(); t_opt_in(); t_all_models_semantic_untouched(); t_step_cap(); t_threads(); t_search_modes(); t_unlock(); t_trials();
    t_determinism_json(); t_real_device();
    std::printf("%d/%d checks passed\n", checks - fails, checks);
    return fails ? 1 : 0;
}
