/*
 * profiler_advisor_test.cpp
 * =========================
 * Tests for Advisor.hpp (step 4 of the profiler rewrite).
 *
 *   g++ -std=c++20 -O2 -Wall -Wextra -Iinclude -Iinclude/core -Iinclude/tools \
 *       tests/profiler_advisor_test.cpp -o bin/profiler_advisor_test -pthread
 *
 * Everything runs on a SYNTHETIC device (round numbers, see make_cv / make_fp), so the advisor's
 * decisions are deterministic and independent of the machine running the test.
 *
 * Ground truth never comes from the code under test: where the question is "did the search find the
 * best configuration?", the test enumerates the space itself with the same factory and predict().
 */

#include <cstdio>
#include <cstring>
#include <set>

#include "profiler/tmp/Advisor.hpp"

using namespace tensorf::prof;
using namespace tensorf::prof::wl;
namespace A = tensorf::prof::adv;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failed; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while (0)
#define CHECK_REL(a, b, rel) do { ++g_checks; double _a=(double)(a), _b=(double)(b); if (!(std::fabs(_a-_b) <= (rel) * std::fabs(_b))) { ++g_failed; \
    std::printf("  FAIL %s:%d  %s ~ %s within %g%%  (%g vs %g)\n", __FILE__, __LINE__, #a, #b, (rel) * 100.0, _a, _b); } } while (0)
#define SECTION(name) std::printf("[%s]\n", name)

// ─── Synthetic device ────────────────────────────────────────────────────────

static CapabilityVector make_cv() {
    CapabilityVector cv;
    cv.f32_gflops_best = 10; cv.f32_gflops_skinny = 4; cv.i8_gops = 20; cv.conv_gflops = 8; cv.nbody_gflops = 12; cv.f64_gflops = 5;
    cv.bw_l1_gbs = 100; cv.bw_l2_gbs = 60; cv.bw_l3_gbs = 30; cv.bw_dram_gbs = 10;
    cv.triad_1t_gbs = 10; cv.triad_max_gbs = 20; cv.triad_sweep = {{1, 10.0}, {2, 18.0}, {4, 20.0}};
    cv.dram_saturation_threads = 4; cv.gather_gbs = 2.0;
    cv.tiny_op_ns = 10; cv.barrier_roundtrip_us = 2.0;
    cv.recommended_class = CoreClass::UNIFORM;
    ClusterScaling sc; sc.cls = CoreClass::UNIFORM; sc.cpus = 4;
    sc.independent.points = {{1, 10.0, 1.0}, {2, 19.0, 0.95}, {4, 32.0, 0.8}};
    sc.independent.ok = true;
    cv.scaling.push_back(sc);
    return cv;
}

static DeviceFingerprint make_fp(uint64_t budget = 4096ull << 20) {
    DeviceFingerprint fp;
    CoreCluster c; c.cpus = {0, 1, 2, 3}; c.physical_cores = 4;
    c.l1d_bytes = 32 << 10; c.l2_bytes = 256 << 10; c.l3_bytes = 8ull << 20;
    fp.clusters.push_back(c);
    fp.logical_cpus = 4;
    fp.memory.total_bytes = budget * 2; fp.memory.available_bytes = budget; fp.memory.safe_budget_bytes = budget;
    return fp;
}

// A 4-layer 512-wide MLP with checkpointing, threads, precision and batch as tunables.
static Algorithm make_alg(bool with_threads = true) {
    Algorithm a; a.name = "mlp";
    a.tunables = {
        {"micro_batch", {8, 16, 32, 64, 128}, true, "step"},
        {"ckpt", {1, 2, 4}, true, "step"},
        {"precision_bits", {32, 16}, false, "step"},
        {"batch", {128, 256}, false, "step"},
    };
    if (with_threads) a.tunables.insert(a.tunables.begin() + 1, Tunable{"threads", {1, 2, 4}, true, "step"});
    a.build = [](const Config& c) {
        const double mb = c.at("micro_batch"), batch = c.at("batch");
        RunSpec r; r.mode = Mode::TRAIN; r.opt_states = 2;
        r.dtype = c.at("precision_bits") == 16 ? Dtype::F16 : Dtype::F32;
        Workload w; w.name = "mlp"; w.units_per_micro_step = mb; w.micro_steps = (int)std::max(1.0, std::round(batch / mb));
        for (int i = 0; i < 4; ++i) w.layers.push_back(dense(mb, 512, 512, r, "fc" + std::to_string(i)));
        w.layers.push_back(elementwise(mb * 512, 1, 2, r, "act"));
        w.layers.push_back(optimizer_update(4.0 * 512 * 512, r));
        const int ck = (int)c.at("ckpt");
        if (ck > 1) w = with_checkpointing(w, ck);
        return w;
    };
    return a;
}

static Config base_cfg() { return {{"micro_batch", 32}, {"threads", 1}, {"ckpt", 1}, {"precision_bits", 32}, {"batch", 128}}; }

// Independent ground truth: enumerate the result-preserving space (altering fixed at base) with the same factory.
struct Truth { bool found = false; Config cfg; double ups = 0; double step = 0; double mem = 0; size_t feasible = 0; };
static Truth brute_force(const Algorithm& alg, const Config& base, const DeviceFingerprint& fp, const CapabilityVector& cv,
                         double mem_fraction, int max_threads = 0, double max_step = 0) {
    Truth t;
    const std::vector<double> mbs = {8, 16, 32, 64, 128}, thr = {1, 2, 4}, cks = {1, 2, 4};
    for (double mb : mbs) for (double th : thr) for (double ck : cks) {
        if (max_threads > 0 && th > max_threads) continue;
        Config c = base; c["micro_batch"] = mb; c["threads"] = th; c["ckpt"] = ck;
        Workload w = alg.build(c);
        PredictOptions po; po.threads = (int)th; po.memory_fraction = mem_fraction; po.mem_overhead_factor = 1.25;
        Prediction p = predict(w, fp, cv, po);
        if (!p.valid || !p.memory_fits) continue;
        if (max_step > 0 && p.step_s > max_step) continue;
        ++t.feasible;
        if (!t.found || p.units_per_s > t.ups * (1 + 1e-12)) { t.found = true; t.cfg = c; t.ups = p.units_per_s; t.step = p.step_s; t.mem = p.memory.process_total; }
    }
    return t;
}

static bool has_note(const A::Advice& a, const char* needle) {
    for (const auto& n : a.notes) if (n.find(needle) != std::string::npos) return true;
    return false;
}

static bool balanced_clean(const std::string& s) {
    int brace = 0, brack = 0; bool in_str = false, esc = false;
    for (char ch : s) {
        if (in_str) { if (esc) esc = false; else if (ch == '\\') esc = true; else if (ch == '"') in_str = false; continue; }
        if (ch == '"') in_str = true;
        else if (ch == '{') ++brace; else if (ch == '}') --brace;
        else if (ch == '[') ++brack; else if (ch == ']') --brack;
        if (brace < 0 || brack < 0) return false;
    }
    const std::string low = to_lower(s);
    for (const char* bad : {":nan", ":-nan", ":inf", ":-inf", ",nan", ",inf", "[nan", "[inf"}) if (low.find(bad) != std::string::npos) return false;
    return !in_str && brace == 0 && brack == 0;
}

// ═════════════════════════════════════════════════════════════════════════════

static void test_finds_the_true_best() {
    SECTION("search: finds the same optimum as an independent brute force");
    Algorithm alg = make_alg(); CapabilityVector cv = make_cv();
    for (uint64_t mb : {4096ull << 20, 40ull << 20, 30ull << 20}) {                     // roomy, memory-limited, tightly memory-limited
        DeviceFingerprint fp = make_fp(mb);
        A::Constraints c;                                                                // memory_fraction 0.6, altering off
        A::Advice a = A::advise(alg, base_cfg(), fp, cv, nullptr, c);
        Truth t = brute_force(alg, base_cfg(), fp, cv, c.memory_fraction);
        CHECK(a.ok == t.found);
        if (!a.ok || !t.found) continue;
        CHECK(a.config == t.cfg);
        CHECK_REL(a.prediction.units_per_s, t.ups, 1e-12);
        CHECK(a.prediction.memory_fits);                                                 // never recommends something that does not fit
        CHECK(a.search_mode == "exhaustive");
        CHECK(a.evaluated == 45);                                                        // 5 micro-batches x 3 threads x 3 checkpoint settings
    }
}

static void test_altering_tunables() {
    SECTION("result-changing tunables are held at the base config unless allowed");
    Algorithm alg = make_alg(); CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    A::Constraints off;                                                                   // allow_altering = false
    A::Advice a = A::advise(alg, base_cfg(), fp, cv, nullptr, off);
    CHECK(a.ok);
    CHECK(a.config.at("precision_bits") == 32 && a.config.at("batch") == 128);            // untouched
    A::Constraints on; on.allow_altering = true;
    A::Advice b = A::advise(alg, base_cfg(), fp, cv, nullptr, on);
    CHECK(b.ok);
    CHECK(b.evaluated == 180);                                                            // x2 precisions x2 batches
    CHECK(b.prediction.units_per_s >= a.prediction.units_per_s * (1 - 1e-12));            // a superset search can never do worse
    // The advisor tells the caller what it left on the table, exactly when it is worth mentioning.
    const double gain = b.prediction.units_per_s / a.prediction.units_per_s - 1.0;
    CHECK(has_note(a, "result-changing tunables could raise throughput") == (gain > 0.02));
    CHECK(!has_note(b, "result-changing tunables could raise throughput"));
}

static void test_binding_constraints() {
    SECTION("binding constraint: lift each in turn, report the one that matters");
    Algorithm alg = make_alg(); CapabilityVector cv = make_cv();
    {   // nothing limits it but the hardware
        A::Advice a = A::advise(alg, base_cfg(), make_fp(), cv);
        CHECK(a.ok && a.binding.rfind("hardware:", 0) == 0);
        CHECK(a.binding == "hardware: compute-bound");
        CHECK(a.binding_gain == 0.0);
    }
    {   // memory binds
        DeviceFingerprint fp = make_fp(40ull << 20);
        A::Advice a = A::advise(alg, base_cfg(), fp, cv);
        CHECK(a.ok && a.binding == "memory");
        CHECK(a.binding_gain > 0.5);                                                      // lifting it would help a lot
        CHECK(a.infeasible_memory > 0);
        A::Advice roomy = A::advise(alg, base_cfg(), make_fp(), cv);
        CHECK(a.prediction.units_per_s < roomy.prediction.units_per_s);                   // and the constrained answer is genuinely slower
        CHECK_REL(roomy.prediction.units_per_s / a.prediction.units_per_s - 1.0, a.binding_gain, 1e-9);   // the reported gain IS that ratio
    }
    {   // thread cap binds
        A::Constraints c; c.max_threads = 2;
        A::Advice a = A::advise(alg, base_cfg(), make_fp(), cv, nullptr, c);
        CHECK(a.ok && a.binding == "thread cap");
        CHECK(a.config.at("threads") <= 2);
        CHECK(a.binding_gain > 0.3);
        Truth t = brute_force(alg, base_cfg(), make_fp(), cv, c.memory_fraction, 2);
        CHECK(a.config == t.cfg);
        // a cap of 1 allows only one thread; a cap of 3 excludes 4 but allows 2
        A::Constraints c1; c1.max_threads = 1;
        CHECK(A::advise(alg, base_cfg(), make_fp(), cv, nullptr, c1).config.at("threads") == 1);
        A::Constraints c3; c3.max_threads = 3;
        CHECK(A::advise(alg, base_cfg(), make_fp(), cv, nullptr, c3).config.at("threads") == 2);
    }
    {   // a step-time ceiling that no configuration can meet
        Truth t = brute_force(alg, base_cfg(), make_fp(), cv, 0.6);
        A::Constraints c; c.max_step_s = t.step * 0.5;
        A::Advice a = A::advise(alg, base_cfg(), make_fp(), cv, nullptr, c);
        CHECK(!a.ok && a.binding == "step time");
        CHECK(has_note(a, "step-time ceiling"));
        // a ceiling with room: respected, and not reported as binding
        A::Constraints loose; loose.max_step_s = t.step * 1.2;
        A::Advice b = A::advise(alg, base_cfg(), make_fp(), cv, nullptr, loose);
        CHECK(b.ok && b.prediction.step_s <= loose.max_step_s);
        CHECK(b.binding != "step time");
    }
    {   // a throughput floor nothing reaches
        A::Constraints c; c.min_units_per_s = 1e9;
        A::Advice a = A::advise(alg, base_cfg(), make_fp(), cv, nullptr, c);
        CHECK(!a.ok && a.binding == "throughput floor");
    }
}

static void test_nothing_fits() {
    SECTION("memory: when nothing fits, say so and say by how much");
    Algorithm alg = make_alg(); CapabilityVector cv = make_cv();
    A::Advice a = A::advise(alg, base_cfg(), make_fp(12ull << 20), cv);
    CHECK(!a.ok);
    CHECK(a.binding == "memory");
    CHECK(has_note(a, "smallest needs about"));
    CHECK(a.infeasible_memory == a.evaluated);                                            // every candidate failed on memory
    // the reported smallest requirement is real: it matches the minimum over an independent enumeration
    double minmem = 1e30;
    for (double mb : {8., 16., 32., 64., 128.}) for (double ck : {1., 2., 4.}) {
        Config c = base_cfg(); c["micro_batch"] = mb; c["ckpt"] = ck;
        minmem = std::min(minmem, predict(alg.build(c), make_fp(12ull << 20), cv, {}, nullptr).memory.process_total);
    }
    char want[64]; std::snprintf(want, sizeof want, "%lld MB", (long long)(minmem / 1048576.0));
    CHECK(has_note(a, want));
    // The memory fraction is the governor's lever: the same device with a larger share fits.
    A::Constraints big; big.memory_fraction = 1.0;
    CHECK(A::advise(alg, base_cfg(), make_fp(26ull << 20), cv, nullptr, big).ok);
    CHECK(!A::advise(alg, base_cfg(), make_fp(26ull << 20), cv).ok);                      // at the default 60% it does not
}

static void test_coordinate_ascent() {
    SECTION("large spaces: coordinate ascent, deterministic and sound");
    Algorithm alg = make_alg(); CapabilityVector cv = make_cv();
    A::AdvisorOptions small; small.max_candidates = 10;                                   // force it on a 45-config space
    for (uint64_t mb : {4096ull << 20, 40ull << 20}) {
        DeviceFingerprint fp = make_fp(mb);
        A::Advice ex = A::advise(alg, base_cfg(), fp, cv);
        A::Advice co = A::advise(alg, base_cfg(), fp, cv, nullptr, {}, small);
        CHECK(co.search_mode == "coordinate ascent" && ex.search_mode == "exhaustive");
        CHECK(co.ok && ex.ok);
        CHECK(co.evaluated <= ex.evaluated);                                              // revisits are cached, never more work than exhaustive
        CHECK(co.prediction.units_per_s >= 0.9 * ex.prediction.units_per_s);
        CHECK(co.prediction.memory_fits);
        std::printf("  (info) budget %llu MB: coordinate ascent reached %.1f%% of the exhaustive optimum in %zu of %zu evaluations\n",
                    (unsigned long long)(mb >> 20), 100.0 * co.prediction.units_per_s / ex.prediction.units_per_s, co.evaluated, ex.evaluated);
        for (int i = 0; i < 3; ++i) CHECK(A::advise(alg, base_cfg(), fp, cv, nullptr, {}, small).config == co.config);   // deterministic
    }
    {   // it recovers from an INFEASIBLE starting point
        Config start = base_cfg(); start["micro_batch"] = 128; start["ckpt"] = 1;         // needs ~28 MB; only 24 MB allowed
        A::Advice co = A::advise(alg, start, make_fp(40ull << 20), cv, nullptr, {}, small);
        CHECK(co.ok && co.prediction.memory_fits);
    }
    {   // a space far beyond max_candidates: 8 dimensions x 4 values = 65,536 configs. Throughput rises strictly with
        // each dimension (a fixed per-op overhead is amortised over more rows), so the optimum is "all at 4".
        Algorithm big; big.name = "big";
        for (int d = 0; d < 8; ++d) big.tunables.push_back({"d" + std::to_string(d), {1, 2, 3, 4}, true, "step"});
        big.build = [](const Config& c) {
            double rows = 0; for (const auto& kv : c) rows += kv.second;
            RunSpec r; r.mode = Mode::INFER;
            Workload w; w.units_per_micro_step = 16.0 * rows;
            w.layers = {dense(16.0 * rows, 256, 256, r, "fc", false)};
            return w;
        };
        Config start; for (int d = 0; d < 8; ++d) start["d" + std::to_string(d)] = 1;
        A::Advice a = A::advise(big, start, make_fp(), cv);
        CHECK(a.ok && a.search_mode == "coordinate ascent");
        CHECK(a.evaluated < 500);                                                         // nowhere near 65,536
        bool all4 = true; for (const auto& kv : a.config) if (kv.second != 4) all4 = false;
        CHECK(all4);
    }
}

static void test_trials() {
    SECTION("trials: measurement overrules the model");
    Algorithm alg = make_alg(); CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    A::Advice model_only = A::advise(alg, base_cfg(), fp, cv);
    CHECK(model_only.top.size() >= 3);
    const Config c0 = model_only.top[0].cfg, c1 = model_only.top[1].cfg, c2 = model_only.top[2].cfg;
    const double p0 = model_only.top[0].pred.step_s, p1 = model_only.top[1].pred.step_s, p2 = model_only.top[2].pred.step_s;
    CHECK(!model_only.confirmed && model_only.trials.empty());
    CHECK_REL(model_only.units_per_s, model_only.prediction.units_per_s, 1e-12);          // unconfirmed: the central estimate IS the model's

    {   // the model's #1 is really slower, and its #2 faster
        A::TrialFn trial = [&](const Config& c) { return c == c0 ? p0 * 1.5 : c == c1 ? p1 * 0.5 : c == c2 ? p2 * 1.0 : -1.0; };
        A::Advice a = A::advise(alg, base_cfg(), fp, cv, nullptr, {}, {}, trial);
        CHECK(a.ok && a.confirmed);
        CHECK(a.ranking_changed);
        CHECK(a.config == c1);                                                            // chosen by measurement, not by the model
        CHECK(a.trials.size() == 3);
        for (const auto& t : a.trials) CHECK(t.ok);
        // central estimate is the MEASURED throughput
        const double units = model_only.top[1].pred.units_per_s * p1;
        CHECK_REL(a.units_per_s, units / (p1 * 0.5), 1e-9);
        // errors: |1.5-1|/1.5, |0.5-1|/0.5, 0  -> median is 1/3 (0.333) ... sorted {0, 0.333, 1.0}
        CHECK_REL(a.trial_error, 1.0 / 3.0, 1e-9);
        CHECK(a.error_basis == "trials");
        CHECK_REL(a.rel_error, std::max(a.trial_error, 0.05), 1e-12);
        CHECK(a.trial_observations.size() == 3);
        for (const auto& o : a.trial_observations) CHECK(o.kind == kOpKinds && o.predicted_s > 0 && o.measured_s > 0);
        // those observations are directly usable to improve the next prediction
        Calibration cal; cal.fit(a.trial_observations);
        CHECK(cal.calibrated(kOpKinds));
        CHECK(!has_note(a, "differ by less than the model's error"));                      // settled by measurement, so no tie warning
    }
    {   // when measurement agrees, nothing changes
        A::TrialFn trial = [&](const Config& c) { return c == c0 ? p0 : c == c1 ? p1 : c == c2 ? p2 : -1.0; };
        A::Advice a = A::advise(alg, base_cfg(), fp, cv, nullptr, {}, {}, trial);
        CHECK(a.confirmed && !a.ranking_changed && a.config == c0);
        CHECK_REL(a.trial_error, 0.0, 1.0);
        CHECK(a.trial_error < 1e-9);
    }
    {   // a trial that fails (throws / NaN / non-positive) is skipped, the others still count
        A::TrialFn trial = [&](const Config& c) -> double {
            if (c == c0) throw std::runtime_error("device busy");
            if (c == c1) return std::numeric_limits<double>::quiet_NaN();
            return p2 * 1.0;
        };
        A::Advice a = A::advise(alg, base_cfg(), fp, cv, nullptr, {}, {}, trial);
        CHECK(a.ok && a.confirmed && a.config == c2);
        CHECK(a.trials.size() == 3 && !a.trials[0].ok && !a.trials[1].ok && a.trials[2].ok);
        CHECK(a.trials[0].note.find("device busy") != std::string::npos);
        CHECK(a.error_basis == "model");                                                  // one good trial is not enough to re-estimate the error
    }
    {   // every trial fails: keep the model's choice and say so
        A::TrialFn trial = [](const Config&) -> double { return -1.0; };
        A::Advice a = A::advise(alg, base_cfg(), fp, cv, nullptr, {}, {}, trial);
        CHECK(a.ok && !a.confirmed && a.config == c0);
        CHECK(has_note(a, "every trial failed"));
    }
    {   // no budget: no trials run
        int calls = 0;
        A::TrialFn trial = [&](const Config&) { ++calls; return 1.0; };
        A::AdvisorOptions o; o.trial_budget_s = 0;
        A::Advice a = A::advise(alg, base_cfg(), fp, cv, nullptr, {}, o, trial);
        CHECK(calls == 0 && !a.confirmed && a.config == c0);
        CHECK(has_note(a, "trial budget is zero"));
    }
    {   // the budget stops later trials
        int calls = 0;
        A::TrialFn trial = [&](const Config& c) { ++calls; { const double t0 = now_s(); while (now_s() - t0 < 0.06) {} } return c == c0 ? p0 : p1; };
        A::AdvisorOptions o; o.trial_budget_s = 0.05; o.trial_top = 3;
        A::Advice a = A::advise(alg, base_cfg(), fp, cv, nullptr, {}, o, trial);
        CHECK(calls == 1 && a.trials.size() == 1);                                        // the first ate the whole budget
    }
    {   // trials only ever run configurations that are feasible
        std::set<Config> asked;
        A::TrialFn trial = [&](const Config& c) { asked.insert(c); return 0.01; };
        DeviceFingerprint tight = make_fp(40ull << 20);
        A::Advice a = A::advise(alg, base_cfg(), tight, cv, nullptr, {}, {}, trial);
        CHECK(a.ok);
        for (const auto& c : asked) {
            PredictOptions po; po.threads = (int)c.at("threads"); po.memory_fraction = 0.6;
            CHECK(predict(alg.build(c), tight, cv, po).memory_fits);
        }
    }
}

static void test_factory_rejection() {
    SECTION("factory: invalid combinations are infeasible, not fatal");
    CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    Algorithm alg = make_alg();
    auto inner = alg.build;
    alg.build = [inner](const Config& c) { if (c.at("micro_batch") == 128) throw std::invalid_argument("128 rows do not divide the graph"); return inner(c); };
    A::Advice a = A::advise(alg, base_cfg(), fp, cv);
    CHECK(a.ok && a.config.at("micro_batch") != 128);
    for (const auto& c : a.top) CHECK(c.cfg.at("micro_batch") != 128);
    CHECK(a.evaluated == 45);                                                             // the rejected ones were still tried

    Algorithm none = make_alg();
    none.build = [](const Config&) -> Workload { throw std::runtime_error("always"); };
    A::Advice b = A::advise(none, base_cfg(), fp, cv);
    CHECK(!b.ok && b.binding == "no feasible configuration");

    Algorithm nofactory; nofactory.name = "x";
    A::Advice c = A::advise(nofactory, {}, fp, cv);
    CHECK(!c.ok && has_note(c, "no factory"));
}

static void test_inputs_and_notes() {
    SECTION("inputs: missing base values, threads convention, degenerate devices");
    CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    {   // base config without a value: the tunable's first candidate, announced
        Algorithm alg = make_alg();
        A::Advice a = A::advise(alg, {}, fp, cv);
        CHECK(a.ok);
        CHECK(has_note(a, "base config has no value for 'batch'"));
        CHECK(a.config.at("batch") == 128 && a.config.at("precision_bits") == 32);        // first candidates
    }
    {   // no 'threads' tunable: single-threaded prediction, announced
        Algorithm alg = make_alg(false);
        Config b = base_cfg(); b.erase("threads");
        A::Advice a = A::advise(alg, b, fp, cv);
        CHECK(a.ok && has_note(a, "single-threaded"));
        CHECK(a.config.count("threads") == 0);
        // threads fixed at 1 in an independent evaluation
        double best = 0; for (double mb : {8., 16., 32., 64., 128.}) for (double ck : {1., 2., 4.}) {
            Config c = b; c["micro_batch"] = mb; c["ckpt"] = ck;
            PredictOptions po; po.threads = 1; po.memory_fraction = 0.6;
            Prediction p = predict(alg.build(c), fp, cv, po);
            if (p.memory_fits) best = std::max(best, p.units_per_s);
        }
        CHECK_REL(a.prediction.units_per_s, best, 1e-12);
        CHECK(!has_note(A::advise(make_alg(), base_cfg(), fp, cv), "single-threaded"));
    }
    {   // an algorithm with no tunables: the base config is evaluated once
        Algorithm fixed; fixed.name = "fixed";
        fixed.build = [](const Config&) { RunSpec r; r.mode = Mode::INFER; Workload w; w.units_per_micro_step = 64; w.layers = {dense(64, 256, 256, r)}; return w; };
        A::Advice a = A::advise(fixed, {}, fp, cv);
        CHECK(a.ok && a.evaluated == 1 && a.search_mode == "exhaustive");
    }
    {   // unusable capability data: no advice, no crash
        A::Advice a = A::advise(make_alg(), base_cfg(), fp, CapabilityVector{});
        CHECK(!a.ok);
        CHECK(a.binding == "no feasible configuration");
    }
    {   // unknown memory budget: advice is given, and the caveat travels with it
        A::Advice a = A::advise(make_alg(), base_cfg(), make_fp(0), cv);
        CHECK(a.ok);
        CHECK(has_note(a, "memory budget unknown"));
    }
    {   // ties are broken deterministically, not by enumeration order
        Algorithm alg; alg.name = "tie";
        alg.tunables = {{"dummy", {1, 2, 3}, true, "step"}};                              // does not affect the workload at all
        alg.build = [](const Config&) { RunSpec r; r.mode = Mode::INFER; Workload w; w.units_per_micro_step = 64; w.layers = {dense(64, 256, 256, r)}; return w; };
        for (int i = 0; i < 3; ++i) CHECK(A::advise(alg, {{"dummy", 3}}, fp, cv).config.at("dummy") == 1);   // the lexicographically first
        A::Advice a = A::advise(alg, {}, fp, cv);
        CHECK(has_note(a, "differ by less than the model's error"));                      // and the tie is announced
    }
}

static void test_error_range_and_validation() {
    SECTION("error range and what the model cannot vouch for");
    Algorithm alg = make_alg(); CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    A::Advice raw = A::advise(alg, base_cfg(), fp, cv);
    CHECK_REL(raw.rel_error, Calibration::kUncalibratedError, 1e-12);
    CHECK_REL(raw.units_per_s_low, raw.units_per_s / 1.6, 1e-12);
    CHECK_REL(raw.units_per_s_high, raw.units_per_s * 1.6, 1e-12);
    CHECK(has_note(raw, "uncalibrated"));
    // every kind in the workload is listed as unvalidated
    CHECK(std::find(raw.unvalidated_kinds.begin(), raw.unvalidated_kinds.end(), "dense") != raw.unvalidated_kinds.end());
    CHECK(std::find(raw.unvalidated_kinds.begin(), raw.unvalidated_kinds.end(), "optimizer") != raw.unvalidated_kinds.end());

    // calibrate only DENSE: the advice gets tighter, and names what is still unmeasured
    Calibration cal;
    cal.fit({{(int)OpKind::DENSE, 1, 1.1}, {(int)OpKind::DENSE, 1, 1.1}, {(int)OpKind::DENSE, 1, 1.1}});
    A::Advice c = A::advise(alg, base_cfg(), fp, cv, &cal);
    CHECK(c.ok);
    CHECK(c.rel_error < raw.rel_error);                                                   // dense layers dominate and are now trusted
    CHECK(c.rel_error > 0.0);                                                             // but not zero: elementwise/optimizer are not measured
    CHECK(std::find(c.unvalidated_kinds.begin(), c.unvalidated_kinds.end(), "dense") == c.unvalidated_kinds.end());
    CHECK(std::find(c.unvalidated_kinds.begin(), c.unvalidated_kinds.end(), "elementwise") != c.unvalidated_kinds.end());
    CHECK(has_note(c, "never measured against real ops"));
    CHECK(c.units_per_s_high / c.units_per_s_low < raw.units_per_s_high / raw.units_per_s_low);   // a tighter range
    // the calibration also changes the prediction itself
    CHECK_REL(c.prediction.step_s / raw.prediction.step_s, 1.1, 0.2);
}

static void test_error_floor() {
    SECTION("error floor: the range is never tighter than the measured noise");
    Algorithm alg = make_alg(); CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    Calibration cal;
    cal.fit({{(int)OpKind::DENSE, 1, 1.0}, {(int)OpKind::DENSE, 1, 1.0}, {(int)OpKind::DENSE, 1, 1.0},
             {(int)OpKind::ELEMENTWISE, 1, 1.0}, {(int)OpKind::ELEMENTWISE, 1, 1.0}, {(int)OpKind::ELEMENTWISE, 1, 1.0},
             {(int)OpKind::OPTIMIZER, 1, 1.0}, {(int)OpKind::OPTIMIZER, 1, 1.0}, {(int)OpKind::OPTIMIZER, 1, 1.0}});
    A::Advice tight = A::advise(alg, base_cfg(), fp, cv, &cal);                           // everything calibrated, zero residual error
    CHECK(tight.ok && tight.rel_error < 0.01);
    A::AdvisorOptions o; o.error_floor = 0.3;
    A::Advice floored = A::advise(alg, base_cfg(), fp, cv, &cal, {}, o);
    CHECK_REL(floored.rel_error, 0.3, 1e-12);
    CHECK_REL(floored.units_per_s_low, floored.units_per_s / 1.3, 1e-12);
    CHECK_REL(floored.units_per_s_high, floored.units_per_s * 1.3, 1e-12);
    CHECK(floored.error_basis.find("floored") != std::string::npos);
    CHECK(has_note(floored, "run-to-run noise measured on this device"));
    CHECK(floored.config == tight.config);                                                // the floor changes the range, never the choice
    // A floor below the model's own error changes nothing and adds no note.
    A::AdvisorOptions low; low.error_floor = 0.1;
    A::Advice raw = A::advise(alg, base_cfg(), fp, cv, nullptr, {}, low);                 // uncalibrated: 0.60 > 0.10
    CHECK_REL(raw.rel_error, 0.6, 1e-12);
    CHECK(raw.error_basis == "model" && !has_note(raw, "run-to-run noise"));
    // The floor also applies to a trial-based error estimate.
    const Config c0 = tight.top[0].cfg, c1 = tight.top[1].cfg;
    const double p0 = tight.top[0].pred.step_s, p1 = tight.top[1].pred.step_s;
    A::TrialFn trial = [&](const Config& c) { return c == c0 ? p0 : c == c1 ? p1 : -1.0; };   // trials agree exactly: trial error 0
    A::AdvisorOptions ot; ot.error_floor = 0.2;
    A::Advice t = A::advise(alg, base_cfg(), fp, cv, &cal, {}, ot, trial);
    CHECK(t.confirmed && t.trial_error < 1e-9);
    CHECK_REL(t.rel_error, 0.2, 1e-12);
}

static void test_output() {
    SECTION("output: JSON and text");
    Algorithm alg = make_alg(); CapabilityVector cv = make_cv();
    A::TrialFn trial = [](const Config&) { return 0.02; };
    for (A::Advice a : {A::advise(alg, base_cfg(), make_fp(), cv), A::advise(alg, base_cfg(), make_fp(12ull << 20), cv),
                        A::advise(alg, base_cfg(), make_fp(), cv, nullptr, {}, {}, trial)}) {
        Json j; j.begin_object(); A::to_json(j, a); j.end_object();
        const std::string s = j.finish();
        CHECK(balanced_clean(s));
        CHECK(s.find("\"binding_constraint\"") != std::string::npos && s.find("\"runners_up\"") != std::string::npos);
        CHECK(s.find("\"config\"") != std::string::npos && s.find("\"unvalidated_kinds\"") != std::string::npos);
        A::print_advice(a);                                                               // both the ok and the not-ok forms must not crash
    }
}

int main() {
    test_finds_the_true_best();
    test_altering_tunables();
    test_binding_constraints();
    test_nothing_fits();
    test_coordinate_ascent();
    test_trials();
    test_factory_rejection();
    test_inputs_and_notes();
    test_error_range_and_validation();
    test_error_floor();
    test_output();
    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
