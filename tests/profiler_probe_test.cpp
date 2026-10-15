/*
 * profiler_probe_test.cpp
 * =======================
 * Tests for WorkloadProbe.hpp (step 2b of the profiler rewrite).
 *
 *   g++ -std=c++20 -O2 -Wall -Wextra -Iinclude -Iinclude/core -Iinclude/tools \
 *       tests/profiler_probe_test.cpp -o bin/profiler_probe_test -pthread
 *   ./bin/profiler_probe_test            # ~15 s: includes one real capability run + real probe
 *   ./bin/profiler_probe_test --quick    # ~3 s
 *
 * Two kinds of test, deliberately separate:
 *
 *   EXACT   The fitting / attribution / guard logic, driven by synthetic steps that busy-wait for
 *           a chosen multiple of the predicted time. The right answer is known, so the assertions
 *           are tight: a factor of 2 must come back as ~2, a held-out composite must be priced
 *           within a few percent, a hot device must stop the probe before anything runs.
 *
 *   REAL    The reference cases run on real Matrix / Tensor ops. Their accuracy depends on the
 *           machine (a shared VM is noisy and the portable cblas shim is not OpenBLAS), so these
 *           assert only that everything RUNS, is measured, and lands in a sane range. They do
 *           not assert that the model is accurate on this hardware: that is what the printed
 *           report is for.
 */

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "profiler/tmp/WorkloadProbe.hpp"

using namespace tensorf::prof;
using namespace tensorf::prof::wl;
namespace fs = std::filesystem;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failed; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while (0)
#define CHECK_REL(a, b, rel) do { ++g_checks; double _a=(double)(a), _b=(double)(b); if (!(std::fabs(_a-_b) <= (rel) * std::fabs(_b))) { ++g_failed; \
    std::printf("  FAIL %s:%d  %s ~ %s within %g%%  (%g vs %g)\n", __FILE__, __LINE__, #a, #b, (rel) * 100.0, _a, _b); } } while (0)
#define SECTION(name) std::printf("[%s]\n", name)

// ─── Synthetic device + helpers ──────────────────────────────────────────────

static CapabilityVector make_cv() {
    CapabilityVector cv;
    cv.f32_gflops_best = 10; cv.f32_gflops_skinny = 4; cv.i8_gops = 20; cv.conv_gflops = 8; cv.nbody_gflops = 12; cv.f64_gflops = 5;
    cv.bw_l1_gbs = 100; cv.bw_l2_gbs = 60; cv.bw_l3_gbs = 30; cv.bw_dram_gbs = 10;
    cv.triad_1t_gbs = 10; cv.triad_max_gbs = 20; cv.triad_sweep = {{1, 10.0}, {2, 18.0}, {4, 20.0}};
    cv.dram_saturation_threads = 4; cv.gather_gbs = 2.0;
    cv.tiny_op_ns = 10; cv.barrier_roundtrip_us = 2.0; cv.thread_spawn_us = 50;
    cv.recommended_class = CoreClass::UNIFORM;
    return cv;
}

static DeviceFingerprint make_fp(uint64_t budget = 4ull << 30) {
    DeviceFingerprint fp;
    CoreCluster c; c.cpus = {0, 1, 2, 3}; c.physical_cores = 4;
    c.l1d_bytes = 32 << 10; c.l2_bytes = 256 << 10; c.l3_bytes = 8ull << 20;
    fp.clusters.push_back(c);
    fp.logical_cpus = 4;
    fp.memory.total_bytes = budget * 2; fp.memory.available_bytes = budget; fp.memory.safe_budget_bytes = budget;
    return fp;
}

static void spin(double seconds) { const double t0 = now_s(); while (now_s() - t0 < seconds) {} }

static RunSpec infer_spec() { RunSpec r; r.mode = Mode::INFER; return r; }

static Workload dense_w(size_t n) {
    Workload w; w.name = "dense" + std::to_string(n); w.units_per_micro_step = (double)n;
    w.layers = {dense((double)n, (double)n, (double)n, infer_spec(), "mm", false)};
    return w;
}
static Workload ew_w(double n) {
    Workload w; w.name = "ew"; w.units_per_micro_step = n;
    w.layers = {elementwise(n, 1.0, 3.0, infer_spec(), "add")};
    return w;
}

// A case whose step busy-waits `factor` times the model's prediction: the true miss is `factor`.
static ProbeCase synth(const std::string& name, int kind, Workload w, double factor, const DeviceFingerprint& fp,
                       const CapabilityVector& cv, std::shared_ptr<int> made = nullptr) {
    ProbeCase c; c.name = name; c.kind = kind;
    const double target = predict(w, fp, cv).step_s * factor;
    c.workload = std::move(w);
    c.make = [target, made] { if (made) ++*made; return std::function<void()>([target] { spin(target); }); };
    return c;
}

static ProbeConfig fast_cfg() {
    ProbeConfig c; c.warmup = 0; c.min_samples = 3; c.max_samples = 5; c.per_case_s = 0.02; c.time_budget_s = 30;
    return c;
}

// Fixture directory for the thermal guard.
class FakeRoot {
public:
    FakeRoot() { path_ = (fs::temp_directory_path() / ("tensorf_probe_" + std::to_string((long long)(now_s() * 1e6)))).string(); fs::create_directories(path_); }
    ~FakeRoot() { std::error_code ec; fs::remove_all(path_, ec); }
    const std::string& root() const { return path_; }
    void thermal(int zone, const char* type, long millideg) const {
        const fs::path d = fs::path(path_) / "sys/class/thermal" / ("thermal_zone" + std::to_string(zone));
        fs::create_directories(d);
        std::ofstream(d / "type") << type << "\n";
        std::ofstream(d / "temp") << millideg << "\n";
    }
private:
    std::string path_;
};

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
    for (const char* bad : {":nan", ":-nan", ":inf", ":-inf", ",nan", ",inf", "[nan", "[inf"})
        if (low.find(bad) != std::string::npos) return false;
    return !in_str && brace == 0 && brack == 0;
}

// ═════════════════════════════════════════════════════════════════════════════
// EXACT tests: fitting, attribution, guards
// ═════════════════════════════════════════════════════════════════════════════

static void test_recovers_known_factor() {
    SECTION("fit: a known 2x miss is recovered and held-out cases are priced within a few percent");
    DeviceFingerprint fp = make_fp(); CapabilityVector cv = make_cv();
    std::vector<ProbeCase> cs;
    for (size_t n : {128u, 192u, 256u, 320u}) cs.push_back(synth("dense" + std::to_string(n), (int)OpKind::DENSE, dense_w(n), 2.0, fp, cv));
    ProbeReport r = measure_cases(std::move(cs), fp, cv, fast_cfg());

    CHECK(r.usable == 4);
    CHECK(r.stopped_reason.empty());
    CHECK(r.calibration.calibrated((int)OpKind::DENSE));
    CHECK_REL(r.calibration.factor_for(OpKind::DENSE), 2.0, 0.10);
    for (const auto& x : r.results) { CHECK(x.ok); CHECK_REL(x.ratio, 2.0, 0.12); CHECK(x.loo_error < 0.15); CHECK(x.samples >= 3); }
    CHECK(r.overall_loo_error < 0.15);
    CHECK(r.overall_raw_error > 0.4);                                    // uncalibrated it was ~50% off (predicted half the truth)
    CHECK(r.overall_loo_error < r.overall_raw_error * 0.5);              // calibration visibly helped
    CHECK(r.blas == "shim" || r.blas == "external" || r.blas == "none");
    CHECK(r.threads_validated == 1);

    // Using the fitted calibration in predict() closes the gap on a layer it was fitted for.
    Prediction raw = predict(dense_w(224), fp, cv), cal = predict(dense_w(224), fp, cv, {}, &r.calibration);
    CHECK_REL(cal.step_s / raw.step_s, 2.0, 0.10);
    CHECK(cal.calibrated);
}

static void test_per_kind_factors_and_fallback() {
    SECTION("fit: factors are per kind; unprobed kinds fall back to the global median");
    DeviceFingerprint fp = make_fp(); CapabilityVector cv = make_cv();
    std::vector<ProbeCase> cs;
    cs.push_back(synth("d1", (int)OpKind::DENSE, dense_w(192), 1.0, fp, cv));
    cs.push_back(synth("d2", (int)OpKind::DENSE, dense_w(256), 1.0, fp, cv));
    cs.push_back(synth("e1", (int)OpKind::ELEMENTWISE, ew_w(1e6), 4.0, fp, cv));
    cs.push_back(synth("e2", (int)OpKind::ELEMENTWISE, ew_w(2e6), 4.0, fp, cv));
    ProbeReport r = measure_cases(std::move(cs), fp, cv, fast_cfg());
    CHECK(r.usable == 4);
    CHECK_REL(r.calibration.factor_for(OpKind::DENSE), 1.0, 0.12);
    CHECK_REL(r.calibration.factor_for(OpKind::ELEMENTWISE), 4.0, 0.12);
    // CONV2D was never probed: it gets the global factor (median of all ratios = between 1 and 4), not 1 and not a kind's own.
    const double conv = r.calibration.factor_for(OpKind::CONV2D);
    CHECK(conv > 1.2 && conv < 3.8);
    CHECK(!r.calibration.calibrated((int)OpKind::CONV2D));
    CHECK(r.kinds.size() == 2);                                          // one row per kind that has cases
    CHECK(r.global_factor > 1.2 && r.global_factor < 3.8);               // the fallback is reported, not hidden
}

static void test_composite_held_out() {
    SECTION("fit: a composite step is held out and priced layer by layer");
    DeviceFingerprint fp = make_fp(); CapabilityVector cv = make_cv();
    // True behaviour: dense matches the model, elementwise is 4x slower. The composite contains both.
    Workload comp; comp.name = "step"; comp.units_per_micro_step = 256;
    comp.layers = {dense(256, 256, 256, infer_spec(), "mm", false), elementwise(2e6, 1.0, 3.0, infer_spec(), "act")};
    Prediction pc = predict(comp, fp, cv);
    CHECK(pc.layers.size() == 2);
    const double truth = pc.layers[0].seconds * 1.0 + pc.layers[1].seconds * 4.0;
    ProbeCase c; c.name = "composite"; c.kind = kOpKinds; c.workload = comp;
    c.make = [truth] { return std::function<void()>([truth] { spin(truth); }); };

    std::vector<ProbeCase> cs;
    cs.push_back(synth("d1", (int)OpKind::DENSE, dense_w(192), 1.0, fp, cv));
    cs.push_back(synth("d2", (int)OpKind::DENSE, dense_w(256), 1.0, fp, cv));
    cs.push_back(synth("e1", (int)OpKind::ELEMENTWISE, ew_w(1e6), 4.0, fp, cv));
    cs.push_back(synth("e2", (int)OpKind::ELEMENTWISE, ew_w(2e6), 4.0, fp, cv));
    cs.push_back(std::move(c));
    ProbeReport r = measure_cases(std::move(cs), fp, cv, fast_cfg());
    CHECK(r.usable == 5);
    const ProbeResult& comp_r = r.results.back();
    CHECK(comp_r.ok);
    CHECK(comp_r.kind == kOpKinds);
    CHECK(comp_r.ratio > 1.5);                                           // uncalibrated it is far off: elementwise dominates the composite
    CHECK(comp_r.loo_error < 0.15);                                      // priced with per-kind factors fitted on the OTHER cases
    CHECK(comp_r.loo_error < std::fabs(comp_r.predicted_s - comp_r.measured_s) / comp_r.measured_s * 0.5);
    double ksum = 0; for (double v : comp_r.kind_s) ksum += v;
    CHECK_REL(ksum, comp_r.predicted_s, 1e-9);                           // breakdown sums to the prediction
    CHECK(comp_r.kind_s[(size_t)OpKind::DENSE] > 0 && comp_r.kind_s[(size_t)OpKind::ELEMENTWISE] > 0);
}

static void test_guards() {
    SECTION("guards: memory, exceptions, budget, thermal, bad prediction");
    DeviceFingerprint fp = make_fp(); CapabilityVector cv = make_cv();

    {   // memory: a case needing more than its share is skipped and never allocated
        auto made = std::make_shared<int>(0);
        DeviceFingerprint small = make_fp(64ull << 20);                  // 64 MB budget, probe share 25% = 16 MB
        std::vector<ProbeCase> cs;
        cs.push_back(synth("fits", (int)OpKind::DENSE, dense_w(64), 1.0, small, cv, made));
        cs.push_back(synth("huge", (int)OpKind::DENSE, dense_w(2048), 1.0, small, cv, made));   // 3 matrices of 16 MB
        ProbeReport r = measure_cases(std::move(cs), small, cv, fast_cfg());
        CHECK(r.results[0].ok);
        CHECK(!r.results[1].ok && r.results[1].note.find("skipped") != std::string::npos);
        CHECK(*made == 1);
        CHECK(r.usable == 1);
    }
    {   // exceptions in make() and in step() are contained; the next case still runs
        std::vector<ProbeCase> cs;
        ProbeCase a = synth("throws_in_make", (int)OpKind::DENSE, dense_w(64), 1.0, fp, cv);
        a.make = []() -> std::function<void()> { throw std::runtime_error("boom make"); };
        ProbeCase b = synth("throws_in_step", (int)OpKind::DENSE, dense_w(64), 1.0, fp, cv);
        b.make = [] { return std::function<void()>([] { throw std::runtime_error("boom step"); }); };
        ProbeCase c = synth("fine", (int)OpKind::DENSE, dense_w(96), 1.0, fp, cv);
        ProbeCase d = synth("non_std_exception", (int)OpKind::DENSE, dense_w(64), 1.0, fp, cv);
        d.make = [] { return std::function<void()>([] { throw 42; }); };
        cs.push_back(std::move(a)); cs.push_back(std::move(b)); cs.push_back(std::move(c)); cs.push_back(std::move(d));
        ProbeReport r = measure_cases(std::move(cs), fp, cv, fast_cfg());
        CHECK(!r.results[0].ok && r.results[0].note.find("boom make") != std::string::npos);
        CHECK(!r.results[1].ok && r.results[1].note.find("boom step") != std::string::npos);
        CHECK(r.results[2].ok);
        CHECK(!r.results[3].ok && !r.results[3].note.empty());
        CHECK(r.usable == 1);
    }
    {   // time budget: later cases are not run, and make() is never called for them
        auto made = std::make_shared<int>(0);
        ProbeConfig c = fast_cfg(); c.time_budget_s = 0.15; c.min_samples = 2; c.per_case_s = 0.05;
        std::vector<ProbeCase> cs;
        for (int i = 0; i < 6; ++i) {
            ProbeCase p = synth("slow" + std::to_string(i), (int)OpKind::DENSE, dense_w(128), 1.0, fp, cv, made);
            p.make = [made] { ++*made; return std::function<void()>([] { spin(0.04); }); };
            cs.push_back(std::move(p));
        }
        Stopwatch sw;
        ProbeReport r = measure_cases(std::move(cs), fp, cv, c);
        CHECK(r.stopped_reason == "time budget exhausted");
        CHECK(*made < 6 && *made >= 1);
        CHECK(!r.results.back().ok && r.results.back().note.find("not run") != std::string::npos);
        CHECK(sw.elapsed() < 1.0);                                       // a 0.15 s budget did not turn into seconds
        CHECK(r.usable >= 1);                                            // partial results still fit
    }
    {   // thermal: a hot device stops the probe before the first case allocates anything
        FakeRoot fx; fx.thermal(0, "cpu-0-0", 61000);
        auto made = std::make_shared<int>(0);
        ProbeConfig c = fast_cfg(); c.sysroot = fx.root();
        std::vector<ProbeCase> cs;
        cs.push_back(synth("a", (int)OpKind::DENSE, dense_w(96), 1.0, fp, cv, made));
        cs.push_back(synth("b", (int)OpKind::DENSE, dense_w(96), 1.0, fp, cv, made));
        ProbeReport r = measure_cases(std::move(cs), fp, cv, c);
        CHECK(r.stopped_reason.find("thermal") != std::string::npos);
        CHECK(*made == 0);
        CHECK(r.usable == 0);
        CHECK_REL(r.overall_loo_error, Calibration::kUncalibratedError, 1e-12);   // nothing measured: honest default, not a made-up number
    }
    {   // charging uses the tighter battery limit
        FakeRoot fx;
        const fs::path d = fs::path(fx.root()) / "sys/class/power_supply/battery"; fs::create_directories(d);
        std::ofstream(d / "type") << "Battery\n"; std::ofstream(d / "temp") << 390 << "\n";   // 39.0 C
        DeviceFingerprint chg = make_fp(); chg.battery.present = true; chg.battery.status = "Charging";
        auto made = std::make_shared<int>(0);
        ProbeConfig c = fast_cfg(); c.sysroot = fx.root();
        std::vector<ProbeCase> cs; cs.push_back(synth("a", (int)OpKind::DENSE, dense_w(96), 1.0, chg, cv, made));
        ProbeReport r = measure_cases(std::move(cs), chg, cv, c);
        CHECK(r.stopped_reason.find("thermal") != std::string::npos && *made == 0);
        // the same temperature while discharging is fine
        DeviceFingerprint dis = make_fp(); dis.battery.present = true; dis.battery.status = "Discharging";
        auto made2 = std::make_shared<int>(0);
        std::vector<ProbeCase> cs2; cs2.push_back(synth("a", (int)OpKind::DENSE, dense_w(96), 1.0, dis, cv, made2));
        ProbeReport r2 = measure_cases(std::move(cs2), dis, cv, c);
        CHECK(r2.stopped_reason.empty() && *made2 == 1);
    }
    {   // an empty capability vector cannot predict, so nothing is measured against it
        CapabilityVector empty;
        std::vector<ProbeCase> cs;
        ProbeCase c; c.name = "x"; c.kind = (int)OpKind::DENSE; c.workload = dense_w(64);
        auto made = std::make_shared<int>(0);
        c.make = [made] { ++*made; return std::function<void()>([] {}); };
        cs.push_back(std::move(c));
        ProbeReport r = measure_cases(std::move(cs), fp, empty, fast_cfg());
        CHECK(!r.results[0].ok && r.results[0].note.find("no prediction") != std::string::npos);
        CHECK(*made == 0);
    }
    {   // no cases at all
        ProbeReport r = measure_cases({}, fp, cv, fast_cfg());
        CHECK(r.results.empty() && r.usable == 0);
        CHECK_REL(r.overall_loo_error, Calibration::kUncalibratedError, 1e-12);
    }
}

static void test_min_samples_even_when_slow() {
    SECTION("measure: a slow case still gets min_samples, never a single noisy sample");
    DeviceFingerprint fp = make_fp(); CapabilityVector cv = make_cv();
    ProbeConfig c = fast_cfg(); c.min_samples = 4; c.max_samples = 10; c.per_case_s = 0.001;   // per-case time is far below one step
    ProbeCase p = synth("slow", (int)OpKind::DENSE, dense_w(128), 1.0, fp, cv);
    p.make = [] { return std::function<void()>([] { spin(0.01); }); };
    std::vector<ProbeCase> cs; cs.push_back(std::move(p));
    ProbeReport r = measure_cases(std::move(cs), fp, cv, c);
    CHECK(r.results[0].ok && r.results[0].samples == 4);
}

static void test_fit_report_directly() {
    SECTION("fit_report: leave-one-out on hand-made results (no timing)");
    auto mk = [](const char* n, OpKind k, double pred, double meas) {
        ProbeResult r; r.name = n; r.kind = (int)k; r.ok = true; r.predicted_s = pred; r.measured_s = meas; r.ratio = meas / pred; r.samples = 5;
        r.kind_s[(size_t)k] = pred; return r;
    };
    {   // one outlier among consistent cases: the held-out error for the outlier is large and the factor is not dragged
        ProbeReport r;
        r.results = {mk("a", OpKind::DENSE, 1, 2), mk("b", OpKind::DENSE, 2, 4), mk("c", OpKind::DENSE, 3, 6), mk("d", OpKind::DENSE, 4, 8),
                     mk("odd", OpKind::DENSE, 1, 30)};
        fit_report(r);
        CHECK_REL(r.calibration.factor_for(OpKind::DENSE), 2.0, 1e-9);   // median shrugs off the outlier
        CHECK(r.results[4].loo_error > 0.9);                             // it is honestly reported as badly predicted
        CHECK(r.results[0].loo_error < 1e-9);
    }
    {   // a kind with ONE case has nothing to hold out against its own kind: it falls back to the global factor of the others
        ProbeReport r;
        r.results = {mk("a", OpKind::DENSE, 1, 2), mk("b", OpKind::DENSE, 2, 4), mk("lone", OpKind::SPMM, 1, 2)};
        fit_report(r);
        CHECK(r.results[2].loo_error < 1e-9);                            // global factor 2 happens to be right here
        CHECK(r.kinds.size() >= 2);
    }
    {   // stored per-kind error is the honest (held-out) one, never below it
        ProbeReport r;
        r.results = {mk("a", OpKind::DENSE, 1, 2.0), mk("b", OpKind::DENSE, 1, 2.0), mk("c", OpKind::DENSE, 1, 2.0), mk("d", OpKind::DENSE, 1, 2.6)};
        fit_report(r);
        double loo = 0; for (const auto& k : r.kinds) if (k.kind == (int)OpKind::DENSE) loo = k.loo_median_error;
        CHECK(r.calibration.error[(size_t)OpKind::DENSE] >= loo - 1e-12);
    }
    {   // failed cases are ignored, none usable is safe
        ProbeReport r; ProbeResult bad; bad.name = "bad"; bad.ok = false; r.results = {bad, bad};
        fit_report(r);
        CHECK(r.usable == 0 && r.kinds.empty());
        CHECK_REL(r.overall_loo_error, Calibration::kUncalibratedError, 1e-12);
    }
}

static void test_output() {
    SECTION("output: JSON and printed report");
    DeviceFingerprint fp = make_fp(); CapabilityVector cv = make_cv();
    std::vector<ProbeCase> cs;
    cs.push_back(synth("d1", (int)OpKind::DENSE, dense_w(128), 1.5, fp, cv));
    cs.push_back(synth("d2", (int)OpKind::DENSE, dense_w(192), 1.5, fp, cv));
    ProbeCase bad = synth("skipped \"quoted\"", (int)OpKind::DENSE, dense_w(64), 1.0, fp, cv);
    bad.make = []() -> std::function<void()> { throw std::runtime_error("bad \"thing\"\n"); };
    cs.push_back(std::move(bad));
    ProbeReport r = measure_cases(std::move(cs), fp, cv, fast_cfg());
    r.not_covered = {"conv2d", "particle"};
    Json j; j.begin_object(); to_json(j, r); j.end_object();
    const std::string s = j.finish();
    CHECK(balanced_clean(s));                                            // quotes and newlines in notes are escaped
    CHECK(s.find("\"overall_loo_error\"") != std::string::npos);
    CHECK(s.find("\"threads_validated\":1") != std::string::npos);
    CHECK(s.find("\"not_covered\"") != std::string::npos && s.find("conv2d") != std::string::npos);
    CHECK(s.find("\"blas\"") != std::string::npos);

    ProbeReport empty;                                                   // empty report still serialises
    Json j2; j2.begin_object(); to_json(j2, empty); j2.end_object();
    CHECK(balanced_clean(j2.finish()));
    print_report(r);                                                     // must not crash on a mix of ok / failed rows
    print_report(empty);
}

// ═════════════════════════════════════════════════════════════════════════════
// REAL ops: they run, are measured, and land in a sane range
// ═════════════════════════════════════════════════════════════════════════════

static void test_reference_case_definitions() {
    SECTION("reference cases: definitions");
    ProbeConfig cfg;
    auto cases = reference_cases(cfg);
    CHECK(cases.size() == 14);
    int n[kOpKinds + 1] = {};
    for (const auto& c : cases) {
        CHECK(!c.name.empty());
        CHECK(c.kind >= 0 && c.kind <= kOpKinds);
        CHECK(!c.workload.layers.empty());
        CHECK(totals(c.workload).flops > 0);
        CHECK(c.workload.units_per_step() > 0);
        CHECK((bool)c.make);
        ++n[c.kind];
    }
    CHECK(n[(int)OpKind::DENSE] == 5 && n[(int)OpKind::ELEMENTWISE] == 2 && n[(int)OpKind::SPMM] == 3 &&
          n[(int)OpKind::RECURRENT] == 2 && n[kOpKinds] == 2);           // every kind has >= 2 cases, so each can be held out
    // the spmm workload uses the REAL nnz of the graph that make() will build
    for (const auto& c : cases) if (c.kind == (int)OpKind::SPMM) {
        const LayerCost& l = c.workload.layers[0];
        CHECK(l.kind == OpKind::SPMM && l.flops_fwd > 0);
        CHECK(l.irregular_frac > 0);
    }
    // MLP composites describe the training path: backward flops present, first layer skips dX
    for (const auto& c : cases) if (c.kind == kOpKinds) {
        // first layer: input needs no gradient, so backward is dW only (== forward flops); fc2 has dW + dX (== 2x forward)
        CHECK_REL(c.workload.layers[0].flops_bwd, c.workload.layers[0].flops_fwd, 1e-9);
        CHECK_REL(c.workload.layers[2].flops_bwd, 2.0 * c.workload.layers[2].flops_fwd, 1e-9);
        CHECK(c.workload.layers.size() == 4);
    }
    // size_scale shrinks everything
    ProbeConfig small; small.size_scale = 0.25;
    auto sc = reference_cases(small);
    CHECK(sc.size() == cases.size());
    double big = 0, tiny = 0;
    for (const auto& c : cases) big += totals(c.workload).flops;
    for (const auto& c : sc) tiny += totals(c.workload).flops;
    CHECK(tiny < big * 0.5);
}

static void test_reference_steps_execute() {
    SECTION("reference cases: every step runs on real ops (tiny scale)");
    ProbeConfig cfg; cfg.size_scale = 0.1;
    int ran = 0;
    for (auto& c : reference_cases(cfg)) {
        try { auto step = c.make(); step(); step(); ++ran; }
        catch (const std::exception& e) { std::printf("  case '%s' threw: %s\n", c.name.c_str(), e.what()); }
    }
    CHECK(ran == 14);
}

static void test_real_probe_mechanics() {
    SECTION("real probe with a synthetic capability vector: mechanics, not accuracy");
    DeviceFingerprint fp = make_fp(); CapabilityVector cv = make_cv();
    ProbeConfig cfg; cfg.size_scale = 0.5; cfg.per_case_s = 0.15; cfg.min_samples = 3; cfg.max_samples = 10; cfg.warmup = 1; cfg.time_budget_s = 40;
    ProbeReport r = run_probe(fp, cv, cfg);
    CHECK(r.results.size() == 14);
    CHECK(r.usable == 14);                                               // every real case ran and was measured
    CHECK(r.stopped_reason.empty());
    for (const auto& x : r.results) {
        CHECK(x.ok && x.measured_s > 0 && x.predicted_s > 0);
        CHECK(x.ratio > 0.01 && x.ratio < 500.0);                        // the synthetic cv is not this machine, so only a wide sanity band
        CHECK(std::isfinite(x.loo_error));
    }
    CHECK(!r.not_covered.empty());
    CHECK(r.blas == "shim" || r.blas == "external" || r.blas == "none");
    Json j; j.begin_object(); to_json(j, r); j.end_object();
    CHECK(balanced_clean(j.finish()));
}

static void test_real_probe_live(bool quick) {
    SECTION("real probe with a REAL capability vector");
    if (quick) { std::printf("  (skipped: --quick)\n"); return; }
    DeviceFingerprint fp = detect_device();
    SuiteConfig sc; sc.time_budget_s = 14.0; sc.run_sustained = false;
    CapabilityVector cv = CapabilitySuite(sc).run(fp);
    if (!cv.stopped_reason.empty()) { std::printf("  (capability run stopped: %s; skipping)\n", cv.stopped_reason.c_str()); return; }
    ProbeConfig cfg; cfg.time_budget_s = 40;
    ProbeReport r = run_probe(fp, cv, cfg);
    print_report(r);
    CHECK(r.usable >= 12);                                               // allow a skip or two on a tiny / busy device
    CHECK(r.overall_loo_error >= 0.0 && r.overall_loo_error < 3.0);      // sanity only: noisy shared VMs vary a lot
    for (const auto& x : r.results) if (x.ok) CHECK(x.ratio > 0.05 && x.ratio < 50.0);
    // Calibration must be usable by predict() and change the answer for a probed kind.
    RunSpec rs; rs.mode = Mode::INFER;
    Workload w; w.units_per_micro_step = 256; w.layers = {dense(256, 256, 256, rs, "mm", false)};
    Prediction a = predict(w, fp, cv), b = predict(w, fp, cv, {}, &r.calibration);
    CHECK(b.calibrated && !a.calibrated);
    CHECK(b.rel_error <= Calibration::kUncalibratedError + 1e-12);
}

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    test_recovers_known_factor();
    test_per_kind_factors_and_fallback();
    test_composite_held_out();
    test_guards();
    test_min_samples_even_when_slow();
    test_fit_report_directly();
    test_output();
    test_reference_case_definitions();
    test_reference_steps_execute();
    test_real_probe_mechanics();
    test_real_probe_live(quick);
    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
