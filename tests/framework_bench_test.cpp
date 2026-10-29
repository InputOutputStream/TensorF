/*
 * framework_bench_test.cpp
 * ========================
 * Tests for FrameworkBench.hpp: the profiler times Matrix::matmul through whichever BLAS the build selected
 * (make BLAS=openblas|shim|none), not its own portable loops.
 *
 *   g++ -std=c++20 -O2 -Wall -Wextra -Iinclude -Iinclude/profiler -Iinclude/core -Iinclude/tools \
 *       tests/framework_bench_test.cpp -o bin/framework_bench_test -pthread
 *
 * Pure parts (FwProfile maths, interpolation, fallbacks) run on hand-made profiles with round numbers.
 * Live parts only assert that the numbers exist, are finite and respect the controls (budget, stop hook).
 * They never assert that OpenBLAS beats the shim: that is what the printed report is for.
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <thread>
#include "profiler/FrameworkBench.hpp"
#include "profiler/DeviceBenchmark.hpp"

using namespace tensorf::fwbench;

static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++fails; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
static bool near(double a, double b, double rel = 1e-9) { return std::abs(a - b) <= rel * std::max(1.0, std::max(std::abs(a), std::abs(b))); }

static FwPoint pt(const char* key, double g, bool f64 = false) { FwPoint p; p.key = key; p.gflops = g; p.f64 = f64; p.ok = true; p.windows = 5; return p; }
static FwScale sc(int t, double g, double su) { FwScale s; s.threads = t; s.gflops = g; s.speedup = su; s.eff = su / t; return s; }

static void t_profile_maths() {
    FwProfile P;
    CHECK(!P.usable() && P.best_f32() == 0 && P.max_threads() == 1 && P.speedup(8) == 1.0);          // nothing measured: neutral
    P.available = true;
    CHECK(!P.usable());                                                                               // available but no points
    P.points = {pt("f32_16", 2), pt("f32_64", 8), pt("f32_256", 20), pt("f32_512", 18), pt("f32_tall", 99), pt("f64_256", 9, true)};
    CHECK(P.usable());
    CHECK(P.best_f32() == 20);                                                                        // tall is not a square shape: never the roof
    CHECK(P.gflops("f64_256") == 9 && P.gflops("nope") == 0);
    FwPoint bad = pt("f32_256", 500); bad.ok = false; P.points.push_back(bad);
    CHECK(P.best_f32() == 20 && P.gflops("f32_256") == 20);                                           // failed points are ignored
    P.scaling = {sc(1, 20, 1.0), sc(2, 36, 1.8), sc(4, 60, 3.0)};
    CHECK(P.max_threads() == 4);
    CHECK(near(P.speedup(1), 1.0) && near(P.speedup(2), 1.8) && near(P.speedup(4), 3.0));
    CHECK(near(P.speedup(3), 2.4));                                                                   // interpolated
    CHECK(near(P.speedup(64), 3.0));                                                                  // beyond the curve: last point, never extrapolated
    P.scaling = {sc(1, 20, 1.0), sc(2, 15, 0.75)};                                                    // threads HURT: speedup never below 1
    CHECK(P.speedup(2) == 1.0);
}

static void t_live() {
    std::printf("---- framework bench on this build: backend '%s' ----\n", backend_name());
    CHECK(available());
    CHECK(std::strlen(backend_name()) > 0);
    Options o; o.window_s = 0.03; o.budget_s = 20; o.max_threads = std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
    FwProfile P = run(o);
    CHECK(P.available && P.usable() && P.backend == backend_name());
    for (const char* k : {"f32_16", "f32_64", "f32_256", "f32_512", "f32_skinny", "f32_gnn", "f32_gnn_t", "f64_256"}) {
        const double g = P.gflops(k);
        CHECK(std::isfinite(g) && g > 0.01 && g < 1e5);
    }
    for (const auto& p : P.points) std::printf("  %-11s %8.2f GFLOP/s (rsd %.2f, %d windows)\n", p.key.c_str(), p.gflops, p.spread, p.windows);
    if (o.max_threads > 1) {
        CHECK(!P.scaling.empty() && P.scaling.front().threads == 1 && near(P.scaling.front().speedup, 1.0));
        CHECK(P.scaling.back().threads == o.max_threads);
        for (const auto& s : P.scaling) {
            CHECK(s.gflops > 0 && s.speedup > 0 && near(s.eff, s.speedup / s.threads, 1e-6));
            CHECK(s.small_gflops > 0 && s.small_speedup > 0 && s.cpu_ratio > 0);                    // the GNN-shaped probe ran at every count
            std::printf("  threads %d: %.1f GFLOP/s x%.2f | gnn-shape x%.2f, cpu/wall %.2f\n", s.threads, s.gflops, s.speedup, s.small_speedup, s.cpu_ratio);
        }
        CHECK(P.best_threads >= 1 && P.best_threads <= o.max_threads);
        CHECK(P.threads_effective == (P.max_speedup >= 1.25));
        CHECK(P.threads_effective || !P.notes.empty());                                              // "threads do not help" is always said
    }
    CHECK(P.single_thread_cpu == -1 && !P.single_thread_pinned);                                     // one core class: nothing is pinned
    CHECK(P.seconds > 0 && P.seconds < o.budget_s + 3);
}

static void t_controls() {
    {   // governor stop before anything runs: no points, a reason, and no time spent
        Options o; o.should_stop = [] { return true; };
        const auto t0 = std::chrono::steady_clock::now();
        FwProfile P = run(o);
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        CHECK(!P.usable() && !P.skipped.empty() && el < 1.0);
    }
    {   // tiny budget: honoured, and the skipped points are listed
        Options o; o.window_s = 0.03; o.budget_s = 0.05; o.max_threads = 2;
        FwProfile P = run(o);
        bool budget = false; for (auto& s : P.skipped) budget |= s.find("budget") != std::string::npos;
        CHECK(budget);
        CHECK(P.seconds < 3.0);
    }
    {   // memory cap: shapes over it are skipped and named, never allocated
        Options o; o.window_s = 0.02; o.max_bytes = 1u << 20; o.max_threads = 1;                    // 1 MB: 512^3 (4 MB) and the GNN shapes do not fit
        FwProfile P = run(o);
        bool cap512 = false, capgnn = false;
        for (auto& s : P.skipped) { cap512 |= s.find("f32_512") != std::string::npos; capgnn |= s.find("f32_gnn") != std::string::npos; }
        CHECK(cap512 && capgnn && P.gflops("f32_512") == 0 && P.gflops("f32_gnn") == 0);
        CHECK(P.gflops("f32_64") > 0);
    }
    {   // one thread: no scaling is attempted and the plan is one thread
        Options o; o.window_s = 0.02; o.max_threads = 1;
        FwProfile P = run(o);
        CHECK(P.scaling.empty() && P.best_threads == 1 && !P.threads_effective);
    }
}

static void t_sustained() {
    double el = 0;
    auto w = sustained_windows(1, 0.4, 0.1, 256ull << 20, {}, &el);
    CHECK(w.size() >= 3 && el >= 0.3 && el < 2.0);
    for (double g : w) CHECK(std::isfinite(g) && g > 0);
    int calls = 0;
    auto s = sustained_windows(1, 5.0, 0.05, 256ull << 20, [&] { return ++calls > 2; }, &el);         // the governor stops it mid-run
    CHECK(s.size() <= 3 && el < 1.0);
    CHECK(sustained_windows(1, 1.0, 0.1, 1024, {}, &el).empty());                                     // cap too small: nothing allocated
}

static void t_pin() {
    CHECK(pinning_supported() == (TENSORF_FWBENCH_HAS_AFFINITY != 0));
#if defined(__linux__)
    {   // pin to cpu 0 and check we really run there; the mask is restored afterwards
        PinGuard g(0);
        CHECK(g.ok());
        CHECK(sched_getcpu() == 0);
    }
    cpu_set_t m; CPU_ZERO(&m);
    CHECK(sched_getaffinity(0, sizeof m, &m) == 0 && CPU_COUNT(&m) >= 1);
    { PinGuard bad(99999); CHECK(!bad.ok()); }                                                       // impossible cpu: reported, not fatal
    cpu_set_t m2; CPU_ZERO(&m2);
    CHECK(sched_getaffinity(0, sizeof m2, &m2) == 0 && CPU_COUNT(&m2) == CPU_COUNT(&m));              // still the original mask
    {   // big.LITTLE path with a fake two-class layout on this machine: one pinned rate per class
        Options o; o.window_s = 0.02; o.budget_s = 60; o.max_threads = 1; o.classes = {{0}, {0}}; o.class_names = {"big", "little"};
        FwProfile P = run(o);
        CHECK(P.classes.size() == 2);
        for (const auto& c : P.classes) CHECK(c.pinned && c.gflops_1t > 0 && c.cpu == 0);
        CHECK(P.single_thread_pinned && P.single_thread_cpu == 0);
        CHECK(P.usable());
    }
    {   // impossible cpu in a class: said in the notes, numbers still produced
        Options o; o.window_s = 0.02; o.max_threads = 1; o.classes = {{99999}, {0}}; o.class_names = {"big", "little"};
        FwProfile P = run(o);
        bool said = false; for (auto& n : P.notes) said |= n.find("pin") != std::string::npos;
        CHECK(said && P.usable() && !P.single_thread_pinned);
    }
#endif
}

static void t_integration() {
    // The old profiler stack carries the framework profile and the cost model reads it.
    auto fp = tensorf::profile::collect();
    auto cfg = tensorf::profile::BenchConfig::quick();
    cfg.sustained_seconds = 0.5;
    auto P = tensorf::profile::run_device_benchmark(fp, cfg);
    CHECK(P.fw.usable() && P.fw.backend == backend_name());
    CHECK(P.sustained_via_framework == (P.sustained.ran && P.fw.usable()));
    const std::string js = P.to_json();
    CHECK(js.find("\"framework\"") != std::string::npos && js.find("\"backend\"") != std::string::npos);
    CHECK(P.summary().find("fw gemm") != std::string::npos);
}

int main() {
    t_profile_maths();
    t_live();
    t_controls();
    t_sustained();
    t_pin();
    t_integration();
    std::printf("%d/%d checks passed\n", checks - fails, checks);
    return fails ? 1 : 0;
}
