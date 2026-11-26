// FrameworkBench.hpp
//
// Measures what the FRAMEWORK gets on this device, not what a hand-written loop gets.
//
// Why this exists
//   The portable kernels in DeviceBenchmark / CapabilityKernels (own GEMM loop, own FMA chain) are limited
//   by the flags the profiler was compiled with: a build without -march=native runs them as SSE2 and reports
//   ~4 GFLOP/s on a CPU where OpenBLAS reaches ~90. Training does not run those loops. It runs Matrix::matmul,
//   which calls whatever BLAS tensorf_blas.h selected (OpenBLAS with runtime CPU dispatch, or the portable
//   shim). So the cost model must be priced against Matrix::matmul, and this header measures exactly that:
//
//     * single-thread GEMM at the shapes the cost model asks about (square, tall, skinny, MLP-like, f64)
//     * BLAS thread scaling (does more threads help at all? the shim and OPENBLAS_NUM_THREADS=1 say no)
//     * per-core-class rates on big.LITTLE phones (single thread pinned to each class)
//     * a sustained-load window series for the throttling probe
//
// Cross-platform rules
//   * Header-only and dependency-free when the framework is not on the include path: it then reports
//     available() == false and every consumer falls back to its portable kernels and says so in `notes`.
//     Opt out explicitly with -DTENSORF_FWBENCH_DISABLE.
//   * Thread pinning uses the raw sched_{get,set}affinity SYSCALLS (Linux and Android). Bionic only exposes
//     the libc wrappers from newer API levels, and the CPU_SET macros need _GNU_SOURCE; the syscalls need
//     neither. On macOS / iOS / Windows pinning is unavailable: per-class numbers are skipped and noted.
//   * Pinning order matters. A thread created while the caller is pinned INHERITS that mask, so pin only
//     around single-threaded measurements and always release before creating any worker thread or changing
//     the BLAS thread count (BlasThreads is constructed BEFORE PinGuard everywhere in this file).
//   * Memory is bounded by Options::max_bytes and every point returns ok=false instead of throwing.
//
// Limits (stated, not hidden)
//   * BLAS worker threads are not pinned: on a big.LITTLE phone the multi-thread curve is what the scheduler
//     gives a plain process, which is also what training gets.
//   * ScopedThreads is the framework's own switch; if the active backend ignores it, the scaling curve is
//     flat and `threads_effective` is false (the planner then assigns one thread to GEMM-bound layers).
#pragma once

#include <algorithm>
#include <chrono>
#include <ctime>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
  #include <sys/syscall.h>
  #include <unistd.h>
  #define TENSORF_FWBENCH_HAS_AFFINITY 1
#else
  #define TENSORF_FWBENCH_HAS_AFFINITY 0
#endif

#if !defined(TENSORF_FWBENCH_DISABLE) && defined(__has_include)
  #if __has_include("DataStructures/Matrix.hpp") && __has_include("tools/tensorf_blas.h")
    #include "DataStructures/Matrix.hpp"
    #include "tools/tensorf_blas.h"
    #define TENSORF_FWBENCH_AVAILABLE 1
  #endif
#endif
#ifndef TENSORF_FWBENCH_AVAILABLE
  #define TENSORF_FWBENCH_AVAILABLE 0
#endif

namespace tensorf::fwbench {

inline constexpr bool available() { return TENSORF_FWBENCH_AVAILABLE != 0; }

inline const char* backend_name() {
#if TENSORF_FWBENCH_AVAILABLE
    return tensorf::blas::backend();
#else
    return "none (framework headers not on the include path)";
#endif
}

// ───────────────────────────── timing ─────────────────────────────
namespace detail {

using Clock = std::chrono::steady_clock;
inline double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

inline void keep(const void* p) {
#if defined(_MSC_VER) && !defined(__clang__)
    (void)*static_cast<const volatile char*>(p);
#else
    asm volatile("" : : "g"(p) : "memory");
#endif
}

}  // namespace detail

struct Rate {
    double median = 0;     // work units per second, scaled by `scale` (GFLOP/s when scale = 1e-9)
    double best = 0;
    double spread = 0;     // (max - min) / median across windows
    int    windows = 0;
    bool   ok = false;
};

// Runs unit() (returns work done by one call) in windows until `min_seconds` have passed and at least 3
// windows exist. Warm-up first: page faults, frequency ramp, BLAS thread wake-up.
template <class F>
Rate measure(F&& unit, double min_seconds, double scale = 1e-9) {
    Rate r;
    unit();
    auto t0 = detail::Clock::now();
    unit();
    const double one = std::max(detail::since(t0), 1e-7);
    const double window = std::max(min_seconds / 4.0, 0.002);
    const int64_t reps = std::max<int64_t>(1, int64_t(std::ceil(window / one)));
    std::vector<double> rates;
    auto start = detail::Clock::now();
    while ((detail::since(start) < min_seconds || rates.size() < 3) && rates.size() < 64) {
        double work = 0;
        auto t = detail::Clock::now();
        for (int64_t i = 0; i < reps; ++i) work += unit();
        rates.push_back(work / std::max(detail::since(t), 1e-9) * scale);
        if (detail::since(start) > min_seconds * 8 + 0.5) break;       // runaway guard on a very slow device
    }
    std::sort(rates.begin(), rates.end());
    r.windows = int(rates.size());
    r.median = rates[rates.size() / 2];
    r.best = rates.back();
    r.spread = r.median > 0 ? (rates.back() - rates.front()) / r.median : 0;
    r.ok = r.median > 0 && std::isfinite(r.median);
    return r;
}

// ───────────────────────────── affinity ─────────────────────────────

inline constexpr bool pinning_supported() { return TENSORF_FWBENCH_HAS_AFFINITY != 0; }

namespace detail {
#if TENSORF_FWBENCH_HAS_AFFINITY
constexpr size_t kMaskWords = 128;   // 8192 CPUs

inline bool get_affinity(std::vector<unsigned long>& mask) {
    mask.assign(kMaskWords, 0);
    const long r = ::syscall(SYS_sched_getaffinity, 0, mask.size() * sizeof(unsigned long), mask.data());
    if (r <= 0) return false;                                       // raw syscall returns the byte count it filled
    mask.resize((size_t(r) + sizeof(unsigned long) - 1) / sizeof(unsigned long));
    return true;
}
inline bool set_affinity(const std::vector<unsigned long>& mask) {
    return ::syscall(SYS_sched_setaffinity, 0, mask.size() * sizeof(unsigned long), mask.data()) == 0;
}
inline bool set_affinity_cpu(int cpu) {
    constexpr size_t bits = sizeof(unsigned long) * 8;
    if (cpu < 0 || size_t(cpu) >= kMaskWords * bits) return false;
    std::vector<unsigned long> m(kMaskWords, 0);
    m[size_t(cpu) / bits] |= 1ul << (size_t(cpu) % bits);
    return set_affinity(m);
}
#endif
}  // namespace detail

// Pins the CALLING thread to one cpu for the guard's lifetime and restores the previous mask afterwards.
class PinGuard {
public:
    explicit PinGuard(int cpu) {
#if TENSORF_FWBENCH_HAS_AFFINITY
        had_ = detail::get_affinity(saved_);
        ok_ = had_ && detail::set_affinity_cpu(cpu);
#else
        (void)cpu;
#endif
    }
    ~PinGuard() { release(); }
    PinGuard(const PinGuard&) = delete;
    PinGuard& operator=(const PinGuard&) = delete;
    bool ok() const { return ok_; }
    void release() {
#if TENSORF_FWBENCH_HAS_AFFINITY
        if (ok_ && had_) detail::set_affinity(saved_);
#endif
        ok_ = false;
    }
private:
#if TENSORF_FWBENCH_HAS_AFFINITY
    std::vector<unsigned long> saved_;
    bool had_ = false;
#endif
    bool ok_ = false;
};

// ───────────────────────────── the framework's GEMM ─────────────────────────────

// Sets the BLAS thread count for the guard's lifetime (a no-op without the framework). Construct it BEFORE a
// PinGuard: changing the thread count can create threads, and they would inherit a pinned mask.
class BlasThreads {
public:
#if TENSORF_FWBENCH_AVAILABLE
    explicit BlasThreads(int n) : st_(std::max(1, n)) {}
private:
    tensorf::blas::ScopedThreads st_;
#else
    explicit BlasThreads(int) {}
#endif
};

#if TENSORF_FWBENCH_AVAILABLE
// C = A(MxK) * B(KxN) through Matrix::matmul, the same call the Tensor ops make.
template <class T>
class GemmRunner {
public:
    GemmRunner(int M, int N, int K) : flops_(2.0 * double(M) * double(N) * double(K)) {
        A_ = std::make_unique<Matrix<T>>(Matrix<T>::randu(T(-0.5), T(0.5), {size_t(M), size_t(K)}));
        B_ = std::make_unique<Matrix<T>>(Matrix<T>::randu(T(-0.5), T(0.5), {size_t(K), size_t(N)}));
    }
    double run() {
        Matrix<T> C = A_->matmul(*B_);
        detail::keep(&C.data[0]);
        return flops_;
    }
private:
    std::unique_ptr<Matrix<T>> A_, B_;
    double flops_;
};
#else
template <class T>
class GemmRunner {
public:
    GemmRunner(int, int, int) {}
    double run() { return 0.0; }
};
#endif

// ───────────────────────────── results ─────────────────────────────

struct FwPoint {
    std::string key;                 // "f32_16" "f32_64" "f32_256" "f32_512" "f32_tall" "f32_skinny" "f32_rect" "f64_256"
    int M = 0, N = 0, K = 0;
    bool f64 = false;
    double gflops = 0, spread = 0;
    int windows = 0;
    bool ok = false;
};

struct FwScale {
    int threads = 0;
    double gflops = 0;
    double speedup = 0;              // vs one thread
    double eff = 0;                  // speedup / threads
    // The same thread count on a narrow-output GEMM (GCN shape 2708x16x1433), where BLAS threading overhead shows.
    double small_gflops = 0;
    double small_speedup = 0;        // vs one thread
    double cpu_ratio = 0;            // process CPU seconds / wall seconds during that run (>> speedup = threads spinning)
};

struct FwClass {
    int index = 0;                   // 0 = fastest class
    std::string name;
    int cpu = -1;                    // the cpu the single thread was pinned to
    double gflops_1t = 0;
    bool pinned = false;
};

struct FwProfile {
    bool available = false;
    std::string backend;
    std::vector<FwPoint> points;     // one thread (on the fastest core class when the device has several)
    std::vector<FwScale> scaling;    // f32 GEMM, BLAS thread counts 1..T
    std::vector<FwClass> classes;    // big.LITTLE: one pinned thread per class
    int  single_thread_cpu = -1;     // >= 0: the 1-thread points were pinned to this cpu
    bool single_thread_pinned = false;
    int  best_threads = 1;           // fewest threads within 5% of the best GEMM throughput
    bool threads_effective = false;  // more threads gave >= 1.25x
    bool small_threads_effective = true;   // ... and on the narrow GNN-shaped GEMM (false: BLAS threads cost more than they give)
    double max_speedup = 1.0;
    double seconds = 0;
    std::vector<std::string> skipped;
    std::vector<std::string> notes;

    double gflops(const std::string& key) const {
        for (const auto& p : points) if (p.ok && p.key == key) return p.gflops;
        return 0.0;
    }
    // Best single-thread f32 rate over the square shapes: the framework's GEMM roof on one thread.
    double best_f32() const {
        double m = 0;
        for (const auto& p : points) if (p.ok && !p.f64 && (p.key == "f32_64" || p.key == "f32_256" || p.key == "f32_512")) m = std::max(m, p.gflops);
        return m;
    }
    bool usable() const { return available && best_f32() > 0; }
    int max_threads() const { int m = 1; for (const auto& s : scaling) m = std::max(m, s.threads); return m; }
    // Speed-up of T BLAS threads over one, interpolated on the measured curve; never below 1.
    double speedup(int T) const {
        if (T <= 1 || scaling.empty()) return 1.0;
        const FwScale* lo = nullptr; const FwScale* hi = nullptr;
        for (const auto& s : scaling) {
            if (s.threads <= T) lo = &s;
            if (s.threads >= T && !hi) hi = &s;
        }
        double v = 1.0;
        if (lo && hi && hi->threads > lo->threads) v = lo->speedup + (hi->speedup - lo->speedup) * double(T - lo->threads) / double(hi->threads - lo->threads);
        else if (lo) v = lo->speedup;
        else if (hi) v = hi->speedup;
        return std::max(1.0, v);
    }
};

struct Options {
    double   window_s = 0.06;                          // timing window per point
    double   budget_s = 5.0;                           // wall budget for run()
    uint64_t max_bytes = 256ull << 20;                 // operand + result bytes per point
    int      max_threads = 1;                          // usable threads on this device
    bool     scaling = true;
    // cpus of each core class, FASTEST FIRST. Two or more entries enable the 1-thread pin and the per-class stage.
    std::vector<std::vector<int>> classes;
    std::vector<std::string>      class_names;
    std::function<bool()> should_stop;                 // governor hook (thermal, battery)
};

namespace detail {
struct Spec { const char* key; int M, N, K; bool f64; };

inline uint64_t bytes_for(const Spec& s) {
    const uint64_t e = s.f64 ? 8 : 4;
    return (uint64_t(s.M) * s.K + uint64_t(s.K) * s.N + 2ull * s.M * s.N) * e;   // A, B, C and the temporary result
}

template <class T>
Rate time_gemm(int M, int N, int K, double window_s) {
    GemmRunner<T> g(M, N, K);
    return measure([&] { return g.run(); }, window_s);
}
}  // namespace detail

// Runs the whole framework stage. Never throws; problems are recorded in notes / skipped.
inline FwProfile run(const Options& o) {
    FwProfile P;
    P.backend = backend_name();
    if (!available()) {
        P.notes.push_back("framework headers (DataStructures/Matrix.hpp, tools/tensorf_blas.h) are not on the include path: "
                          "GEMM numbers are portable-kernel numbers, NOT what Matrix::matmul will get");
        return P;
    }
    P.available = true;
    const auto t_start = detail::Clock::now();
    auto stopped = [&] { return (o.should_stop && o.should_stop()) || detail::since(t_start) > o.budget_s; };
    const bool multi_class = o.classes.size() >= 2 && pinning_supported();
    const int fast_cpu = multi_class && !o.classes.front().empty() ? o.classes.front().front() : -1;

    static const detail::Spec specs[] = {
        {"f32_16", 16, 16, 16, false},    {"f32_64", 64, 64, 64, false},        {"f32_256", 256, 256, 256, false},
        {"f32_512", 512, 512, 512, false}, {"f32_tall", 8192, 32, 32, false},    {"f32_skinny", 4, 1024, 1024, false},
        {"f32_rect", 64, 784, 256, false}, {"f64_256", 256, 256, 256, true},
        // GNN / GCN shapes (Cora: X[2708x1433] * W[1433x16] and the backward X^T * dY): tall, very narrow output.
        {"f32_gnn", 2708, 16, 1433, false}, {"f32_gnn_t", 1433, 16, 2708, false},
    };

    try {
        // ---- single thread, on the fastest core class ----
        {
            BlasThreads one(1);                                       // before the pin: see the header comment
            std::unique_ptr<PinGuard> pin;
            if (fast_cpu >= 0) {
                pin = std::make_unique<PinGuard>(fast_cpu);
                P.single_thread_cpu = fast_cpu;
                P.single_thread_pinned = pin->ok();
                if (!pin->ok()) P.notes.push_back("could not pin to the fastest core class: 1-thread numbers are whatever core the scheduler chose");
            }
            for (const auto& s : specs) {
                if (stopped()) { P.skipped.push_back(std::string("framework ") + s.key + ": budget or governor stop"); continue; }
                if (detail::bytes_for(s) > o.max_bytes) { P.skipped.push_back(std::string("framework ") + s.key + ": over the memory cap"); continue; }
                FwPoint pt; pt.key = s.key; pt.M = s.M; pt.N = s.N; pt.K = s.K; pt.f64 = s.f64;
                Rate r = s.f64 ? detail::time_gemm<double>(s.M, s.N, s.K, o.window_s) : detail::time_gemm<float>(s.M, s.N, s.K, o.window_s);
                pt.gflops = r.median; pt.spread = r.spread; pt.windows = r.windows; pt.ok = r.ok;
                P.points.push_back(pt);
            }
        }   // pin released, BLAS threads restored

        // ---- BLAS thread scaling (unpinned: what a plain process gets) ----
        if (o.scaling && o.max_threads > 1 && !stopped()) {
            const double g512 = P.gflops("f32_512");
            int n = (g512 >= 3.0 && detail::bytes_for({"", 512, 512, 512, false}) <= o.max_bytes) ? 512 : 256;
            std::vector<int> counts;
            for (int t = 1; t < o.max_threads; t *= 2) counts.push_back(t);
            counts.push_back(o.max_threads);
            double g1 = 0, q1 = 0;
            for (int t : counts) {
                if (stopped()) { P.skipped.push_back("framework scaling: remaining thread counts skipped (budget or governor)"); break; }
                BlasThreads bt(t);
                Rate r = detail::time_gemm<float>(n, n, n, o.window_s);
                if (!r.ok) break;
                if (t == 1) g1 = r.median;
                FwScale s; s.threads = t; s.gflops = r.median;
                s.speedup = g1 > 0 ? r.median / g1 : 1.0;
                s.eff = s.speedup / double(t);
                // Narrow-output GEMM (GNN layers): BLAS packs/forks per call, so threads can lose here even when they win above.
                if (detail::bytes_for({"", 2708, 16, 1433, false}) <= o.max_bytes && !stopped()) {
                    const std::clock_t c0 = std::clock(); const auto w0 = detail::Clock::now();
                    Rate q = detail::time_gemm<float>(2708, 16, 1433, o.window_s);
                    const double wall = std::max(detail::since(w0), 1e-9);
                    if (q.ok) {
                        if (t == 1) q1 = q.median;
                        s.small_gflops = q.median;
                        s.small_speedup = q1 > 0 ? q.median / q1 : 1.0;
                        s.cpu_ratio = double(std::clock() - c0) / CLOCKS_PER_SEC / wall;
                    }
                }
                P.scaling.push_back(s);
            }
            for (const auto& s : P.scaling)
                if (s.threads > 1 && s.small_speedup > 0 && s.small_speedup < 0.95) P.small_threads_effective = false;
            for (const auto& s : P.scaling)
                if (s.threads > 1 && s.small_speedup > 0 && s.small_speedup < 1.15 && s.cpu_ratio > 1.5 * std::max(1.0, s.small_speedup)) {
                    P.notes.push_back("BLAS threads spin on narrow GEMMs (" + std::to_string(s.threads) + " threads use " +
                        std::to_string(s.cpu_ratio).substr(0, 4) + " cores of CPU for " + std::to_string(s.small_speedup).substr(0, 4) +
                        "x the speed): for GNN/small layers run BLAS on 1 thread (OPENBLAS_NUM_THREADS=1 or tensorf::blas::ScopedThreads(1))");
                    break;
                }
            double best = 0;
            for (const auto& s : P.scaling) { best = std::max(best, s.gflops); P.max_speedup = std::max(P.max_speedup, s.speedup); }
            for (const auto& s : P.scaling) if (s.gflops >= best * 0.95) { P.best_threads = s.threads; break; }
            P.threads_effective = P.max_speedup >= 1.25;
            if (!P.threads_effective && P.scaling.size() >= 2)
                P.notes.push_back(std::string("BLAS backend '") + P.backend + "' gained < 1.25x from more threads: it is single-threaded here "
                                  "(the shim, a build without threads, or OPENBLAS_NUM_THREADS=1). GEMM-bound layers are planned on one thread");
        } else if (o.max_threads <= 1) {
            P.best_threads = 1;
        }

        // ---- per core class (big.LITTLE / hybrid) ----
        if (multi_class && !stopped()) {
            for (size_t i = 0; i < o.classes.size(); ++i) {
                if (o.classes[i].empty()) continue;
                if (stopped()) { P.skipped.push_back("framework classes: remaining classes skipped (budget or governor)"); break; }
                FwClass c; c.index = int(i); c.cpu = o.classes[i].front();
                c.name = i < o.class_names.size() ? o.class_names[i] : std::string("class") + std::to_string(i);
                BlasThreads one(1);
                GemmRunner<float> g(256, 256, 256);
                PinGuard pin(c.cpu);
                c.pinned = pin.ok();
                if (c.pinned) c.gflops_1t = measure([&] { return g.run(); }, o.window_s).median;
                else P.notes.push_back("could not pin to class " + c.name + ": its framework rate is omitted");
                P.classes.push_back(c);
            }
        } else if (o.classes.size() >= 2 && !pinning_supported()) {
            P.skipped.push_back("framework classes: thread pinning is not supported on this platform");
        }
    } catch (const std::exception& e) {
        P.notes.push_back(std::string("framework stage aborted: ") + e.what());
    } catch (...) {
        P.notes.push_back("framework stage aborted by an unknown exception");
    }
    P.seconds = detail::since(t_start);
    if (!P.usable()) P.notes.push_back("no framework GEMM point was measured: the cost model falls back to its portable kernels");
    return P;
}

// GFLOP/s of a long f32 GEMM load, one entry per `window_s` window, for the throttling probe. Empty when the
// framework is unavailable. `stop` is polled between windows. BLAS runs on `threads` threads, like training.
inline std::vector<double> sustained_windows(int threads, double duration_s, double window_s, uint64_t max_bytes,
                                             const std::function<bool()>& stop = {}, double* elapsed = nullptr) {
    std::vector<double> out;
    if (!available()) return out;
    const int n = 256;
    if (detail::bytes_for({"", n, n, n, false}) > max_bytes) return out;
    try {
        BlasThreads bt(threads);
        GemmRunner<float> g(n, n, n);
        g.run();
        const auto t0 = detail::Clock::now();
        while (detail::since(t0) < duration_s) {
            if (stop && stop()) break;
            const auto w0 = detail::Clock::now();
            double work = 0;
            do { work += g.run(); } while (detail::since(w0) < window_s);
            out.push_back(work / std::max(detail::since(w0), 1e-9) * 1e-9);
        }
        if (elapsed) *elapsed = detail::since(t0);
    } catch (...) {
        out.clear();
    }
    return out;
}

}  // namespace tensorf::fwbench
