// DeviceBenchmark.hpp
//
// An algorithm-independent capability suite. It answers "what can this device
// actually do" with a vector of measurements, not one score, so that any
// algorithm's cost model (CNN, GNN, RNN, Markov chain, Transformer, particle
// simulation, ...) can be priced against it later.
//
// Kernels span the workload classes that matter for those algorithms:
//
//   compute bound       fma_f32 / fma_f64 (vector FMA throughput), gemm at several
//                       shapes (tiny, cache-resident, tall-skinny, larger)
//   bandwidth bound     STREAM triad across working-set sizes (L1 .. DRAM)
//   latency bound       dependent pointer chase over one random cycle (L1 .. DRAM);
//                       this is also the bound for Markov-chain style sampling
//   irregular           CSR gather/accumulate on a power-law graph (GNN aggregation)
//   stencil             direct 3x3 convolution (CNN)
//   pairwise            Lennard-Jones style n-body (particle simulation)
//   sequential          recurrent matvec chain (RNN)
//   overhead            fork-join latency of a worker pool, tiny-call dispatch cost
//   parallel            thread scaling, per-core-class throughput, per-thread rates
//                       when every core runs at once (big.LITTLE aware)
//   sustained           a throttling probe: all cores busy for a while, windowed
//
// Rules the suite follows
//   * Self-contained kernels, compiled with the same flags as the library, so
//     the numbers are what OUR code gets on this device (including a build that
//     forgot -O2 or -march; `build` records that).
//   * Time-boxed. A hard budget, a per-measurement window, a stop callback the
//     governor can use (thermal, battery, user activity), and a memory cap for
//     working sets. Nothing allocates more than `max_working_set_bytes`.
//   * Medians over several windows, with a spread value, so the advisor can
//     tell a stable number from a noisy one. `best` is kept for reference.
//   * Every kernel returns a checksum and is verified against a plain reference
//     in tests/device_benchmark_test.cpp, so a kernel the compiler deleted or
//     broke cannot silently report an absurd speed.
//
// Limits (stated, not hidden)
//   * The sustained probe is seconds long by default. Real phone throttling can
//     take minutes; `sustained.duration_s` is recorded so consumers know how
//     much to trust `fraction`. The governor can request a longer run.
//   * Core pinning is implemented for Linux/Android and Windows. Where pinning
//     is unavailable (macOS/iOS) per-class numbers are skipped and noted.
//   * Reported flop/s are for these kernels under this compiler, not the
//     silicon's theoretical peak.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "HardwareFingerprint.hpp"

#if defined(_WIN32)
  // windows.h already included through HardwareFingerprint.hpp
#elif !defined(__APPLE__)
  #include <pthread.h>
  #include <sched.h>
#endif

namespace tensorf::profile {

inline constexpr int kBenchmarkSchema = 2;

// ───────────────────────────── configuration ─────────────────────────────

struct BenchConfig {
    double budget_seconds = 12.0;          // hard wall-clock budget for the whole suite
    double window_seconds = 0.06;          // minimum time spent measuring each kernel
    uint64_t max_working_set_bytes = 256ull << 20;   // largest single working set
    double sustained_seconds = 6.0;        // 0 disables the sustained probe
    double sustained_window_seconds = 0.25;
    int max_threads = 0;                   // 0 = fingerprint.usable_threads
    bool per_class = true;                 // measure each core class separately (needs pinning)
    // Governor hooks. Both are optional.
    std::function<bool()> should_stop;                      // true = abort between measurements
    std::function<std::optional<double>()> read_temp_c;     // hottest relevant sensor, deg C

    static BenchConfig quick() {           // unit tests / first-run probe
        BenchConfig c;
        c.budget_seconds = 6.0;
        c.window_seconds = 0.015;
        c.max_working_set_bytes = 64ull << 20;
        c.sustained_seconds = 1.0;
        c.sustained_window_seconds = 0.1;
        return c;
    }
};

// ───────────────────────────── results ─────────────────────────────

struct Metric {
    double median = 0;     // what planning should use
    double best = 0;       // fastest window (reference)
    double rsd = 0;        // (max-min)/median over windows; large = noisy device
    int windows = 0;
    bool ok = false;
};

struct LevelPoint {
    uint64_t bytes = 0;    // working-set size
    double ns = 0;         // latency sweep: ns per dependent load
    double gbps = 0;       // stream sweep: GB/s
};

struct ThreadRates {
    std::vector<int> cpus;           // cpu each worker was pinned to (empty = unpinned)
    std::vector<double> gflops;      // per-thread rate while all ran together
    double total = 0;
    bool pinned = false;
};

struct ScalePoint {
    int threads = 0;
    double gflops = 0;               // fma_f32 summed over threads
    double gbps = 0;                 // stream triad summed over threads (DRAM-sized, private arrays)
    double compute_eff = 0;          // gflops / (threads * single-thread)
    double bw_eff = 0;               // gbps / single-thread gbps (not divided by threads: bandwidth saturates)
};

struct ClassPerf {
    int index = 0;
    std::string name;
    std::vector<int> cpus;
    bool pinned = false;
    double gflops_1t = 0;
    double gflops_all = 0;
    double gbps_1t = 0;
    double gbps_all = 0;
};

struct SustainedResult {
    bool ran = false;
    int threads = 0;
    double window_s = 0;
    double duration_s = 0;
    std::vector<double> window_gflops;
    double peak_gflops = 0;          // median of the first third of windows
    double sustained_gflops = 0;     // median of the last third
    double fraction = 1.0;           // sustained / peak
    bool throttled = false;          // fraction < 0.90
    std::optional<double> temp_start_c, temp_end_c;
};

struct CapabilityProfile {
    int schema = kBenchmarkSchema;
    std::string device_id;
    struct Build {
        bool optimized = false;
        std::string simd;            // widest vector ISA the binary was compiled for
    } build;

    Metric fma_f32, fma_f64;                        // GFLOP/s, one thread
    std::map<std::string, Metric> gemm;             // GFLOP/s, one thread: "f32_16", "f32_64", "f32_256", "f32_512", "f32_tall", "f64_256"
    std::vector<LevelPoint> stream_1t;              // GB/s by working set
    std::vector<LevelPoint> latency;                // ns per dependent load by working set
    Metric gather_edges_per_s;                      // power-law CSR aggregation, 32 float features per node
    double gather_gbps = 0;                         // feature bytes gathered per second
    uint64_t gather_ws_bytes = 0;                   // size of the feature table it was measured on (gather speed depends on it)
    double gather_efficiency = 0;                   // gather_gbps / DRAM stream gbps (single thread)
    Metric conv_gflops;                             // direct 3x3, 16->16 channels, 64x64
    Metric md_ginteractions;                        // LJ pair interactions, 1e9/s
    Metric rnn_steps_per_s;                         // 256-hidden recurrent step
    double rnn_gflops = 0;
    double fork_join_us = 0;                        // dispatch + join of the worker pool
    double call_overhead_ns = 0;                    // one tiny op through std::function

    std::vector<ScalePoint> scaling;
    ThreadRates all_cores;
    std::vector<ClassPerf> classes;
    SustainedResult sustained;

    // Convenience values the advisor reads.
    double l1_latency_ns = 0;
    double dram_latency_ns = 0;
    double dram_gbps_1t = 0;
    double dram_gbps_all = 0;
    uint64_t dram_probe_bytes = 0;                  // size used for the DRAM numbers
    bool dram_probe_valid = false;                  // true when it is >= 4x the last-level cache

    std::vector<std::string> skipped;               // stages not run, with the reason
    std::vector<std::string> notes;
    double duration_s = 0;
    bool stopped_early = false;

    bool derived_ok() const {           // derived values agree with the raw sweeps
        if (latency.empty() || stream_1t.empty()) return false;
        return l1_latency_ns == latency.front().ns && dram_latency_ns == latency.back().ns &&
               dram_gbps_1t == stream_1t.back().gbps && dram_probe_bytes == latency.back().bytes;
    }
    std::string to_json() const;
    std::string summary() const;
};

// ───────────────────────────── kernels ─────────────────────────────
namespace kernels {

inline uint64_t rng_next(uint64_t& s) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return s * 2685821657736338717ull;
}
inline double rng_unit(uint64_t& s) { return double(rng_next(s) >> 11) * (1.0 / 9007199254740992.0); }

// Stops the optimiser from deleting results or hoisting work out of loops.
inline void keep(const void* p) {
#if defined(_MSC_VER) && !defined(__clang__)
    _ReadWriteBarrier();
    (void)*static_cast<const volatile char*>(p);
#else
    asm volatile("" : : "g"(p) : "memory");
#endif
}

// ---- vector FMA throughput ----
// 64 independent lanes, 16 dependent FMAs per lane per pass: enough independent
// work to hide FMA latency on 128..512-bit units, small enough to stay in registers/L1.
inline constexpr int kFmaLanes = 64, kFmaDepth = 16;
template <class T>
double fma_kernel(int64_t passes, T* a) {
    const T x = T(1.0000001), y = T(1e-7);
    for (int64_t it = 0; it < passes; ++it) {
        for (int d = 0; d < kFmaDepth; ++d)
            for (int i = 0; i < kFmaLanes; ++i) a[i] = a[i] * x + y;
        keep(a);
    }
    return 2.0 * kFmaLanes * kFmaDepth * double(passes);
}

// ---- GEMM: C += A(MxK) * B(KxN), row-major, i-k-j with 4 rows per pass ----
template <class T>
void gemm_acc(const T* __restrict A, const T* __restrict B, T* __restrict C, int M, int N, int K) {
    int i = 0;
    for (; i + 4 <= M; i += 4) {
        T* c0 = C + size_t(i) * N; T* c1 = c0 + N; T* c2 = c1 + N; T* c3 = c2 + N;
        for (int k = 0; k < K; ++k) {
            const T a0 = A[size_t(i) * K + k], a1 = A[size_t(i + 1) * K + k];
            const T a2 = A[size_t(i + 2) * K + k], a3 = A[size_t(i + 3) * K + k];
            const T* b = B + size_t(k) * N;
            for (int j = 0; j < N; ++j) {
                const T bj = b[j];
                c0[j] += a0 * bj; c1[j] += a1 * bj; c2[j] += a2 * bj; c3[j] += a3 * bj;
            }
        }
    }
    for (; i < M; ++i) {
        T* c = C + size_t(i) * N;
        for (int k = 0; k < K; ++k) {
            const T a = A[size_t(i) * K + k];
            const T* b = B + size_t(k) * N;
            for (int j = 0; j < N; ++j) c[j] += a * b[j];
        }
    }
}

template <class T>
struct GemmCase {
    int M, N, K;
    std::vector<T> A, B, C;
    GemmCase(int m, int n, int k) : M(m), N(n), K(k), A(size_t(m) * k), B(size_t(k) * n), C(size_t(m) * n) {
        uint64_t s = 0x9e3779b97f4a7c15ull ^ uint64_t(m * 131 + n * 17 + k);
        for (auto& v : A) v = T(rng_unit(s) - 0.5);
        for (auto& v : B) v = T(rng_unit(s) - 0.5);
        for (auto& v : C) v = T(0);
    }
    double run() {                       // returns flops; C is kept bounded by scaling
        gemm_acc<T>(A.data(), B.data(), C.data(), M, N, K);
        if (++reps % 8 == 0) for (auto& v : C) v *= T(0.5);   // avoid unbounded growth (cheap vs the gemm)
        keep(C.data());
        return 2.0 * M * N * K;
    }
    int reps = 0;
};

// ---- STREAM triad: a = b + s*c ----
struct StreamCase {
    size_t n;
    std::vector<float> a, b, c;
    explicit StreamCase(size_t bytes) : n(std::max<size_t>(bytes / 12, 16)), a(n, 0.f), b(n, 1.f), c(n, 2.f) {}
    double run() {                       // returns bytes moved
        float* __restrict pa = a.data(); const float* __restrict pb = b.data(); const float* __restrict pc = c.data();
        const float s = 0.5f;
        for (size_t i = 0; i < n; ++i) pa[i] = pb[i] + s * pc[i];
        keep(pa);
        return 12.0 * double(n);
    }
};

// ---- dependent pointer chase over one random cycle (Sattolo) ----
struct ChaseTable {
    static constexpr size_t kStride = 16;        // uint32 entries per node = one 64-byte line
    size_t nodes = 0;
    std::vector<uint32_t> t;
    explicit ChaseTable(size_t bytes, uint64_t seed = 12345) {
        nodes = std::max<size_t>(bytes / 64, 8);
        t.assign(nodes * kStride, 0);
        std::vector<uint32_t> p(nodes);
        std::iota(p.begin(), p.end(), 0u);
        uint64_t s = seed;
        for (size_t i = nodes - 1; i >= 1; --i) {   // Sattolo: yields exactly one cycle
            size_t j = size_t(rng_next(s) % i);
            std::swap(p[i], p[j]);
        }
        for (size_t i = 0; i < nodes; ++i) t[i * kStride] = p[i];
    }
    uint32_t walk(uint32_t start, int64_t steps) const {
        uint32_t q = start;
        const uint32_t* tp = t.data();
        for (int64_t s = 0; s < steps; ++s) q = tp[size_t(q) * kStride];
        return q;
    }
};

// ---- power-law CSR graph and neighbour aggregation (GNN) ----
struct Graph {
    int n = 0;
    std::vector<uint32_t> rowptr, col;
    size_t edges() const { return col.size(); }
};
inline Graph make_powerlaw_graph(int n, uint64_t seed = 777, double min_deg = 4.0, double alpha = 1.5, int max_deg = 512) {
    Graph g;
    g.n = n;
    g.rowptr.assign(size_t(n) + 1, 0);
    uint64_t s = seed;
    std::vector<int> deg(n);
    for (int i = 0; i < n; ++i) {
        double u = rng_unit(s);
        int d = int(min_deg / std::pow(1.0 - u, 1.0 / alpha));   // Pareto
        deg[i] = std::clamp(d, 1, std::min(max_deg, std::max(1, n - 1)));
        g.rowptr[i + 1] = g.rowptr[i] + uint32_t(deg[i]);
    }
    g.col.resize(g.rowptr[n]);
    for (int i = 0; i < n; ++i)
        for (uint32_t e = g.rowptr[i]; e < g.rowptr[i + 1]; ++e) g.col[e] = uint32_t(rng_next(s) % uint64_t(n));
    return g;
}
template <int F>
void spmm_sum(const Graph& g, const float* __restrict X, float* __restrict Y) {
    for (int i = 0; i < g.n; ++i) {
        float acc[F] = {0};
        for (uint32_t e = g.rowptr[i]; e < g.rowptr[i + 1]; ++e) {
            const float* xj = X + size_t(g.col[e]) * F;
            for (int f = 0; f < F; ++f) acc[f] += xj[f];
        }
        float* yi = Y + size_t(i) * F;
        for (int f = 0; f < F; ++f) yi[f] = acc[f];
    }
}
struct GatherCase {
    static constexpr int F = 32;
    Graph g;
    std::vector<float> X, Y;
    explicit GatherCase(int n) : g(make_powerlaw_graph(n)), X(size_t(n) * F), Y(size_t(n) * F) {
        uint64_t s = 99;
        for (auto& v : X) v = float(rng_unit(s));
    }
    double run() { spmm_sum<F>(g, X.data(), Y.data()); keep(Y.data()); return double(g.edges()); }   // edges
};

// ---- direct 3x3 convolution, valid padding ----
struct ConvCase {
    int C, H, W;
    std::vector<float> in, w, out;
    ConvCase(int c, int h, int ww) : C(c), H(h), W(ww), in(size_t(c) * h * ww), w(size_t(c) * c * 9), out(size_t(c) * (h - 2) * (ww - 2)) {
        uint64_t s = 4242;
        for (auto& v : in) v = float(rng_unit(s) - 0.5);
        for (auto& v : w) v = float(rng_unit(s) - 0.5);
    }
    void conv() {
        const int OH = H - 2, OW = W - 2;
        std::fill(out.begin(), out.end(), 0.f);
        for (int oc = 0; oc < C; ++oc) {
            float* o = out.data() + size_t(oc) * OH * OW;
            for (int ic = 0; ic < C; ++ic) {
                const float* ip = in.data() + size_t(ic) * H * W;
                for (int ky = 0; ky < 3; ++ky)
                    for (int kx = 0; kx < 3; ++kx) {
                        const float wv = w[((size_t(oc) * C + ic) * 3 + ky) * 3 + kx];
                        for (int y = 0; y < OH; ++y) {
                            const float* src = ip + size_t(y + ky) * W + kx;
                            float* dst = o + size_t(y) * OW;
                            for (int x = 0; x < OW; ++x) dst[x] += wv * src[x];
                        }
                    }
            }
        }
        keep(out.data());
    }
    double run() { conv(); return 2.0 * C * C * 9.0 * (H - 2) * (W - 2); }
};

// ---- Lennard-Jones style pairwise force, 8 independent lanes (no FP reduction needed to vectorise) ----
struct MdCase {
    int n;
    std::vector<float> x, y, z, fx, fy, fz;
    explicit MdCase(int count) : n(count & ~7), x(n), y(n), z(n), fx(n), fy(n), fz(n) {
        uint64_t s = 31337;
        const double box = std::cbrt(double(n)) * 1.1;       // roughly liquid density
        for (int i = 0; i < n; ++i) { x[i] = float(rng_unit(s) * box); y[i] = float(rng_unit(s) * box); z[i] = float(rng_unit(s) * box); }
    }
    void forces() {
        constexpr int L = 8;
        for (int i = 0; i < n; ++i) {
            float ax[L] = {0}, ay[L] = {0}, az[L] = {0};
            const float xi = x[i], yi = y[i], zi = z[i];
            for (int j = 0; j < n; j += L)
                for (int l = 0; l < L; ++l) {
                    const float dx = x[j + l] - xi, dy = y[j + l] - yi, dz = z[j + l] - zi;
                    const float r2 = dx * dx + dy * dy + dz * dz + 0.25f;      // softened: no self-singularity, no branch
                    const float inv2 = 1.0f / r2;
                    const float inv6 = inv2 * inv2 * inv2;
                    const float f = (inv6 * inv6 - inv6) * inv2;
                    ax[l] += dx * f; ay[l] += dy * f; az[l] += dz * f;
                }
            float sx = 0, sy = 0, sz = 0;
            for (int l = 0; l < L; ++l) { sx += ax[l]; sy += ay[l]; sz += az[l]; }
            fx[i] = sx; fy[i] = sy; fz[i] = sz;
        }
        keep(fx.data());
    }
    double run() { forces(); return double(n) * double(n); }    // pair interactions
};

// ---- recurrent chain: h <- softsign(W h + x), W stored transposed so the update is axpy (vectorises, no reduction) ----
struct RnnCase {
    int H;
    std::vector<float> Wt, h, nh, x;
    explicit RnnCase(int hidden) : H(hidden), Wt(size_t(hidden) * hidden), h(hidden, 0.1f), nh(hidden), x(hidden) {
        uint64_t s = 2024;
        const float sc = 1.0f / std::sqrt(float(hidden));
        for (auto& v : Wt) v = float(rng_unit(s) - 0.5) * sc;
        for (auto& v : x) v = float(rng_unit(s) - 0.5) * 0.1f;
    }
    void step() {
        for (int j = 0; j < H; ++j) nh[j] = x[j];
        for (int k = 0; k < H; ++k) {
            const float hk = h[k];
            const float* w = Wt.data() + size_t(k) * H;
            for (int j = 0; j < H; ++j) nh[j] += w[j] * hk;
        }
        for (int j = 0; j < H; ++j) h[j] = nh[j] / (1.0f + std::fabs(nh[j]));
        keep(h.data());
    }
    double run() { step(); return 1.0; }                        // one sequential step
    double flops_per_step() const { return 2.0 * H * H + 3.0 * H; }
};

}  // namespace kernels

// ───────────────────────────── pinning and worker pool ─────────────────────────────

inline bool pin_current_thread(int cpu) {
    if (cpu < 0) return false;
#if defined(_WIN32)
    if (cpu >= 64) return false;
    return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << cpu) != 0;
#elif defined(__APPLE__)
    (void)cpu;
    return false;
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof set, &set) == 0;     // pid 0 = calling thread on Linux/Android
#endif
}

inline bool pinning_supported() {
#if defined(__APPLE__)
    return false;
#else
    return true;
#endif
}

// Persistent fork-join pool for benchmarking: spin briefly then yield, so
// dispatch latency is representative of a real executor.
class BenchPool {
public:
    // cpus[i] = cpu to pin worker i to, or -1 for no pin.
    explicit BenchPool(std::vector<int> cpus) : cpus_(std::move(cpus)) {
        const int n = int(cpus_.size());
        ok_.assign(n, 0);
        for (int i = 0; i < n; ++i) th_.emplace_back([this, i] { loop(i); });
        while (started_.load(std::memory_order_acquire) < n) std::this_thread::yield();
    }
    ~BenchPool() {
        stop_.store(true, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        for (auto& t : th_) t.join();
    }
    BenchPool(const BenchPool&) = delete;
    BenchPool& operator=(const BenchPool&) = delete;

    int size() const { return int(cpus_.size()); }
    // True when every worker that asked for a cpu got it.
    bool all_pinned() const {
        for (int i = 0; i < size(); ++i) if (cpus_[i] >= 0 && !ok_[i]) return false;
        return true;
    }
    void run(const std::function<void(int)>& fn) {
        fn_ = &fn;
        pending_.store(size(), std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        int spins = 0;
        while (pending_.load(std::memory_order_acquire) != 0) { if (++spins > 100) std::this_thread::yield(); }
    }

private:
    void loop(int tid) {
        if (cpus_[tid] >= 0) ok_[tid] = pin_current_thread(cpus_[tid]) ? 1 : 0;
        started_.fetch_add(1, std::memory_order_release);
        uint64_t seen = 0;
        for (;;) {
            int spins = 0;
            while (gen_.load(std::memory_order_acquire) == seen) { if (++spins > 200) std::this_thread::yield(); }
            seen = gen_.load(std::memory_order_acquire);
            if (stop_.load(std::memory_order_relaxed)) return;
            (*fn_)(tid);
            pending_.fetch_sub(1, std::memory_order_release);
        }
    }
    std::vector<int> cpus_;
    std::vector<char> ok_;
    std::vector<std::thread> th_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> pending_{0};
    std::atomic<int> started_{0};
    std::atomic<bool> stop_{false};
    const std::function<void(int)>* fn_ = nullptr;
};

// ───────────────────────────── measurement helpers ─────────────────────────────
namespace detail {

using Clock = std::chrono::steady_clock;
inline double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// Runs `unit()` (returns work units per call) in windows until `min_seconds`
// have been spent (and at least 3 windows). Rate = work / second.
template <class F>
Metric measure(F&& unit, double min_seconds) {
    Metric m;
    unit();                                           // warm-up: page faults, frequency ramp, caches
    auto t0 = Clock::now();
    double w0 = unit();
    double one = std::max(seconds_since(t0), 1e-7);
    const double window = std::max(min_seconds / 4.0, 0.002);
    int64_t reps = std::max<int64_t>(1, int64_t(std::ceil(window / one)));
    std::vector<double> rates;
    auto start = Clock::now();
    while ((seconds_since(start) < min_seconds || rates.size() < 3) && rates.size() < 64) {
        double work = 0;
        auto t = Clock::now();
        for (int64_t r = 0; r < reps; ++r) work += unit();
        double el = std::max(seconds_since(t), 1e-9);
        rates.push_back(work / el);
        if (seconds_since(start) > min_seconds * 8 + 0.5) break;     // runaway guard on a very slow device
    }
    (void)w0;
    std::sort(rates.begin(), rates.end());
    m.windows = int(rates.size());
    m.median = rates[rates.size() / 2];
    m.best = rates.back();
    m.rsd = m.median > 0 ? (rates.back() - rates.front()) / m.median : 0;
    m.ok = m.median > 0 && std::isfinite(m.median);
    return m;
}

// Per-thread timed loop: run unit() until `seconds` elapse, return work / second.
template <class F>
double timed_rate(F&& unit, double seconds) {
    auto t0 = Clock::now();
    double work = 0;
    do { work += unit(); } while (seconds_since(t0) < seconds);
    return work / std::max(seconds_since(t0), 1e-9);
}

inline std::optional<double> read_max_temp_c(FsSource& fs) {
    std::optional<double> best;
    for (const auto& z : fs.list("/sys/class/thermal")) {
        if (z.rfind("thermal_zone", 0) != 0) continue;
        auto t = fs.read("/sys/class/thermal/" + z + "/temp");
        if (!t) continue;
        auto v = parse_i64(*t);
        if (!v) continue;
        double c = std::abs(*v) >= 1000 ? double(*v) / 1000.0 : double(*v);
        if (c < -40 || c > 150) continue;
        if (!best || c > *best) best = c;
    }
    return best;
}

inline std::string compiled_simd_name() {
#if defined(__AVX512F__)
    return "avx512";
#elif defined(__AVX2__)
    return "avx2";
#elif defined(__AVX__)
    return "avx";
#elif defined(__ARM_FEATURE_SVE)
    return "sve";
#elif defined(__ARM_NEON) || defined(__aarch64__)
    return "neon";
#elif defined(__SSE2__) || defined(_M_X64)
    return "sse2";
#else
    return "scalar";
#endif
}

inline bool compiled_optimized() {
#if defined(__OPTIMIZE__)
    return true;
#elif defined(_MSC_VER) && !defined(_DEBUG)
    return true;
#else
    return false;
#endif
}

}  // namespace detail

// ───────────────────────────── the suite ─────────────────────────────

class DeviceBenchmark {
public:
    DeviceBenchmark(const HardwareFingerprint& fp, BenchConfig cfg = {}) : fp_(fp), cfg_(std::move(cfg)) {}

    CapabilityProfile run() {
        using namespace detail;
        CapabilityProfile P;
        P.device_id = fp_.id();
        P.build.optimized = compiled_optimized();
        P.build.simd = compiled_simd_name();
        if (!P.build.optimized) P.notes.push_back("built without optimisation: every number here is a debug-build number");
        {   // compiled ISA vs what the cpu can do: a portable binary leaves performance on the table
            const auto& s = fp_.cpu.simd;
            if ((P.build.simd == "sse2" && (s.avx2 || s.avx)) || (P.build.simd == "avx2" && s.avx512f) ||
                (P.build.simd == "scalar" && s.neon))
                P.notes.push_back("binary compiled for " + P.build.simd + " but the cpu supports wider vectors: build flags decide the speed");
        }
        t_start_ = Clock::now();
        threads_ = cfg_.max_threads > 0 ? cfg_.max_threads : fp_.usable_threads;
        threads_ = std::max(1, threads_);
        working_cap_ = cfg_.max_working_set_bytes;
        if (fp_.memory.effective_available > 0)
            working_cap_ = std::min<uint64_t>(working_cap_, fp_.memory.effective_available / 5);   // never more than 20% of what is free
        working_cap_ = std::max<uint64_t>(working_cap_, 1ull << 20);

        stage(P, "compute_fma", [&] { stage_fma(P); });
        stage(P, "memory_latency", [&] { stage_latency(P); });
        stage(P, "memory_stream", [&] { stage_stream(P); });
        stage(P, "gemm", [&] { stage_gemm(P); });
        stage(P, "gather", [&] { stage_gather(P); });
        stage(P, "conv_md_rnn", [&] { stage_apps(P); });
        stage(P, "overhead", [&] { stage_overhead(P); });
        stage(P, "thread_scaling", [&] { stage_scaling(P); });
        stage(P, "core_classes", [&] { stage_classes(P); });
        if (cfg_.sustained_seconds > 0) stage(P, "sustained", [&] { stage_sustained(P); }, cfg_.sustained_seconds);

        derive(P);
        P.duration_s = seconds_since(t_start_);
        return P;
    }

private:
    const HardwareFingerprint& fp_;
    BenchConfig cfg_;
    std::chrono::steady_clock::time_point t_start_;
    int threads_ = 1;
    uint64_t working_cap_ = 0;
    bool stop_ = false;

    double elapsed() const { return detail::seconds_since(t_start_); }

    template <class F>
    void stage(CapabilityProfile& P, const char* name, F&& fn, double need_seconds = 0.5) {
        if (stop_) { P.skipped.push_back(std::string(name) + ": stopped"); return; }
        if (cfg_.should_stop && cfg_.should_stop()) {
            stop_ = true; P.stopped_early = true;
            P.skipped.push_back(std::string(name) + ": stop requested by governor");
            return;
        }
        if (elapsed() + need_seconds > cfg_.budget_seconds) {
            P.skipped.push_back(std::string(name) + ": out of time budget");
            return;
        }
        fn();
    }

    // sizes: 16 KB x4 steps up to the working-set cap
    std::vector<uint64_t> sweep_sizes() const {
        std::vector<uint64_t> v;
        for (uint64_t s = 16 << 10; s <= working_cap_; s *= 4) v.push_back(s);
        if (v.empty() || v.back() < working_cap_ / 2) v.push_back(working_cap_);
        return v;
    }

    void stage_fma(CapabilityProfile& P) {
        std::vector<float> a(kernels::kFmaLanes, 1.0f);
        std::vector<double> b(kernels::kFmaLanes, 1.0);
        P.fma_f32 = scale(detail::measure([&] { return kernels::fma_kernel<float>(400, a.data()); }, cfg_.window_seconds), 1e-9);
        P.fma_f64 = scale(detail::measure([&] { return kernels::fma_kernel<double>(400, b.data()); }, cfg_.window_seconds), 1e-9);
    }

    void stage_latency(CapabilityProfile& P) {
        for (uint64_t bytes : sweep_sizes()) {
            if (elapsed() > cfg_.budget_seconds * 0.8) { P.skipped.push_back("latency: remaining sizes skipped (budget)"); break; }
            kernels::ChaseTable ct(bytes);
            uint32_t pos = 0;
            const int64_t steps = std::max<int64_t>(2000, std::min<int64_t>(int64_t(ct.nodes) * 2, 200000));
            auto m = detail::measure([&] { pos = ct.walk(pos, steps); kernels::keep(&pos); return double(steps); }, cfg_.window_seconds);
            LevelPoint lp; lp.bytes = bytes; lp.ns = m.median > 0 ? 1e9 / m.median : 0;
            P.latency.push_back(lp);
        }
    }

    void stage_stream(CapabilityProfile& P) {
        for (uint64_t bytes : sweep_sizes()) {
            if (elapsed() > cfg_.budget_seconds * 0.8) { P.skipped.push_back("stream: remaining sizes skipped (budget)"); break; }
            kernels::StreamCase sc(bytes);
            auto m = detail::measure([&] { return sc.run(); }, cfg_.window_seconds);
            LevelPoint lp; lp.bytes = bytes; lp.gbps = m.median * 1e-9;
            P.stream_1t.push_back(lp);
        }
    }

    void stage_gemm(CapabilityProfile& P) {
        auto run = [&](const std::string& name, auto& gc) {
            if (elapsed() > cfg_.budget_seconds * 0.85) { P.skipped.push_back("gemm " + name + ": budget"); return; }
            P.gemm[name] = scale(detail::measure([&] { return gc.run(); }, cfg_.window_seconds), 1e-9);
        };
        { kernels::GemmCase<float> g(16, 16, 16);    run("f32_16", g); }       // dispatch-dominated regime
        { kernels::GemmCase<float> g(64, 64, 64);    run("f32_64", g); }       // L1/L2
        { kernels::GemmCase<float> g(256, 256, 256); run("f32_256", g); }      // L2/L3
        { kernels::GemmCase<float> g(8192, 32, 32);  run("f32_tall", g); }     // feature transform: many rows, few columns
        { kernels::GemmCase<double> g(256, 256, 256); run("f64_256", g); }
        if (working_cap_ >= (16u << 20)) { kernels::GemmCase<float> g(512, 512, 512); run("f32_512", g); }
    }

    void stage_gather(CapabilityProfile& P) {
        // feature matrix + output = 2 * n * 128 B must fit comfortably in the cap; cap n for runtime
        const uint64_t per_node = 2ull * kernels::GatherCase::F * sizeof(float) + 12ull * 4;
        int n = int(std::min<uint64_t>(200000, working_cap_ / per_node));
        n = std::max(n, 2000);
        kernels::GatherCase gc(n);
        auto m = detail::measure([&] { return gc.run(); }, cfg_.window_seconds);
        P.gather_edges_per_s = m;
        P.gather_gbps = m.median * kernels::GatherCase::F * sizeof(float) * 1e-9;
        P.gather_ws_bytes = uint64_t(n) * kernels::GatherCase::F * sizeof(float);
    }

    void stage_apps(CapabilityProfile& P) {
        { kernels::ConvCase c(16, 64, 64); P.conv_gflops = scale(detail::measure([&] { return c.run(); }, cfg_.window_seconds), 1e-9); }
        { kernels::MdCase md(1024);        P.md_ginteractions = scale(detail::measure([&] { return md.run(); }, cfg_.window_seconds), 1e-9); }
        {
            kernels::RnnCase r(256);
            P.rnn_steps_per_s = detail::measure([&] { return r.run(); }, cfg_.window_seconds);
            P.rnn_gflops = P.rnn_steps_per_s.median * r.flops_per_step() * 1e-9;
        }
    }

    void stage_overhead(CapabilityProfile& P) {
        {   // fork-join of an empty task over the full pool
            BenchPool pool(std::vector<int>(size_t(threads_), -1));
            std::function<void(int)> nop = [](int) {};
            for (int i = 0; i < 50; ++i) pool.run(nop);
            const int N = 1000;
            auto t0 = detail::Clock::now();
            for (int i = 0; i < N; ++i) pool.run(nop);
            P.fork_join_us = detail::seconds_since(t0) / N * 1e6;
        }
        {   // a tiny op through a type-erased call: what a graph executor pays per node
            std::vector<float> a(64, 1.f), b(64, 2.f), c(64);
            std::function<void()> op = [&] { for (int i = 0; i < 64; ++i) c[i] = a[i] + b[i]; kernels::keep(c.data()); };
            const int N = 200000;
            for (int i = 0; i < 1000; ++i) op();
            auto t0 = detail::Clock::now();
            for (int i = 0; i < N; ++i) op();
            P.call_overhead_ns = detail::seconds_since(t0) / N * 1e9;
        }
    }

    // per-thread compute and bandwidth state used by the parallel stages
    struct Worker {
        std::vector<float> a = std::vector<float>(kernels::kFmaLanes, 1.0f);
        std::unique_ptr<kernels::StreamCase> sc;
    };

    double parallel_gflops(BenchPool& pool, double seconds, std::vector<double>* per_thread = nullptr) {
        const int T = pool.size();
        std::vector<Worker> w(T);
        std::vector<double> rate(T, 0.0);
        std::function<void(int)> fn = [&](int t) {
            rate[t] = detail::timed_rate([&] { return kernels::fma_kernel<float>(200, w[t].a.data()); }, seconds);
        };
        pool.run(fn);
        double total = 0;
        for (double r : rate) total += r;
        if (per_thread) { per_thread->resize(T); for (int i = 0; i < T; ++i) (*per_thread)[i] = rate[i] * 1e-9; }
        return total * 1e-9;
    }

    double parallel_gbps(BenchPool& pool, double seconds) {
        const int T = pool.size();
        const uint64_t per = std::clamp<uint64_t>(working_cap_ / uint64_t(T), 4ull << 20, 64ull << 20);
        std::vector<Worker> w(T);
        for (auto& x : w) x.sc = std::make_unique<kernels::StreamCase>(per);
        std::vector<double> rate(T, 0.0);
        std::function<void(int)> fn = [&](int t) { rate[t] = detail::timed_rate([&] { return w[t].sc->run(); }, seconds); };
        pool.run(fn);
        double total = 0;
        for (double r : rate) total += r;
        return total * 1e-9;
    }

    void stage_scaling(CapabilityProfile& P) {
        const double sec = std::max(cfg_.window_seconds, 0.03);
        std::vector<int> counts;
        for (int t = 1; t < threads_; t *= 2) counts.push_back(t);
        counts.push_back(threads_);
        double g1 = 0, b1 = 0;
        for (int t : counts) {
            if (elapsed() > cfg_.budget_seconds * 0.85) { P.skipped.push_back("scaling: remaining thread counts skipped (budget)"); break; }
            BenchPool pool(std::vector<int>(size_t(t), -1));       // unpinned: what the OS scheduler gives a plain pool
            ScalePoint sp;
            sp.threads = t;
            parallel_gflops(pool, sec * 0.5);                      // ramp
            sp.gflops = parallel_gflops(pool, sec);
            sp.gbps = parallel_gbps(pool, sec);
            if (t == 1) { g1 = sp.gflops; b1 = sp.gbps; }
            sp.compute_eff = (g1 > 0) ? sp.gflops / (g1 * t) : 0;
            sp.bw_eff = (b1 > 0) ? sp.gbps / b1 : 0;
            P.scaling.push_back(sp);
        }
        // all usable cpus at once, one worker per cpu, fastest class first: per-thread rates for chunk weighting
        std::vector<int> cpus = cpu_order();
        if (static_cast<int>(cpus.size()) > threads_) cpus.resize(size_t(threads_));
        const bool pin = pinning_supported() && cpus.size() == size_t(threads_) && !cpus.empty();
        BenchPool pool(pin ? cpus : std::vector<int>(size_t(threads_), -1));
        ThreadRates& tr = P.all_cores;
        tr.pinned = pin && pool.all_pinned();
        if (tr.pinned) tr.cpus = cpus;
        parallel_gflops(pool, sec * 0.5);
        tr.total = parallel_gflops(pool, sec, &tr.gflops);
    }

    // online cpus, fastest class first
    std::vector<int> cpu_order() const {
        std::vector<int> out;
        for (auto it = fp_.cpu.classes.rbegin(); it != fp_.cpu.classes.rend(); ++it)
            for (int id : it->cpus) {
                bool online = true;
                for (const auto& c : fp_.cpu.cores) if (c.id == id) online = c.online;
                if (online) out.push_back(id);
            }
        return out;
    }

    void stage_classes(CapabilityProfile& P) {
        if (!cfg_.per_class) return;
        if (!pinning_supported()) { P.skipped.push_back("core_classes: pinning not supported on this platform"); return; }
        if (fp_.cpu.classes.size() < 2) { P.notes.push_back("single core class: per-class stage not needed"); return; }
        const double sec = std::max(cfg_.window_seconds, 0.03);
        for (const auto& cls : fp_.cpu.classes) {
            if (elapsed() > cfg_.budget_seconds * 0.9) { P.skipped.push_back("core_classes: remaining classes skipped (budget)"); break; }
            std::vector<int> cpus;
            for (int id : cls.cpus) for (const auto& c : fp_.cpu.cores) if (c.id == id && c.online) cpus.push_back(id);
            if (cpus.empty()) continue;
            ClassPerf cp; cp.index = cls.index; cp.name = cls.name; cp.cpus = cpus;
            {
                BenchPool one({cpus.front()});
                cp.pinned = one.all_pinned();
                if (!cp.pinned) { P.classes.push_back(cp); P.notes.push_back("could not pin to class " + cls.name + "; its numbers are omitted"); continue; }
                parallel_gflops(one, sec * 0.5);
                cp.gflops_1t = parallel_gflops(one, sec);
                cp.gbps_1t = parallel_gbps(one, sec);
            }
            {
                BenchPool all(cpus);
                parallel_gflops(all, sec * 0.5);
                cp.gflops_all = parallel_gflops(all, sec);
                cp.gbps_all = parallel_gbps(all, sec);
            }
            P.classes.push_back(cp);
        }
    }

    void stage_sustained(CapabilityProfile& P) {
        SustainedResult& S = P.sustained;
        const double remaining = cfg_.budget_seconds - elapsed();
        const double dur = std::min(cfg_.sustained_seconds, remaining);
        if (dur < 2 * cfg_.sustained_window_seconds) { P.skipped.push_back("sustained: not enough budget left"); return; }
        S.ran = true;
        S.threads = threads_;
        S.window_s = cfg_.sustained_window_seconds;
        std::optional<double> t0 = cfg_.read_temp_c ? cfg_.read_temp_c() : std::nullopt;
#if !defined(_WIN32) && !defined(__APPLE__)
        if (!cfg_.read_temp_c) { RealFs fs; t0 = detail::read_max_temp_c(fs); }
#endif
        S.temp_start_c = t0;
        // GEMM-128 per thread: compute heavy with a real working set, so it heats the chip like training does
        std::vector<std::unique_ptr<kernels::GemmCase<float>>> gcs;
        for (int i = 0; i < threads_; ++i) gcs.push_back(std::make_unique<kernels::GemmCase<float>>(128, 128, 128));
        BenchPool pool(std::vector<int>(size_t(threads_), -1));
        auto start = detail::Clock::now();
        while (detail::seconds_since(start) < dur) {
            if (cfg_.should_stop && cfg_.should_stop()) { P.stopped_early = true; stop_ = true; break; }
            std::vector<double> rate(size_t(threads_), 0.0);
            std::function<void(int)> fn = [&](int t) {
                rate[t] = detail::timed_rate([&] { return gcs[t]->run(); }, cfg_.sustained_window_seconds);
            };
            pool.run(fn);
            double total = 0;
            for (double r : rate) total += r;
            S.window_gflops.push_back(total * 1e-9);
        }
        S.duration_s = detail::seconds_since(start);
        std::optional<double> t1 = cfg_.read_temp_c ? cfg_.read_temp_c() : std::nullopt;
#if !defined(_WIN32) && !defined(__APPLE__)
        if (!cfg_.read_temp_c) { RealFs fs; t1 = detail::read_max_temp_c(fs); }
#endif
        S.temp_end_c = t1;
        const size_t n = S.window_gflops.size();
        if (n >= 3) {
            auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            const size_t third = std::max<size_t>(1, n / 3);
            S.peak_gflops = med({S.window_gflops.begin(), S.window_gflops.begin() + long(third)});
            S.sustained_gflops = med({S.window_gflops.end() - long(third), S.window_gflops.end()});
            S.fraction = S.peak_gflops > 0 ? S.sustained_gflops / S.peak_gflops : 1.0;
            S.throttled = S.fraction < 0.90;
        }
    }

    static Metric scale(Metric m, double k) { m.median *= k; m.best *= k; return m; }

    void derive(CapabilityProfile& P) {
        if (!P.latency.empty()) {
            P.l1_latency_ns = P.latency.front().ns;
            P.dram_latency_ns = P.latency.back().ns;
            P.dram_probe_bytes = P.latency.back().bytes;
        }
        if (!P.stream_1t.empty()) {
            P.dram_gbps_1t = P.stream_1t.back().gbps;
            if (P.dram_probe_bytes == 0) P.dram_probe_bytes = P.stream_1t.back().bytes;
        }
        uint64_t llc = 0;
        for (const auto& c : fp_.cpu.classes) for (const auto& k : c.caches) llc = std::max(llc, k.size_bytes);
        P.dram_probe_valid = llc > 0 ? P.dram_probe_bytes >= 4 * llc : false;
        if (!P.dram_probe_valid && !P.latency.empty())
            P.notes.push_back(llc > 0 ? "largest working set is under 4x the last-level cache: DRAM latency/bandwidth are optimistic (memory cap too small)"
                                      : "cache sizes unknown: cannot confirm the DRAM probe left the cache");
        if (!P.scaling.empty()) P.dram_gbps_all = P.scaling.back().gbps;
        if (P.dram_gbps_1t > 0) P.gather_efficiency = P.gather_gbps / P.dram_gbps_1t;
    }
};

inline CapabilityProfile run_device_benchmark(const HardwareFingerprint& fp, BenchConfig cfg = {}) {
    return DeviceBenchmark(fp, std::move(cfg)).run();
}

// ───────────────────────────── output ─────────────────────────────

namespace detail {
inline void write_metric(JsonOut& j, const char* k, const Metric& m) {
    j.key(k); j.begin_obj();
    j.kv("median", m.median); j.kv("best", m.best); j.kv("rsd", m.rsd); j.kv("windows", m.windows); j.kv("ok", m.ok);
    j.end_obj();
}
}  // namespace detail

inline std::string CapabilityProfile::to_json() const {
    detail::JsonOut j;
    j.begin_obj();
    j.kv("schema", schema); j.kv("device_id", device_id);
    j.key("build"); j.begin_obj(); j.kv("optimized", build.optimized); j.kv("simd", build.simd); j.end_obj();
    detail::write_metric(j, "fma_f32_gflops", fma_f32);
    detail::write_metric(j, "fma_f64_gflops", fma_f64);
    j.key("gemm_gflops"); j.begin_obj();
    for (const auto& kv : gemm) detail::write_metric(j, kv.first.c_str(), kv.second);
    j.end_obj();
    j.key("stream_1t"); j.begin_arr();
    for (const auto& p : stream_1t) { j.begin_obj(); j.kv("bytes", uint64_t(p.bytes)); j.kv("gbps", p.gbps); j.end_obj(); }
    j.end_arr();
    j.key("latency"); j.begin_arr();
    for (const auto& p : latency) { j.begin_obj(); j.kv("bytes", uint64_t(p.bytes)); j.kv("ns", p.ns); j.end_obj(); }
    j.end_arr();
    detail::write_metric(j, "gather_edges_per_s", gather_edges_per_s);
    j.kv("gather_gbps", gather_gbps); j.kv("gather_ws_bytes", uint64_t(gather_ws_bytes)); j.kv("gather_efficiency", gather_efficiency);
    detail::write_metric(j, "conv_gflops", conv_gflops);
    detail::write_metric(j, "md_ginteractions", md_ginteractions);
    detail::write_metric(j, "rnn_steps_per_s", rnn_steps_per_s);
    j.kv("rnn_gflops", rnn_gflops);
    j.kv("fork_join_us", fork_join_us); j.kv("call_overhead_ns", call_overhead_ns);
    j.key("scaling"); j.begin_arr();
    for (const auto& s : scaling) {
        j.begin_obj(); j.kv("threads", s.threads); j.kv("gflops", s.gflops); j.kv("gbps", s.gbps);
        j.kv("compute_eff", s.compute_eff); j.kv("bw_eff", s.bw_eff); j.end_obj();
    }
    j.end_arr();
    j.key("all_cores"); j.begin_obj();
    j.kv("pinned", all_cores.pinned); j.ints("cpus", all_cores.cpus); j.kv("total_gflops", all_cores.total);
    j.key("per_thread_gflops"); j.begin_arr(); for (double g : all_cores.gflops) j.num(g); j.end_arr();
    j.end_obj();
    j.key("classes"); j.begin_arr();
    for (const auto& c : classes) {
        j.begin_obj(); j.kv("index", c.index); j.kv("name", c.name); j.ints("cpus", c.cpus); j.kv("pinned", c.pinned);
        j.kv("gflops_1t", c.gflops_1t); j.kv("gflops_all", c.gflops_all); j.kv("gbps_1t", c.gbps_1t); j.kv("gbps_all", c.gbps_all);
        j.end_obj();
    }
    j.end_arr();
    j.key("sustained"); j.begin_obj();
    j.kv("ran", sustained.ran); j.kv("threads", sustained.threads); j.kv("window_s", sustained.window_s);
    j.kv("duration_s", sustained.duration_s); j.kv("peak_gflops", sustained.peak_gflops);
    j.kv("sustained_gflops", sustained.sustained_gflops); j.kv("fraction", sustained.fraction);
    j.kv("throttled", sustained.throttled);
    j.kv("temp_start_c", sustained.temp_start_c); j.kv("temp_end_c", sustained.temp_end_c);
    j.key("window_gflops"); j.begin_arr(); for (double g : sustained.window_gflops) j.num(g); j.end_arr();
    j.end_obj();
    j.key("derived"); j.begin_obj();
    j.kv("l1_latency_ns", l1_latency_ns); j.kv("dram_latency_ns", dram_latency_ns);
    j.kv("dram_gbps_1t", dram_gbps_1t); j.kv("dram_gbps_all", dram_gbps_all);
    j.kv("dram_probe_bytes", uint64_t(dram_probe_bytes)); j.kv("dram_probe_valid", dram_probe_valid);
    j.end_obj();
    j.kv("duration_s", duration_s); j.kv("stopped_early", stopped_early);
    j.key("skipped"); j.begin_arr(); for (const auto& s : skipped) j.str(s); j.end_arr();
    j.key("notes"); j.begin_arr(); for (const auto& s : notes) j.str(s); j.end_arr();
    j.end_obj();
    return j.str();
}

inline std::string CapabilityProfile::summary() const {
    char b[256];
    std::string o;
    auto line = [&](const char* fmt, auto... args) { std::snprintf(b, sizeof b, fmt, args...); o += b; o += '\n'; };
    line("build: %s, simd=%s", build.optimized ? "optimized" : "NOT optimized", build.simd.c_str());
    line("fma f32 %.1f GFLOP/s (1 thread), f64 %.1f", fma_f32.median, fma_f64.median);
    for (const auto& kv : gemm) line("gemm %-8s %.2f GFLOP/s (rsd %.2f)", kv.first.c_str(), kv.second.median, kv.second.rsd);
    for (size_t i = 0; i < latency.size(); ++i)
        line("working set %7llu KB: latency %7.2f ns, stream %7.2f GB/s", (unsigned long long)(latency[i].bytes >> 10), latency[i].ns,
             i < stream_1t.size() ? stream_1t[i].gbps : 0.0);
    line("gather %.1f M edges/s (%.2f GB/s, %.2f of stream)", gather_edges_per_s.median * 1e-6, gather_gbps, gather_efficiency);
    line("conv %.2f GFLOP/s, md %.3f Ginteractions/s, rnn %.0f steps/s (%.2f GFLOP/s)", conv_gflops.median, md_ginteractions.median,
         rnn_steps_per_s.median, rnn_gflops);
    line("fork-join %.1f us, tiny call %.1f ns", fork_join_us, call_overhead_ns);
    for (const auto& s : scaling) line("threads %2d: %.1f GFLOP/s (eff %.2f), %.1f GB/s", s.threads, s.gflops, s.compute_eff, s.gbps);
    for (const auto& c : classes) line("class %-7s pinned=%d 1t %.1f GFLOP/s, all %.1f GFLOP/s", c.name.c_str(), int(c.pinned), c.gflops_1t, c.gflops_all);
    if (sustained.ran) line("sustained: peak %.1f, last third %.1f GFLOP/s, fraction %.2f%s over %.1f s", sustained.peak_gflops,
                            sustained.sustained_gflops, sustained.fraction, sustained.throttled ? " (THROTTLED)" : "", sustained.duration_s);
    for (const auto& s : skipped) line("skipped: %s", s.c_str());
    for (const auto& s : notes) line("note: %s", s.c_str());
    line("total %.2f s", duration_s);
    return o;
}

}  // namespace tensorf::profile
