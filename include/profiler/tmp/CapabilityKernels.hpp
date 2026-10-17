#pragma once

/*
 * CapabilityKernels.hpp
 * =====================
 * The individual micro-kernels of the capability suite. Each one probes ONE
 * resource that some class of algorithm is limited by, and returns plain
 * numbers. Orchestration, time budgets and thermal stops live in
 * CapabilitySuite.hpp; this file only measures.
 *
 *   gemm            dense GEMM, f32 / f64 / int8, several shapes     (CNN, MLP, Transformer)
 *   stream_*        multi-threaded read + triad bandwidth            (everything memory bound)
 *   chase           single-cycle pointer chase, L1 -> DRAM           (latency, Markov, trees)
 *   gather/scatter  CSR SpMM on a power-law graph                    (GNN)
 *   conv3x3         direct 3x3 convolution                           (CNN, stencils)
 *   dependent_*     serial FMA chain + recurrent cell                (RNN)
 *   nbody           O(N^2) pair interactions                         (particles)
 *   overheads       thread spawn, pool barrier, tiny-op floor        (small-model regimes)
 *   thread_scaling  weak scaling per core class                      (OMP_NUM_THREADS advice)
 *   sync_gemm       barrier-paced GEMM: does mixing core classes hurt?
 *   sustained       fixed-length run reported in 1 s windows         (throttling)
 *
 * Honesty notes:
 *   - These are portable C++ kernels compiled with the caller's flags. They
 *     measure what portable, reasonably blocked code reaches on this device,
 *     not the vendor-BLAS peak. Step 2 of the design calibrates them against the
 *     real Matrix ops.
 *   - A Debug build or a build without -O2 gives meaningless numbers; the suite
 *     records the compile target for that reason.
 *   - Every kernel bounds its memory use (`max_bytes`) and returns ok=false
 *     instead of throwing.
 */

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdlib.h>   // posix_memalign
#include <cstring>
#include <functional>
#include <memory>
#include <type_traits>
#include <new>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include "ProfilerCommon.hpp"

namespace tensorf::prof::kernels {

// ─── Memory + threading helpers ──────────────────────────────────────────────

// std::aligned_alloc is missing below Android API 28 (libc++ marks it using_if_exists) and on MSVC.
// posix_memalign exists on every Android level and on Linux/macOS; Windows has _aligned_malloc.
inline void* aligned_alloc_portable(size_t alignment, size_t bytes) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, alignment);
#else
    void* p = nullptr;
    if (posix_memalign(&p, alignment, bytes) != 0) return nullptr;
    return p;
#endif
}
inline void aligned_free_portable(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

template <class T>
class AlignedBuf {
public:
    AlignedBuf() = default;
    explicit AlignedBuf(size_t n) { alloc(n); }
    AlignedBuf(const AlignedBuf&) = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
    AlignedBuf(AlignedBuf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    AlignedBuf& operator=(AlignedBuf&& o) noexcept { if (this != &o) { release(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; } return *this; }
    ~AlignedBuf() { release(); }

    bool alloc(size_t n) {
        release();
        if (n == 0) return true;
        const size_t bytes = ((n * sizeof(T) + 63) / 64) * 64;
        void* p = aligned_alloc_portable(64, bytes);
        if (!p) return false;
        p_ = static_cast<T*>(p); n_ = n;
        return true;
    }
    T*       data()       { return p_; }
    const T* data() const { return p_; }
    size_t   size() const { return n_; }
    T& operator[](size_t i)       { return p_[i]; }
    const T& operator[](size_t i) const { return p_[i]; }
    explicit operator bool() const { return p_ != nullptr; }
private:
    void release() { aligned_free_portable(p_); p_ = nullptr; n_ = 0; }
    T* p_ = nullptr; size_t n_ = 0;
};

// Deterministic xorshift64*: kernels must be reproducible run to run.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed = 0x9E3779B97F4A7C15ull) : s(seed ? seed : 1) {}
    uint64_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 0x2545F4914F6CDD1Dull; }
    double   uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }   // [0,1)
    float    uniformf(float lo, float hi) { return lo + (hi - lo) * (float)uniform(); }
};

inline bool pin_this_thread(int cpu) {
#if defined(__linux__)
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
#else
    (void)cpu; return false;
#endif
}

struct WindowResult {
    double units   = 0.0;   // total work units completed by all threads
    double seconds = 0.0;   // wall time of the slowest thread's window
    bool   ok      = true;
    double per_second() const { return seconds > 0 ? units / seconds : 0.0; }
};

// Run `nthreads` workers for ~window_s. Each builds its own state (first-touch
// allocation happens on the worker, which matters for NUMA and for honest cache
// behaviour), waits at a start gate, then loops body(state) -> units until the
// window ends. `cpus`, if non-empty, pins worker i to cpus[i % size].
//
// `collective_stop`: bodies that contain a barrier MUST pass this. With per-thread
// clocks one worker can leave its loop while a peer enters one more iteration and
// waits at the barrier for it forever. With a collective flag, the decision is made
// once per iteration at the barrier (SpinBarrier::wait(on_last)) and every worker
// reads the same value.
template <class Setup, class Body>
WindowResult run_window(int nthreads, const std::vector<int>& cpus, double window_s, Setup&& setup, Body&& body,
                        std::atomic<bool>* collective_stop = nullptr) {
    nthreads = std::max(1, nthreads);
    std::atomic<int>  ready{0};
    std::atomic<bool> go{false}, failed{false};
    std::vector<double> units((size_t)nthreads, 0.0), secs((size_t)nthreads, 0.0);
    std::vector<std::thread> pool;
    pool.reserve((size_t)nthreads);
    for (int t = 0; t < nthreads; ++t) {
        pool.emplace_back([&, t] {
            if (!cpus.empty()) pin_this_thread(cpus[(size_t)t % cpus.size()]);
            try {
                auto state = setup(t);
                ready.fetch_add(1);
                while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
                if (failed.load()) return;      // a peer's setup threw: never enter a barrier it will not reach
                const double t0 = now_s();
                double u = 0.0, el = 0.0;
                if (collective_stop) {
                    do { u += body(state); el = now_s() - t0; } while (!collective_stop->load(std::memory_order_acquire));
                } else {
                    do { u += body(state); el = now_s() - t0; } while (el < window_s);
                }
                units[(size_t)t] = u; secs[(size_t)t] = el;
            } catch (...) {
                failed.store(true);
                ready.fetch_add(1);
            }
        });
    }
    while (ready.load() < nthreads) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    for (auto& th : pool) th.join();

    WindowResult r;
    r.ok = !failed.load();
    for (int t = 0; t < nthreads; ++t) { r.units += units[(size_t)t]; r.seconds = std::max(r.seconds, secs[(size_t)t]); }
    return r;
}

// Median throughput over several windows (stops early when the deadline hits).
struct Measured { double value = 0.0; double spread = 0.0; int samples = 0; bool ok = false; };

template <class Setup, class Body>
Measured measure_rate(int nthreads, const std::vector<int>& cpus, double window_s, int samples,
                      const Deadline& dl, Setup&& setup, Body&& body) {
    std::vector<double> v;
    for (int i = 0; i < samples; ++i) {
        WindowResult w = run_window(nthreads, cpus, window_s, setup, body);
        if (!w.ok) return {};
        v.push_back(w.per_second());
        if (i >= 1 && dl.expired()) break;
    }
    Measured m; m.value = median(v); m.spread = rel_spread(v); m.samples = (int)v.size(); m.ok = m.value > 0;
    return m;
}

// ─── 1. GEMM ─────────────────────────────────────────────────────────────────

enum class Precision { F32, F64, I8 };
inline const char* to_string(Precision p) { return p == Precision::F32 ? "f32" : p == Precision::F64 ? "f64" : "i8"; }
inline size_t bytes_of(Precision p) { return p == Precision::F32 ? 4 : p == Precision::F64 ? 8 : 1; }

struct GemmShape { size_t m, n, k; const char* tag; };

// C[M,N] += A[M,K] * B[K,N], row-major. Blocked over j and k so the C slice stays
// in L1 and the B panel in L2; the inner j loop is unit stride and vectorises.
// Not a BLAS, but representative of cache-blocked portable code.
template <class TA, class TC>
inline void gemm_blocked(const TA* A, const TA* B, TC* C, size_t M, size_t N, size_t K) {
    constexpr size_t JB = 256, KB = 128;
    for (size_t jj = 0; jj < N; jj += JB) {
        const size_t jend = std::min(N, jj + JB);
        for (size_t kk = 0; kk < K; kk += KB) {
            const size_t kend = std::min(K, kk + KB);
            for (size_t i = 0; i < M; ++i) {
                TC* __restrict c = C + i * N;
                for (size_t k = kk; k < kend; ++k) {
                    const TC a = (TC)A[i * K + k];
                    const TA* __restrict b = B + k * N;
                    for (size_t j = jj; j < jend; ++j) c[j] += a * (TC)b[j];
                }
            }
        }
    }
}

// Reference implementation for the self-test.
template <class TA, class TC>
inline void gemm_naive(const TA* A, const TA* B, TC* C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            TC s = 0;
            for (size_t k = 0; k < K; ++k) s += (TC)A[i * K + k] * (TC)B[k * N + j];
            C[i * N + j] += s;
        }
}

struct GemmPoint {
    Precision prec = Precision::F32;
    GemmShape shape{0, 0, 0, ""};
    int       threads = 1;
    double    gops = 0.0;           // giga multiply-add pairs x2 per second (flop/s, or int8 op/s)
    double    spread = 0.0;
    double    arithmetic_intensity = 0.0;   // flop per byte of A+B+C touched once
    bool      ok = false;
};

namespace detail {
template <class TA, class TC>
struct GemmState {
    AlignedBuf<TA> A, B; AlignedBuf<TC> C; size_t M = 0, N = 0, K = 0;
};
template <class TA, class TC>
GemmState<TA, TC> make_gemm_state(size_t M, size_t N, size_t K, uint64_t seed) {
    GemmState<TA, TC> s; s.M = M; s.N = N; s.K = K;
    if (!s.A.alloc(M * K) || !s.B.alloc(K * N) || !s.C.alloc(M * N)) throw std::bad_alloc();
    Rng r(seed);
    for (size_t i = 0; i < M * K; ++i) { if constexpr (std::is_integral_v<TA>) s.A[i] = (TA)((int)(r.next() % 7) - 3); else s.A[i] = (TA)r.uniformf(-1.f, 1.f); }
    for (size_t i = 0; i < K * N; ++i) { if constexpr (std::is_integral_v<TA>) s.B[i] = (TA)((int)(r.next() % 7) - 3); else s.B[i] = (TA)r.uniformf(-1.f, 1.f); }
    std::memset(s.C.data(), 0, M * N * sizeof(TC));
    return s;
}
}  // namespace detail

template <class TA, class TC>
inline GemmPoint gemm_typed(Precision p, GemmShape sh, int threads, const std::vector<int>& cpus,
                            double window_s, int samples, const Deadline& dl, size_t max_bytes) {
    GemmPoint pt; pt.prec = p; pt.shape = sh; pt.threads = threads;
    const size_t per_thread = sh.m * sh.k * sizeof(TA) + sh.k * sh.n * sizeof(TA) + sh.m * sh.n * sizeof(TC);
    if (per_thread * (size_t)threads > max_bytes) return pt;      // would not fit the budget
    const double work = 2.0 * (double)sh.m * (double)sh.n * (double)sh.k;
    Measured m = measure_rate(threads, cpus, window_s, samples, dl,
        [&](int t) { return detail::make_gemm_state<TA, TC>(sh.m, sh.n, sh.k, 1234u + (uint64_t)t); },
        [&](detail::GemmState<TA, TC>& s) {
            gemm_blocked<TA, TC>(s.A.data(), s.B.data(), s.C.data(), s.M, s.N, s.K);
            do_not_optimize(s.C[0]);
            return work;
        });
    pt.ok = m.ok; pt.gops = m.value / 1e9; pt.spread = m.spread;
    pt.arithmetic_intensity = work / (double)per_thread;
    return pt;
}

inline GemmPoint gemm(Precision p, GemmShape sh, int threads, const std::vector<int>& cpus,
                      double window_s, int samples, const Deadline& dl, size_t max_bytes) {
    switch (p) {
        case Precision::F32: return gemm_typed<float,  float  >(p, sh, threads, cpus, window_s, samples, dl, max_bytes);
        case Precision::F64: return gemm_typed<double, double >(p, sh, threads, cpus, window_s, samples, dl, max_bytes);
        case Precision::I8:  return gemm_typed<int8_t, int32_t>(p, sh, threads, cpus, window_s, samples, dl, max_bytes);
    }
    return {};
}

// ─── 2. Bandwidth ────────────────────────────────────────────────────────────

// Sum of a buffer using 4 independent integer accumulators: pure load throughput,
// no float-reduction ordering problem, no dependence on -ffast-math.
inline uint64_t read_pass(const uint64_t* p, size_t n) {
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) { a0 += p[i]; a1 += p[i + 1]; a2 += p[i + 2]; a3 += p[i + 3]; }
    for (; i < n; ++i) a0 += p[i];
    return a0 + a1 + a2 + a3;
}

inline void triad_pass(float* __restrict a, const float* __restrict b, const float* __restrict c, float s, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = b[i] + s * c[i];
}

struct BandwidthPoint {
    std::string level;          // "L1" | "L2" | "L3" | "DRAM"
    size_t      bytes_per_thread = 0;
    int         threads = 1;
    double      read_gbs = 0.0;
    bool        ok = false;
};

// Each thread repeatedly reads its own private working set of `bytes`.
inline BandwidthPoint read_bandwidth(const std::string& level, size_t bytes, int threads, const std::vector<int>& cpus,
                                     double window_s, int samples, const Deadline& dl, size_t max_bytes) {
    BandwidthPoint bp; bp.level = level; bp.bytes_per_thread = bytes; bp.threads = threads;
    if (bytes * (size_t)threads > max_bytes) return bp;
    const size_t n = bytes / sizeof(uint64_t);
    Measured m = measure_rate(threads, cpus, window_s, samples, dl,
        [&](int t) {
            AlignedBuf<uint64_t> b(n);
            if (!b) throw std::bad_alloc();
            for (size_t i = 0; i < n; ++i) b[i] = i * 0x9E3779B97F4A7C15ull + (uint64_t)t;
            return b;
        },
        [&](AlignedBuf<uint64_t>& b) {
            // Several passes per body call so the clock read is amortised for tiny buffers.
            const size_t passes = std::max<size_t>(1, (4u << 20) / std::max<size_t>(bytes, 1));
            uint64_t acc = 0;
            for (size_t r = 0; r < passes; ++r) acc += read_pass(b.data(), n);
            do_not_optimize(acc);
            return (double)(passes * n * sizeof(uint64_t));
        });
    bp.ok = m.ok; bp.read_gbs = m.value / 1e9;
    return bp;
}

// STREAM triad: a = b + s*c over three arrays, 12 bytes of traffic per element.
inline Measured triad_bandwidth(size_t bytes_per_array_per_thread, int threads, const std::vector<int>& cpus,
                                double window_s, int samples, const Deadline& dl, size_t max_bytes) {
    if (3 * bytes_per_array_per_thread * (size_t)threads > max_bytes) return {};
    const size_t n = bytes_per_array_per_thread / sizeof(float);
    struct S { AlignedBuf<float> a, b, c; };
    Measured m = measure_rate(threads, cpus, window_s, samples, dl,
        [&](int t) {
            S s;
            if (!s.a.alloc(n) || !s.b.alloc(n) || !s.c.alloc(n)) throw std::bad_alloc();
            for (size_t i = 0; i < n; ++i) { s.a[i] = 0.f; s.b[i] = (float)((i + (size_t)t) & 255); s.c[i] = 1.f; }
            return s;
        },
        [&](S& s) { triad_pass(s.a.data(), s.b.data(), s.c.data(), 1.0001f, n); do_not_optimize(s.a[0]); return (double)(3 * n * sizeof(float)); });
    m.value /= 1e9;
    return m;
}

// ─── 3. Pointer chase (latency) ──────────────────────────────────────────────

struct LatencyPoint { size_t bytes = 0; double ns = 0.0; bool ok = false; };

// next[i] = the slot visited after slot i. Slots are linked in a random ORDER, so
// the result is one cycle through all `n` slots by construction (every slot is
// visited before any repeats). The old benchmark followed arr[idx] where arr was a
// shuffled array; that is a random permutation FUNCTION, which splits into many
// short cycles and can leave the chase inside a tiny subset of the buffer.
inline std::vector<uint32_t> make_chase_next(size_t n, uint64_t seed = 0xC0FFEEull) {
    std::vector<uint32_t> order(n), next(n);
    std::iota(order.begin(), order.end(), 0u);
    Rng rng(seed);
    for (size_t i = n - 1; i > 0; --i) std::swap(order[i], order[(size_t)(rng.next() % (i + 1))]);   // Fisher-Yates
    for (size_t i = 0; i < n; ++i) next[order[i]] = order[(i + 1) % n];
    return next;
}

inline LatencyPoint chase_latency(size_t bytes, const Deadline& dl, size_t max_bytes) {
    LatencyPoint lp; lp.bytes = bytes;
    constexpr size_t SLOT = 64 / sizeof(uint32_t);       // one index per 64 B line
    const size_t lines = std::max<size_t>(bytes / 64, 8);
    if (lines * 64 > max_bytes) return lp;
    AlignedBuf<uint32_t> buf(lines * SLOT);
    if (!buf) return lp;

    const std::vector<uint32_t> next = make_chase_next(lines);
    for (size_t i = 0; i < lines; ++i) buf[i * SLOT] = next[i];

    uint32_t idx = 0;
    // Warm the whole cycle once (also faults the pages in).
    for (size_t i = 0; i < lines; ++i) idx = buf[(size_t)idx * SLOT];

    std::vector<double> ns;
    const size_t steps = std::max<size_t>(1u << 16, std::min<size_t>(lines * 4, 1u << 22));
    for (int rep = 0; rep < 5; ++rep) {
        const double t0 = now_s();
        for (size_t i = 0; i < steps; ++i) idx = buf[(size_t)idx * SLOT];   // each load depends on the last
        const double s = now_s() - t0;
        do_not_optimize(idx);
        ns.push_back(s / (double)steps * 1e9);
        if (rep >= 1 && dl.expired()) break;
    }
    lp.ns = median(ns); lp.ok = lp.ns > 0;
    return lp;
}

// ─── 4. Sparse gather / scatter (GNN) ────────────────────────────────────────

struct SparseResult {
    size_t nodes = 0, edges = 0, feat = 0;
    double gather_gbs = 0.0;      // SpMM aggregate: out[i] += sum feat[col[e]]
    double scatter_gbs = 0.0;     // out[col[e]] += feat[i]
    double edges_per_s = 0.0;     // gather edges per second
    bool   ok = false;
};

// CSR graph with a heavy-tailed degree distribution (deg ~ rank^-0.7) and hub-
// biased targets (low node ids are hit most often), like citation/social graphs.
struct CsrGraph {
    std::vector<uint32_t> rowptr, col;
    size_t nodes = 0;
};

inline CsrGraph make_power_law_graph(size_t nodes, double avg_degree, uint64_t seed = 7) {
    CsrGraph g; g.nodes = nodes;
    std::vector<double> w(nodes);
    double sum = 0;
    for (size_t i = 0; i < nodes; ++i) { w[i] = std::pow((double)(i + 1), -0.7); sum += w[i]; }
    const double scale = avg_degree * (double)nodes / sum;
    g.rowptr.assign(nodes + 1, 0);
    for (size_t i = 0; i < nodes; ++i) g.rowptr[i + 1] = g.rowptr[i] + (uint32_t)std::max(1.0, std::floor(w[i] * scale));
    g.col.resize(g.rowptr[nodes]);
    Rng r(seed);
    // Node ids are shuffled in rank order so hubs are not contiguous in memory.
    std::vector<uint32_t> id(nodes); std::iota(id.begin(), id.end(), 0u);
    for (size_t i = nodes - 1; i > 0; --i) std::swap(id[i], id[(size_t)(r.next() % (i + 1))]);
    for (size_t e = 0; e < g.col.size(); ++e) {
        const double u = r.uniform();
        g.col[e] = id[(size_t)(u * u * u * (double)nodes) % nodes];     // hub-biased target
    }
    return g;
}

inline SparseResult sparse_gather_scatter(size_t nodes, size_t feat, double avg_degree, const Deadline& dl, size_t max_bytes) {
    SparseResult r; r.nodes = nodes; r.feat = feat;
    const size_t feat_bytes = nodes * feat * sizeof(float);
    if (2 * feat_bytes + nodes * 16 * (size_t)avg_degree > max_bytes) return r;
    CsrGraph g = make_power_law_graph(nodes, avg_degree);
    r.edges = g.col.size();
    AlignedBuf<float> X(nodes * feat), Y(nodes * feat);
    if (!X || !Y) return r;
    Rng rng(99);
    for (size_t i = 0; i < nodes * feat; ++i) X[i] = rng.uniformf(-1.f, 1.f);

    auto gather_pass = [&] {
        std::memset(Y.data(), 0, feat_bytes);
        for (size_t i = 0; i < nodes; ++i) {
            float* __restrict y = Y.data() + i * feat;
            for (uint32_t e = g.rowptr[i]; e < g.rowptr[i + 1]; ++e) {
                const float* __restrict x = X.data() + (size_t)g.col[e] * feat;
                for (size_t f = 0; f < feat; ++f) y[f] += x[f];
            }
        }
        do_not_optimize(Y[0]);
    };
    auto scatter_pass = [&] {
        std::memset(Y.data(), 0, feat_bytes);
        for (size_t i = 0; i < nodes; ++i) {
            const float* __restrict x = X.data() + i * feat;
            for (uint32_t e = g.rowptr[i]; e < g.rowptr[i + 1]; ++e) {
                float* __restrict y = Y.data() + (size_t)g.col[e] * feat;
                for (size_t f = 0; f < feat; ++f) y[f] += x[f];
            }
        }
        do_not_optimize(Y[0]);
    };
    // Bytes counted: one feature row read + one row accumulated per edge.
    const double bytes_per_pass = (double)r.edges * (double)feat * sizeof(float) * 2.0;
    auto time_pass = [&](auto&& pass) {
        std::vector<double> s;
        pass();   // warm
        for (int i = 0; i < 5; ++i) { const double t0 = now_s(); pass(); s.push_back(now_s() - t0); if (i >= 1 && dl.expired()) break; }
        return median(s);
    };
    const double tg = time_pass(gather_pass), ts = time_pass(scatter_pass);
    r.gather_gbs  = bytes_per_pass / tg / 1e9;
    r.scatter_gbs = bytes_per_pass / ts / 1e9;
    r.edges_per_s = (double)r.edges / tg;
    r.ok = tg > 0 && ts > 0;
    return r;
}

// ─── 5. Conv / stencil ───────────────────────────────────────────────────────

struct ConvResult { double gflops = 0.0; double spread = 0.0; int channels = 0, hw = 0; bool ok = false; };

// Direct 3x3, stride 1, same-size output (input is pre-padded). NCHW, no batch.
inline ConvResult conv3x3(int C, int HW, int threads, const std::vector<int>& cpus, double window_s, int samples,
                          const Deadline& dl, size_t max_bytes) {
    ConvResult cr; cr.channels = C; cr.hw = HW;
    const size_t P = (size_t)HW + 2;
    const size_t in_n = (size_t)C * P * P, out_n = (size_t)C * HW * HW, w_n = (size_t)C * C * 9;
    if ((in_n + out_n + w_n) * sizeof(float) * (size_t)threads > max_bytes) return cr;
    struct S { AlignedBuf<float> in, out, w; };
    const double work = 2.0 * 9.0 * (double)C * (double)C * (double)HW * (double)HW;
    Measured m = measure_rate(threads, cpus, window_s, samples, dl,
        [&](int t) {
            S s; if (!s.in.alloc(in_n) || !s.out.alloc(out_n) || !s.w.alloc(w_n)) throw std::bad_alloc();
            Rng r(5u + (uint64_t)t);
            for (size_t i = 0; i < in_n; ++i) s.in[i] = r.uniformf(-1.f, 1.f);
            for (size_t i = 0; i < w_n; ++i)  s.w[i]  = r.uniformf(-.1f, .1f);
            return s;
        },
        [&](S& s) {
            std::memset(s.out.data(), 0, out_n * sizeof(float));
            for (int oc = 0; oc < C; ++oc) {
                float* __restrict o = s.out.data() + (size_t)oc * HW * HW;
                for (int ic = 0; ic < C; ++ic) {
                    const float* wk = s.w.data() + ((size_t)oc * C + (size_t)ic) * 9;
                    const float* in = s.in.data() + (size_t)ic * P * P;
                    for (int ky = 0; ky < 3; ++ky)
                        for (int kx = 0; kx < 3; ++kx) {
                            const float wv = wk[ky * 3 + kx];
                            for (int y = 0; y < HW; ++y) {
                                const float* __restrict ir = in + (size_t)(y + ky) * P + (size_t)kx;
                                float* __restrict orow = o + (size_t)y * HW;
                                for (int x = 0; x < HW; ++x) orow[x] += wv * ir[x];
                            }
                        }
                }
            }
            do_not_optimize(s.out[0]);
            return work;
        });
    cr.ok = m.ok; cr.gflops = m.value / 1e9; cr.spread = m.spread;
    return cr;
}

// ─── 6. Dependent chains (RNN, Markov) ───────────────────────────────────────

struct DependentResult {
    double fma_latency_ns = 0.0;       // one dependent multiply-add
    double rnn_step_us = 0.0;          // one H x H recurrent step, strictly sequential
    double rnn_steps_per_s = 0.0;
    int    hidden = 0;
    bool   ok = false;
};

inline DependentResult dependent_chains(int hidden, const Deadline& dl) {
    DependentResult r; r.hidden = hidden;
    // (a) pure latency: x = x*a + b, N times, each depends on the previous.
    {
        volatile float va = 0.9999f, vb = 0.0001f;    // volatile: the compiler cannot fold the chain
        const float a = va, b = vb;
        const size_t N = 1u << 22;
        std::vector<double> ns;
        for (int rep = 0; rep < 5; ++rep) {
            float x = 1.0f;
            const double t0 = now_s();
            for (size_t i = 0; i < N; ++i) x = x * a + b;
            const double s = now_s() - t0;
            do_not_optimize(x);
            ns.push_back(s / (double)N * 1e9);
            if (rep >= 1 && dl.expired()) break;
        }
        r.fma_latency_ns = median(ns);
    }
    // (b) recurrent cell h' = softsign(W h + x), steps must run one after another.
    {
        const size_t H = (size_t)hidden;
        AlignedBuf<float> W(H * H), h(H), nh(H);
        if (!W || !h || !nh) return r;
        Rng rng(11);
        for (size_t i = 0; i < H * H; ++i) W[i] = rng.uniformf(-1.f, 1.f) / (float)H;
        for (size_t i = 0; i < H; ++i) h[i] = rng.uniformf(-1.f, 1.f);
        std::vector<double> us;
        const int steps = 2000;
        for (int rep = 0; rep < 5; ++rep) {
            const double t0 = now_s();
            for (int s = 0; s < steps; ++s) {
                for (size_t i = 0; i < H; ++i) {
                    const float* __restrict w = W.data() + i * H;
                    float acc = 0.f;
                    for (size_t j = 0; j < H; ++j) acc += w[j] * h[j];
                    nh[i] = acc / (1.0f + std::fabs(acc)) + 0.01f;
                }
                std::swap_ranges(h.data(), h.data() + H, nh.data());
            }
            const double t = now_s() - t0;
            do_not_optimize(h[0]);
            us.push_back(t / steps * 1e6);
            if (rep >= 1 && dl.expired()) break;
        }
        r.rnn_step_us = median(us);
        r.rnn_steps_per_s = r.rnn_step_us > 0 ? 1e6 / r.rnn_step_us : 0.0;
    }
    r.ok = r.fma_latency_ns > 0 && r.rnn_step_us > 0;
    return r;
}

// ─── 7. N-body ───────────────────────────────────────────────────────────────

struct NBodyResult { double interactions_per_s = 0.0; double gflops_est = 0.0; size_t n = 0; bool ok = false; };

inline NBodyResult nbody(size_t n, int threads, const std::vector<int>& cpus, double window_s, int samples,
                         const Deadline& dl, size_t max_bytes) {
    NBodyResult nr; nr.n = n;
    if (n * 7 * sizeof(float) * (size_t)threads > max_bytes) return nr;
    struct S { AlignedBuf<float> x, y, z, m, ax, ay, az; };
    const double pairs = (double)n * (double)n;
    Measured m = measure_rate(threads, cpus, window_s, samples, dl,
        [&](int t) {
            S s;
            if (!s.x.alloc(n) || !s.y.alloc(n) || !s.z.alloc(n) || !s.m.alloc(n) || !s.ax.alloc(n) || !s.ay.alloc(n) || !s.az.alloc(n))
                throw std::bad_alloc();
            Rng r(21u + (uint64_t)t);
            for (size_t i = 0; i < n; ++i) { s.x[i] = r.uniformf(-1, 1); s.y[i] = r.uniformf(-1, 1); s.z[i] = r.uniformf(-1, 1); s.m[i] = r.uniformf(.5f, 1.5f); }
            return s;
        },
        [&](S& s) {
            constexpr float eps2 = 1e-3f;
            for (size_t i = 0; i < n; ++i) {
                const float xi = s.x[i], yi = s.y[i], zi = s.z[i];
                float fx = 0, fy = 0, fz = 0;
                for (size_t j = 0; j < n; ++j) {
                    const float dx = s.x[j] - xi, dy = s.y[j] - yi, dz = s.z[j] - zi;
                    const float d2 = dx * dx + dy * dy + dz * dz + eps2;
                    const float inv = 1.0f / std::sqrt(d2);
                    const float f = s.m[j] * inv * inv * inv;
                    fx += f * dx; fy += f * dy; fz += f * dz;
                }
                s.ax[i] = fx; s.ay[i] = fy; s.az[i] = fz;
            }
            do_not_optimize(s.ax[0]);
            return pairs;
        });
    nr.ok = m.ok; nr.interactions_per_s = m.value;
    nr.gflops_est = m.value * 20.0 / 1e9;       // ~20 flops per pair (sqrt and div counted as one each)
    return nr;
}

// ─── 8. Overheads ────────────────────────────────────────────────────────────

struct OverheadResult {
    double thread_spawn_us = 0.0;       // create + join one std::thread doing nothing
    double barrier_roundtrip_us = 0.0;  // fork-join through a persistent spinning pool (0 if < 2 CPUs)
    double tiny_op_ns = 0.0;            // 64-float add through a non-inlined call
    int    pool_threads = 0;
    bool   ok = false;
};

#if defined(__GNUC__) || defined(__clang__)
#define TENSORF_NOINLINE __attribute__((noinline))
#else
#define TENSORF_NOINLINE
#endif

TENSORF_NOINLINE inline void tiny_add64(float* __restrict o, const float* __restrict a, const float* __restrict b) {
    for (int i = 0; i < 64; ++i) o[i] = a[i] + b[i];
}

inline OverheadResult overheads(int max_pool_threads, const Deadline& dl) {
    OverheadResult r;
    // thread spawn + join
    {
        std::vector<double> us;
        for (int rep = 0; rep < 50; ++rep) {
            const double t0 = now_s();
            std::thread t([] {}); t.join();
            us.push_back((now_s() - t0) * 1e6);
            if (rep >= 10 && dl.expired()) break;
        }
        r.thread_spawn_us = median(us);
    }
    // persistent-pool fork/join: main publishes a generation, workers ack it.
    if (max_pool_threads >= 2) {
        const int workers = max_pool_threads - 1;
        r.pool_threads = max_pool_threads;
        std::atomic<uint64_t> gen{0};
        std::atomic<int> acks{0};
        std::atomic<bool> stop{false};
        std::vector<std::thread> pool;
        for (int w = 0; w < workers; ++w)
            pool.emplace_back([&] {
                uint64_t seen = 0;
                while (!stop.load(std::memory_order_acquire)) {
                    const uint64_t g = gen.load(std::memory_order_acquire);
                    if (g != seen) { seen = g; acks.fetch_add(1, std::memory_order_acq_rel); }
                    else std::this_thread::yield();
                }
            });
        std::vector<double> us;
        for (int rep = 0; rep < 2000; ++rep) {
            acks.store(0, std::memory_order_release);
            const double t0 = now_s();
            gen.fetch_add(1, std::memory_order_acq_rel);
            while (acks.load(std::memory_order_acquire) < workers) std::this_thread::yield();
            us.push_back((now_s() - t0) * 1e6);
            if (rep >= 200 && dl.expired()) break;
        }
        stop.store(true, std::memory_order_release);
        for (auto& t : pool) t.join();
        r.barrier_roundtrip_us = median(us);
    }
    // tiny op floor
    {
        alignas(64) float a[64], b[64], o[64];
        for (int i = 0; i < 64; ++i) { a[i] = (float)i; b[i] = 1.f; }
        const size_t N = 1u << 20;
        std::vector<double> ns;
        for (int rep = 0; rep < 5; ++rep) {
            const double t0 = now_s();
            for (size_t i = 0; i < N; ++i) { tiny_add64(o, a, b); clobber_memory(); }
            ns.push_back((now_s() - t0) / (double)N * 1e9);
            if (rep >= 1 && dl.expired()) break;
        }
        do_not_optimize(o[0]);
        r.tiny_op_ns = median(ns);
    }
    r.ok = r.thread_spawn_us > 0 && r.tiny_op_ns > 0;
    return r;
}

// ─── 9. Thread scaling ───────────────────────────────────────────────────────

struct ScalingPoint { int threads = 0; double gflops = 0.0; double efficiency = 0.0; };

struct ScalingResult {
    std::vector<ScalingPoint> points;
    int best_threads = 1;          // smallest count within `tolerance` of the best throughput
    int peak_threads = 1;          // count with the highest throughput
    bool ok = false;
};

// Weak scaling of independent f32 GEMMs, one per thread, pinned inside `cpus`.
// efficiency = gflops / (threads * single-thread gflops). Never claims more cores
// help than the data shows: best_threads is the SMALLEST count within 5% of peak.
inline ScalingResult thread_scaling(const std::vector<int>& cpus, int max_threads, size_t n, double window_s,
                                    int samples, const Deadline& dl, size_t max_bytes, double tolerance = 0.05) {
    ScalingResult sr;
    const GemmShape sh{n, n, n, "scaling"};
    double base = 0.0;
    for (int t = 1; t <= max_threads; ++t) {
        if (t > 1 && dl.expired()) break;
        GemmPoint p = gemm(Precision::F32, sh, t, cpus, window_s, samples, dl, max_bytes);
        if (!p.ok) break;
        if (t == 1) base = p.gops;
        ScalingPoint sp; sp.threads = t; sp.gflops = p.gops;
        sp.efficiency = base > 0 ? p.gops / ((double)t * base) : 0.0;
        sr.points.push_back(sp);
    }
    if (sr.points.empty()) return sr;
    double best = 0; for (const auto& p : sr.points) best = std::max(best, p.gflops);
    for (const auto& p : sr.points) { if (p.gflops >= best * (1.0 - tolerance)) { sr.best_threads = p.threads; break; } }
    for (const auto& p : sr.points) if (p.gflops >= best) { sr.peak_threads = p.threads; break; }
    sr.ok = true;
    return sr;
}

// ─── 9b. Barrier-synchronised GEMM (the slowest core sets the pace) ──────────

// Sense-reversing spin barrier. Spins briefly, then yields, so it stays usable on
// an oversubscribed machine.
class SpinBarrier {
public:
    explicit SpinBarrier(int n) : n_(n) {}
    void wait() { wait([] {}); }
    // `on_last` runs on the last thread to arrive, BEFORE the others are released, so
    // anything it writes is visible to every thread when wait() returns.
    template <class F>
    void wait(F&& on_last) {
        const uint64_t gen = gen_.load(std::memory_order_acquire);
        if (count_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
            on_last();
            count_.store(0, std::memory_order_release);
            gen_.fetch_add(1, std::memory_order_acq_rel);
        } else {
            int spins = 0;
            while (gen_.load(std::memory_order_acquire) == gen) { if (++spins > 200) std::this_thread::yield(); }
        }
    }
private:
    const int n_;
    std::atomic<int> count_{0};
    std::atomic<uint64_t> gen_{0};
};

// ONE n x n x n GEMM split by rows over `threads`, with a barrier after every
// iteration, like a fork-join parallel_for. Throughput is therefore paced by the
// slowest participating core: mixing efficiency cores into the team shows up here
// as a LOWER number than the performance cluster alone, which independent
// (weak-scaling) GEMMs cannot reveal.
inline Measured sync_gemm(const std::vector<int>& cpus, int threads, size_t n, double window_s, int samples,
                          const Deadline& dl, size_t max_bytes) {
    threads = std::max(1, threads);
    const size_t rows = std::max<size_t>(1, n / (size_t)threads);
    const size_t per_thread = (rows * n * 2 + n * n) * sizeof(float);
    if (per_thread * (size_t)threads > max_bytes) return {};
    const double work_per_thread = 2.0 * (double)rows * (double)n * (double)n;
    std::vector<double> v;
    for (int i = 0; i < samples; ++i) {
        SpinBarrier bar(threads);
        std::atomic<bool> stop{false};
        double win_start = -1.0;                   // touched only by the last arriver, ordered by the barrier
        struct S { AlignedBuf<float> A, B, C; };
        WindowResult w = run_window(threads, cpus, window_s,
            [&](int t) {
                S s; if (!s.A.alloc(rows * n) || !s.B.alloc(n * n) || !s.C.alloc(rows * n)) throw std::bad_alloc();
                Rng r(77u + (uint64_t)t);
                for (size_t k = 0; k < rows * n; ++k) s.A[k] = r.uniformf(-1.f, 1.f);
                for (size_t k = 0; k < n * n; ++k)    s.B[k] = r.uniformf(-1.f, 1.f);
                std::memset(s.C.data(), 0, rows * n * sizeof(float));
                return s;
            },
            [&](S& s) {
                gemm_blocked<float, float>(s.A.data(), s.B.data(), s.C.data(), rows, n, n);
                do_not_optimize(s.C[0]);
                bar.wait([&] {
                    const double t = now_s();
                    if (win_start < 0) win_start = t;
                    if (t - win_start >= window_s) stop.store(true, std::memory_order_release);
                });
                return work_per_thread;
            }, &stop);
        if (!w.ok) return {};
        v.push_back(w.per_second());
        if (i >= 1 && dl.expired()) break;
    }
    Measured m; m.value = median(v) / 1e9; m.spread = rel_spread(v); m.samples = (int)v.size(); m.ok = m.value > 0;
    return m;
}

// ─── 10. Sustained run ───────────────────────────────────────────────────────

struct SustainedResult {
    std::vector<double> window_gflops;   // one per window_s
    double first_gflops = 0.0;           // median of the first windows
    double last_gflops  = 0.0;           // median of the last third
    double fraction     = 1.0;           // last / first (1.0 = no throttling seen)
    double seconds_run  = 0.0;
    std::string stopped_reason;          // "" = ran to completion
    bool ok = false;
};

// Runs f32 GEMM for up to `duration_s`, reports each `window_s` window, and asks
// `should_stop(elapsed_s)` between windows (return a non-empty string to stop,
// e.g. thermal limit reached). With fewer than ~4 windows no fraction is claimed.
inline SustainedResult sustained_gemm(int threads, const std::vector<int>& cpus, size_t n, double duration_s,
                                      double window_s, size_t max_bytes,
                                      const std::function<std::string(double)>& should_stop) {
    SustainedResult sr;
    const GemmShape sh{n, n, n, "sustained"};
    const double t0 = now_s();
    while (now_s() - t0 < duration_s) {
        if (should_stop) {
            std::string why = should_stop(now_s() - t0);
            if (!why.empty()) { sr.stopped_reason = why; break; }
        }
        Deadline one(window_s * 3);
        GemmPoint p = gemm(Precision::F32, sh, threads, cpus, window_s, 1, one, max_bytes);
        if (!p.ok) break;
        sr.window_gflops.push_back(p.gops);
    }
    sr.seconds_run = now_s() - t0;
    const size_t w = sr.window_gflops.size();
    if (w >= 4) {
        const size_t head = std::max<size_t>(1, w / 4), tail = std::max<size_t>(1, w / 3);
        sr.first_gflops = median(std::vector<double>(sr.window_gflops.begin(), sr.window_gflops.begin() + (long)head));
        sr.last_gflops  = median(std::vector<double>(sr.window_gflops.end() - (long)tail, sr.window_gflops.end()));
        sr.fraction     = sr.first_gflops > 0 ? sr.last_gflops / sr.first_gflops : 1.0;
        sr.ok = true;
    } else if (w > 0) {
        sr.first_gflops = sr.last_gflops = median(sr.window_gflops);
        sr.ok = false;    // too short to say anything about throttling
    }
    return sr;
}

}  // namespace tensorf::prof::kernels
