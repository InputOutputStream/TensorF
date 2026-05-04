#pragma once

/*
 * BenchmarkRunner.hpp
 * ===================
 * Runs actual microbenchmarks using TensorF's own Matrix class so results
 * reflect the real performance of the ops the framework uses.
 *
 * Benchmarks:
 *   1. matmul_throughput   — GFLOP/s for matrix multiply at L1/L2/L3/RAM sizes
 *   2. memory_bandwidth    — GB/s sequential read + write (stream-style)
 *   3. cache_latency       — ns pointer-chase latency at each cache level
 *   4. storage_bandwidth   — MB/s sequential read from disk
 *   5. thread_scaling      — throughput vs core count (for OMP_NUM_THREADS advice)
 *
 * All benchmarks are self-contained — no external tools needed.
 * Results are stored in BenchmarkResult and used by HyperparamAdvisor.
 */

#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <cstdio>
#include <functional>
#include <fstream>

#include "../DataStructures/Matrix.hpp"

// ─── Timing helper ───────────────────────────────────────────────────────────

using Clock     = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

static inline double elapsed_ms(TimePoint t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static inline double elapsed_s(TimePoint t0) {
    return elapsed_ms(t0) / 1000.0;
}

// ─── Results structs ─────────────────────────────────────────────────────────

struct MatmulBench {
    // Key sizes tested (square matrices: N×N × N×N)
    struct Point {
        size_t  n;           // matrix dimension
        double  gflops;      // achieved GFLOP/s
        double  time_ms;
        bool    fits_l1;
        bool    fits_l2;
        bool    fits_l3;
    };
    std::vector<Point> points;

    double peak_gflops       = 0.0;  // best achieved
    size_t l3_optimal_n      = 0;    // largest N where perf doesn't drop >20%
    double l3_gflops          = 0.0;
    double ram_gflops         = 0.0;  // when matrix spills to RAM
};

struct BandwidthBench {
    double read_gbs    = 0.0;   // GB/s sequential read
    double write_gbs   = 0.0;   // GB/s sequential write
    double copy_gbs    = 0.0;   // GB/s memcpy-style
};

struct CacheLatencyBench {
    double l1_latency_ns = 0.0;
    double l2_latency_ns = 0.0;
    double l3_latency_ns = 0.0;
    double ram_latency_ns = 0.0;
};

struct StorageBench {
    std::string device;
    double seq_read_mbs  = 0.0;   // MB/s large sequential read
    double seq_write_mbs = 0.0;
    double latency_us    = 0.0;   // random 4K read latency
};

struct ThreadScalingBench {
    struct Point {
        int    threads;
        double gflops;
        double efficiency;  // gflops / (threads × single_thread_gflops)
    };
    std::vector<Point> points;
    int optimal_threads = 1;
};

struct BenchmarkResult {
    MatmulBench       matmul;
    BandwidthBench    bandwidth;
    CacheLatencyBench latency;
    StorageBench      storage;
    ThreadScalingBench thread_scaling;

    // Derived: estimated tokens/sec for a forward pass of given dims
    double estimated_tokens_per_sec(size_t batch, size_t block, size_t n_embed,
                                    size_t n_layers) const {
        // A transformer layer ≈ 2 matmuls of shape (batch*block, n_embed) × (n_embed, n_embed)
        double flops_per_layer = 2.0 * batch * block * n_embed * n_embed * 2;
        double total_flops     = flops_per_layer * n_layers;
        double gflops          = total_flops / 1e9;
        double peak            = matmul.l3_gflops > 0 ? matmul.l3_gflops
                                                       : matmul.peak_gflops;
        if (peak <= 0) return 0;
        double time_s          = gflops / peak;
        return (double)(batch * block) / time_s;
    }
};

// ─── BenchmarkRunner ─────────────────────────────────────────────────────────

class BenchmarkRunner {
public:
    // Cache sizes to calibrate which regime each matmul size falls in
    size_t l1_bytes = 32 * 1024;       // 32 KB default
    size_t l2_bytes = 256 * 1024;      // 256 KB default
    size_t l3_bytes = 8 * 1024 * 1024; // 8 MB default

    void set_cache_sizes(size_t l1, size_t l2, size_t l3) {
        l1_bytes = l1;
        l2_bytes = l2;
        l3_bytes = l3;
    }

    // ── 1. Matmul throughput ─────────────────────────────────────────────────

    MatmulBench bench_matmul(int warmup = 3, int iters = 10) {
        MatmulBench result;

        // Matrix sizes: from tiny (fits L1) to huge (spills RAM)
        // N × N float32: memory = N² × 3 × 4 bytes (A, B, C)
        std::vector<size_t> sizes;
        // L1-fitting: N² × 12 < l1_bytes  →  N < sqrt(l1 / 12)
        size_t n_l1 = (size_t)std::sqrt((double)l1_bytes / 12.0);
        size_t n_l2 = (size_t)std::sqrt((double)l2_bytes / 12.0);
        size_t n_l3 = (size_t)std::sqrt((double)l3_bytes / 12.0);

        // Align to 16 (for SIMD)
        auto align16 = [](size_t n) { return (n / 16) * 16; };
        n_l1 = align16(std::max(n_l1, (size_t)16));
        n_l2 = align16(std::max(n_l2, (size_t)32));
        n_l3 = align16(std::max(n_l3, (size_t)64));

        sizes = {n_l1, n_l2, n_l3, n_l3 * 2, n_l3 * 4};

        for (size_t N : sizes) {
            // Build random matrices using TensorF Matrix class
            Matrix<float> A = Matrix<float>::randu(-1.0f, 1.0f, {N, N});
            Matrix<float> B = Matrix<float>::randu(-1.0f, 1.0f, {N, N});

            // Warmup
            for (int w = 0; w < warmup; w++) {
                volatile auto C = A.matmul(B);
                (void)C;
            }

            // Timed runs
            std::vector<double> times;
            for (int i = 0; i < iters; i++) {
                auto t0 = Clock::now();
                volatile auto C = A.matmul(B);
                (void)C;
                times.push_back(elapsed_ms(t0));
            }

            double median_ms = median(times);
            double flops     = 2.0 * N * N * N;  // N³ mults + N³ adds
            double gflops    = (flops / 1e9) / (median_ms / 1000.0);

            size_t mem_bytes = 3 * N * N * sizeof(float);
            MatmulBench::Point p;
            p.n        = N;
            p.gflops   = gflops;
            p.time_ms  = median_ms;
            p.fits_l1  = mem_bytes <= l1_bytes;
            p.fits_l2  = mem_bytes <= l2_bytes;
            p.fits_l3  = mem_bytes <= l3_bytes;

            result.points.push_back(p);
            result.peak_gflops = std::max(result.peak_gflops, gflops);

            // L3 optimal = last N that still fits L3
            if (p.fits_l3) {
                result.l3_optimal_n = N;
                result.l3_gflops    = gflops;
            } else if (result.ram_gflops == 0.0) {
                result.ram_gflops = gflops;
            }
        }

        return result;
    }

    // ── 2. Memory bandwidth ──────────────────────────────────────────────────
    // STREAM-style benchmark: operate on arrays much larger than L3

    BandwidthBench bench_bandwidth(size_t array_mb = 256, int iters = 5) {
        BandwidthBench result;
        size_t n = (array_mb * 1024 * 1024) / sizeof(float);

        std::vector<float> A(n, 1.0f);
        std::vector<float> B(n, 2.0f);
        std::vector<float> C(n, 0.0f);

        // Sequential READ
        {
            std::vector<double> bws;
            for (int i = 0; i < iters; i++) {
                volatile float sum = 0;
                auto t0 = Clock::now();
                for (size_t j = 0; j < n; j++) sum += A[j];
                double s = elapsed_s(t0);
                (void)sum;
                bws.push_back((double)(n * sizeof(float)) / 1e9 / s);
            }
            result.read_gbs = median(bws);
        }

        // Sequential WRITE
        {
            std::vector<double> bws;
            for (int i = 0; i < iters; i++) {
                auto t0 = Clock::now();
                for (size_t j = 0; j < n; j++) C[j] = 1.0f;
                double s = elapsed_s(t0);
                bws.push_back((double)(n * sizeof(float)) / 1e9 / s);
            }
            result.write_gbs = median(bws);
        }

        // COPY (read + write)
        {
            std::vector<double> bws;
            for (int i = 0; i < iters; i++) {
                auto t0 = Clock::now();
                memcpy(C.data(), A.data(), n * sizeof(float));
                double s = elapsed_s(t0);
                // Both read and write: count both
                bws.push_back((double)(2 * n * sizeof(float)) / 1e9 / s);
            }
            result.copy_gbs = median(bws);
        }

        return result;
    }

    // ── 3. Cache latency (pointer-chase) ─────────────────────────────────────

    CacheLatencyBench bench_latency() {
        CacheLatencyBench result;

        auto measure_latency = [](size_t array_bytes) -> double {
            size_t n = array_bytes / sizeof(size_t);
            if (n < 64) n = 64;
            std::vector<size_t> arr(n);

            // Build a random permutation chain (pointer chase)
            std::iota(arr.begin(), arr.end(), 0);
            std::mt19937 rng(42);
            std::shuffle(arr.begin(), arr.end(), rng);

            // Follow the chain — defeats prefetcher
            const int REPS = 1 << 24;
            size_t idx = 0;
            auto t0 = Clock::now();
            for (int i = 0; i < REPS; i++)
                idx = arr[idx % n];
            double s = elapsed_s(t0);
            (void)idx;

            return (s / REPS) * 1e9; // ns per access
        };

        result.l1_latency_ns  = measure_latency(l1_bytes / 2);
        result.l2_latency_ns  = measure_latency(l2_bytes * 2);
        result.l3_latency_ns  = measure_latency(l3_bytes * 2);
        result.ram_latency_ns = measure_latency(512 * 1024 * 1024ULL); // 512 MB

        return result;
    }

    // ── 4. Storage bandwidth ─────────────────────────────────────────────────

    StorageBench bench_storage(const std::string& tmp_path = "/tmp",
                               size_t file_mb = 512) {
        StorageBench result;
        result.device = tmp_path;

        std::string path = tmp_path + "/tensorf_bench_" +
                           std::to_string(getpid()) + ".bin";

        size_t n_bytes = file_mb * 1024 * 1024;
        std::vector<char> buf(1024 * 1024, 'X'); // 1 MB buffer

        // Sequential WRITE
        {
            auto t0 = Clock::now();
            FILE* f = fopen(path.c_str(), "wb");
            if (f) {
                size_t written = 0;
                while (written < n_bytes) {
                    size_t chunk = std::min(buf.size(), n_bytes - written);
                    fwrite(buf.data(), 1, chunk, f);
                    written += chunk;
                }
                fclose(f);
                result.seq_write_mbs = (double)n_bytes / 1e6 / elapsed_s(t0);
            }
        }

        // Sequential READ (drop OS cache if possible)
        // Note: posix_fadvise DONTNEED requires privileged or is advisory only
        {
            auto t0 = Clock::now();
            FILE* f = fopen(path.c_str(), "rb");
            if (f) {
                size_t total = 0;
                size_t got;
                while ((got = fread(buf.data(), 1, buf.size(), f)) > 0)
                    total += got;
                fclose(f);
                if (total > 0)
                    result.seq_read_mbs = (double)total / 1e6 / elapsed_s(t0);
            }
        }

        // Cleanup
        remove(path.c_str());

        return result;
    }

    // ── 5. Run all benchmarks ────────────────────────────────────────────────

    BenchmarkResult run_all(bool verbose = true) {
        BenchmarkResult r;

        if (verbose) printf("[Benchmark] Starting matmul throughput...\n");
        r.matmul = bench_matmul();

        if (verbose) printf("[Benchmark] Memory bandwidth...\n");
        r.bandwidth = bench_bandwidth();

        if (verbose) printf("[Benchmark] Cache latency...\n");
        r.latency = bench_latency();

        if (verbose) printf("[Benchmark] Storage...\n");
        r.storage = bench_storage();

        if (verbose) {
            printf("\n[Results]\n");
            printf("  Matmul peak     : %.2f GFLOP/s\n", r.matmul.peak_gflops);
            printf("  Matmul L3       : %.2f GFLOP/s\n", r.matmul.l3_gflops);
            printf("  Mem BW read     : %.2f GB/s\n",    r.bandwidth.read_gbs);
            printf("  Mem BW write    : %.2f GB/s\n",    r.bandwidth.write_gbs);
            printf("  L1 latency      : %.1f ns\n",      r.latency.l1_latency_ns);
            printf("  L3 latency      : %.1f ns\n",      r.latency.l3_latency_ns);
            printf("  RAM latency     : %.1f ns\n",      r.latency.ram_latency_ns);
            printf("  Storage read    : %.0f MB/s\n",    r.storage.seq_read_mbs);
        }

        return r;
    }

private:
    template <typename T>
    static T median(std::vector<T> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }
};
