#pragma once

/*
 * HyperparamAdvisor.hpp
 * =====================
 * Takes the HardwareFingerprint + BenchmarkResult and produces:
 *
 *   HyperparamConfig  — the optimal hyperparameters for this client
 *   AlgoPolicy        — which algorithms + quantization to use
 *
 * Decision logic:
 *
 *   batch_size  → constrained by L3 cache (keep KV-cache + activations in L3)
 *   block_size  → constrained by available RAM minus param footprint
 *   n_embed     → aligned to vector register width (AVX/AVX512)
 *   n_layers    → scaled by RAM/compute budget
 *   quant       → selected by ISA capability + RAM pressure
 *   algo        → matmul tiling, threading, prefetch strategy
 */

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>

#include "HardwareFingerprint.hpp"
#include "BenchmarkRunner.hpp"
#include "MemoryProfiler.hpp"

// ─── Quantization policy ─────────────────────────────────────────────────────

enum class QuantPolicy {
    NONE,       // float32 — no quantization
    FP16,       // half precision — needs AVX-512 or hardware support
    INT8,       // 8-bit integer — needs AVX2 at minimum (VNNI ideal)
    FP8_E4M3,   // FP8 — needs AVX-512 + explicit dequant
    FP8_E5M2,   // FP8 — better range, lower precision
    INT4,       // compression — use when RAM very limited
    // Needed to add a 3 or 3.5 bit quant according to a AQML paper
};

static const char* quant_name(QuantPolicy q) {
    switch (q) {
        case QuantPolicy::NONE:     return "float32";
        case QuantPolicy::FP16:     return "float16";
        case QuantPolicy::INT8:     return "int8";
        case QuantPolicy::FP8_E4M3: return "fp8_e4m3";
        case QuantPolicy::FP8_E5M2: return "fp8_e5m2";
        case QuantPolicy::INT4:     return "int4";
        default:                    return "unknown";
    }
}

static size_t quant_bytes(QuantPolicy q) {
    switch (q) {
        case QuantPolicy::NONE:     return 4;
        case QuantPolicy::FP16:     return 2;
        case QuantPolicy::INT8:     return 1;
        case QuantPolicy::FP8_E4M3: return 1;
        case QuantPolicy::FP8_E5M2: return 1;
        case QuantPolicy::INT4:     return 1; // packed 2/byte, approx
        default:                    return 4;
    }
}

// ─── Algorithm policy ────────────────────────────────────────────────────────

struct AlgoPolicy {
    // Matmul
    size_t matmul_tile_size   = 32;   // tile NxN for cache blocking
    bool   use_blas           = true;  // TensorF delagates ops to BLASS by default 

    // Threading
    int    num_threads        = 1;    // OMP_NUM_THREADS recommendation
    bool   use_numa_bind      = false; // bind threads to NUMA nodes // was not implemented by default 

    // Memory
    bool   use_mmap_weights   = false; // memory-map weight files (good for NVMe Gen4+)
    bool   prefetch_next_batch = false; // async prefetch from storage

    // Quantization
    QuantPolicy quant         = QuantPolicy::NONE;
    bool   dequant_on_the_fly = false; // dequantize each layer at inference time

    // Attention
    bool   use_flash_attn_style = false; // tiled attention to avoid O(T²) materialization
    size_t attn_chunk_size      = 256;
};

// ─── Hyperparameters ─────────────────────────────────────────────────────────

struct HyperparamConfig {
    // Model dimensions
    size_t batch_size  = 8;
    size_t block_size  = 128;
    size_t n_embed     = 256;
    size_t n_layers    = 4;
    size_t n_heads     = 4;

    // Derived
    size_t head_dim()     const { return n_embed / n_heads; }
    size_t ffn_dim()      const { return n_embed * 4; }

    // Training
    float  learning_rate  = 3e-4f;
    size_t grad_accum_steps = 1;    // accumulate if batch doesn't fit

    // Runtime policy
    AlgoPolicy algo;

    // RAM estimates (MB)
    uint64_t estimated_param_mb = 0;
    uint64_t estimated_train_mb = 0;
    uint64_t estimated_infer_mb = 0;

    // Confidence: how well this machine can run the config (0–100)
    int fit_score = 0;
};

// ─── HyperparamAdvisor ───────────────────────────────────────────────────────

class HyperparamAdvisor {
public:
    // Derive optimal config from hardware fingerprint and benchmark results
    static HyperparamConfig advise(
        const HardwareFingerprint& fp,
        const BenchmarkResult&     bench,
        size_t vocab_size = 50257)   // Distill GPT-2 default
    {
        HyperparamConfig cfg;
        cfg.algo = build_algo_policy(fp, bench);

        // ── Step 1: n_embed — align to vector register width ─────────────────
        //
        // n_embed must be a multiple of floats-per-vector-register so that
        // every SIMD lane is used. With AVX-512: 16 floats/reg → multiples of 16.
        // We then pick the largest n_embed that keeps params + KV cache in L3.

        size_t vec_align = fp.cpu.embed_alignment(); // 4, 8, or 16

        // Target: KV cache for one layer at block_size=128 fits in L3
        // kv_per_layer = 2 × block_size × n_embed × sizeof(float)
        // We want: n_layers × kv_per_layer < L3 * 0.5  (leave half L3 for activations)
        size_t l3 = fp.cache.l3_bytes;
        size_t tentative_n_embed = 256;

        if (l3 >= 64 * 1024 * 1024) tentative_n_embed = 1024;
        else if (l3 >= 32 * 1024 * 1024) tentative_n_embed = 768;
        else if (l3 >= 16 * 1024 * 1024) tentative_n_embed = 512;
        else if (l3 >= 8  * 1024 * 1024) tentative_n_embed = 384;
        else                              tentative_n_embed = 256;

        // Boost if AVX-512 available (wider registers → prefer multiples of 16)
        if (fp.cpu.has_avx512f && tentative_n_embed < 512)
            tentative_n_embed = 512;

        // Align down to vec_align
        cfg.n_embed = (tentative_n_embed / vec_align) * vec_align;
        if (cfg.n_embed == 0) cfg.n_embed = vec_align;

        // ── Step 2: n_heads — must divide n_embed ────────────────────────────
        // head_dim should be ≥ 32 (too small → poor attention quality)
        // head_dim should be ≤ 128 (too large → L1 pressure in attn)
        cfg.n_heads = choose_n_heads(cfg.n_embed);

        // ── Step 3: n_layers — scale by RAM budget ───────────────────────────
        uint64_t ram_available_mb = fp.ram.available_mb;
        // Reserve 30% for OS + other processes
        uint64_t usable_mb = (uint64_t)(ram_available_mb * 0.7);

        // Estimate param cost at current n_embed for increasing n_layers
        cfg.n_layers = choose_n_layers(vocab_size, cfg.n_embed, usable_mb,
                                       cfg.algo.quant);

        // ── Step 4: block_size — constrained by RAM ───────────────────────────
        // KV cache = 2 × n_layers × block_size × n_embed × bytes_per_param
        // Must leave room for activations (~same size as KV cache)
        size_t bytes_pp = quant_bytes(cfg.algo.quant);
        uint64_t param_mb = MemoryProfiler::estimate_param_ram_mb(
            vocab_size, cfg.n_embed, cfg.n_layers, cfg.n_heads, 1, 1, bytes_pp);

        uint64_t remaining_mb = (usable_mb > param_mb) ? (usable_mb - param_mb) : 0;

        // kv_cache per token = 2 × n_layers × n_embed × bytes_pp
        size_t kv_per_token = 2 * cfg.n_layers * cfg.n_embed * bytes_pp;
        // activations per token ≈ same
        size_t bytes_per_token = kv_per_token * 2;

        // Max block_size given remaining RAM, accounting for batch
        size_t tentative_block = (remaining_mb * 1024 * 1024)
                                 / (bytes_per_token + 1);
        tentative_block = std::max(tentative_block, (size_t)64);
        tentative_block = std::min(tentative_block, (size_t)2048);

        // Round down to power of 2
        cfg.block_size = prev_pow2(tentative_block);

        // ── Step 5: batch_size — keep hot data in L3 ─────────────────────────
        // During forward: activations = batch × block × n_embed × bytes_pp × n_layers
        // We want activations to fit in L3 (for max matmul throughput)
        size_t act_per_sample = cfg.block_size * cfg.n_embed * bytes_pp * cfg.n_layers;

        // Also factor in the matmul optimal batch from benchmark
        size_t batch_from_l3   = (l3 / 2) / (act_per_sample + 1);
        size_t batch_from_bench = (bench.matmul.l3_optimal_n > 0)
            ? std::max((size_t)1, bench.matmul.l3_optimal_n / cfg.block_size)
            : 4;

        cfg.batch_size = std::max((size_t)1,
                         std::min(batch_from_l3, batch_from_bench));
        // Cap at 64 — beyond this, marginal returns drop fast on CPU
        cfg.batch_size = std::min(cfg.batch_size, (size_t)64);
        // Must be ≥ 1
        cfg.batch_size = std::max(cfg.batch_size, (size_t)1);

        // ── Step 6: gradient accumulation ────────────────────────────────────
        // If optimal effective batch for convergence is 32+ but we can only
        // fit 4 in memory, accumulate 8 steps.
        size_t desired_effective_batch = 32;
        cfg.grad_accum_steps = std::max((size_t)1,
            desired_effective_batch / cfg.batch_size);

        // ── Step 7: RAM estimates ─────────────────────────────────────────────
        cfg.estimated_param_mb = MemoryProfiler::estimate_param_ram_mb(
            vocab_size, cfg.n_embed, cfg.n_layers, cfg.n_heads,
            cfg.block_size, cfg.batch_size, bytes_pp);
        cfg.estimated_train_mb = cfg.estimated_param_mb
            + MemoryProfiler::estimate_optimizer_ram_mb(
                vocab_size, cfg.n_embed, cfg.n_layers, cfg.n_heads, bytes_pp);
        cfg.estimated_infer_mb = cfg.estimated_param_mb
            + (uint64_t)(cfg.n_layers * cfg.n_embed * cfg.block_size
                         * cfg.batch_size * bytes_pp) / (1024 * 1024);

        // ── Step 8: fit score ─────────────────────────────────────────────────
        cfg.fit_score = compute_fit_score(fp, bench, cfg);

        return cfg;
    }

    static void print_config(const HyperparamConfig& cfg,
                              const HardwareFingerprint& fp) {
        printf("\n╔══════════════════════════════════════════════════╗\n");
        printf("║         HYPERPARAMETER ADVISOR REPORT            ║\n");
        printf("╠══════════════════════════════════════════════════╣\n");
        printf("║  Hardware score   : %3d / 100                    ║\n",
               fp.capability_score);
        printf("║  Config fit score : %3d / 100                    ║\n",
               cfg.fit_score);
        printf("╠══════════════════════════════════════════════════╣\n");
        printf("║  MODEL DIMENSIONS                                 ║\n");
        printf("║    n_embed        : %-6zu  (align=%zu, %s)   ║\n",
               cfg.n_embed, fp.cpu.embed_alignment(),
               fp.cpu.has_avx512f ? "AVX-512" :
               fp.cpu.has_avx2    ? "AVX2" : "SSE2");
        printf("║    n_heads        : %-6zu  (head_dim=%zu)         ║\n",
               cfg.n_heads, cfg.head_dim());
        printf("║    n_layers       : %-6zu                         ║\n",
               cfg.n_layers);
        printf("║    block_size     : %-6zu  (context length)       ║\n",
               cfg.block_size);
        printf("║    batch_size     : %-6zu                         ║\n",
               cfg.batch_size);
        printf("║    grad_accum     : %-6zu  (effective_batch=%zu)   ║\n",
               cfg.grad_accum_steps, cfg.batch_size * cfg.grad_accum_steps);
        printf("╠══════════════════════════════════════════════════╣\n");
        printf("║  ALGORITHM POLICY                                 ║\n");
        printf("║    quantization   : %-10s                    ║\n",
               quant_name(cfg.algo.quant));
        printf("║    num_threads    : %-4d                          ║\n",
               cfg.algo.num_threads);
        printf("║    matmul tile    : %-4zu                          ║\n",
               cfg.algo.matmul_tile_size);
        printf("║    use BLAS       : %-3s                           ║\n",
               cfg.algo.use_blas ? "yes" : "no");
        printf("║    mmap weights   : %-3s                           ║\n",
               cfg.algo.use_mmap_weights ? "yes" : "no");
        printf("║    flash attn     : %-3s  (chunk=%zu)              ║\n",
               cfg.algo.use_flash_attn_style ? "yes" : "no",
               cfg.algo.attn_chunk_size);
        printf("╠══════════════════════════════════════════════════╣\n");
        printf("║  MEMORY ESTIMATES                                 ║\n");
        printf("║    Params only    : %5llu MB                      ║\n",
               (unsigned long long)cfg.estimated_param_mb);
        printf("║    Training peak  : %5llu MB                      ║\n",
               (unsigned long long)cfg.estimated_train_mb);
        printf("║    Inference peak : %5llu MB                      ║\n",
               (unsigned long long)cfg.estimated_infer_mb);
        printf("║    Available RAM  : %5u MB                      ║\n",
               fp.ram.available_mb);
        printf("╚══════════════════════════════════════════════════╝\n\n");
    }

private:
    // ── Algo policy builder ──────────────────────────────────────────────────

    static AlgoPolicy build_algo_policy(const HardwareFingerprint& fp,
                                        const BenchmarkResult& bench) {
        AlgoPolicy p;

        // Quantization: choose based on ISA and RAM pressure
        p.quant = choose_quant(fp);

        // Threading: use physical cores (not hyperthreaded)
        // For memory-bound workloads (LLMs), HT adds contention not speed
        p.num_threads = (int)fp.cpu.physical_cores;

        // Matmul tile: calibrate to L2 cache
        // tile² × 3 × sizeof(float) ≤ L2/2
        size_t l2 = fp.cache.l2_bytes;
        p.matmul_tile_size = (size_t)std::sqrt((double)(l2 / 2) / (3 * sizeof(float)));
        p.matmul_tile_size = (p.matmul_tile_size / 16) * 16; // align to 16
        p.matmul_tile_size = std::max(p.matmul_tile_size, (size_t)16);
        p.matmul_tile_size = std::min(p.matmul_tile_size, (size_t)128);

        // BLAS: always prefer it if available
        p.use_blas = (fp.software.blas_backend != "unknown");

        // Memory-map weights: only worth it if NVMe Gen4+
        for (const auto& s : fp.storage) {
            if (s.type == StorageType::NVME_GEN4 ||
                s.type == StorageType::NVME_GEN5) {
                p.use_mmap_weights = true;
                break;
            }
        }

        // Flash-attention style chunking: worthwhile when block_size > 512
        // and RAM bandwidth is the bottleneck (< 50 GB/s)
        p.use_flash_attn_style = (bench.bandwidth.read_gbs < 50.0);
        p.attn_chunk_size      = 128; // conservative chunk

        // Prefetch from storage: only if storage is fast enough
        p.prefetch_next_batch = (bench.storage.seq_read_mbs > 2000.0);

        return p;
    }

    // ── Quantization selection ───────────────────────────────────────────────

    static QuantPolicy choose_quant(const HardwareFingerprint& fp) {
        uint64_t ram_mb = fp.ram.available_mb;

        // If plenty of RAM and good ISA → no quant (best quality)
        if (ram_mb >= 32768 && fp.cpu.has_avx2)
            return QuantPolicy::NONE;

        // AMX available → INT8 is extremely fast (hardware native)
        if (fp.cpu.has_amx)
            return QuantPolicy::INT8;

        // VNNI → INT8 dot product native
        if (fp.cpu.has_avx512_vnni)
            return QuantPolicy::INT8;

        // AVX-512 → FP8 is feasible (TensorF has fp8 types)
        if (fp.cpu.has_avx512f && ram_mb < 16384)
            return QuantPolicy::FP8_E4M3;

        // AVX2 → INT8 via software
        if (fp.cpu.has_avx2 && ram_mb < 8192)
            return QuantPolicy::INT8;

        // Very constrained RAM → INT4
        if (ram_mb < 4096)
            return QuantPolicy::INT4;

        // Moderate RAM → FP16
        if (ram_mb < 16384)
            return QuantPolicy::FP16;

        return QuantPolicy::NONE;
    }

    // ── n_heads selection ────────────────────────────────────────────────────

    static size_t choose_n_heads(size_t n_embed) {
        // head_dim candidates: 32, 64, 96, 128
        for (size_t hd : {128, 96, 64, 32}) {
            if (n_embed % hd == 0) {
                size_t nh = n_embed / hd;
                if (nh >= 2 && nh <= 32) return nh;
            }
        }
        // Fallback: largest divisor ≤ 16
        for (size_t nh = 16; nh >= 1; nh--) {
            if (n_embed % nh == 0) return nh;
        }
        return 1;
    }

    // ── n_layers selection ───────────────────────────────────────────────────

    static size_t choose_n_layers(size_t vocab_size, size_t n_embed,
                                   uint64_t usable_mb, QuantPolicy quant) {
        size_t bytes_pp = quant_bytes(quant);

        // Reserve 50% of usable RAM for parameters
        uint64_t param_budget_mb = usable_mb / 2;

        // Binary-search the max n_layers that fits
        size_t lo = 1, hi = 96;
        while (lo < hi) {
            size_t mid = (lo + hi + 1) / 2;
            uint64_t est = MemoryProfiler::estimate_param_ram_mb(
                vocab_size, n_embed, mid, n_embed / 64, 128, 1, bytes_pp);
            if (est <= param_budget_mb) lo = mid;
            else                         hi = mid - 1;
        }

        return std::max(lo, (size_t)2);
    }

    // ── Fit score ────────────────────────────────────────────────────────────

    static int compute_fit_score(const HardwareFingerprint& fp,
                                  const BenchmarkResult& bench,
                                  const HyperparamConfig& cfg) {
        int score = 0;

        // Does estimated training peak fit in available RAM? (40 pts)
        if (cfg.estimated_train_mb <= fp.ram.available_mb) {
            double ratio = (double)cfg.estimated_train_mb / fp.ram.available_mb;
            score += (ratio < 0.5) ? 40 : (int)(40 * (1.0 - ratio));
        }

        // Can matmul throughput sustain > 1 GFLOP/s? (30 pts)
        if (bench.matmul.l3_gflops >= 10.0)  score += 30;
        else if (bench.matmul.l3_gflops >= 4.0) score += 20;
        else if (bench.matmul.l3_gflops >= 1.0) score += 10;

        // Storage fast enough for weight streaming? (15 pts)
        if (bench.storage.seq_read_mbs >= 3000.0) score += 15;
        else if (bench.storage.seq_read_mbs >= 500.0) score += 10;
        else score += 5;

        // ISA bonus (15 pts)
        switch (fp.cpu.best_isa()) {
            case ISA::AMX:          score += 15; break;
            case ISA::AVX512_VNNI:  score += 13; break;
            case ISA::AVX512:       score += 10; break;
            case ISA::AVX2:         score += 7;  break;
            default:                score += 2;  break;
        }

        return std::min(score, 100);
    }

    // ── Utilities ────────────────────────────────────────────────────────────

    static size_t prev_pow2(size_t n) {
        if (n == 0) return 1;
        size_t p = 1;
        while (p * 2 <= n) p *= 2;
        return p;
    }
};
