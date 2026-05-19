#pragma once

/*
 * MemoryProfiler.hpp
 * ==================
 * Tracks real memory consumption at three stages of model lifecycle:
 *
 *   STAGE_BASELINE  — before anything is loaded
 *   STAGE_LOADED    — after model parameters are loaded into RAM
 *   STAGE_TRAIN     — peak during a training step (forward + backward + grad)
 *   STAGE_INFER     — peak during inference (forward only)
 *
 * For each stage we capture:
 *   - RSS  (Resident Set Size)   — physical RAM actually used (basically one of the key infos)
 *   - VmPeak                     — peak virtual memory ever reached
 *   - VmRSS                      — current RSS from /proc/self/status
 *   - heap_alloc                 — bytes allocated via new/malloc/shared_ptr since last reset
 *
 * This lets the server predict, for ANY model config, how much RAM this
 * client will need, with safety margins.
 *
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <array>
#include <chrono>
#include <fstream>
#include <sstream>
#include <atomic>
#include <functional>

// ─── Stage enum ──────────────────────────────────────────────────────────────

enum class MemStage : int {
    BASELINE = 0,
    LOADED   = 1,
    TRAIN    = 2,
    INFER    = 3,
    COUNT    = 4
};

static constexpr const char* stage_name(MemStage s) {
    switch (s) {
        case MemStage::BASELINE: return "baseline";
        case MemStage::LOADED:   return "loaded";
        case MemStage::TRAIN:    return "train";
        case MemStage::INFER:    return "infer";
        default:                 return "unknown";
    }
}

// ─── Single memory snapshot ──────────────────────────────────────────────────

struct MemSnapshot {
    uint64_t rss_kb      = 0;  // Resident Set Size (physical RAM)
    uint64_t vm_peak_kb  = 0;  // Peak virtual memory ever
    uint64_t vm_size_kb  = 0;  // Current virtual memory size
    uint64_t shared_kb   = 0;  // Shared pages
    uint64_t stack_kb    = 0;  // Stack size
    uint64_t heap_kb     = 0;  // Heap via /proc/self/status VmData

    int64_t  timestamp_us = 0; // microseconds since profiler start

    bool valid = false;

    uint64_t rss_mb()     const { return rss_kb     / 1024; }
    uint64_t vm_peak_mb() const { return vm_peak_kb / 1024; }
    uint64_t heap_mb()    const { return heap_kb    / 1024; }
};

// ─── Delta between two snapshots ─────────────────────────────────────────────

struct MemDelta {
    int64_t rss_kb_delta  = 0;
    int64_t heap_kb_delta = 0;
    int64_t vm_kb_delta   = 0;

    int64_t rss_mb_delta()  const { return rss_kb_delta  / 1024; }
    int64_t heap_mb_delta() const { return heap_kb_delta / 1024; }
};

static MemDelta diff(const MemSnapshot& after, const MemSnapshot& before) {
    MemDelta d;
    d.rss_kb_delta  = (int64_t)after.rss_kb   - (int64_t)before.rss_kb;
    d.heap_kb_delta = (int64_t)after.heap_kb  - (int64_t)before.heap_kb;
    d.vm_kb_delta   = (int64_t)after.vm_size_kb - (int64_t)before.vm_size_kb;
    return d;
}

// ─── Full report ─────────────────────────────────────────────────────────────

struct MemoryReport {
    // Absolute values at each stage (MB)
    uint64_t baseline_rss_mb = 0;
    uint64_t loaded_rss_mb   = 0;
    uint64_t train_rss_mb    = 0;
    uint64_t infer_rss_mb    = 0;

    // Deltas (MB)
    int64_t param_ram_mb      = 0;  // cost of loading parameters
    int64_t train_overhead_mb = 0;  // extra RAM for gradients + optimizer state
    int64_t infer_overhead_mb = 0;  // extra RAM for inference (activations)

    // Peak across all stages
    uint64_t total_peak_mb    = 0;
    uint64_t vm_peak_mb       = 0;

    // Recommended minimum free RAM the client needs (with 20% safety margin)
    uint64_t recommended_free_mb = 0;

    // Per-stage snapshots for detailed analysis
    std::array<MemSnapshot, (int)MemStage::COUNT> snapshots;
};

// ─── MemoryProfiler class ────────────────────────────────────────────────────

class MemoryProfiler {
public:
    using Clock = std::chrono::steady_clock;

    MemoryProfiler() {
        start_time_ = Clock::now();
        // Initialize all snapshots as invalid
        for (auto& s : snapshots_) s.valid = false;
    }

    // Take a snapshot at the given stage
    void snapshot(MemStage stage) {
        snapshots_[(int)stage] = read_proc_status();
        snapshots_[(int)stage].timestamp_us = elapsed_us();
    }

    // Run a workload and capture peak memory during it
    // Polls /proc/self/status every `poll_interval_ms` milliseconds
    void profile_workload(MemStage stage, std::function<void()> workload,
                          int poll_interval_ms = 5) {
        MemSnapshot peak = read_proc_status();

        // Launch workload in foreground — poll in a tight loop
        // For simplicity (no threads), we snapshot before and after.
        snapshot_before_workload_ = read_proc_status();

        // code for the workload goes here .........................................................
        // During test we will de define what to pro
        workload();
        MemSnapshot after = read_proc_status();

        // Take max of before/after for rss (conservative)
        peak.rss_kb     = std::max(snapshot_before_workload_.rss_kb, after.rss_kb);
        peak.vm_peak_kb = std::max(snapshot_before_workload_.vm_peak_kb, after.vm_peak_kb);
        peak.heap_kb    = std::max(snapshot_before_workload_.heap_kb, after.heap_kb);
        peak.vm_size_kb = std::max(snapshot_before_workload_.vm_size_kb, after.vm_size_kb);
        peak.timestamp_us = elapsed_us();
        peak.valid = true;

        snapshots_[(int)stage] = peak;
    }

    // Build the report
    MemoryReport report() const {
        MemoryReport r;
        r.snapshots = snapshots_;

        const auto& base  = snapshots_[(int)MemStage::BASELINE];
        const auto& load  = snapshots_[(int)MemStage::LOADED];
        const auto& train = snapshots_[(int)MemStage::TRAIN];
        const auto& infer = snapshots_[(int)MemStage::INFER];

        if (base.valid)  r.baseline_rss_mb = base.rss_mb();
        if (load.valid)  r.loaded_rss_mb   = load.rss_mb();
        if (train.valid) r.train_rss_mb    = train.rss_mb();
        if (infer.valid) r.infer_rss_mb    = infer.rss_mb();

        // Deltas
        if (base.valid && load.valid)
            r.param_ram_mb = (int64_t)load.rss_mb() - (int64_t)base.rss_mb();
        if (load.valid && train.valid)
            r.train_overhead_mb = (int64_t)train.rss_mb() - (int64_t)load.rss_mb();
        if (load.valid && infer.valid)
            r.infer_overhead_mb = (int64_t)infer.rss_mb() - (int64_t)load.rss_mb();

        // Peak RSS across all valid stages
        for (const auto& s : snapshots_) {
            if (s.valid) {
                r.total_peak_mb = std::max(r.total_peak_mb, s.rss_mb());
                r.vm_peak_mb    = std::max(r.vm_peak_mb, s.vm_peak_mb());
            }
        }

        // Recommended free RAM = peak + 20% margin
        r.recommended_free_mb = (uint64_t)(r.total_peak_mb * 1.2);

        return r;
    }

    // Utility: estimate RAM needed for a given model config (no measurement)
    // Based on standard formulas for transformer-style models
    static uint64_t estimate_param_ram_mb(
        size_t vocab_size, size_t n_embed, size_t n_layers,
        size_t n_heads, size_t block_size, size_t batch_size,
        size_t bytes_per_param = 4)   // float32 default
    {
        // Parameters
        uint64_t embed_params   = (uint64_t)vocab_size * n_embed;
        uint64_t attn_params    = (uint64_t)n_layers * 4 * n_embed * n_embed; // Q,K,V,O
        uint64_t ffn_params     = (uint64_t)n_layers * 3 * n_embed * (4 * n_embed); // 3 projections × 4x expansion
        uint64_t ln_params      = (uint64_t)n_layers * 2 * 2 * n_embed; // 2 LN per layer, γ+β each
        uint64_t total_params   = embed_params + attn_params + ffn_params + ln_params;
        uint64_t param_mb       = (total_params * bytes_per_param) / (1024 * 1024);

        // KV cache per token: 2 (K+V) × n_layers × n_embed × bytes
        uint64_t kv_per_token   = 2 * n_layers * n_embed * bytes_per_param;
        uint64_t kv_cache_mb    = (uint64_t)(kv_per_token * block_size * batch_size)
                                  / (1024 * 1024);

        // Activation memory (forward): batch × block × n_embed × n_layers × factor
        uint64_t act_mb         = (uint64_t)(batch_size * block_size * n_embed
                                  * n_layers * bytes_per_param * 4)
                                  / (1024 * 1024);

        return param_mb + kv_cache_mb + act_mb;
    }

    // Estimate gradient + optimizer state (Adam: 2 moment tensors)
    static uint64_t estimate_optimizer_ram_mb(
        size_t vocab_size, size_t n_embed, size_t n_layers,
        size_t n_heads, size_t bytes_per_param = 4)
    {
        uint64_t param_mb = estimate_param_ram_mb(
            vocab_size, n_embed, n_layers, n_heads, 1, 1, bytes_per_param);
        // Gradients: same size as params
        // Adam m1 + m2: 2x params
        // Total overhead: 3x params
        return param_mb * 3;
    }

    // Print a human-readable summary
    void print_summary() const {
        MemoryReport r = report();
        printf("\n╔══════════════════════════════════════════╗\n");
        printf("║          MEMORY PROFILE REPORT           ║\n");
        printf("╠══════════════════════════════════════════╣\n");

        const char* labels[] = {"BASELINE", "LOADED", "TRAIN", "INFER"};
        for (int i = 0; i < (int)MemStage::COUNT; i++) {
            const auto& s = snapshots_[i];
            if (s.valid)
                printf("║  %-10s  RSS: %5llu MB  Heap: %5llu MB ║\n",
                    labels[i],
                    (unsigned long long)s.rss_mb(),
                    (unsigned long long)s.heap_mb());
        }

        printf("╠══════════════════════════════════════════╣\n");
        printf("║  Param load cost    : %+6lld MB          ║\n",
            (long long)r.param_ram_mb);
        printf("║  Training overhead  : %+6lld MB          ║\n",
            (long long)r.train_overhead_mb);
        printf("║  Inference overhead : %+6lld MB          ║\n",
            (long long)r.infer_overhead_mb);
        printf("╠══════════════════════════════════════════╣\n");
        printf("║  Peak RSS           :  %5llu MB          ║\n",
            (unsigned long long)r.total_peak_mb);
        printf("║  Recommended free   :  %5llu MB          ║\n",
            (unsigned long long)r.recommended_free_mb);
        printf("╚══════════════════════════════════════════╝\n\n");
    }

private:
    std::array<MemSnapshot, (int)MemStage::COUNT> snapshots_;
    MemSnapshot snapshot_before_workload_;
    Clock::time_point start_time_;

    int64_t elapsed_us() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - start_time_).count();
    }

    // Read /proc/self/status and parse memory fields
    static MemSnapshot read_proc_status() {
        MemSnapshot s;
        std::ifstream f("/proc/self/status");
        if (!f.is_open()) return s;

        std::string line;
        while (std::getline(f, line)) {
            uint64_t val = 0;
            if      (sscanf(line.c_str(), "VmRSS:   %llu kB", &val) == 1)
                s.rss_kb = val;
            else if (sscanf(line.c_str(), "VmPeak:  %llu kB", &val) == 1)
                s.vm_peak_kb = val;
            else if (sscanf(line.c_str(), "VmSize:  %llu kB", &val) == 1)
                s.vm_size_kb = val;
            else if (sscanf(line.c_str(), "VmData:  %llu kB", &val) == 1)
                s.heap_kb = val;
            else if (sscanf(line.c_str(), "VmStk:   %llu kB", &val) == 1)
                s.stack_kb = val;
        }
        s.valid = (s.rss_kb > 0);
        return s;
    }
};

// ─── RAII scope profiler ─────────────────────────────────────────────────────
// Automatically snapshots on construction and destruction.
//
struct ScopedMemSnapshot {
    MemoryProfiler& profiler;
    MemStage        stage;

    ScopedMemSnapshot(MemoryProfiler& p, MemStage s)
        : profiler(p), stage(s) {}

    ~ScopedMemSnapshot() {
        profiler.snapshot(stage);
    }
};
