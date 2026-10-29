#pragma once

/*
 * CapabilitySuite.hpp
 * ===================
 * Runs the micro-kernels from CapabilityKernels.hpp under a TIME BUDGET and a
 * THERMAL GUARD, and assembles the result into a CapabilityVector:
 *
 *     flop/s per precision and shape      bandwidth + latency per cache level
 *     framework GEMM (Matrix::matmul,     DRAM saturation point (threads)
 *       BLAS rate, thread curve, per      gather/scatter efficiency (GNN)
 *       core class: FrameworkBench.hpp)
 *     conv, dependent-chain, n-body       fork-join / thread-spawn / tiny-op overheads
 *     thread scaling per core class       does mixing core classes hurt?
 *     sustained fraction (throttling)     noise estimate and per-group validity
 *
 * There is deliberately no single score. The advisor and the workload model read
 * the vector (e.g. a GNN looks at gather efficiency, an RNN at dependent-chain
 * latency, a transformer at skinny-GEMM flop/s against DRAM bandwidth).
 *
 * Safety (this runs on phones):
 *   - Hard wall-clock budget; every phase gets a slice and checks the deadline.
 *   - Memory use is capped at a fraction of DeviceFingerprint::memory.safe_budget.
 *   - Before every phase the thermal guard is consulted. The suite stops (and says
 *     why) when the hottest sensor passes an absolute limit OR rises too far above
 *     where it started. The absolute defaults are STARTING POINTS taken from the
 *     governor design document, which marks them as unverified; real limits must be
 *     calibrated per device. Charging uses a tighter battery limit.
 *   - No disk writes at all (the old storage benchmark wrote 512 MB to /tmp).
 *
 * Usage:
 *     DeviceFingerprint fp = detect_device();
 *     CapabilitySuite suite;
 *     CapabilityVector cv = suite.run(fp);        // default: ~20 s budget
 *     std::string json = suite_to_json(fp, cv);
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "CapabilityKernels.hpp"
#include "DeviceFingerprint.hpp"
#include "FrameworkBench.hpp"
#include "ProfilerCommon.hpp"

namespace tensorf::prof {

// ─── Configuration ───────────────────────────────────────────────────────────

struct SuiteConfig {
    double time_budget_s   = 20.0;     // total wall-clock cap for run()
    bool   run_sustained   = true;     // the throttling probe is the longest phase
    double sustained_s     = 6.0;      // target length, clamped to what the budget leaves
    double window_s        = 0.04;     // one timing window
    int    samples         = 5;        // windows per measurement (median)
    double memory_fraction = 0.25;     // of fingerprint safe budget
    size_t memory_hard_cap = 512ull << 20;
    size_t memory_floor    = 64ull << 20;   // used when the budget is unknown
    int    max_scaling_threads = 16;
    bool   pin_threads     = true;

    // Thermal guard. STARTING POINTS, unverified; calibrate per device.
    double stop_cpu_c        = 45.0;   // hottest non-battery zone
    double stop_battery_c    = 40.0;   // battery temperature while discharging
    double charging_margin_c = 2.0;    // battery limit is lowered by this while charging
    double max_rise_c        = 8.0;    // stop if the hottest sensor rose this far from the start

    std::string sysroot;               // "" = real machine; tests point this at a fixture
    bool   verbose         = false;
};

// ─── Result ──────────────────────────────────────────────────────────────────

struct LevelPlan {
    size_t l1_ws = 0, l2_ws = 0, l3_ws = 0, dram_ws = 0;   // per-thread working sets (bytes); 0 = level absent
    bool   cache_sizes_assumed = false;                     // a size was unknown and a default was used
    bool   dram_may_be_cached  = false;                     // L3 so large the DRAM set might still hit it
};

struct ClusterScaling {
    CoreClass cls = CoreClass::UNIFORM;
    int       cpus = 0;
    kernels::ScalingResult independent;   // weak scaling, no synchronisation
    double    sync_gflops = 0.0;          // barrier-paced GEMM on all of this cluster's cores
    bool      sync_ok = false;
};

struct GroupStatus {                      // why a number is missing or weak
    std::string name;
    bool        ran = false;
    std::string note;
};

struct CapabilityVector {
    // --- compute ---
    std::vector<kernels::GemmPoint> gemm;     // every GEMM point measured
    double f32_gflops_best = 0.0;             // best single-thread f32 over square shapes
    double f32_gflops_skinny = 0.0;           // M=8: the decode / small-batch shape
    double f64_gflops = 0.0;
    double i8_gops = 0.0;
    double conv_gflops = 0.0;
    double nbody_gflops = 0.0;
    double nbody_pairs_per_s = 0.0;

    // --- the framework itself: Matrix::matmul through the BLAS it selected ---
    // The portable gemm above depends on compile flags (SSE2 without -march=native) and can sit 20x below BLAS.
    // Training runs Matrix::matmul, so GEMM-bound layers are priced with these numbers when `fw.usable()`.
    fwbench::FwProfile fw;
    // Scalar views of the framework rates (what the cost model reads; hand-built vectors can set them directly).
    std::string blas_backend;                 // tensorf::blas::backend(): "openblas", "shim", "none"
    double blas_f32_gflops = 0.0;             // Matrix::matmul, best square f32, one thread
    double blas_f32_gflops_skinny = 0.0;      // M = 4: decode / small batch
    double blas_f64_gflops = 0.0;
    std::vector<fwbench::FwPoint> blas_gemm;  // every framework GEMM point measured
    // Roof for the ridge point: the framework rate when measured, else the portable loop.
    double compute_best_gflops() const { return blas_f32_gflops > 0 ? blas_f32_gflops : f32_gflops_best; }

    // --- memory ---
    LevelPlan levels;
    double bw_l1_gbs = 0, bw_l2_gbs = 0, bw_l3_gbs = 0, bw_dram_gbs = 0;     // 1 thread, read
    double triad_1t_gbs = 0, triad_max_gbs = 0;
    int    dram_saturation_threads = 1;       // fewest threads within 5% of the best DRAM triad
    std::vector<std::pair<int, double>> triad_sweep;   // (threads, GB/s)
    double lat_l1_ns = 0, lat_l2_ns = 0, lat_l3_ns = 0, lat_dram_ns = 0;

    // --- irregular / sequential ---
    double gather_gbs = 0, scatter_gbs = 0, gather_edges_per_s = 0;
    double gather_efficiency = 0;             // gather_gbs / bw_dram_gbs
    double fma_latency_ns = 0, rnn_step_us = 0;

    // --- overheads ---
    double thread_spawn_us = 0, barrier_roundtrip_us = 0, tiny_op_ns = 0;

    // --- parallelism ---
    std::vector<ClusterScaling> scaling;      // one per core cluster
    double sync_all_gflops = 0.0;             // barrier-paced GEMM over ALL cores
    bool   mixing_hurts = false;              // sync_all < best single cluster's sync
    int    recommended_threads = 1;           // best cluster's best_threads
    CoreClass recommended_class = CoreClass::UNIFORM;

    // --- throttling ---
    kernels::SustainedResult sustained;
    double temp_start_c = std::numeric_limits<double>::quiet_NaN();
    double temp_end_c   = std::numeric_limits<double>::quiet_NaN();

    // --- derived ---
    double ridge_flop_per_byte = 0.0;         // f32_gflops_best / bw_dram_gbs: below this, DRAM-bound

    // --- trust ---
    std::string stopped_reason;               // "" = every phase ran
    double time_used_s = 0.0, time_budget_s = 0.0;
    double noise = 0.0;                       // worst relative spread among key measurements
    bool   build_is_optimised = true;
    std::vector<GroupStatus> groups;

    bool group_ran(const std::string& name) const {
        for (const auto& g : groups) if (g.name == name) return g.ran;
        return false;
    }
};

// ─── Pure planning helpers (unit-tested) ─────────────────────────────────────

// Per-thread working sets for each memory level, from cache sizes (0 = unknown).
inline LevelPlan plan_levels(uint64_t l1, uint64_t l2, uint64_t l3, size_t max_bytes_per_thread) {
    LevelPlan p;
    if (l1 == 0) { l1 = 32ull << 10;  p.cache_sizes_assumed = true; }
    if (l2 == 0) { l2 = 256ull << 10; p.cache_sizes_assumed = true; }
    p.l1_ws = (size_t)std::max<uint64_t>(4096, l1 / 2);
    p.l2_ws = (size_t)std::max<uint64_t>(l2 / 2, 2 * l1);
    if (l3 > 0) {
        p.l3_ws = (size_t)std::min<uint64_t>(std::max<uint64_t>(l3 / 2, 2 * l2), 32ull << 20);   // capped: huge VM L3s
        if (p.l3_ws <= p.l2_ws) p.l3_ws = 0;                                                      // no room above L2
    }
    const uint64_t above = std::max<uint64_t>(l3, l2);
    uint64_t dram = std::max<uint64_t>(64ull << 20, 4 * above);
    dram = std::min<uint64_t>(dram, 256ull << 20);
    dram = std::min<uint64_t>(dram, (uint64_t)max_bytes_per_thread);
    p.dram_ws = (size_t)dram;
    p.dram_may_be_cached = l3 > 0 && p.dram_ws < 2 * l3;
    return p;
}

// Thermal guard decision. Returns "" to continue, else the reason to stop.
inline std::string thermal_verdict(const ThermalProbe::Reading& now, double start_hottest_c, bool charging, const SuiteConfig& c) {
    if (!now.ok) return "";
    if (std::isfinite(now.max_cpu_like_c) && now.max_cpu_like_c >= c.stop_cpu_c) {
        char b[96]; std::snprintf(b, sizeof b, "thermal: zone %.1f C >= %.1f C limit", now.max_cpu_like_c, c.stop_cpu_c); return b;
    }
    const double bat_limit = c.stop_battery_c - (charging ? c.charging_margin_c : 0.0);
    if (std::isfinite(now.battery_c) && now.battery_c >= bat_limit) {
        char b[96]; std::snprintf(b, sizeof b, "thermal: battery %.1f C >= %.1f C limit%s", now.battery_c, bat_limit, charging ? " (charging)" : ""); return b;
    }
    double hot = std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(now.max_cpu_like_c)) hot = now.max_cpu_like_c;
    if (std::isfinite(now.battery_c)) hot = std::isfinite(hot) ? std::max(hot, now.battery_c) : now.battery_c;
    if (std::isfinite(hot) && std::isfinite(start_hottest_c) && hot - start_hottest_c >= c.max_rise_c) {
        char b[96]; std::snprintf(b, sizeof b, "thermal: rose %.1f C above start (limit %.1f C)", hot - start_hottest_c, c.max_rise_c); return b;
    }
    return "";
}

inline size_t memory_cap_bytes(const DeviceFingerprint& fp, const SuiteConfig& c) {
    size_t cap = c.memory_floor;
    if (fp.memory.safe_budget_bytes > 0) cap = (size_t)((double)fp.memory.safe_budget_bytes * c.memory_fraction);
    cap = std::min(cap, c.memory_hard_cap);
    return std::max<size_t>(cap, 8ull << 20);
}

// ─── The suite ───────────────────────────────────────────────────────────────

class CapabilitySuite {
public:
    explicit CapabilitySuite(SuiteConfig cfg = {}) : cfg_(std::move(cfg)) {}

    CapabilityVector run(const DeviceFingerprint& fp) {
        using namespace kernels;
        CapabilityVector cv;
        cv.time_budget_s = cfg_.time_budget_s;
#if !defined(__OPTIMIZE__) && !defined(_MSC_VER)
        cv.build_is_optimised = false;      // no -O flag: the numbers say nothing about the hardware
#endif
        Deadline total(cfg_.time_budget_s);
        const size_t cap = memory_cap_bytes(fp, cfg_);
        const CoreCluster* strong = fp.strongest();
        const std::vector<int> strong_cpus = (cfg_.pin_threads && strong) ? strong->cpus : std::vector<int>{};
        // All CPUs, strongest cluster first.
        std::vector<int> all_cpus;
        for (const auto& c : fp.clusters) all_cpus.insert(all_cpus.end(), c.cpus.begin(), c.cpus.end());
        const int n_all = std::max(1, (int)all_cpus.size());
        const double win = total.limit() < 6.0 ? std::min(cfg_.window_s, 0.02) : cfg_.window_s;
        const int S = cfg_.samples;

        // thermal baseline
        ThermalProbe probe(cfg_.sysroot);
        const auto t0r = probe.read();
        cv.temp_start_c = probe.hottest_c();
        const bool charging = fp.battery.charging();
        auto thermal_ok = [&](const char* phase) -> bool {
            if (!cv.stopped_reason.empty()) return false;
            const std::string why = thermal_verdict(probe.read(), cv.temp_start_c, charging, cfg_);
            if (!why.empty()) { cv.stopped_reason = why + " (before " + phase + ")"; return false; }
            if (total.expired()) { cv.stopped_reason = std::string("time budget exhausted (before ") + phase + ")"; return false; }
            return true;
        };
        (void)t0r;
        auto note = [&](const char* name, bool ran, std::string why = "") { cv.groups.push_back({name, ran, std::move(why)}); };
        std::vector<double> spreads;

        // Cache plan for the strongest cluster
        cv.levels = plan_levels(strong ? strong->l1d_bytes : 0, strong ? strong->l2_bytes : 0, strong ? strong->l3_bytes : 0,
                                cap / 2);

        // 1. overheads --------------------------------------------------------
        if (thermal_ok("overheads")) {
            Deadline dl = total.slice(0.04, 1.0);
            auto o = overheads(std::min(4, n_all), dl);
            cv.thread_spawn_us = o.thread_spawn_us; cv.barrier_roundtrip_us = o.barrier_roundtrip_us; cv.tiny_op_ns = o.tiny_op_ns;
            note("overheads", o.ok, n_all < 2 ? "single CPU: pool barrier not measured" : "");
        } else note("overheads", false, cv.stopped_reason);

        // 2. GEMM -------------------------------------------------------------
        if (thermal_ok("gemm")) {
            Deadline dl = total.slice(0.15, 4.0);
            const GemmShape shapes[] = {{128, 128, 128, "square_s"}, {256, 256, 256, "square_m"}, {512, 512, 512, "square_l"}, {8, 1024, 1024, "skinny_m8"}};
            for (const auto& sh : shapes) {
                if (dl.expired() && !cv.gemm.empty()) break;
                GemmPoint p = gemm(Precision::F32, sh, 1, strong_cpus, win, S, dl, cap);
                if (p.ok) { cv.gemm.push_back(p); spreads.push_back(p.spread);
                    if (std::string(sh.tag) == "skinny_m8") cv.f32_gflops_skinny = p.gops; else cv.f32_gflops_best = std::max(cv.f32_gflops_best, p.gops); }
            }
            GemmPoint p64 = gemm(Precision::F64, {256, 256, 256, "square_m"}, 1, strong_cpus, win, S, dl, cap);
            if (p64.ok) { cv.gemm.push_back(p64); cv.f64_gflops = p64.gops; spreads.push_back(p64.spread); }
            GemmPoint p8 = gemm(Precision::I8, {256, 256, 256, "square_m"}, 1, strong_cpus, win, S, dl, cap);
            if (p8.ok) { cv.gemm.push_back(p8); cv.i8_gops = p8.gops; spreads.push_back(p8.spread); }
            note("gemm", cv.f32_gflops_best > 0);
        } else note("gemm", false, cv.stopped_reason);

        // 2b. the framework's own GEMM (Matrix::matmul through BLAS) -----------
        // Single-thread points on the fastest core class, the BLAS thread curve, and one pinned thread per class on
        // big.LITTLE. Runs before any worker pool of this suite exists, and releases its pin before returning.
        if (thermal_ok("blas_gemm")) {
            fwbench::Options fo;
            fo.window_s = win;
            fo.budget_s = std::max(1.0, std::min(total.remaining() * 0.25, 5.0));
            fo.max_bytes = cap;
            fo.max_threads = std::min(n_all, cfg_.max_scaling_threads);
            fo.scaling = n_all >= 2;
            fo.should_stop = [&] { return !thermal_verdict(probe.read(), cv.temp_start_c, charging, cfg_).empty(); };
            if (cfg_.pin_threads)
                for (const auto& c : fp.clusters) if (!c.cpus.empty()) { fo.classes.push_back(c.cpus); fo.class_names.push_back(to_string(c.cls)); }
            cv.fw = fwbench::run(fo);
            cv.blas_backend = cv.fw.backend;
            cv.blas_f32_gflops = cv.fw.best_f32();
            cv.blas_f32_gflops_skinny = cv.fw.gflops("f32_skinny");
            cv.blas_f64_gflops = cv.fw.gflops("f64_256");
            for (const auto& p : cv.fw.points) if (p.ok) cv.blas_gemm.push_back(p);
            if (cv.fw.usable() && cv.f32_gflops_best > 0 && cv.fw.best_f32() > 1.5 * cv.f32_gflops_best)
                cv.fw.notes.push_back("the portable gemm loop is far below the framework's BLAS (flags-limited): GEMM layers are priced with the framework rate");
            note("blas_gemm", cv.fw.usable(),
                 cv.fw.usable() ? (cv.fw.threads_effective ? "" : "BLAS gained nothing from more threads here") : (cv.fw.notes.empty() ? "no framework GEMM measured" : cv.fw.notes.front()));
        } else note("blas_gemm", false, cv.stopped_reason);

        // 3. latency ----------------------------------------------------------
        if (thermal_ok("latency")) {
            Deadline dl = total.slice(0.10, 3.0);
            const auto& L = cv.levels;
            auto lat = [&](size_t b) { return b ? chase_latency(b, dl, cap).ns : 0.0; };
            cv.lat_l1_ns = lat(L.l1_ws); cv.lat_l2_ns = lat(L.l2_ws); cv.lat_l3_ns = lat(L.l3_ws); cv.lat_dram_ns = lat(L.dram_ws);
            note("latency", cv.lat_l1_ns > 0 && cv.lat_dram_ns > 0, L.dram_may_be_cached ? "L3 reported larger than the DRAM working set: dram latency may include L3 hits" : "");
        } else note("latency", false, cv.stopped_reason);

        // 4. bandwidth per level (1 thread) -----------------------------------
        if (thermal_ok("bandwidth")) {
            Deadline dl = total.slice(0.12, 3.0);
            const auto& L = cv.levels;
            auto bw = [&](const char* nm, size_t b) { return b ? read_bandwidth(nm, b, 1, strong_cpus, win, S, dl, cap) : BandwidthPoint{}; };
            cv.bw_l1_gbs = bw("L1", L.l1_ws).read_gbs; cv.bw_l2_gbs = bw("L2", L.l2_ws).read_gbs;
            cv.bw_l3_gbs = bw("L3", L.l3_ws).read_gbs; cv.bw_dram_gbs = bw("DRAM", L.dram_ws).read_gbs;
            note("bandwidth", cv.bw_dram_gbs > 0, L.cache_sizes_assumed ? "some cache sizes unknown: defaults used for working sets" : "");
        } else note("bandwidth", false, cv.stopped_reason);

        // 5. DRAM saturation sweep (triad, all cores) -------------------------
        if (thermal_ok("stream")) {
            Deadline dl = total.slice(0.10, 3.0);
            const int maxT = std::min(n_all, cfg_.max_scaling_threads);
            const size_t per_array = std::max<size_t>(1u << 20, std::min<size_t>(cv.levels.dram_ws / 3, 48u << 20));
            double best = 0;
            for (int t = 1; t <= maxT; ++t) {
                if (t > 1 && dl.expired()) break;
                Measured m = triad_bandwidth(per_array, t, cfg_.pin_threads ? all_cpus : std::vector<int>{}, win, S, dl, cap);
                if (!m.ok) break;
                cv.triad_sweep.push_back({t, m.value}); spreads.push_back(m.spread);
                best = std::max(best, m.value);
            }
            if (!cv.triad_sweep.empty()) {
                cv.triad_1t_gbs = cv.triad_sweep.front().second; cv.triad_max_gbs = best;
                for (const auto& [t, g] : cv.triad_sweep) if (g >= best * 0.95) { cv.dram_saturation_threads = t; break; }
            }
            note("stream", best > 0);
        } else note("stream", false, cv.stopped_reason);

        // 6. sparse gather / scatter -------------------------------------------
        if (thermal_ok("sparse")) {
            Deadline dl = total.slice(0.08, 3.0);
            // Features table well past L2 but bounded by the memory cap.
            size_t nodes = 1u << 17; const size_t F = 64;
            while (nodes * F * 4 * 2 + nodes * 16 * 8 > cap && nodes > 4096) nodes >>= 1;
            auto sp = sparse_gather_scatter(nodes, F, 8.0, dl, cap);
            cv.gather_gbs = sp.gather_gbs; cv.scatter_gbs = sp.scatter_gbs; cv.gather_edges_per_s = sp.edges_per_s;
            note("sparse", sp.ok);
        } else note("sparse", false, cv.stopped_reason);

        // 7. conv, dependent chains, n-body -------------------------------------
        if (thermal_ok("conv")) {
            Deadline dl = total.slice(0.05, 2.0);
            auto c = conv3x3(16, 64, 1, strong_cpus, win, S, dl, cap);
            cv.conv_gflops = c.gflops; if (c.ok) spreads.push_back(c.spread);
            note("conv", c.ok);
        } else note("conv", false, cv.stopped_reason);
        if (thermal_ok("dependent")) {
            Deadline dl = total.slice(0.05, 2.0);
            auto d = dependent_chains(128, dl);
            cv.fma_latency_ns = d.fma_latency_ns; cv.rnn_step_us = d.rnn_step_us;
            note("dependent", d.ok);
        } else note("dependent", false, cv.stopped_reason);
        if (thermal_ok("nbody")) {
            Deadline dl = total.slice(0.05, 2.0);
            auto nb = nbody(1024, 1, strong_cpus, win, S, dl, cap);
            cv.nbody_gflops = nb.gflops_est; cv.nbody_pairs_per_s = nb.interactions_per_s;
            note("nbody", nb.ok);
        } else note("nbody", false, cv.stopped_reason);

        // 8. thread scaling per core class + mixing test -------------------------
        if (thermal_ok("scaling")) {
            Deadline dl = total.slice(0.20, 6.0);
            double best_cluster_sync = 0.0;
            for (const auto& c : fp.clusters) {
                if (dl.expired() && !cv.scaling.empty()) break;
                ClusterScaling cs; cs.cls = c.cls; cs.cpus = (int)c.cpus.size();
                const int maxT = std::min(cs.cpus, cfg_.max_scaling_threads);
                cs.independent = thread_scaling(cfg_.pin_threads ? c.cpus : std::vector<int>{}, maxT, 256, win, std::max(3, S - 2), dl, cap);
                if (cs.cpus >= 2) {
                    Measured m = sync_gemm(cfg_.pin_threads ? c.cpus : std::vector<int>{}, std::min(cs.cpus, cfg_.max_scaling_threads), 256, win, std::max(3, S - 2), dl, cap);
                    cs.sync_ok = m.ok; cs.sync_gflops = m.value;
                    best_cluster_sync = std::max(best_cluster_sync, m.value);
                }
                cv.scaling.push_back(std::move(cs));
            }
            if (n_all >= 2 && fp.clusters.size() > 1 && !dl.expired()) {
                Measured m = sync_gemm(cfg_.pin_threads ? all_cpus : std::vector<int>{}, std::min(n_all, cfg_.max_scaling_threads), 256, win, std::max(3, S - 2), dl, cap);
                if (m.ok) { cv.sync_all_gflops = m.value; cv.mixing_hurts = best_cluster_sync > 0 && m.value < best_cluster_sync * 0.95; }
            }
            if (!cv.scaling.empty()) {
                // Recommendation: the strongest cluster's best thread count, unless mixing helped.
                const auto& s0 = cv.scaling.front();
                cv.recommended_threads = s0.independent.ok ? s0.independent.best_threads : 1;
                cv.recommended_class = s0.cls;
            }
            note("scaling", !cv.scaling.empty(), n_all < 2 ? "single CPU: scaling cannot be measured" : "");
            // The framework's own GEMM decides: if BLAS threads do not help, plan on one thread.
            if (cv.fw.usable() && cv.fw.scaling.size() >= 2)
                cv.recommended_threads = cv.fw.threads_effective ? cv.fw.best_threads : 1;
        } else note("scaling", false, cv.stopped_reason);

        // 9. sustained -----------------------------------------------------------
        if (cfg_.run_sustained && thermal_ok("sustained")) {
            const double dur = std::min(cfg_.sustained_s, std::max(0.0, total.remaining() - 0.5));
            if (dur >= 2.0) {
                const int T = strong ? std::min<int>((int)strong->cpus.size(), cfg_.max_scaling_threads) : 1;
                cv.sustained = sustained_gemm(T, strong_cpus, 256, dur, std::max(0.25, dur / 12.0), cap,
                    [&](double) { return thermal_verdict(probe.read(), cv.temp_start_c, charging, cfg_); });
                if (!cv.sustained.stopped_reason.empty() && cv.stopped_reason.empty()) cv.stopped_reason = cv.sustained.stopped_reason;
                note("sustained", cv.sustained.ok, cv.sustained.ok ? "" : "run too short to claim a throttling fraction");
            } else note("sustained", false, "not enough budget left");
        } else if (!cfg_.run_sustained) note("sustained", false, "disabled in config");
        else note("sustained", false, cv.stopped_reason);

        // derived ---------------------------------------------------------------
        cv.temp_end_c = probe.hottest_c();
        if (cv.bw_dram_gbs > 0) {
            cv.gather_efficiency = cv.gather_gbs / cv.bw_dram_gbs;
            cv.ridge_flop_per_byte = cv.compute_best_gflops() / cv.bw_dram_gbs;
        }
        cv.noise = spreads.empty() ? 0.0 : max_of(spreads);
        cv.time_used_s = total.elapsed();
        if (cfg_.verbose) print_summary(fp, cv);
        return cv;
    }

    static void print_summary(const DeviceFingerprint& fp, const CapabilityVector& cv) {
        std::printf("\n[Capability] %s | %s | %s\n", fp.model_name.c_str(), fp.microarch.c_str(), fp.simd.summary().c_str());
        for (const auto& c : fp.clusters)
            std::printf("  cluster %-11s %2zu cpus  %6.0f MHz  L1d %4llu K  L2 %5llu K  L3 %5llu K\n", to_string(c.cls), c.cpus.size(),
                        c.max_freq_mhz, (unsigned long long)(c.l1d_bytes >> 10), (unsigned long long)(c.l2_bytes >> 10), (unsigned long long)(c.l3_bytes >> 10));
        std::printf("  compute  f32 %.1f  skinny %.1f  f64 %.1f  i8 %.1f  conv %.1f  nbody %.1f  (GFLOP/s)\n",
                    cv.f32_gflops_best, cv.f32_gflops_skinny, cv.f64_gflops, cv.i8_gops, cv.conv_gflops, cv.nbody_gflops);
        if (cv.fw.usable()) {
            std::printf("  framework (%s)  f32 %.1f  skinny %.1f  f64 %.1f GFLOP/s%s\n", cv.fw.backend.c_str(), cv.fw.best_f32(),
                        cv.fw.gflops("f32_skinny"), cv.fw.gflops("f64_256"),
                        cv.fw.single_thread_pinned ? (" [1t pinned to cpu " + std::to_string(cv.fw.single_thread_cpu) + "]").c_str() : "");
            if (!cv.fw.scaling.empty()) {
                std::printf("    BLAS threads:");
                for (const auto& s : cv.fw.scaling) std::printf("  %dT %.1f (x%.2f; gnn-shape x%.2f, cpu %.1fx)", s.threads, s.gflops, s.speedup, s.small_speedup, s.cpu_ratio);
                std::printf("%s\n", cv.fw.threads_effective ? "" : "  -> threads do not help, planning on 1");
            }
            for (const auto& c : cv.fw.classes) std::printf("    class %-8s cpu %d  %.1f GFLOP/s\n", c.name.c_str(), c.cpu, c.gflops_1t);
        } else {
            std::printf("  framework: not linked/measured (rates above are portable kernels, not Matrix::matmul)\n");
        }
        std::printf("  read BW  L1 %.1f  L2 %.1f  L3 %.1f  DRAM %.1f GB/s | triad 1t %.1f max %.1f (saturates at %d threads)\n",
                    cv.bw_l1_gbs, cv.bw_l2_gbs, cv.bw_l3_gbs, cv.bw_dram_gbs, cv.triad_1t_gbs, cv.triad_max_gbs, cv.dram_saturation_threads);
        std::printf("  latency  L1 %.1f  L2 %.1f  L3 %.1f  DRAM %.1f ns\n", cv.lat_l1_ns, cv.lat_l2_ns, cv.lat_l3_ns, cv.lat_dram_ns);
        std::printf("  GNN      gather %.1f GB/s (%.0f%% of DRAM read)  scatter %.1f GB/s\n", cv.gather_gbs, cv.gather_efficiency * 100, cv.scatter_gbs);
        std::printf("  RNN      dep-FMA %.2f ns  recurrent step %.1f us\n", cv.fma_latency_ns, cv.rnn_step_us);
        std::printf("  overhead spawn %.1f us  barrier %.1f us  tiny op %.1f ns\n", cv.thread_spawn_us, cv.barrier_roundtrip_us, cv.tiny_op_ns);
        std::printf("  sustained fraction %.2f over %zu windows%s\n", cv.sustained.fraction, cv.sustained.window_gflops.size(),
                    cv.sustained.ok ? "" : " (not claimed)");
        std::printf("  ridge %.1f flop/byte | noise %.0f%% | %.1f of %.1f s%s%s\n", cv.ridge_flop_per_byte, cv.noise * 100, cv.time_used_s,
                    cv.time_budget_s, cv.build_is_optimised ? "" : " | NOT AN OPTIMISED BUILD", cv.stopped_reason.empty() ? "" : (" | STOPPED: " + cv.stopped_reason).c_str());
    }

private:
    SuiteConfig cfg_;
};

// ─── JSON ────────────────────────────────────────────────────────────────────

inline void to_json(Json& j, const CapabilityVector& cv, const char* key = "capability") {
    j.begin_object(key);
    j.begin_object("compute")
        .num("f32_gflops", cv.f32_gflops_best).num("f32_skinny_gflops", cv.f32_gflops_skinny).num("f64_gflops", cv.f64_gflops)
        .str("blas_backend", cv.blas_backend).num("blas_f32_gflops", cv.blas_f32_gflops).num("blas_f32_gflops_skinny", cv.blas_f32_gflops_skinny)
        .num("blas_f64_gflops", cv.blas_f64_gflops).num("i8_gops", cv.i8_gops).num("conv_gflops", cv.conv_gflops).num("nbody_gflops", cv.nbody_gflops).end_object();
    j.begin_object("framework")
        .boolean("available", cv.fw.available).str("backend", cv.fw.backend).boolean("usable", cv.fw.usable())
        .num("f32_gflops", cv.fw.best_f32()).num("f32_skinny_gflops", cv.fw.gflops("f32_skinny")).num("f64_gflops", cv.fw.gflops("f64_256"))
        .integer("best_threads", cv.fw.best_threads).boolean("threads_effective", cv.fw.threads_effective).boolean("small_threads_effective", cv.fw.small_threads_effective).num("max_speedup", cv.fw.max_speedup)
        .integer("single_thread_cpu", cv.fw.single_thread_cpu).boolean("single_thread_pinned", cv.fw.single_thread_pinned);
    j.begin_array("points");
    for (const auto& p : cv.fw.points)
        j.begin_object().str("key", p.key).integer("m", (int64_t)p.M).integer("n", (int64_t)p.N).integer("k", (int64_t)p.K)
            .num("gflops", p.gflops).num("spread", p.spread).boolean("ok", p.ok).end_object();
    j.end_array();
    j.begin_array("scaling");
    for (const auto& s : cv.fw.scaling)
        j.begin_object().integer("threads", s.threads).num("gflops", s.gflops).num("speedup", s.speedup).num("efficiency", s.eff)
            .num("gnn_gflops", s.small_gflops).num("gnn_speedup", s.small_speedup).num("cpu_ratio", s.cpu_ratio).end_object();
    j.end_array();
    j.begin_array("classes");
    for (const auto& c : cv.fw.classes)
        j.begin_object().str("name", c.name).integer("cpu", c.cpu).num("gflops_1t", c.gflops_1t).boolean("pinned", c.pinned).end_object();
    j.end_array();
    j.begin_array("notes");
    for (const auto& n : cv.fw.notes) j.str(nullptr, n);
    j.end_array();
    j.end_object();
    j.begin_array("gemm");
    for (const auto& g : cv.gemm)
        j.begin_object().str("prec", kernels::to_string(g.prec)).str("shape", g.shape.tag).integer("m", (int64_t)g.shape.m)
            .integer("n", (int64_t)g.shape.n).integer("k", (int64_t)g.shape.k).num("gops", g.gops).num("flop_per_byte", g.arithmetic_intensity).end_object();
    j.end_array();
    j.begin_object("memory")
        .num("bw_l1_gbs", cv.bw_l1_gbs).num("bw_l2_gbs", cv.bw_l2_gbs).num("bw_l3_gbs", cv.bw_l3_gbs).num("bw_dram_gbs", cv.bw_dram_gbs)
        .num("triad_1t_gbs", cv.triad_1t_gbs).num("triad_max_gbs", cv.triad_max_gbs).integer("dram_saturation_threads", cv.dram_saturation_threads)
        .num("lat_l1_ns", cv.lat_l1_ns).num("lat_l2_ns", cv.lat_l2_ns).num("lat_l3_ns", cv.lat_l3_ns).num("lat_dram_ns", cv.lat_dram_ns)
        .boolean("cache_sizes_assumed", cv.levels.cache_sizes_assumed).boolean("dram_may_be_cached", cv.levels.dram_may_be_cached)
        .num("ridge_flop_per_byte", cv.ridge_flop_per_byte).end_object();
    j.begin_object("irregular")
        .num("gather_gbs", cv.gather_gbs).num("scatter_gbs", cv.scatter_gbs).num("gather_efficiency", cv.gather_efficiency)
        .num("gather_edges_per_s", cv.gather_edges_per_s).num("fma_latency_ns", cv.fma_latency_ns).num("rnn_step_us", cv.rnn_step_us).end_object();
    j.begin_object("overheads").num("thread_spawn_us", cv.thread_spawn_us).num("barrier_roundtrip_us", cv.barrier_roundtrip_us)
        .num("tiny_op_ns", cv.tiny_op_ns).end_object();
    j.begin_object("parallelism");
    j.integer("recommended_threads", cv.recommended_threads).str("recommended_class", to_string(cv.recommended_class))
        .num("sync_all_gflops", cv.sync_all_gflops).boolean("mixing_hurts", cv.mixing_hurts);
    j.begin_array("clusters");
    for (const auto& s : cv.scaling) {
        j.begin_object().str("class", to_string(s.cls)).integer("cpus", s.cpus).integer("best_threads", s.independent.best_threads)
            .integer("peak_threads", s.independent.peak_threads).num("sync_gflops", s.sync_gflops);
        j.begin_array("scaling");
        for (const auto& p : s.independent.points) j.begin_object().integer("threads", p.threads).num("gflops", p.gflops).num("efficiency", p.efficiency).end_object();
        j.end_array(); j.end_object();
    }
    j.end_array(); j.end_object();
    j.begin_object("sustained").num("fraction", cv.sustained.fraction).num("first_gflops", cv.sustained.first_gflops)
        .num("last_gflops", cv.sustained.last_gflops).num("seconds", cv.sustained.seconds_run).boolean("claimed", cv.sustained.ok)
        .num("temp_start_c", cv.temp_start_c).num("temp_end_c", cv.temp_end_c).end_object();
    j.begin_object("trust").num("noise", cv.noise).num("time_used_s", cv.time_used_s).num("time_budget_s", cv.time_budget_s)
        .boolean("optimised_build", cv.build_is_optimised).str("stopped_reason", cv.stopped_reason);
    j.begin_array("groups");
    for (const auto& g : cv.groups) j.begin_object().str("name", g.name).boolean("ran", g.ran).str("note", g.note).end_object();
    j.end_array(); j.end_object();
    j.end_object();
}

inline std::string suite_to_json(const DeviceFingerprint& fp, const CapabilityVector& cv) {
    Json j;
    j.begin_object();
    j.str("profiler_version", "2.0");
    to_json(j, fp);
    to_json(j, cv);
    j.end_object();
    return j.finish();
}

}  // namespace tensorf::prof
