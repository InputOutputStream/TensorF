#pragma once

/*
 * ProfilerSuite.hpp
 * =================
 * The orchestrator of the profiler rewrite. It replaces the old Profiler.hpp pipeline (which benchmarked one
 * operation and let one function guess a whole model) with the four layers of the design document:
 *
 *   1. device       DeviceFingerprint   facts only: core classes, caches, SIMD, safe memory budget, sensors
 *   2. memory       TensorAccounting    measured RSS-per-tensor-byte overhead factor (needs the allocation hook);
 *                                       runs BEFORE the heavy stages, because they leave freed pages resident
 *   3. capability   CapabilitySuite     roofline-style kernels -> a capability VECTOR, not one score
 *   4. validation   WorkloadProbe       the analytic model vs real Matrix/Tensor ops -> Calibration + honest error
 *   then, per algorithm:
 *   5. advice       Advisor             tunables chosen by the calibrated model, optionally confirmed by a trial
 *
 * Usage:
 *
 *     ProfilerSuite suite;
 *     Profile p = suite.run();                                   // stages 1-4 under one time budget
 *     adv::Advice a = suite.advise(p, my_algorithm, base_config, constraints, {}, trial_fn);
 *     std::string payload = ProfilerSuite::to_json(p, &a);       // one JSON document for the server
 *     ProfilerSuite::print_summary(p);
 *
 * Behaviour that matters on a phone:
 *   - The advice's error range is floored at this device's measured run-to-run noise (noise_floor()).
 *   - ONE wall-clock budget for everything. Each stage gets what remains, never more than its own cap, and a
 *     stage that cannot run is recorded as skipped WITH A REASON; nothing disappears silently.
 *   - The thermal guard of the capability suite governs the run: if the device is hot at the start or heats up,
 *     the capability stage stops and every later stage is skipped. A profile of a throttling phone is worse than none.
 *   - The memory stage needs the allocation hook (define TENSORF_MEMTRACK_IMPLEMENT in exactly one translation
 *     unit). Without it the stage is skipped and the advisor keeps the default overhead factor, and says so.
 *   - Nothing here chooses a model size. Advice only searches the tunables an Algorithm declares.
 *
 * Needs the framework on the include path as well (it measures real ops): -Iinclude/core -Iinclude/tools.
 */

#include <chrono>
#include <ctime>
#include <string>
#include <vector>

#include "Advisor.hpp"
#include "CapabilitySuite.hpp"
#include "DeviceFingerprint.hpp"
#include "TensorAccounting.hpp"
#include "WorkloadProbe.hpp"

namespace tensorf::prof {

struct ProfileOptions {
    double total_budget_s      = 60.0;     // everything
    double capability_budget_s = 20.0;     // cap for stage 2
    bool   capability_sustained = false;   // the throttling probe is the longest capability phase; opt in
    bool   run_probe           = true;     // stage 3
    double probe_budget_s      = 25.0;
    double probe_size_scale    = 1.0;
    bool   measure_memory      = true;     // stage 4 (needs the allocation hook)
    double min_stage_s         = 3.0;      // below this much remaining time a stage is skipped, not started
    std::string sysroot;                   // "" = this machine; tests point it at a fixture
    bool   verbose             = false;
};

struct StageRecord {
    std::string name;
    std::string status;                    // "ok" | "skipped" | "stopped" | "failed"
    double      seconds = 0;
    std::string note;
};

struct Profile {
    DeviceFingerprint fp;
    CapabilityVector  cv;           bool have_capability = false;
    wl::ProbeReport   probe;        bool have_probe = false;
    mem::MemoryMeasurement memory;  bool have_memory = false;
    double mem_overhead_factor = 1.25;     // measured when stage 4 succeeded, else the default
    bool   mem_factor_measured = false;
    std::vector<StageRecord> stages;
    std::string stopped_reason;
    double total_seconds = 0;

    // The calibration to hand to predict()/advise(), or nullptr when no real measurement backs the model.
    const wl::Calibration* calibration() const { return have_probe && probe.usable > 0 ? &probe.calibration : nullptr; }
    const StageRecord* stage(const std::string& name) const {
        for (const auto& s : stages) if (s.name == name) return &s;
        return nullptr;
    }
};

namespace detail {

// One real training step of a two-layer MLP through Tensor autograd, sized so its memory growth clears the
// meaningful-ratio threshold of TensorAccounting. Returns a closure that owns the net.
struct MemoryStep { std::function<void()> step; double modelled_bytes = 0; size_t H = 0; };

inline MemoryStep make_memory_step(size_t H, size_t B = 256, size_t I = 784, size_t O = 10) {
    struct Net { Tensor_t<float> x, w1, w2; };
    auto n = std::make_shared<Net>();
    {
        mem::MemScope s(mem::Cat::PARAMS);
        n->w1 = make_tensor<float>(Matrix<float>::randu(-.05f, .05f, {I, H}));
        n->w2 = make_tensor<float>(Matrix<float>::randu(-.05f, .05f, {H, O}));
    }
    n->x = make_tensor<float>(Matrix<float>::randu(-1.f, 1.f, {B, I}));
    n->x->requires_grad = false;
    wl::RunSpec r; r.mode = wl::Mode::TRAIN; r.opt_states = 0;
    wl::Workload w;
    w.layers = {wl::dense((double)B, (double)I, (double)H, r, "fc1", false), wl::elementwise((double)(B * H), 1, 2, r, "relu"),
                wl::dense((double)B, (double)H, (double)O, r, "fc2", false), wl::elementwise((double)(B * O), 2, 2, r, "sq")};
    MemoryStep ms; ms.H = H; ms.modelled_bytes = wl::estimate_memory(w, 1.0).tensor_total;
    ms.step = [n] {
        auto y = n->x->matmul(n->w1)->relu()->matmul(n->w2);
        auto loss = (y * y)->sum();
        loss->backward(Matrix<float>(std::vector<float>{1.f}, shape_t{1}));
        n->w1->zero_grad(); n->w2->zero_grad();
    };
    return ms;
}

inline std::string utc_timestamp() {
    const std::time_t tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char buf[32];
    std::tm tmv{};
#if defined(_WIN32)
    gmtime_s(&tmv, &tt);
#else
    gmtime_r(&tt, &tmv);
#endif
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tmv);
    return buf;
}

}  // namespace detail

class ProfilerSuite {
public:
    Profile run(const ProfileOptions& opt = {}) const {
        Profile p;
        Stopwatch total;
        auto remaining = [&] { return opt.total_budget_s - total.elapsed(); };
        auto record = [&](const char* name, const char* status, double secs, const std::string& note) {
            p.stages.push_back({name, status, secs, note});
            if (opt.verbose) std::printf("[profiler] %-11s %-8s %6.2fs  %s\n", name, status, secs, note.c_str());
        };

        // 1. device (always runs: cheap, and every later stage needs it)
        {
            Stopwatch sw;
            p.fp = opt.sysroot.empty() ? detect_device() : detect_device_linux(opt.sysroot);
            record("device", "ok", sw.elapsed(), p.fp.model_name + " | " + p.fp.microarch);
        }

        // 2. memory overhead factor (needs the allocation hook).
        // Runs BEFORE the heavy stages on purpose: the capability suite and the probe touch hundreds of MB, and the
        // allocator keeps freed pages resident, so a later step would reuse them and peak RSS would not move.
        // Light (< 1 s) and thermally pre-checked.
        const std::string thermal_now = [&] {
            ThermalProbe tp(opt.sysroot);
            return thermal_verdict(tp.read(), tp.hottest_c(), p.fp.battery.charging(), SuiteConfig{});
        }();
        if (!opt.measure_memory) record("memory", "skipped", 0, "disabled by options");
        else if (!thermal_now.empty()) record("memory", "skipped", 0, "device too hot: " + thermal_now);
        else if (!mem::hook_active()) record("memory", "skipped", 0, "allocation hook not linked (define TENSORF_MEMTRACK_IMPLEMENT in one translation unit); default overhead factor kept");
        else if (std::min(5.0, remaining()) < 1.0) record("memory", "skipped", 0, "not enough time budget left");
        else {
            Stopwatch sw;
            // Largest net that is safe: its modelled memory must stay under 10% of the device's safe budget and 96 MB.
            const double cap = p.fp.memory.safe_budget_bytes > 0 ? std::min(0.10 * (double)p.fp.memory.safe_budget_bytes, 96.0 * 1048576.0) : 48.0 * 1048576.0;
            bool done = false;
            for (size_t H : {(size_t)768, (size_t)512, (size_t)256, (size_t)128}) {
                try {
                    detail::MemoryStep ms = detail::make_memory_step(H);
                    if (ms.modelled_bytes * 1.25 > cap) continue;
                    // COLD single step, on purpose: the peak a real run reaches is its first step in a fresh heap. A warm-up (or
                    // a second repeat) reuses pages the allocator already holds, so RSS barely rises and the ratio reads
                    // far too low (measured: 0.99 cold, 0.54 warmed, 0.17 after the heavy stages).
                    mem::MeasureOptions mo; mo.warmup = 0; mo.repeats = 1;
                    p.memory = mem::measure_memory(ms.step, mo);
                    p.have_memory = true;
                    done = true;
                    if (p.memory.factor_valid) {
                        p.mem_overhead_factor = p.memory.recommended_factor();
                        p.mem_factor_measured = true;
                        record("memory", "ok", sw.elapsed(), "overhead factor " + std::to_string(p.mem_overhead_factor).substr(0, 5) + " (H=" + std::to_string(H) + ")");
                    } else {
                        std::string why = p.memory.notes.empty() ? "no valid ratio" : p.memory.notes.front();
                        record("memory", "skipped", sw.elapsed(), "factor not measurable here: " + why + "; default kept");
                    }
                    break;
                } catch (const std::exception& e) {
                    record("memory", "failed", sw.elapsed(), std::string("exception: ") + e.what());
                    done = true;
                    break;
                }
            }
            if (!done) record("memory", "skipped", sw.elapsed(), "device memory budget too small for the reference step; default kept");
        }

        // 3. capability
        {
            const double budget = std::min(opt.capability_budget_s, remaining());
            if (budget < opt.min_stage_s) {
                record("capability", "skipped", 0, "not enough time budget left");
                p.stopped_reason = "time budget exhausted before the capability stage";
            } else {
                Stopwatch sw;
                SuiteConfig sc;
                sc.time_budget_s = budget; sc.run_sustained = opt.capability_sustained; sc.sysroot = opt.sysroot; sc.verbose = false;
                p.cv = CapabilitySuite(sc).run(p.fp);
                p.have_capability = p.cv.compute_best_gflops() > 0;
                if (!p.cv.stopped_reason.empty()) {
                    const bool thermal = p.cv.stopped_reason.find("thermal") != std::string::npos;
                    record("capability", "stopped", sw.elapsed(), p.cv.stopped_reason);       // the guard working is a stop, not a failure
                    if (thermal) p.stopped_reason = p.cv.stopped_reason;
                } else {
                    record("capability", p.have_capability ? "ok" : "failed", sw.elapsed(),
                           p.have_capability ? "" : "no compute rate was measured");
                }
            }
        }
        const bool thermal_stop = p.stopped_reason.find("thermal") != std::string::npos;

        // 4. validation of the workload model against real ops
        if (!opt.run_probe) record("validation", "skipped", 0, "disabled by options");
        else if (!p.have_capability) record("validation", "skipped", 0, thermal_stop ? "device too hot" : "no capability data to validate");
        else if (thermal_stop) record("validation", "skipped", 0, "device too hot");
        else if (std::min(opt.probe_budget_s, remaining()) < opt.min_stage_s) record("validation", "skipped", 0, "not enough time budget left");
        else {
            Stopwatch sw;
            wl::ProbeConfig pc;
            pc.time_budget_s = std::min(opt.probe_budget_s, remaining());
            pc.size_scale = opt.probe_size_scale; pc.sysroot = opt.sysroot; pc.guard.sysroot = opt.sysroot;
            p.probe = wl::run_probe(p.fp, p.cv, pc);
            p.have_probe = p.probe.usable > 0;
            const std::string note = std::to_string(p.probe.usable) + " cases usable; leave-one-out error " +
                                     std::to_string((int)std::lround(p.probe.overall_loo_error * 100)) + "%" +
                                     (p.probe.stopped_reason.empty() ? "" : "; stopped: " + p.probe.stopped_reason);
            record("validation", !p.have_probe ? "failed" : p.probe.stopped_reason.empty() ? "ok" : "stopped", sw.elapsed(), note);
            if (p.probe.stopped_reason.find("thermal") != std::string::npos) p.stopped_reason = p.probe.stopped_reason;
        }

        p.total_seconds = total.elapsed();
        return p;
    }

    // Advice for one algorithm, using what the profile learned: the calibration, the measured overhead factor.
    adv::Advice advise(const Profile& p, const wl::Algorithm& alg, const wl::Config& base, const adv::Constraints& cons = {},
                       adv::AdvisorOptions opt = {}, const adv::TrialFn& trial = nullptr) const {
        opt.mem_overhead_factor = p.mem_overhead_factor;
        opt.error_floor = std::max(opt.error_floor, noise_floor(p));
        adv::Advice a = adv::advise(alg, base, p.fp, p.cv, p.calibration(), cons, opt, trial);
        if (!p.have_capability) a.notes.push_back("the profile has no capability data" + (p.stopped_reason.empty() ? std::string() : " (" + p.stopped_reason + ")"));
        if (!p.mem_factor_measured) a.notes.push_back("memory overhead factor is the default " + std::to_string(p.mem_overhead_factor).substr(0, 4) + ", not measured on this device");
        return a;
    }

    // The advised range must not be tighter than this device's own repeatability. Standard deviation of ONE timed
    // step, estimated from the measured spread: (p90 - p10) / median is 2.56 sigma for a normal distribution, so
    // sigma_rel ~= spread / 2.56. Uses the median spread of the validation cases (what real ops did), else the
    // capability suite's worst-case noise. 0 when nothing was measured.
    static double noise_floor(const Profile& p) {
        constexpr double kSpreadPerSigma = 2.56;
        std::vector<double> sp;
        for (const auto& r : p.probe.results) if (r.ok) sp.push_back(r.spread);
        if (!sp.empty()) return median(sp) / kSpreadPerSigma;
        return p.have_capability ? p.cv.noise / kSpreadPerSigma : 0.0;
    }

    static std::string to_json(const Profile& p, const adv::Advice* advice = nullptr) {
        Json j;
        j.begin_object();
        j.str("schema", "tensorf.profile/2").str("timestamp", detail::utc_timestamp()).num("total_seconds", p.total_seconds)
         .str("stopped_reason", p.stopped_reason).num("memory_overhead_factor", p.mem_overhead_factor)
         .boolean("memory_overhead_measured", p.mem_factor_measured);
        j.begin_array("stages");
        for (const auto& s : p.stages) j.begin_object().str("name", s.name).str("status", s.status).num("seconds", s.seconds).str("note", s.note).end_object();
        j.end_array();
        ::tensorf::prof::to_json(j, p.fp);
        if (p.have_capability) ::tensorf::prof::to_json(j, p.cv);
        if (p.have_probe) wl::to_json(j, p.probe);
        if (p.have_memory) mem::to_json(j, p.memory);
        if (advice) adv::to_json(j, *advice);
        j.end_object();
        return j.finish();
    }

    static void print_summary(const Profile& p) {
        std::printf("\n══════════════════════════════════════════════════════\n  TensorF profile  (%.1f s)\n══════════════════════════════════════════════════════\n", p.total_seconds);
        std::printf("\n[ Device ]\n  %s  (%s)\n  %d logical / %d physical cores, %zu cluster(s)\n  SIMD: %s\n  cache: L1d %llu KB, L2 %llu KB, L3 %llu MB\n  memory: %llu MB total, safe budget %llu MB\n",
                    p.fp.model_name.c_str(), p.fp.microarch.c_str(), p.fp.logical_cpus, p.fp.physical_cores, p.fp.clusters.size(), p.fp.simd.summary().c_str(),
                    (unsigned long long)(p.fp.l1d_bytes() >> 10), (unsigned long long)(p.fp.l2_bytes() >> 10), (unsigned long long)(p.fp.l3_bytes() >> 20),
                    (unsigned long long)(p.fp.memory.total_bytes >> 20), (unsigned long long)(p.fp.memory.safe_budget_bytes >> 20));
        if (p.have_capability)
            std::printf("\n[ Capability ]\n  f32 GEMM %.1f GFLOP/s (skinny %.1f), int8 %.1f GOP/s\n  bandwidth: L1 %.0f / L2 %.0f / L3 %.0f / DRAM %.1f GB/s\n  latency: L1 %.1f / L2 %.1f / L3 %.1f / DRAM %.0f ns\n  gather %.2f GB/s, per-op %.0f ns, noise %.0f%%\n",
                        p.cv.f32_gflops_best, p.cv.f32_gflops_skinny, p.cv.i8_gops, p.cv.bw_l1_gbs, p.cv.bw_l2_gbs, p.cv.bw_l3_gbs, p.cv.bw_dram_gbs,
                        p.cv.lat_l1_ns, p.cv.lat_l2_ns, p.cv.lat_l3_ns, p.cv.lat_dram_ns, p.cv.gather_gbs, p.cv.tiny_op_ns, p.cv.noise * 100);
        if (p.have_capability && p.cv.fw.usable())
            std::printf("  framework (%s): f32 %.1f, skinny %.1f GFLOP/s; BLAS threads %s (x%.2f)\n", p.cv.fw.backend.c_str(), p.cv.fw.best_f32(),
                        p.cv.fw.gflops("f32_skinny"), p.cv.fw.threads_effective ? "help" : "do not help", p.cv.fw.max_speedup);
        if (p.have_probe) wl::print_report(p.probe);
        if (p.have_memory) std::printf("\n[ Memory ]\n  overhead factor %s%.2f   (tensor growth %.1f MB, RSS growth %.1f MB)\n", p.mem_factor_measured ? "" : "(default) ", p.mem_overhead_factor,
                                       p.memory.tensor_growth() / 1048576.0, p.memory.rss_growth() / 1048576.0);
        std::printf("\n[ Stages ]\n");
        for (const auto& s : p.stages) std::printf("  %-11s %-8s %6.2fs  %s\n", s.name.c_str(), s.status.c_str(), s.seconds, s.note.c_str());
        if (!p.stopped_reason.empty()) std::printf("  STOPPED: %s\n", p.stopped_reason.c_str());
    }
};

}  // namespace tensorf::prof
