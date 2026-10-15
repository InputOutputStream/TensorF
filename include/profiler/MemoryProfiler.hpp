// MemoryProfiler.hpp
//
// Memory knowledge for a background training/simulation workload, in four
// independent parts that do not assume any particular algorithm:
//
//   1. MemoryLedger / LedgerAllocator / PhaseRecorder
//        Exact accounting of tensor storage, split by category (parameters,
//        gradients, optimizer state, activations, scratch, data) with live and
//        peak bytes, and peak per phase (forward, backward, update, ...).
//        Works for any algorithm; the algorithm only decides which category
//        scope it allocates under.
//
//   2. ProcessMemory / RssSampler / measure_allocator_overhead
//        What the OS says the process costs (RSS, peak RSS, swap, PSS where
//        readable, iOS phys_footprint). Used to turn "tensor bytes" into "real
//        bytes": allocator and fragmentation overhead is measured on the device,
//        not guessed.
//
//   3. plan_budget / MemoryBudget
//        How much memory the workload may ADD right now without disturbing the
//        owner: available memory, minus a reserve that keeps the system away
//        from its low-memory-killer thresholds, scaled by run mode (background
//        or foreground) and by memory pressure (PSI), capped by cgroup limits.
//
//   4. MemoryGuard
//        The runtime check the governor calls between steps: Ok / Shrink / Stop,
//        with hysteresis, from fresh available-memory and pressure readings.
//
// Plus the types the per-algorithm profilers use to describe their needs
// (Dtype, tensor_bytes, MemoryEstimate, OptimizerKind) and check_fit().
//
// Integrating the ledger with Matrix (one line, not done here because Matrix.hpp
// is yours):
//     template <class T> using TrackedVector = tensorf::profile::TrackedVector<T>;
//     // in Matrix:   TrackedVector<T> data;      instead of   std::vector<T> data;
//   Parameters are created under   ScopedCategory c(MemCat::Parameters);
//   Tensor::backward allocates gradients under MemCat::Gradients, and so on.
//   If Matrix constructors take std::vector<T> by value, those signatures must
//   take TrackedVector<T> (or a template) so storage is moved, not copied.
//   Without the integration everything else here still works, using process
//   numbers instead of categories.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "HardwareFingerprint.hpp"

#if defined(__APPLE__)
  #include <mach/mach.h>
#elif defined(_WIN32)
  #include <psapi.h>
#endif

namespace tensorf::profile {

// ───────────────────────────── categories and the ledger ─────────────────────────────

enum class MemCat : uint8_t { Parameters = 0, Gradients, OptimizerState, Activations, Scratch, Data, Other, Count };
inline constexpr size_t kMemCats = size_t(MemCat::Count);

inline const char* to_string(MemCat c) {
    switch (c) {
        case MemCat::Parameters: return "parameters";
        case MemCat::Gradients: return "gradients";
        case MemCat::OptimizerState: return "optimizer_state";
        case MemCat::Activations: return "activations";
        case MemCat::Scratch: return "scratch";
        case MemCat::Data: return "data";
        default: return "other";
    }
}

namespace detail {
inline MemCat& tls_category() { thread_local MemCat c = MemCat::Other; return c; }
inline void bump_max(std::atomic<uint64_t>& peak, uint64_t v) {
    uint64_t p = peak.load(std::memory_order_relaxed);
    while (v > p && !peak.compare_exchange_weak(p, v, std::memory_order_relaxed)) {}
}
}  // namespace detail

inline MemCat current_category() { return detail::tls_category(); }

// Everything allocated on this thread while the scope lives is attributed to `c`.
class ScopedCategory {
public:
    explicit ScopedCategory(MemCat c) : prev_(detail::tls_category()) { detail::tls_category() = c; }
    ~ScopedCategory() { detail::tls_category() = prev_; }
    ScopedCategory(const ScopedCategory&) = delete;
    ScopedCategory& operator=(const ScopedCategory&) = delete;
private:
    MemCat prev_;
};

struct LedgerSnapshot {
    std::array<uint64_t, kMemCats> live{};
    std::array<uint64_t, kMemCats> peak{};      // since start or last reset
    uint64_t live_total = 0;
    uint64_t peak_total = 0;
    uint64_t allocs = 0;
    uint64_t frees = 0;
};

class MemoryLedger {
public:
    static MemoryLedger& global() { static MemoryLedger l; return l; }

    void on_alloc(MemCat c, uint64_t bytes) noexcept {
        const size_t i = size_t(c);
        const uint64_t l = live_[i].fetch_add(bytes, std::memory_order_relaxed) + bytes;
        detail::bump_max(peak_[i], l);
        detail::bump_max(phase_peak_[i], l);
        const uint64_t t = live_total_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        detail::bump_max(peak_total_, t);
        detail::bump_max(phase_peak_total_, t);
        allocs_.fetch_add(1, std::memory_order_relaxed);
    }
    void on_free(MemCat c, uint64_t bytes) noexcept {
        live_[size_t(c)].fetch_sub(bytes, std::memory_order_relaxed);
        live_total_.fetch_sub(bytes, std::memory_order_relaxed);
        frees_.fetch_add(1, std::memory_order_relaxed);
    }
    LedgerSnapshot snapshot() const {
        LedgerSnapshot s;
        for (size_t i = 0; i < kMemCats; ++i) { s.live[i] = live_[i].load(); s.peak[i] = peak_[i].load(); }
        s.live_total = live_total_.load(); s.peak_total = peak_total_.load();
        s.allocs = allocs_.load(); s.frees = frees_.load();
        return s;
    }
    // Peaks restart from the current live bytes.
    void reset_peaks() {
        for (size_t i = 0; i < kMemCats; ++i) peak_[i].store(live_[i].load());
        peak_total_.store(live_total_.load());
        reset_phase_peaks();
    }
    void reset_phase_peaks() {
        for (size_t i = 0; i < kMemCats; ++i) phase_peak_[i].store(live_[i].load());
        phase_peak_total_.store(live_total_.load());
    }
    LedgerSnapshot phase_snapshot() const {          // `peak` fields hold the phase peaks
        LedgerSnapshot s = snapshot();
        for (size_t i = 0; i < kMemCats; ++i) s.peak[i] = phase_peak_[i].load();
        s.peak_total = phase_peak_total_.load();
        return s;
    }

private:
    std::array<std::atomic<uint64_t>, kMemCats> live_{}, peak_{}, phase_peak_{};
    std::atomic<uint64_t> live_total_{0}, peak_total_{0}, phase_peak_total_{0}, allocs_{0}, frees_{0};
};

// std-compatible allocator that reports to the ledger. The category is fixed
// when the allocator is constructed (from the current ScopedCategory) and
// travels with the storage on move and swap; a copy takes the CURRENT category.
template <class T>
class LedgerAllocator {
public:
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    LedgerAllocator() noexcept : cat_(current_category()) {}
    explicit LedgerAllocator(MemCat c) noexcept : cat_(c) {}
    template <class U>
    LedgerAllocator(const LedgerAllocator<U>& o) noexcept : cat_(o.category()) {}

    T* allocate(size_t n) {
        T* p = std::allocator<T>{}.allocate(n);
        MemoryLedger::global().on_alloc(cat_, uint64_t(n) * sizeof(T));
        return p;
    }
    void deallocate(T* p, size_t n) noexcept {
        MemoryLedger::global().on_free(cat_, uint64_t(n) * sizeof(T));
        std::allocator<T>{}.deallocate(p, n);
    }
    LedgerAllocator select_on_container_copy_construction() const { return LedgerAllocator(current_category()); }
    MemCat category() const noexcept { return cat_; }

    template <class U> bool operator==(const LedgerAllocator<U>& o) const noexcept { return cat_ == o.category(); }
    template <class U> bool operator!=(const LedgerAllocator<U>& o) const noexcept { return cat_ != o.category(); }
private:
    MemCat cat_;
};

template <class T> using TrackedVector = std::vector<T, LedgerAllocator<T>>;

struct PhaseRecord {
    std::string name;
    std::array<uint64_t, kMemCats> peak{};     // per-category peak inside the phase
    uint64_t peak_total = 0;
    uint64_t live_at_start = 0;
    uint64_t live_at_end = 0;
    double seconds = 0;
};

// Records the peak inside each named phase. Phases do not nest: begin one at a time.
class PhaseRecorder {
public:
    class Scope {
    public:
        Scope(PhaseRecorder& r, std::string name) : r_(r), name_(std::move(name)), t0_(std::chrono::steady_clock::now()) {
            auto& l = MemoryLedger::global();
            start_live_ = l.snapshot().live_total;
            l.reset_phase_peaks();
        }
        ~Scope() {
            auto s = MemoryLedger::global().phase_snapshot();
            PhaseRecord pr;
            pr.name = name_;
            pr.peak = s.peak; pr.peak_total = s.peak_total;
            pr.live_at_start = start_live_; pr.live_at_end = s.live_total;
            pr.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
            std::lock_guard<std::mutex> g(r_.mu_);
            r_.records_.push_back(std::move(pr));
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
    private:
        PhaseRecorder& r_;
        std::string name_;
        std::chrono::steady_clock::time_point t0_;
        uint64_t start_live_ = 0;
    };
    Scope phase(std::string name) { return Scope(*this, std::move(name)); }
    std::vector<PhaseRecord> records() const { std::lock_guard<std::mutex> g(mu_); return records_; }
    // Peak over all recorded phases of one name (steady-state iterations repeat phases).
    uint64_t max_peak(const std::string& name) const {
        std::lock_guard<std::mutex> g(mu_);
        uint64_t m = 0;
        for (const auto& r : records_) if (r.name == name) m = std::max(m, r.peak_total);
        return m;
    }
private:
    mutable std::mutex mu_;
    std::vector<PhaseRecord> records_;
};

// ───────────────────────────── process memory ─────────────────────────────

struct ProcessMemory {
    bool ok = false;
    uint64_t rss = 0;                       // resident set now
    uint64_t rss_peak = 0;                  // high-water mark since process start
    uint64_t swap = 0;
    std::optional<uint64_t> pss;            // proportional set size (Linux smaps_rollup)
    std::optional<uint64_t> footprint;      // Apple phys_footprint: the number jetsam acts on
};

namespace detail {
// /proc/self/status text -> ProcessMemory (values are kB).
inline ProcessMemory parse_proc_status(const std::string& text) {
    ProcessMemory pm;
    bool have_rss = false;
    for (const auto& line : split(text, '\n')) {
        auto c = line.find(':');
        if (c == std::string::npos) continue;
        const std::string k = line.substr(0, c);
        auto toks = split_ws(line.substr(c + 1));
        if (toks.empty()) continue;
        auto v = parse_u64(toks[0]);
        if (!v) continue;
        const uint64_t bytes = *v * 1024;
        if (k == "VmRSS") { pm.rss = bytes; have_rss = true; }
        else if (k == "VmHWM") pm.rss_peak = bytes;
        else if (k == "VmSwap") pm.swap = bytes;
    }
    pm.ok = have_rss;
    if (pm.rss_peak < pm.rss) pm.rss_peak = pm.rss;
    return pm;
}
inline std::optional<uint64_t> parse_smaps_rollup_pss(const std::string& text) {
    for (const auto& line : split(text, '\n'))
        if (starts_with(line, "Pss:")) {
            auto t = split_ws(line.substr(4));
            if (!t.empty()) if (auto v = parse_u64(t[0])) return *v * 1024;
        }
    return std::nullopt;
}
}  // namespace detail

inline ProcessMemory read_process_memory_linux(FsSource& fs, bool want_pss = false) {
    ProcessMemory pm;
    if (auto t = fs.read("/proc/self/status")) pm = detail::parse_proc_status(*t);
    if (want_pss)
        if (auto r = fs.read("/proc/self/smaps_rollup")) pm.pss = detail::parse_smaps_rollup_pss(*r);   // kernel 4.14+; absent on Android 7 era kernels
    return pm;
}

inline ProcessMemory read_process_memory(bool want_pss = false) {
#if defined(__APPLE__)
    ProcessMemory pm;
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        pm.ok = true; pm.rss = info.resident_size; pm.rss_peak = info.resident_size_peak; pm.footprint = info.phys_footprint;
    }
    (void)want_pss;
    return pm;
#elif defined(_WIN32)
    ProcessMemory pm;
    PROCESS_MEMORY_COUNTERS_EX c{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&c), sizeof c)) {
        pm.ok = true; pm.rss = c.WorkingSetSize; pm.rss_peak = c.PeakWorkingSetSize; pm.footprint = c.PrivateUsage;
    }
    (void)want_pss;
    return pm;
#else
    static RealFs fs;
    return read_process_memory_linux(fs, want_pss);
#endif
}

// Samples RSS on a background thread so short spikes between phase boundaries are seen.
// (VmHWM catches them too, but it is a lifetime maximum and cannot be reset portably.)
class RssSampler {
public:
    using Fn = std::function<std::optional<uint64_t>()>;
    RssSampler(Fn fn, std::chrono::milliseconds interval) : fn_(std::move(fn)), interval_(interval) {}
    ~RssSampler() { stop(); }
    void start() {
        if (th_.joinable()) return;
        stop_ = false;
        if (auto v = fn_()) peak_.store(*v);
        th_ = std::thread([this] {
            std::unique_lock<std::mutex> lk(mu_);
            while (!stop_) {
                cv_.wait_for(lk, interval_, [this] { return stop_; });
                if (stop_) break;
                lk.unlock();
                if (auto v = fn_()) detail::bump_max(peak_, *v);
                lk.lock();
            }
        });
    }
    // Stops sampling and returns the largest value seen (one last sample is taken).
    uint64_t stop() {
        if (th_.joinable()) {
            { std::lock_guard<std::mutex> g(mu_); stop_ = true; }
            cv_.notify_all();
            th_.join();
            if (auto v = fn_()) detail::bump_max(peak_, *v);
        }
        return peak_.load();
    }
    uint64_t peak() const { return peak_.load(); }
private:
    Fn fn_;
    std::chrono::milliseconds interval_;
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::atomic<uint64_t> peak_{0};
};

struct AllocatorOverhead {
    uint64_t requested = 0;          // bytes asked for
    uint64_t rss_delta = 0;          // bytes the process grew by
    double factor = 1.25;            // real / requested, >= 1; the default is used when not measured
    bool valid = false;              // false: could not be measured (e.g. heap pages were being reused)
};

// Allocates a spread of tensor-like sizes (256 B .. 1 MB), touches every page, and
// compares the growth of RSS with the bytes requested. This captures the allocator's
// size classes, headers and fragmentation on THIS device (bionic/scudo, glibc, ...).
inline AllocatorOverhead measure_allocator_overhead(const std::function<std::optional<uint64_t>()>& rss_fn,
                                                    uint64_t target_bytes = 24ull << 20) {
    AllocatorOverhead r;
    auto before = rss_fn();
    if (!before) return r;
    std::vector<void*> ptrs;
    uint64_t seed = 0xabcdef12345ull, requested = 0;
    while (requested < target_bytes) {
        seed ^= seed >> 12; seed ^= seed << 25; seed ^= seed >> 27;
        const double u = double((seed * 2685821657736338717ull) >> 11) / 9007199254740992.0;
        const size_t sz = size_t(256.0 * std::pow(2.0, u * 12.0));       // 256 B .. 1 MB, log-uniform
        void* p = std::malloc(sz);
        if (!p) break;
        for (size_t off = 0; off < sz; off += 4096) static_cast<volatile char*>(p)[off] = 1;
        static_cast<volatile char*>(p)[sz - 1] = 1;
        ptrs.push_back(p);
        requested += sz;
    }
    auto after = rss_fn();
    for (void* p : ptrs) std::free(p);
    if (!after || *after < *before) return r;
    r.requested = requested;
    r.rss_delta = *after - *before;
    r.valid = requested >= (4ull << 20) && double(r.rss_delta) >= 0.9 * double(requested);   // less growth than requested = freed heap was reused, the cost is hidden
    if (r.valid) r.factor = std::max(1.0, double(r.rss_delta) / double(requested));
    return r;
}

// ───────────────────────────── estimates: what an algorithm profiler hands over ─────────────────────────────

enum class Dtype { F32, F16, BF16, FP8, FP4, I32, I8 };

// Bytes for n elements. FP8/FP4 are block-quantised: one scale per `group` elements
// (Quantizer.hpp: group 32, absmax scale), FP4 packs two values per byte.
inline uint64_t tensor_bytes(Dtype d, uint64_t n, int group = 32, int scale_bytes = 4) {
    const uint64_t blocks = group > 0 ? (n + uint64_t(group) - 1) / uint64_t(group) : 0;
    switch (d) {
        case Dtype::F32: case Dtype::I32: return 4 * n;
        case Dtype::F16: case Dtype::BF16: return 2 * n;
        case Dtype::I8: return n;
        case Dtype::FP8: return n + blocks * uint64_t(scale_bytes);
        case Dtype::FP4: return (n + 1) / 2 + blocks * uint64_t(scale_bytes);
    }
    return 4 * n;
}

enum class OptimizerKind { SGD, Momentum, Adam };
inline uint64_t optimizer_state_bytes(OptimizerKind k, uint64_t n_params, Dtype state = Dtype::F32) {
    switch (k) {
        case OptimizerKind::SGD: return 0;
        case OptimizerKind::Momentum: return tensor_bytes(state, n_params);
        case OptimizerKind::Adam: return 2 * tensor_bytes(state, n_params);
    }
    return 0;
}

struct MemoryEstimate {
    std::array<uint64_t, kMemCats> bytes{};
    uint64_t& operator[](MemCat c) { return bytes[size_t(c)]; }
    uint64_t operator[](MemCat c) const { return bytes[size_t(c)]; }
    uint64_t total() const { uint64_t t = 0; for (auto b : bytes) t += b; return t; }
    MemoryEstimate& operator+=(const MemoryEstimate& o) { for (size_t i = 0; i < kMemCats; ++i) bytes[i] += o.bytes[i]; return *this; }
};

// ───────────────────────────── budget planning ─────────────────────────────

enum class RunMode { Background, Foreground };

struct BudgetPolicy {
    RunMode mode = RunMode::Background;
    double fraction = -1;                    // share of headroom to use; < 0 = by mode (bg 0.40, fg 0.65)
    double reserve_fraction_of_total = 0.08; // kept free for the rest of the system
    uint64_t min_reserve_bytes = 128ull << 20;
    double overhead_factor = 1.25;           // real/requested; replace with measure_allocator_overhead().factor
    double psi_soft = 10.0;                  // memory "some" stall %, avg10: halve the budget
    double psi_hard = 25.0;                  // above this: no budget
};

struct MemoryBudget {
    uint64_t available_now = 0;              // effective_available at planning time
    uint64_t reserve_bytes = 0;
    uint64_t headroom_bytes = 0;             // available - reserve
    double fraction = 0;                     // after mode and pressure scaling
    uint64_t max_additional_bytes = 0;       // real bytes the workload may add to the process
    double overhead_factor = 1.25;
    uint64_t tensor_budget_bytes = 0;        // requested tensor bytes that fit (max_additional / overhead)
    uint64_t max_total_process_bytes = 0;    // current RSS + max_additional
    bool pressured = false;
    std::vector<std::string> notes;
};

inline MemoryBudget plan_budget(const HardwareFingerprint& fp, const ProcessMemory& proc, const BudgetPolicy& pol = {}) {
    MemoryBudget b;
    b.overhead_factor = std::max(1.0, pol.overhead_factor);
    const MemoryInfo& m = fp.memory;
    b.available_now = m.effective_available;
    if (m.total == 0 || m.effective_available == 0) {
        b.notes.push_back("available memory unknown: budget is zero until it can be read");
        return b;
    }
    uint64_t reserve = std::max<uint64_t>(pol.min_reserve_bytes, uint64_t(double(m.total) * pol.reserve_fraction_of_total));
    if (m.lmk_min_free_bytes) {
        reserve = std::max(reserve, *m.lmk_min_free_bytes * 2);
        b.notes.push_back("reserve raised to stay clear of the low-memory-killer threshold");
    }
    b.reserve_bytes = reserve;
    b.headroom_bytes = m.effective_available > reserve ? m.effective_available - reserve : 0;

    double fraction = pol.fraction >= 0 ? pol.fraction : (pol.mode == RunMode::Background ? 0.40 : 0.65);
    fraction = std::clamp(fraction, 0.0, 1.0);
    if (m.psi_some_avg10) {
        if (*m.psi_some_avg10 >= pol.psi_hard) { fraction = 0; b.pressured = true; b.notes.push_back("memory pressure above hard limit: budget is zero"); }
        else if (*m.psi_some_avg10 >= pol.psi_soft) { fraction *= 0.5; b.pressured = true; b.notes.push_back("memory pressure: budget halved"); }
    }
    b.fraction = fraction;
    b.max_additional_bytes = uint64_t(double(b.headroom_bytes) * fraction);
    b.tensor_budget_bytes = uint64_t(double(b.max_additional_bytes) / b.overhead_factor);
    b.max_total_process_bytes = (proc.ok ? proc.rss : 0) + b.max_additional_bytes;
    if (m.cgroup_limit) b.notes.push_back("a container/cgroup memory limit applies");
    if (m.swap_is_zram) b.notes.push_back("swap is compressed RAM: it is not extra capacity");
    if (b.headroom_bytes == 0) b.notes.push_back("available memory is below the reserve: nothing can be added safely");
    return b;
}

struct FitResult {
    bool fits = false;
    uint64_t required_tensor_bytes = 0;
    uint64_t tensor_budget_bytes = 0;
    double utilization = 0;                  // required / budget (inf when budget is 0 and something is required)
    MemCat largest = MemCat::Other;          // category that costs the most
};

inline FitResult check_fit(const MemoryEstimate& e, const MemoryBudget& b) {
    FitResult r;
    r.required_tensor_bytes = e.total();
    r.tensor_budget_bytes = b.tensor_budget_bytes;
    r.fits = r.required_tensor_bytes <= r.tensor_budget_bytes;
    r.utilization = r.tensor_budget_bytes ? double(r.required_tensor_bytes) / double(r.tensor_budget_bytes)
                                          : (r.required_tensor_bytes ? INFINITY : 0.0);
    uint64_t best = 0;
    for (size_t i = 0; i < kMemCats; ++i) if (e.bytes[i] > best) { best = e.bytes[i]; r.largest = MemCat(i); }
    return r;
}

// ───────────────────────────── runtime guard ─────────────────────────────

enum class MemAction { Ok, Shrink, Stop };

struct MemState {
    uint64_t available = 0;                  // effective available bytes now
    std::optional<double> psi_some_avg10;
};

inline MemState read_mem_state() {
    MemState s;
#if defined(__APPLE__)
    s.available = collect_apple().memory.effective_available;
#elif defined(_WIN32)
    s.available = collect_windows().memory.effective_available;
#else
    static RealFs fs;
    MemoryInfo m = read_memory_linux(fs);
    s.available = m.effective_available;
    s.psi_some_avg10 = m.psi_some_avg10;
#endif
    return s;
}

struct GuardConfig {
    uint64_t reserve_bytes = 0;              // from MemoryBudget::reserve_bytes
    double shrink_factor = 1.5;              // Shrink when available < reserve * shrink_factor
    double psi_soft = 10.0, psi_hard = 25.0;
    double recover_margin = 0.15;            // must clear a threshold by this much to go back down
};

// Call between steps. Stop = we are eating the reserve or the system is stalling on
// memory: release or pause. Shrink = reduce footprint or batch. Hysteresis stops
// flapping around a threshold.
class MemoryGuard {
public:
    using Reader = std::function<MemState()>;
    MemoryGuard(GuardConfig cfg, Reader reader = read_mem_state) : cfg_(cfg), reader_(std::move(reader)) {}

    MemAction check() {
        const MemState s = reader_();
        const double res = double(cfg_.reserve_bytes);
        const double psi = s.psi_some_avg10.value_or(0.0);
        const double avail = double(s.available);
        const double up = 1.0 + cfg_.recover_margin;     // threshold multiplier needed to step down a level
        MemAction target = MemAction::Ok;
        if (avail < res || psi >= cfg_.psi_hard) target = MemAction::Stop;
        else if (avail < res * cfg_.shrink_factor || psi >= cfg_.psi_soft) target = MemAction::Shrink;
        // stepping down needs margin
        if (int(target) < int(last_)) {
            bool clear = true;
            if (last_ == MemAction::Stop) clear = avail >= res * up && psi < cfg_.psi_hard / up;
            if (target == MemAction::Ok && clear) clear = avail >= res * cfg_.shrink_factor * up && psi < cfg_.psi_soft / up;
            if (!clear) target = last_;
        }
        last_ = target;
        return target;
    }
    MemAction last() const { return last_; }

private:
    GuardConfig cfg_;
    Reader reader_;
    MemAction last_ = MemAction::Ok;
};

// ───────────────────────────── report ─────────────────────────────

struct MemoryReport {
    ProcessMemory process;
    LedgerSnapshot ledger;
    std::vector<PhaseRecord> phases;
    AllocatorOverhead overhead;
    MemoryBudget budget;

    std::string to_json() const {
        detail::JsonOut j;
        j.begin_obj();
        j.key("process"); j.begin_obj();
        j.kv("ok", process.ok); j.kv("rss", uint64_t(process.rss)); j.kv("rss_peak", uint64_t(process.rss_peak));
        j.kv("swap", uint64_t(process.swap)); j.kv("pss", process.pss); j.kv("footprint", process.footprint);
        j.end_obj();
        j.key("ledger"); j.begin_obj();
        j.kv("live_total", uint64_t(ledger.live_total)); j.kv("peak_total", uint64_t(ledger.peak_total));
        j.kv("allocs", uint64_t(ledger.allocs)); j.kv("frees", uint64_t(ledger.frees));
        j.key("by_category"); j.begin_obj();
        for (size_t i = 0; i < kMemCats; ++i) {
            j.key(to_string(MemCat(i))); j.begin_obj();
            j.kv("live", uint64_t(ledger.live[i])); j.kv("peak", uint64_t(ledger.peak[i]));
            j.end_obj();
        }
        j.end_obj();
        j.end_obj();
        j.key("phases"); j.begin_arr();
        for (const auto& p : phases) {
            j.begin_obj();
            j.kv("name", p.name); j.kv("peak_total", uint64_t(p.peak_total));
            j.kv("live_at_start", uint64_t(p.live_at_start)); j.kv("live_at_end", uint64_t(p.live_at_end));
            j.kv("seconds", p.seconds);
            j.end_obj();
        }
        j.end_arr();
        j.key("allocator_overhead"); j.begin_obj();
        j.kv("valid", overhead.valid); j.kv("factor", overhead.factor);
        j.kv("requested", uint64_t(overhead.requested)); j.kv("rss_delta", uint64_t(overhead.rss_delta));
        j.end_obj();
        j.key("budget"); j.begin_obj();
        j.kv("available_now", uint64_t(budget.available_now)); j.kv("reserve_bytes", uint64_t(budget.reserve_bytes));
        j.kv("headroom_bytes", uint64_t(budget.headroom_bytes)); j.kv("fraction", budget.fraction);
        j.kv("max_additional_bytes", uint64_t(budget.max_additional_bytes));
        j.kv("tensor_budget_bytes", uint64_t(budget.tensor_budget_bytes));
        j.kv("max_total_process_bytes", uint64_t(budget.max_total_process_bytes));
        j.kv("overhead_factor", budget.overhead_factor); j.kv("pressured", budget.pressured);
        j.key("notes"); j.begin_arr(); for (const auto& n : budget.notes) j.str(n); j.end_arr();
        j.end_obj();
        j.end_obj();
        return j.str();
    }
};

}  // namespace tensorf::profile
