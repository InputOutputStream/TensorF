#pragma once

/*
 * TensorAccounting.hpp
 * ====================
 * Memory accounting for any TensorF algorithm on any OS (design doc, section 2).
 *
 *   1. ALLOCATION HOOK     live + peak bytes by category, caught at allocation time, so transient
 *                          peaks (a backward pass, a transpose copy) are seen, not just steady state.
 *   2. GRAPH CENSUS        exact bytes by category for everything reachable from an autograd root.
 *                          No hook, no tagging, works on every platform.
 *   3. PROCESS PEAK        VmHWM (kernel-tracked peak RSS) instead of sampling, so a 5 ms spike is
 *                          not missed; the ratio of peak RSS to tensor peak is the measured
 *                          overhead factor that replaces WorkloadModel's guessed 1.25.
 *   4. measure_memory()    runs a step and returns all of the above for it.
 *
 * Why the hook is a global operator new replacement:
 *   Matrix stores its data in a plain std::vector<T>, and that type appears in dozens of signatures
 *   (vecmath::add returns one, binop takes one). A counting allocator on the container would mean
 *   rewriting Matrix. Every byte of every Matrix already passes through operator new, so replacing
 *   that function counts all of them with zero changes to the framework.
 *
 * Using the hook (opt-in, exactly ONE translation unit per program):
 *
 *     #define TENSORF_MEMTRACK_IMPLEMENT
 *     #include "tools/Profiler/TensorAccounting.hpp"
 *
 *   Replacement allocation functions must be non-inline and unique, which is why this is a macro and
 *   not automatic. Without it everything still compiles; hook_active() is false and measurements
 *   report that instead of returning zeros. Unsupported on Windows (#error if requested).
 *
 * Categories are assigned by RAII scope at allocation time:
 *
 *     { MemScope s(Cat::PARAMS);  w = make_tensor<float>(...); }   // bytes allocated here count as PARAMS
 *
 *   The category is stored in a 16-byte header in front of each block, so a free is credited to the
 *   category the block was ALLOCATED under, even if it is freed on another thread or under another
 *   scope. Bytes allocated outside any scope count as OTHER. Threads start in OTHER: a worker pool
 *   should open its own MemScope.
 *
 * Costs and limits, stated plainly:
 *   - 16 bytes of header per allocation (32-byte aligned blocks: still 16, plus alignment padding).
 *     Fine for a profiling build; do not ship the hook enabled.
 *   - Counters are relaxed atomics. live is exact; peak is exact single-threaded and may under-report
 *     by a few bytes under contention (a lower bound).
 *   - Only operator new is seen. Memory from malloc/mmap/BLAS internals is invisible to the hook but
 *     is part of RSS; that gap is exactly what the overhead factor measures.
 *   - A pointer that did not come from this operator new but is passed to operator delete (a foreign
 *     library with its own new) is detected by a magic word and freed with free(); counted in
 *     foreign_frees. It cannot be told apart perfectly: this is a diagnostic build, not production.
 *   - new_handler is not consulted on allocation failure (throws bad_alloc / returns nullptr directly).
 *   - Peak RSS includes file-backed pages (code, mmap'd weights) touched during the window, so the
 *     factor can read high for workloads that load a lot of code or mmap a model.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <string>
#include <unordered_set>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#include "ProfilerCommon.hpp"
#include "DataStructures/Tensor.hpp"

namespace tensorf::prof::mem {

// ─── Categories ──────────────────────────────────────────────────────────────

enum class Cat : uint8_t { PARAMS = 0, GRADS, OPTIMIZER, ACTIVATIONS, SCRATCH, OTHER, COUNT };
constexpr int kCats = (int)Cat::COUNT;

inline const char* to_string(Cat c) {
    switch (c) {
        case Cat::PARAMS: return "params";       case Cat::GRADS: return "grads";
        case Cat::OPTIMIZER: return "optimizer"; case Cat::ACTIVATIONS: return "activations";
        case Cat::SCRATCH: return "scratch";     case Cat::OTHER: return "other";
        default: return "?";
    }
}

struct CatBytes {
    std::array<int64_t, kCats> v{};
    int64_t& operator[](Cat c) { return v[(size_t)c]; }
    int64_t  operator[](Cat c) const { return v[(size_t)c]; }
    int64_t total() const { int64_t t = 0; for (int64_t x : v) t += x; return t; }
    int64_t tagged() const { return total() - v[(size_t)Cat::OTHER]; }    // everything that was put in a real category
    int64_t persistent() const { return v[(size_t)Cat::PARAMS] + v[(size_t)Cat::GRADS] + v[(size_t)Cat::OPTIMIZER]; }
};

// ─── Global counters (constant-initialised: safe to touch before main) ───────

namespace detail {

struct State {
    std::atomic<int64_t>  live[kCats];
    std::atomic<int64_t>  peak[kCats];
    std::atomic<int64_t>  live_total;
    std::atomic<int64_t>  peak_total;
    std::atomic<uint64_t> allocs, frees, foreign_frees, bytes_cumulative;
};

inline constinit State g_state{};
inline constinit thread_local Cat t_cat = Cat::OTHER;

inline void bump_peak(std::atomic<int64_t>& peak, int64_t v) {
    int64_t p = peak.load(std::memory_order_relaxed);
    while (v > p && !peak.compare_exchange_weak(p, v, std::memory_order_relaxed)) {}
}

inline void record_alloc(size_t bytes, Cat c) {
    const int64_t b = (int64_t)bytes;
    const int64_t l = g_state.live[(int)c].fetch_add(b, std::memory_order_relaxed) + b;
    bump_peak(g_state.peak[(int)c], l);
    const int64_t t = g_state.live_total.fetch_add(b, std::memory_order_relaxed) + b;
    bump_peak(g_state.peak_total, t);
    g_state.allocs.fetch_add(1, std::memory_order_relaxed);
    g_state.bytes_cumulative.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
}

inline void record_free(size_t bytes, Cat c) {
    const int64_t b = (int64_t)bytes;
    g_state.live[(int)c].fetch_sub(b, std::memory_order_relaxed);
    g_state.live_total.fetch_sub(b, std::memory_order_relaxed);
    g_state.frees.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace detail

// ─── Public hook API ─────────────────────────────────────────────────────────

struct Snapshot {
    CatBytes live, peak;                 // peak: per-category high-water mark since the last reset_peaks()
    int64_t  live_total = 0, peak_total = 0;
    uint64_t allocs = 0, frees = 0, foreign_frees = 0, bytes_cumulative = 0;
    uint64_t live_blocks() const { return allocs >= frees ? allocs - frees : 0; }
};

inline Snapshot snapshot() {
    Snapshot s;
    for (int c = 0; c < kCats; ++c) {
        s.live.v[(size_t)c] = detail::g_state.live[c].load(std::memory_order_relaxed);
        s.peak.v[(size_t)c] = detail::g_state.peak[c].load(std::memory_order_relaxed);
    }
    s.live_total = detail::g_state.live_total.load(std::memory_order_relaxed);
    s.peak_total = detail::g_state.peak_total.load(std::memory_order_relaxed);
    s.allocs = detail::g_state.allocs.load(std::memory_order_relaxed);
    s.frees = detail::g_state.frees.load(std::memory_order_relaxed);
    s.foreign_frees = detail::g_state.foreign_frees.load(std::memory_order_relaxed);
    s.bytes_cumulative = detail::g_state.bytes_cumulative.load(std::memory_order_relaxed);
    return s;
}

// Start a measurement window: peaks restart from what is live now. Live counts are never reset.
inline void reset_peaks() {
    for (int c = 0; c < kCats; ++c)
        detail::g_state.peak[c].store(detail::g_state.live[c].load(std::memory_order_relaxed), std::memory_order_relaxed);
    detail::g_state.peak_total.store(detail::g_state.live_total.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

// True if the replacement operator new is linked into this program (probed, not assumed).
inline bool hook_active() {
    const uint64_t before = detail::g_state.allocs.load(std::memory_order_relaxed);
    void* p = ::operator new(8);
    do_not_optimize(p);
    const bool counted = detail::g_state.allocs.load(std::memory_order_relaxed) > before;
    ::operator delete(p);
    return counted;
}

class MemScope {
public:
    explicit MemScope(Cat c) : prev_(detail::t_cat) { detail::t_cat = c; }
    ~MemScope() { detail::t_cat = prev_; }
    MemScope(const MemScope&) = delete;
    MemScope& operator=(const MemScope&) = delete;
private:
    Cat prev_;
};
inline Cat current_category() { return detail::t_cat; }

// ─── Process memory (VmRSS / VmHWM) ──────────────────────────────────────────

struct ProcessMem {
    bool     ok = false;
    uint64_t rss = 0, hwm = 0, vm_size = 0, vm_peak = 0, rss_anon = 0, rss_file = 0;   // bytes
    bool     hwm_from_rusage = false;     // /proc unreadable: hwm is getrusage's ru_maxrss, rss is unknown
};

// `root` redirects /proc for tests ("" = real filesystem).
inline ProcessMem read_process_mem(const std::string& root = "") {
    ProcessMem m;
    const std::string text = read_file(root + "/proc/self/status");
    auto field = [&](const char* key, uint64_t& out) { uint64_t kb = 0; if (meminfo_kb(text, key, kb)) { out = kb * 1024ull; return true; } return false; };
    if (!text.empty()) {
        const bool a = field("VmRSS", m.rss);
        field("VmHWM", m.hwm); field("VmSize", m.vm_size); field("VmPeak", m.vm_peak);
        field("RssAnon", m.rss_anon); field("RssFile", m.rss_file);
        m.ok = a;
        if (m.ok && m.hwm == 0) m.hwm = m.rss;        // very old kernels: no VmHWM line
        return m;
    }
#if defined(__unix__) || defined(__APPLE__)
    struct rusage ru;
    if (root.empty() && getrusage(RUSAGE_SELF, &ru) == 0) {
#if defined(__APPLE__)
        m.hwm = (uint64_t)ru.ru_maxrss;               // bytes on Apple
#else
        m.hwm = (uint64_t)ru.ru_maxrss * 1024ull;     // kilobytes on Linux / Android
#endif
        m.hwm_from_rusage = true;
        m.ok = true;
    }
#endif
    return m;
}

// Restart the kernel's peak-RSS tracking from the current RSS (Linux >= 4.0: write "5" to clear_refs).
// May be denied (Android 8+ app sandbox); then the peak is "since process start" and cannot isolate a window.
// Opens WITHOUT O_CREAT: if the node does not exist this must fail, never create a regular file and claim success.
inline bool reset_peak_rss(const std::string& root = "") {
#if defined(__unix__) || defined(__APPLE__)
    const int fd = ::open((root + "/proc/self/clear_refs").c_str(), O_WRONLY);
    if (fd < 0) return false;
    const bool wrote = ::write(fd, "5", 1) == 1;       // a denied or unsupported reset surfaces here (EINVAL / EACCES)
    const bool closed = ::close(fd) == 0;
    return wrote && closed;
#else
    (void)root;
    return false;
#endif
}

// ─── Graph census ────────────────────────────────────────────────────────────

// Exact bytes by category of every tensor reachable from `root` through backOp->inputs().
//   leaf that requires grad     val -> PARAMS         grad -> GRADS
//   leaf without grad (a batch) val -> ACTIVATIONS    (kept for backward, as the workload model counts it)
//   non-leaf                    val -> ACTIVATIONS    grad -> SCRATCH (activation gradients: transient backward buffers)
// Uses vector capacity, i.e. bytes actually reserved. Each tensor is counted once however often it is shared.
template <class T>
CatBytes census(const Tensor_t<T>& root) {
    CatBytes out;
    if (!root) return out;
    std::unordered_set<const Tensor<T>*> seen;
    std::vector<Tensor<T>*> stack{root.get()};
    while (!stack.empty()) {
        Tensor<T>* t = stack.back(); stack.pop_back();
        if (!t || !seen.insert(t).second) continue;
        const int64_t vb = (int64_t)(t->val.data.capacity() * sizeof(T));
        const int64_t gb = (int64_t)(t->grad.data.capacity() * sizeof(T));
        const bool leaf = (t->backOp == nullptr);
        if (leaf && t->requires_grad) { out[Cat::PARAMS] += vb; out[Cat::GRADS] += gb; }
        else if (leaf)                { out[Cat::ACTIVATIONS] += vb; out[Cat::SCRATCH] += gb; }
        else                          { out[Cat::ACTIVATIONS] += vb; out[Cat::SCRATCH] += gb; }
        if (t->backOp) for (const auto& in : t->backOp->inputs()) if (in) stack.push_back(in.get());
    }
    return out;
}

// ─── Measuring a step ────────────────────────────────────────────────────────

struct MeasureOptions {
    int warmup = 1;                      // untimed runs first, so lazy one-time allocations are not charged to the step
    int repeats = 1;                     // measured runs; the largest peak is kept
    std::string proc_root;               // "" = real /proc
    int64_t min_bytes_for_factor = 8ll << 20;   // below this, page granularity and allocator slack swamp the ratio
    double  default_factor = 1.25;       // used when no valid measurement exists
};

struct MemoryMeasurement {
    bool hook_active = false;
    bool rss_available = false;
    bool hwm_reset = false;              // the kernel peak could be restarted for this window
    bool rss_peak_valid = false;         // the window's peak RSS is known to belong to the window
    CatBytes live_before, live_after;
    CatBytes peak_by_cat;                // per-category peaks; they need not coincide in time, so their sum >= peak_total
    int64_t  live_before_total = 0, live_after_total = 0, peak_total = 0;
    uint64_t rss_before = 0, rss_peak = 0;
    uint64_t blocks_leaked = 0;          // blocks still live after the step minus before (a growing number = a leak or a cache)
    double   overhead_factor = std::numeric_limits<double>::quiet_NaN();   // (RSS growth) / (tensor-byte growth)
    bool     factor_valid = false;
    std::vector<std::string> notes;

    int64_t tensor_growth() const { return std::max<int64_t>(0, peak_total - live_before_total); }
    uint64_t rss_growth() const { return rss_peak > rss_before ? rss_peak - rss_before : 0; }
    // What to feed predict(): the measured factor if it is trustworthy, bounded to a plausible range, else the default.
    double recommended_factor(double fallback = 1.25) const {
        return factor_valid ? std::clamp(overhead_factor, 1.0, 4.0) : fallback;
    }
};

inline MemoryMeasurement measure_memory(const std::function<void()>& step, const MeasureOptions& opt = {}) {
    MemoryMeasurement m;
    m.hook_active = hook_active();
    if (!m.hook_active) m.notes.push_back("allocation hook not linked (define TENSORF_MEMTRACK_IMPLEMENT in one translation unit): tensor bytes unavailable");

    for (int i = 0; i < opt.warmup; ++i) step();

    const int reps = std::max(1, opt.repeats);
    bool first = true;
    for (int r = 0; r < reps; ++r) {
        const bool reset_ok = reset_peak_rss(opt.proc_root);
        const ProcessMem p0 = read_process_mem(opt.proc_root);
        reset_peaks();
        const Snapshot s0 = snapshot();

        step();

        const Snapshot s1 = snapshot();
        const ProcessMem p1 = read_process_mem(opt.proc_root);

        const int64_t growth = std::max<int64_t>(0, s1.peak_total - s0.live_total);
        if (first || growth > m.tensor_growth()) {
            m.live_before = s0.live; m.live_after = s1.live; m.peak_by_cat = s1.peak;
            m.live_before_total = s0.live_total; m.live_after_total = s1.live_total; m.peak_total = s1.peak_total;
            m.blocks_leaked = s1.live_blocks() > s0.live_blocks() ? s1.live_blocks() - s0.live_blocks() : 0;
            m.rss_available = p0.ok && p1.ok && !p1.hwm_from_rusage;
            m.hwm_reset = reset_ok;
            m.rss_before = p0.rss;
            m.rss_peak = p1.hwm;
            // The peak belongs to this window if the kernel counter was restarted, or if it rose during the window.
            m.rss_peak_valid = m.rss_available && (reset_ok || p1.hwm > p0.hwm);
            first = false;
        }
    }

    if (m.rss_available && !m.rss_peak_valid)
        m.notes.push_back("peak RSS cannot be isolated to this step (kernel peak counter not resettable and the process peak is older): overhead factor unavailable");
    if (!m.rss_available)
        m.notes.push_back("process memory unavailable (/proc not readable): overhead factor unavailable");
    if (m.hook_active && m.rss_peak_valid) {
        if (m.tensor_growth() >= opt.min_bytes_for_factor) {
            m.overhead_factor = (double)m.rss_growth() / (double)m.tensor_growth();
            m.factor_valid = std::isfinite(m.overhead_factor) && m.overhead_factor > 0;
            if (m.rss_growth() == 0)
                m.notes.push_back("peak RSS did not rise during a step that allocated " + std::to_string(m.tensor_growth() >> 20) +
                                  " MB: those pages were already resident (freed earlier but kept by the allocator). Measure in a fresh process, or before other heavy work");
        } else {
            m.notes.push_back("step allocated too little (" + std::to_string(m.tensor_growth() >> 10) + " KB) for a meaningful overhead ratio; use a larger workload");
        }
    }
    if (m.blocks_leaked > 0)
        m.notes.push_back(std::to_string(m.blocks_leaked) + " blocks were still live after the step (kept by the workload, or a leak)");
    return m;
}

// ─── JSON ────────────────────────────────────────────────────────────────────

inline void to_json(Json& j, const CatBytes& c, const char* key) {
    j.begin_object(key);
    for (int i = 0; i < kCats; ++i) j.num(to_string((Cat)i), (double)c.v[(size_t)i]);
    j.end_object();
}

inline void to_json(Json& j, const MemoryMeasurement& m, const char* key = "memory_measurement") {
    j.begin_object(key);
    j.boolean("hook_active", m.hook_active).boolean("rss_available", m.rss_available).boolean("hwm_reset", m.hwm_reset)
     .boolean("rss_peak_valid", m.rss_peak_valid).boolean("factor_valid", m.factor_valid)
     .num("overhead_factor", m.overhead_factor).num("recommended_factor", m.recommended_factor())
     .num("tensor_peak_bytes", (double)m.peak_total).num("tensor_growth_bytes", (double)m.tensor_growth())
     .num("rss_before", (double)m.rss_before).num("rss_peak", (double)m.rss_peak).num("rss_growth", (double)m.rss_growth())
     .uinteger("blocks_leaked", m.blocks_leaked);
    to_json(j, m.live_before, "live_before");
    to_json(j, m.peak_by_cat, "peak_by_category");
    j.begin_array("notes");
    for (const auto& n : m.notes) { j.begin_object().str("note", n).end_object(); }
    j.end_array();
    j.end_object();
}

}  // namespace tensorf::prof::mem

// ═════════════════════════════════════════════════════════════════════════════
// The hook itself. Define TENSORF_MEMTRACK_IMPLEMENT in exactly one translation unit.
// ═════════════════════════════════════════════════════════════════════════════
#if defined(TENSORF_MEMTRACK_IMPLEMENT)

#if defined(_WIN32)
#error "TENSORF_MEMTRACK_IMPLEMENT is not supported on Windows"
#endif

namespace tensorf::prof::mem::detail {

constexpr uint64_t kMagic  = 0x7E450F4D454D5452ull;
constexpr size_t   kHdr    = 16;                       // keeps default (16-byte) alignment of the user pointer
constexpr uint64_t kSizeMask = (1ull << 56) - 1;

// Block layout: [ raw ... | 16-byte header | user bytes ]
//   header word 0: size (56 bits) | category (8 bits)
//   header word 1: kMagic ^ user pointer     (so a stale or foreign header does not match by accident)
// Over-aligned blocks put the user pointer `align` bytes after the raw start; the header sits right before it.
inline void* hook_alloc(size_t size, size_t align) noexcept {
    if (size > kSizeMask - 4096) return nullptr;
    char* user = nullptr;
    if (align <= 16) {
        char* raw = static_cast<char*>(std::malloc(size + kHdr));
        if (!raw) return nullptr;
        user = raw + kHdr;
    } else {
        void* raw = nullptr;
        if (posix_memalign(&raw, align, size + align) != 0) return nullptr;
        user = static_cast<char*>(raw) + align;
    }
    const Cat c = t_cat;
    uint64_t hdr[2] = {(uint64_t)size | ((uint64_t)c << 56), kMagic ^ (uint64_t)(uintptr_t)user};
    std::memcpy(user - kHdr, hdr, kHdr);
    record_alloc(size, c);
    return user;
}

inline void hook_free(void* p, size_t align) noexcept {
    if (!p) return;
    char* user = static_cast<char*>(p);
    uint64_t hdr[2];
    std::memcpy(hdr, user - kHdr, kHdr);
    if (hdr[1] != (kMagic ^ (uint64_t)(uintptr_t)user)) {      // not ours
        g_state.foreign_frees.fetch_add(1, std::memory_order_relaxed);
        std::free(p);
        return;
    }
    record_free((size_t)(hdr[0] & kSizeMask), (Cat)(hdr[0] >> 56));
    const uint64_t dead[2] = {0, 0};
    std::memcpy(user - kHdr, dead, kHdr);                      // a double free no longer matches the magic
    std::free(align <= 16 ? static_cast<void*>(user - kHdr) : static_cast<void*>(user - align));
}

inline void* hook_new(size_t n, size_t align) {
    void* p = hook_alloc(n, align);
    if (!p) throw std::bad_alloc();
    return p;
}

}  // namespace tensorf::prof::mem::detail

namespace tpm = tensorf::prof::mem::detail;

void* operator new(std::size_t n)                                         { return tpm::hook_new(n, 16); }
void* operator new[](std::size_t n)                                       { return tpm::hook_new(n, 16); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept         { return tpm::hook_alloc(n, 16); }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept       { return tpm::hook_alloc(n, 16); }
void* operator new(std::size_t n, std::align_val_t a)                     { return tpm::hook_new(n, (size_t)a); }
void* operator new[](std::size_t n, std::align_val_t a)                   { return tpm::hook_new(n, (size_t)a); }
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept   { return tpm::hook_alloc(n, (size_t)a); }
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept { return tpm::hook_alloc(n, (size_t)a); }

void operator delete(void* p) noexcept                                    { tpm::hook_free(p, 16); }
void operator delete[](void* p) noexcept                                  { tpm::hook_free(p, 16); }
void operator delete(void* p, std::size_t) noexcept                       { tpm::hook_free(p, 16); }
void operator delete[](void* p, std::size_t) noexcept                     { tpm::hook_free(p, 16); }
void operator delete(void* p, const std::nothrow_t&) noexcept             { tpm::hook_free(p, 16); }
void operator delete[](void* p, const std::nothrow_t&) noexcept           { tpm::hook_free(p, 16); }
void operator delete(void* p, std::align_val_t a) noexcept                { tpm::hook_free(p, (size_t)a); }
void operator delete[](void* p, std::align_val_t a) noexcept              { tpm::hook_free(p, (size_t)a); }
void operator delete(void* p, std::size_t, std::align_val_t a) noexcept   { tpm::hook_free(p, (size_t)a); }
void operator delete[](void* p, std::size_t, std::align_val_t a) noexcept { tpm::hook_free(p, (size_t)a); }
void operator delete(void* p, std::align_val_t a, const std::nothrow_t&) noexcept   { tpm::hook_free(p, (size_t)a); }
void operator delete[](void* p, std::align_val_t a, const std::nothrow_t&) noexcept { tpm::hook_free(p, (size_t)a); }

#endif  // TENSORF_MEMTRACK_IMPLEMENT
