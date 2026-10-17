/*
 * profiler_memory_test.cpp
 * ========================
 * Tests for TensorAccounting.hpp (step 3 of the profiler rewrite): the allocation hook, the graph
 * census, process peak (VmHWM), measure_memory(), and the validation of WorkloadModel's memory
 * estimates against what TensorF's autograd really allocates.
 *
 *   g++ -std=c++20 -O2 -Wall -Wextra -Iinclude -Iinclude/core -Iinclude/tools \
 *       tests/profiler_memory_test.cpp -o bin/profiler_memory_test -pthread
 *   ./bin/profiler_memory_test
 *
 * This translation unit defines TENSORF_MEMTRACK_IMPLEMENT, so it REPLACES global operator new/delete.
 * tests/profiler_memory_nohook_test.cpp covers the same API in a program where the hook is not linked.
 *
 * Sections:
 *   A  hook: exact counts, categories, scopes, threads, alignment, peaks, concurrency
 *   B  process memory: /proc parsing on fixtures and on this machine
 *   C  measure_memory: deterministic fixture arithmetic + a real allocation
 *   D  census: exact bytes by category for a real autograd graph
 *   E  model vs measurement: the memory estimates of WorkloadModel against real Tensor autograd
 */

#define TENSORF_MEMTRACK_IMPLEMENT
#include "profiler/tmp/TensorAccounting.hpp"
#include "profiler/tmp/WorkloadProbe.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>

using namespace tensorf::prof;
using namespace tensorf::prof::mem;
namespace wl = tensorf::prof::wl;
namespace fs = std::filesystem;

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define UNDER_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define UNDER_SANITIZER 1
#endif
#endif
#ifndef UNDER_SANITIZER
#define UNDER_SANITIZER 0
#endif

static int g_checks = 0, g_failed = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failed; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while (0)
#define CHECK_EQ(a, b) do { ++g_checks; auto _a=(a); auto _b=(b); if (!(_a == _b)) { ++g_failed; \
    std::printf("  FAIL %s:%d  %s == %s  (%lld vs %lld)\n", __FILE__, __LINE__, #a, #b, (long long)_a, (long long)_b); } } while (0)
#define CHECK_REL(a, b, rel) do { ++g_checks; double _a=(double)(a), _b=(double)(b); if (!(std::fabs(_a-_b) <= (rel) * std::fabs(_b))) { ++g_failed; \
    std::printf("  FAIL %s:%d  %s ~ %s within %g%%  (%g vs %g)\n", __FILE__, __LINE__, #a, #b, (rel) * 100.0, _a, _b); } } while (0)
#define SECTION(name) std::printf("[%s]\n", name)

static int64_t live(Cat c) { return snapshot().live[c]; }
static int64_t live_total() { return snapshot().live_total; }

class FakeRoot {
public:
    FakeRoot() { path_ = (fs::temp_directory_path() / ("tensorf_mem_" + std::to_string((long long)(now_s() * 1e6)))).string(); fs::create_directories(path_ + "/proc/self"); }
    ~FakeRoot() { std::error_code ec; fs::remove_all(path_, ec); }
    const std::string& root() const { return path_; }
    void status(uint64_t rss_kb, uint64_t hwm_kb, bool with_hwm = true) const {
        std::ofstream f(path_ + "/proc/self/status", std::ios::trunc);
        f << "Name:\tfake\nVmPeak:\t  900000 kB\nVmSize:\t  800000 kB\n";
        if (with_hwm) f << "VmHWM:\t  " << hwm_kb << " kB\n";
        f << "VmRSS:\t  " << rss_kb << " kB\nRssAnon:\t  " << rss_kb / 2 << " kB\nRssFile:\t  " << rss_kb / 4 << " kB\nThreads:\t1\n";
    }
    void clear_refs() const { std::ofstream(path_ + "/proc/self/clear_refs") << ""; }
private:
    std::string path_;
};

// ═════════════════════════════════════════════════════════════════════════════
// A. The allocation hook
// ═════════════════════════════════════════════════════════════════════════════

static void test_hook_exact_counts() {
    SECTION("A1 hook: exact byte counts and categories");
    CHECK(hook_active());
    const int64_t base_params = live(Cat::PARAMS);
    const uint64_t a0 = snapshot().allocs, f0 = snapshot().frees;
    {
        MemScope s(Cat::PARAMS);
        std::vector<float> v(1u << 20);                          // one allocation of exactly 4 MiB
        CHECK_EQ(live(Cat::PARAMS) - base_params, (int64_t)4 * (1 << 20));
        CHECK_EQ(snapshot().allocs - a0, (uint64_t)1);
    }
    CHECK_EQ(live(Cat::PARAMS), base_params);                    // freed: back to where it was, to the byte
    CHECK_EQ(snapshot().frees - f0, (uint64_t)1);

    // Only the scope's category moves.
    const CatBytes b = snapshot().live;
    {
        MemScope s(Cat::OPTIMIZER);
        std::vector<char> v(100000);
        const CatBytes a = snapshot().live;
        for (int c = 0; c < kCats; ++c)
            CHECK_EQ(a.v[(size_t)c] - b.v[(size_t)c], c == (int)Cat::OPTIMIZER ? 100000 : 0);
    }
    // Outside any scope: OTHER.
    const int64_t o0 = live(Cat::OTHER);
    { std::vector<char> v(5000); CHECK_EQ(live(Cat::OTHER) - o0, (int64_t)5000); }
    CHECK_EQ(live(Cat::OTHER), o0);
}

static void test_scopes() {
    SECTION("A2 hook: scopes nest and restore");
    CHECK(current_category() == Cat::OTHER);
    {
        MemScope a(Cat::ACTIVATIONS);
        CHECK(current_category() == Cat::ACTIVATIONS);
        { MemScope b(Cat::GRADS); CHECK(current_category() == Cat::GRADS); { MemScope c(Cat::SCRATCH); CHECK(current_category() == Cat::SCRATCH); } CHECK(current_category() == Cat::GRADS); }
        CHECK(current_category() == Cat::ACTIVATIONS);
    }
    CHECK(current_category() == Cat::OTHER);
}

static void test_free_credited_to_alloc_category() {
    SECTION("A3 hook: a free is credited to the category the block was ALLOCATED under");
    const CatBytes before = snapshot().live;
    std::vector<float>* v;
    { MemScope s(Cat::PARAMS); v = new std::vector<float>(10000); }       // 40000 B (+ the vector object itself, tiny)
    const int64_t p_mid = live(Cat::PARAMS);
    CHECK(p_mid - before[Cat::PARAMS] >= 40000);
    {
        MemScope s(Cat::SCRATCH);                                         // free under a DIFFERENT scope
        const int64_t scratch_mid = live(Cat::SCRATCH);
        delete v;
        CHECK_EQ(live(Cat::SCRATCH), scratch_mid);                        // scratch is untouched, never driven negative
    }
    CHECK_EQ(live(Cat::PARAMS), before[Cat::PARAMS]);                     // params went back down
    CHECK_EQ(live(Cat::SCRATCH), before[Cat::SCRATCH]);
}

static void test_threads_and_cross_thread_free() {
    SECTION("A4 hook: threads start in OTHER; cross-thread frees credit the right category");
    MemScope outer(Cat::ACTIVATIONS);                                     // main thread is in a scope...
    Cat seen = Cat::COUNT;
    std::thread([&] { seen = current_category(); }).join();
    CHECK(seen == Cat::OTHER);                                            // ...a new thread does not inherit it

    const int64_t opt0 = live(Cat::OPTIMIZER);
    char* block = nullptr;
    std::thread([&] { MemScope s(Cat::OPTIMIZER); block = new char[123456]; }).join();
    CHECK_EQ(live(Cat::OPTIMIZER) - opt0, (int64_t)123456);
    delete[] block;                                                       // freed on the main thread, under ACTIVATIONS
    CHECK_EQ(live(Cat::OPTIMIZER), opt0);
}

static void test_peaks() {
    SECTION("A5 hook: peaks and reset_peaks");
    const int64_t base = live_total();
    reset_peaks();
    CHECK_EQ(snapshot().peak_total, base);                                // a fresh window starts at the current live bytes
    {
        MemScope s(Cat::ACTIVATIONS);
        auto* a = new char[8 << 20];
        auto* b = new char[8 << 20];
        a[0] = 1; b[0] = 2; do_not_optimize(a); do_not_optimize(b);       // observable: an unused new/delete pair may legally be elided
        delete[] a;
        const Snapshot sn = snapshot();
        CHECK(sn.peak_total >= base + (16 << 20));                        // both were alive together
        CHECK(sn.live_total - base == (8 << 20));                         // only b is alive now
        CHECK(sn.peak[Cat::ACTIVATIONS] >= (16 << 20));
        delete[] b;
    }
    CHECK(snapshot().peak_total >= base + (16 << 20));                    // the peak is remembered after the frees
    CHECK_EQ(live_total(), base);
    reset_peaks();
    CHECK_EQ(snapshot().peak_total, live_total());                        // and a new window forgets it
    for (int c = 0; c < kCats; ++c) CHECK_EQ(snapshot().peak.v[(size_t)c], snapshot().live.v[(size_t)c]);
}

struct alignas(64)  Aligned64  { char c[64]; };
struct alignas(256) Aligned256 { char c[300]; };

static void test_sizes_and_alignment() {
    SECTION("A6 hook: every new/delete form, zero size, over-alignment");
    const int64_t base = live_total();
    const uint64_t blocks0 = snapshot().live_blocks();
    {
        char* z = new char[0];                    CHECK(z != nullptr); delete[] z;
        int* a = new int(7);                      CHECK(*a == 7); delete a;
        int* arr = new int[50];                   arr[49] = 1; delete[] arr;
        int* nt = new (std::nothrow) int[10];     CHECK(nt != nullptr); delete[] nt;
        struct S { double d[5]; };
        S* s = new S;                             delete s;                           // sized delete
        std::unique_ptr<std::string> str = std::make_unique<std::string>(200, 'x');
        std::vector<std::string> many; for (int i = 0; i < 1000; ++i) many.emplace_back(40, 'y');

        Aligned64*  p64  = new Aligned64;         CHECK(((uintptr_t)p64  % 64)  == 0); delete p64;
        Aligned256* p256 = new Aligned256;        CHECK(((uintptr_t)p256 % 256) == 0); p256->c[299] = 1; delete p256;
        Aligned64*  a64  = new Aligned64[9];      CHECK(((uintptr_t)a64  % 64)  == 0); delete[] a64;
        Aligned256* n256 = new (std::nothrow) Aligned256[3]; CHECK(n256 && ((uintptr_t)n256 % 256) == 0); delete[] n256;
        std::vector<Aligned64> v64(100);          CHECK(((uintptr_t)v64.data() % 64) == 0);
        void* raw = ::operator new(1000, std::align_val_t(128)); CHECK(((uintptr_t)raw % 128) == 0); ::operator delete(raw, std::align_val_t(128));
    }
    CHECK_EQ(live_total(), base);                                                     // everything balanced to the byte
    CHECK_EQ(snapshot().live_blocks(), blocks0);                                      // and to the block
    CHECK_EQ(snapshot().foreign_frees, (uint64_t)0);
}

static void test_bad_alloc() {
    SECTION("A7 hook: failure paths");
    if (UNDER_SANITIZER) { std::printf("  (skipped: sanitizer allocators abort on absurd sizes)\n"); return; }
    const int64_t base = live_total();
    bool threw = false;
    volatile size_t big = (size_t)1 << 56;                       // volatile: stop the compiler folding the size at compile time
    try { char* p = new char[big]; do_not_optimize(p); delete[] p; } catch (const std::bad_alloc&) { threw = true; }
    CHECK(threw);
    char* q = new (std::nothrow) char[big];
    CHECK(q == nullptr);
    CHECK_EQ(live_total(), base);                                                     // a failed allocation counts nothing
}

static void test_concurrency() {
    SECTION("A8 hook: 4 threads hammer alloc/free; counters balance exactly");
    constexpr int kThreads = 4, kIters = 40000;
    std::vector<std::thread> ts;
    ts.reserve(kThreads);                                                              // its buffer must exist BEFORE the baseline
    const CatBytes before = snapshot().live;
    const uint64_t blocks0 = snapshot().live_blocks();
    for (int t = 0; t < kThreads; ++t)
        ts.emplace_back([t] {
            std::mt19937 rng(1234 + t);
            std::vector<std::pair<char*, size_t>> held;
            for (int i = 0; i < kIters; ++i) {
                MemScope s((Cat)(rng() % (kCats)));
                const size_t n = 1 + rng() % 4096;
                char* p = new char[n]; p[0] = 1; p[n - 1] = 2;
                held.emplace_back(p, n);
                if (held.size() > 64 || (rng() & 3) == 0) {
                    const size_t k = rng() % held.size();
                    delete[] held[k].first; held[k] = held.back(); held.pop_back();
                }
            }
            for (auto& h : held) delete[] h.first;
        });
    for (auto& th : ts) th.join();
    const CatBytes after = snapshot().live;
    for (int c = 0; c < kCats; ++c) CHECK_EQ(after.v[(size_t)c], before.v[(size_t)c]);   // every category back to its start
    CHECK_EQ(snapshot().live_blocks(), blocks0);
    CHECK(snapshot().allocs > (uint64_t)kThreads * kIters);
    CHECK_EQ(snapshot().foreign_frees, (uint64_t)0);
}

static void test_foreign_free() {
    SECTION("A9 hook: a pointer from malloc passed to operator delete is detected, not crashed on");
    if (UNDER_SANITIZER) { std::printf("  (skipped: deliberate allocator mismatch)\n"); return; }
    const uint64_t f0 = snapshot().foreign_frees;
    const int64_t base = live_total();
    void* (*volatile alloc_fn)(size_t) = std::malloc;                                  // via a pointer: the mismatch is deliberate
    void* p = alloc_fn(256);
    ::operator delete(p);                                                              // not ours: magic does not match
    CHECK_EQ(snapshot().foreign_frees - f0, (uint64_t)1);
    CHECK_EQ(live_total(), base);                                                      // and nothing was subtracted for it
}

// ═════════════════════════════════════════════════════════════════════════════
// B. Process memory
// ═════════════════════════════════════════════════════════════════════════════

static void test_process_memory_fixture() {
    SECTION("B1 process memory: /proc/self/status parsing (fixture)");
    FakeRoot fx; fx.status(204800, 307200);
    ProcessMem m = read_process_mem(fx.root());
    CHECK(m.ok);
    CHECK_EQ(m.rss, (uint64_t)204800 * 1024);
    CHECK_EQ(m.hwm, (uint64_t)307200 * 1024);
    CHECK_EQ(m.vm_size, (uint64_t)800000 * 1024);
    CHECK_EQ(m.vm_peak, (uint64_t)900000 * 1024);
    CHECK_EQ(m.rss_anon, (uint64_t)102400 * 1024);
    CHECK_EQ(m.rss_file, (uint64_t)51200 * 1024);
    CHECK(!m.hwm_from_rusage);

    FakeRoot old; old.status(5000, 0, /*with_hwm=*/false);                             // very old kernel: no VmHWM
    ProcessMem o = read_process_mem(old.root());
    CHECK(o.ok && o.hwm == o.rss);

    FakeRoot empty; fs::remove(empty.root() + "/proc/self/status");
    CHECK(!read_process_mem(empty.root()).ok);                                         // unreadable: not ok, no invented numbers
    CHECK(!read_process_mem("/nonexistent_tensorf_root").ok);
}

static void test_process_memory_real() {
    SECTION("B2 process memory: this machine");
    ProcessMem m = read_process_mem();
    CHECK(m.ok);
    CHECK(m.rss > 0 && m.hwm >= m.rss && m.vm_size > m.rss);
    CHECK(m.rss < ((uint64_t)1 << 40));                                                // sane: under a terabyte
    // touching memory raises RSS and the kernel peak
    const uint64_t r0 = m.rss;
    std::vector<char> big(48u << 20, 1);
    do_not_optimize(big[big.size() - 1]);
    ProcessMem m2 = read_process_mem();
    CHECK(m2.rss >= r0 + (40u << 20));
    CHECK(m2.hwm >= m2.rss);
}

static void test_reset_peak_rss() {
    SECTION("B3 process memory: resetting the kernel peak");
    FakeRoot fx;
    CHECK(!reset_peak_rss(fx.root()));                                                 // no clear_refs node: must fail
    CHECK(!fs::exists(fx.root() + "/proc/self/clear_refs"));                           // and must NOT have created one (the old fopen("w") did)
    fx.clear_refs();
    CHECK(reset_peak_rss(fx.root()));
    std::ifstream f(fx.root() + "/proc/self/clear_refs"); std::string s; std::getline(f, s);
    CHECK(s == "5");                                                                   // the documented "reset VmHWM" command
    CHECK(!reset_peak_rss("/nonexistent_tensorf_root"));
    if (!UNDER_SANITIZER) {
        const bool ok = reset_peak_rss();                                              // real: may be denied (Android 8+)
        std::printf("  (info) real clear_refs reset %s\n", ok ? "works" : "is denied here");
        if (ok) {
            { std::vector<char> spike(64u << 20, 1); do_not_optimize(spike[0]); }      // a spike, gone before anyone samples it
            const uint64_t hwm_spike = read_process_mem().hwm;
            CHECK(reset_peak_rss());
            const ProcessMem after = read_process_mem();
            CHECK(after.hwm < hwm_spike);                                              // the peak restarted
            CHECK(after.hwm >= after.rss);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// C. measure_memory
// ═════════════════════════════════════════════════════════════════════════════

static void test_measure_fixture_arithmetic() {
    SECTION("C1 measure_memory: factor arithmetic on a controlled fixture");
    FakeRoot fx; fx.clear_refs();
    fx.status(100 * 1024, 100 * 1024);                                                  // 100 MB resident, peak 100 MB
    MeasureOptions o; o.proc_root = fx.root(); o.warmup = 0;
    // The step allocates 64 MB of tensors and, as the "kernel", raises the peak by 80 MB.
    auto step = [&] {
        MemScope s(Cat::ACTIVATIONS);
        std::vector<char> v(64u << 20);
        do_not_optimize(v[0]);
        fx.status(100 * 1024, 180 * 1024);
    };
    MemoryMeasurement m = measure_memory(step, o);
    CHECK(m.hook_active && m.rss_available && m.hwm_reset && m.rss_peak_valid);
    CHECK_EQ(m.rss_before, (uint64_t)100 * 1024 * 1024);
    CHECK_EQ(m.rss_peak, (uint64_t)180 * 1024 * 1024);
    CHECK_EQ(m.rss_growth(), (uint64_t)80 * 1024 * 1024);
    CHECK(m.tensor_growth() >= (64 << 20) && m.tensor_growth() < (64 << 20) + 65536);   // the fixture rewrite itself allocates a few KB
    CHECK(m.factor_valid);
    CHECK_REL(m.overhead_factor, 80.0 / 64.0, 0.001);                                   // 1.25: RSS growth / tensor growth
    CHECK_REL(m.recommended_factor(), 1.25, 0.001);
    CHECK(m.peak_by_cat[Cat::ACTIVATIONS] >= (64 << 20));
    CHECK_EQ(m.live_after_total, m.live_before_total);                                  // the step freed everything
    CHECK_EQ(m.blocks_leaked, (uint64_t)0);
    CHECK(m.notes.empty());

    // A pathological reading (RSS grew less than the tensors) is clamped into the plausible range for use.
    fx.status(100 * 1024, 100 * 1024);
    auto less = [&] { std::vector<char> v(64u << 20); do_not_optimize(v[0]); fx.status(100 * 1024, 110 * 1024); };
    MemoryMeasurement lo = measure_memory(less, o);
    CHECK(lo.factor_valid && lo.overhead_factor < 1.0);
    CHECK_REL(lo.recommended_factor(), 1.0, 1e-12);
    auto more = [&] { std::vector<char> v(64u << 20); do_not_optimize(v[0]); fx.status(100 * 1024, 100 * 1024 + 900 * 1024); };
    MemoryMeasurement hi = measure_memory(more, o);
    CHECK_REL(hi.recommended_factor(), 4.0, 1e-12);                                     // 900 MB / 64 MB = 14 -> capped at 4
}

static void test_measure_validity_rules() {
    SECTION("C2 measure_memory: when the factor must NOT be trusted");
    {   // too small a step
        FakeRoot fx; fx.clear_refs(); fx.status(50 * 1024, 50 * 1024);
        MeasureOptions o; o.proc_root = fx.root(); o.warmup = 0;
        MemoryMeasurement m = measure_memory([&] { std::vector<char> v(1 << 20); do_not_optimize(v[0]); fx.status(50 * 1024, 52 * 1024); }, o);
        CHECK(!m.factor_valid);
        CHECK(std::isnan(m.overhead_factor));
        CHECK_REL(m.recommended_factor(1.25), 1.25, 1e-12);                             // falls back to the default
        bool said = false; for (const auto& n : m.notes) if (n.find("too little") != std::string::npos) said = true;
        CHECK(said);
    }
    {   // kernel peak not resettable AND the old process peak was higher: cannot isolate the window
        FakeRoot fx; fx.status(100 * 1024, 500 * 1024);                                 // no clear_refs file
        MeasureOptions o; o.proc_root = fx.root(); o.warmup = 0;
        MemoryMeasurement m = measure_memory([&] { std::vector<char> v(32u << 20); do_not_optimize(v[0]); }, o);
        CHECK(m.rss_available && !m.hwm_reset && !m.rss_peak_valid);
        CHECK(!m.factor_valid);
        bool said = false; for (const auto& n : m.notes) if (n.find("cannot be isolated") != std::string::npos) said = true;
        CHECK(said);
    }
    {   // same, but the peak ROSE during the window: it must belong to the window, so it is usable
        FakeRoot fx; fx.status(100 * 1024, 120 * 1024);
        MeasureOptions o; o.proc_root = fx.root(); o.warmup = 0;
        MemoryMeasurement m = measure_memory([&] { std::vector<char> v(32u << 20); do_not_optimize(v[0]); fx.status(100 * 1024, 160 * 1024); }, o);
        CHECK(!m.hwm_reset && m.rss_peak_valid);
        CHECK_REL(m.overhead_factor, 60.0 / 32.0, 0.001);                               // no reset: before = 100 MB, peak 160 MB -> 60 MB over 32 MB
        CHECK(m.factor_valid);
    }
    {   // the step allocated plenty but peak RSS did not move: pages were already resident. Say so; never fake a ratio.
        FakeRoot fx; fx.clear_refs(); fx.status(300 * 1024, 300 * 1024);
        MeasureOptions o; o.proc_root = fx.root(); o.warmup = 0;
        MemoryMeasurement m = measure_memory([&] { std::vector<char> v(32u << 20); do_not_optimize(v[0]); }, o);   // status file never changes
        CHECK(m.rss_peak_valid && m.tensor_growth() >= (32 << 20));
        CHECK(m.rss_growth() == 0 && !m.factor_valid);
        CHECK_REL(m.recommended_factor(1.25), 1.25, 1e-12);
        bool said = false; for (const auto& n : m.notes) if (n.find("already resident") != std::string::npos) said = true;
        CHECK(said);
    }
    {   // /proc unreadable
        FakeRoot fx; fs::remove(fx.root() + "/proc/self/status");
        MeasureOptions o; o.proc_root = fx.root(); o.warmup = 0;
        MemoryMeasurement m = measure_memory([&] { std::vector<char> v(32u << 20); do_not_optimize(v[0]); }, o);
        CHECK(!m.rss_available && !m.factor_valid);
        CHECK(m.tensor_growth() >= (32 << 20));                                          // tensor bytes are still reported
    }
    {   // leak detection and warm-up: a step that keeps a block alive is reported
        FakeRoot fx; fx.clear_refs(); fx.status(10 * 1024, 10 * 1024);
        std::vector<std::unique_ptr<char[]>> keep;
        MeasureOptions o; o.proc_root = fx.root(); o.warmup = 1;
        MemoryMeasurement m = measure_memory([&] { keep.emplace_back(new char[10000]); }, o);
        CHECK(m.blocks_leaked >= 1);
    }
    {   // repeats keep the LARGEST window
        FakeRoot fx; fx.clear_refs(); fx.status(10 * 1024, 10 * 1024);
        int call = 0;
        MeasureOptions o; o.proc_root = fx.root(); o.warmup = 0; o.repeats = 3;
        MemoryMeasurement m = measure_memory([&] { const size_t n = (call++ == 1) ? (40u << 20) : (10u << 20); std::vector<char> v(n); do_not_optimize(v[0]); }, o);
        CHECK(m.tensor_growth() >= (40 << 20));
    }
}

static void test_measure_real_allocation() {
    SECTION("C3 measure_memory: a real 64 MB allocation on this machine");
    MeasureOptions o; o.warmup = 1;
    auto step = [] { MemScope s(Cat::SCRATCH); std::vector<char> v(64u << 20, 3); do_not_optimize(v[v.size() / 2]); };
    MemoryMeasurement m = measure_memory(step, o);
    CHECK(m.hook_active);
    CHECK(m.tensor_growth() >= (64 << 20) && m.tensor_growth() < (64 << 20) + 65536);
    CHECK(m.peak_by_cat[Cat::SCRATCH] >= (64 << 20));
    CHECK_EQ(m.live_after_total, m.live_before_total);
    if (m.factor_valid) {
        std::printf("  (info) real overhead factor for a 64 MB buffer: %.3f (RSS growth %.1f MB vs tensor %.1f MB, kernel peak reset %s)\n",
                    m.overhead_factor, m.rss_growth() / 1e6, m.tensor_growth() / 1e6, m.hwm_reset ? "yes" : "no");
        // A flat buffer costs about its size. Sanitizer runtimes add shadow memory to RSS, so the ratio is only meaningful
        // (and only asserted) in a normal build: there it is a true measurement of that build, not an error.
        if (!UNDER_SANITIZER) CHECK(m.overhead_factor > 0.9 && m.overhead_factor < 1.6);
    } else {
        for (const auto& n : m.notes) std::printf("  (info) factor not available: %s\n", n.c_str());
    }
    Json j; j.begin_object(); to_json(j, m); j.end_object();
    const std::string s = j.finish();
    CHECK(s.find("\"tensor_growth_bytes\"") != std::string::npos && s.find("\"peak_by_category\"") != std::string::npos);
    CHECK(s.find("\"scratch\"") != std::string::npos);
    // The measured factor feeds the prediction path directly.
    wl::PredictOptions po; po.mem_overhead_factor = m.recommended_factor();
    CHECK(po.mem_overhead_factor >= 1.0 && po.mem_overhead_factor <= 4.0);
}

// ═════════════════════════════════════════════════════════════════════════════
// D. Census
// ═════════════════════════════════════════════════════════════════════════════

struct Mlp {
    size_t B, I, H, O;
    Tensor_t<float> x, w1, w2, loss;
    Mlp(size_t b, size_t i, size_t h, size_t o) : B(b), I(i), H(h), O(o) {
        x  = make_tensor<float>(Matrix<float>::randu(-1.f, 1.f, {B, I})); x->requires_grad = false;
        w1 = make_tensor<float>(Matrix<float>::randu(-.05f, .05f, {I, H}));
        w2 = make_tensor<float>(Matrix<float>::randu(-.05f, .05f, {H, O}));
    }
    void forward() { auto y = x->matmul(w1)->relu()->matmul(w2); loss = (y * y)->sum(); }
    void backward() { loss->backward(Matrix<float>(std::vector<float>{1.f}, shape_t{1})); }
};

static void test_census() {
    SECTION("D census: exact bytes by category on a real autograd graph");
    CHECK(census<float>(nullptr).total() == 0);

    Mlp n(32, 64, 48, 10);
    n.forward();
    const double B = 32, I = 64, H = 48, O = 10;
    CatBytes c = census<float>(n.loss);
    CHECK_EQ(c[Cat::PARAMS], (int64_t)((I * H + H * O) * 4));                            // both weight matrices, exactly
    CHECK_EQ(c[Cat::GRADS], (int64_t)0);                                                 // no backward yet: no gradient buffers
    // activations: the input batch + fc1 out + relu out + fc2 out + y*y + the scalar sum
    CHECK_EQ(c[Cat::ACTIVATIONS], (int64_t)((B * I + 2 * B * H + 2 * B * O + 1) * 4));
    CHECK_EQ(c[Cat::SCRATCH], (int64_t)0);

    n.backward();
    CatBytes d = census<float>(n.loss);
    CHECK_EQ(d[Cat::PARAMS], c[Cat::PARAMS]);
    CHECK_EQ(d[Cat::GRADS], (int64_t)((I * H + H * O) * 4));                             // one gradient per parameter
    CHECK_EQ(d[Cat::ACTIVATIONS], c[Cat::ACTIVATIONS]);
    // intermediate gradients: one per non-leaf tensor = the sum of the layers' output sizes (the rule WorkloadModel uses)
    CHECK_EQ(d[Cat::SCRATCH], (int64_t)((2 * B * H + 2 * B * O + 1) * 4));

    // A tensor reached through two paths is counted once.
    auto a = make_tensor<float>(Matrix<float>::randu(-1.f, 1.f, {10, 10}));
    auto sq = (a * a)->sum();
    CatBytes e = census<float>(sq);
    CHECK_EQ(e[Cat::PARAMS], (int64_t)400);
    // A bare leaf
    CatBytes f = census<float>(a);
    CHECK_EQ(f[Cat::PARAMS], (int64_t)400);
    CHECK_EQ(f.total(), (int64_t)400);
}

// ═════════════════════════════════════════════════════════════════════════════
// E. Model vs measurement
// ═════════════════════════════════════════════════════════════════════════════

static void test_matmul_backward_temporaries() {
    SECTION("E1 model vs measurement: matmul_grads temporaries = 3MK + 3KN + MN elements");
    const size_t shapes[][3] = {{64, 64, 64}, {128, 32, 64}, {32, 200, 100}, {200, 16, 256}};
    for (const auto& sh : shapes) {
        const size_t M = sh[0], K = sh[1], N = sh[2];
        auto A = Matrix<float>::randu(-1.f, 1.f, {M, K}), B = Matrix<float>::randu(-1.f, 1.f, {K, N}), G = Matrix<float>::randu(-1.f, 1.f, {M, N});
        { auto w = MatmulOperation<float>::matmul_grads(A, B, G); do_not_optimize(w.first.data[0]); }   // warm
        reset_peaks();
        const int64_t base = live_total();
        { auto w = MatmulOperation<float>::matmul_grads(A, B, G); do_not_optimize(w.first.data[0]); }
        const double measured = (double)(snapshot().peak_total - base);
        wl::RunSpec r; r.mode = wl::Mode::TRAIN;
        const double modelled = wl::dense((double)M, (double)K, (double)N, r, "d", false).mem.scratch;
        CHECK_REL(modelled, measured, 0.005);                                              // within half a percent
        CHECK_REL(measured, (3.0 * M * K + 3.0 * K * N + 1.0 * M * N) * 4.0, 0.005);
    }
}

static void test_mlp_memory_model() {
    SECTION("E2 model vs measurement: a full MLP training step");
    struct Cfg { size_t B, I, H, O; } cfgs[] = {{32, 784, 128, 10}, {128, 784, 256, 10}};
    for (const Cfg& g : cfgs) {
        const int64_t L0 = live_total();                                                   // everything alive that is NOT this net
        Mlp n(g.B, g.I, g.H, g.O);
        // description of exactly what the net does (the same one WorkloadProbe uses)
        wl::RunSpec r; r.mode = wl::Mode::TRAIN; r.opt_states = 0;
        wl::Workload w; w.units_per_micro_step = (double)g.B;
        w.layers = {wl::dense((double)g.B, (double)g.I, (double)g.H, r, "fc1", false),
                    wl::elementwise((double)(g.B * g.H), 1.0, 2.0, r, "relu"),
                    wl::dense((double)g.B, (double)g.H, (double)g.O, r, "fc2", false),
                    wl::elementwise((double)(g.B * g.O), 2.0, 2.0, r, "square+sum")};
        const wl::MemoryEstimate est = wl::estimate_memory(w, 1.0);

        // steady-state breakdown from the census, after a full step
        n.forward(); n.backward();
        const CatBytes c = census<float>(n.loss);
        CHECK_REL(est.params, c[Cat::PARAMS], 1e-9);                                       // parameters: exact
        CHECK_REL(est.grads, c[Cat::GRADS], 1e-9);                                         // parameter gradients: exact
        // activations + retained intermediate grads, vs the model's (saved inputs + retained output grads):
        const double act_meas = (double)(c[Cat::ACTIVATIONS] + c[Cat::SCRATCH]);
        CHECK_REL(est.activations, act_meas, 0.02);                                        // within 2%

        // peak, from the hook: persistent state + the step's own peak
        n.loss.reset();
        auto step = [&] { n.forward(); n.backward(); n.w1->zero_grad(); n.w2->zero_grad(); n.loss.reset(); };
        MeasureOptions o; o.warmup = 1;
        const MemoryMeasurement m = measure_memory(step, o);
        // m.peak_total is ABSOLUTE live bytes at the peak; subtract what was alive before the net existed.
        const double measured_peak = (double)(m.peak_total - L0);
        const double ratio = measured_peak / est.tensor_total;
        std::printf("  (info) MLP B=%zu H=%zu: modelled %.2f MB, measured peak %.2f MB, measured/modelled %.3f\n",
                    g.B, g.H, est.tensor_total / 1e6, measured_peak / 1e6, ratio);
        CHECK(ratio > 0.90 && ratio < 1.10);                                               // the model now tracks the real peak to ~10%
        CHECK(measured_peak <= est.tensor_total * 1.10);                                   // and does not UNDER-estimate by more than 10%
    }
}

static void test_old_model_would_have_failed() {
    SECTION("E3 model vs measurement: the pre-hook estimate was off by 2.5x (regression guard)");
    // The memory model used to count one layer-output of scratch per layer. Reconstruct that and show the real
    // peak is far above it, so nobody quietly "simplifies" the backward temporaries away again.
    Mlp n(128, 784, 256, 10);
    auto step = [&] { n.forward(); n.backward(); n.w1->zero_grad(); n.w2->zero_grad(); n.loss.reset(); };
    MeasureOptions o; o.warmup = 1;
    const MemoryMeasurement m = measure_memory(step, o);
    const double growth = (double)m.tensor_growth();
    const double old_scratch_only = 128.0 * 256 * 4;                                       // what the old dense() charged as scratch
    CHECK(growth > 10.0 * old_scratch_only);
}

int main() {
    std::printf("TensorF profiler step 3: memory accounting tests\n================================================\n");
    test_hook_exact_counts();
    test_scopes();
    test_free_credited_to_alloc_category();
    test_threads_and_cross_thread_free();
    test_peaks();
    test_sizes_and_alignment();
    test_bad_alloc();
    test_concurrency();
    test_foreign_free();
    test_process_memory_fixture();
    test_process_memory_real();
    test_reset_peak_rss();
    test_measure_fixture_arithmetic();
    test_measure_validity_rules();
    test_measure_real_allocation();
    test_census();
    test_matmul_backward_temporaries();
    test_mlp_memory_model();
    test_old_model_would_have_failed();
    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
