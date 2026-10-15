#include <cstdio>
#include <fstream>
#include <thread>
#include "profiler/MemoryProfiler.hpp"
using namespace tensorf::profile;

static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++fails; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#if defined(__SANITIZE_ADDRESS__)
  constexpr bool kAsan = true;          // ASan's allocator changes RSS behaviour: skip the OS-level checks
#else
  constexpr bool kAsan = false;
#endif
constexpr uint64_t MB = 1ull << 20;

static LedgerSnapshot snap() { return MemoryLedger::global().snapshot(); }

static void t_ledger_basic() {
    auto& L = MemoryLedger::global();
    auto s0 = snap();
    L.reset_peaks();
    L.on_alloc(MemCat::Parameters, 1000);
    L.on_alloc(MemCat::Activations, 500);
    auto s = snap();
    CHECK(s.live[size_t(MemCat::Parameters)] - s0.live[size_t(MemCat::Parameters)] == 1000);
    CHECK(s.live_total - s0.live_total == 1500);
    L.on_free(MemCat::Activations, 500);
    s = snap();
    CHECK(s.live_total - s0.live_total == 1000);
    CHECK(s.peak_total >= s0.live_total + 1500);            // peak remembers the 1500 high point
    CHECK(s.allocs - s0.allocs == 2 && s.frees - s0.frees == 1);
    L.on_free(MemCat::Parameters, 1000);
    CHECK(snap().live_total == s0.live_total);
}

static void t_allocator() {
    auto s0 = snap();
    {
        ScopedCategory c(MemCat::Parameters);
        TrackedVector<float> w(1000);                       // 4000 B under Parameters
        auto s = snap();
        CHECK(s.live[size_t(MemCat::Parameters)] - s0.live[size_t(MemCat::Parameters)] == 4000);
        {
            ScopedCategory c2(MemCat::Activations);         // nested scope, restored afterwards
            TrackedVector<float> a(250);
            CHECK(snap().live[size_t(MemCat::Activations)] - s0.live[size_t(MemCat::Activations)] == 1000);
            CHECK(current_category() == MemCat::Activations);
            TrackedVector<float> copy = w;                  // a copy takes the CURRENT category
            CHECK(snap().live[size_t(MemCat::Activations)] - s0.live[size_t(MemCat::Activations)] == 1000 + 4000);
            CHECK(copy.get_allocator().category() == MemCat::Activations);
        }
        CHECK(current_category() == MemCat::Parameters);
        CHECK(snap().live[size_t(MemCat::Activations)] == s0.live[size_t(MemCat::Activations)]);
        // growth: reallocation frees the old block and counts the new one
        TrackedVector<float> g;
        for (int i = 0; i < 10000; ++i) g.push_back(1.f);
        const uint64_t cap_bytes = g.capacity() * sizeof(float);
        CHECK(snap().live[size_t(MemCat::Parameters)] - s0.live[size_t(MemCat::Parameters)] == 4000 + cap_bytes);
        // move construction keeps the block and its category; no new allocation
        auto allocs = snap().allocs;
        TrackedVector<float> moved = std::move(g);
        CHECK(snap().allocs == allocs && moved.get_allocator().category() == MemCat::Parameters);
        // move assignment propagates the allocator (and category) with the storage
        TrackedVector<float> other(1, 0.f, LedgerAllocator<float>(MemCat::Scratch));
        other = std::move(moved);
        CHECK(other.get_allocator().category() == MemCat::Parameters);
        CHECK(snap().live[size_t(MemCat::Scratch)] == s0.live[size_t(MemCat::Scratch)]);   // the Scratch element was released
    }
    auto s1 = snap();
    for (size_t i = 0; i < kMemCats; ++i) CHECK(s1.live[i] == s0.live[i]);       // everything returned
    CHECK(s1.live_total == s0.live_total);
    CHECK(LedgerAllocator<int>(MemCat::Data) == LedgerAllocator<double>(MemCat::Data));
    CHECK(LedgerAllocator<int>(MemCat::Data) != LedgerAllocator<int>(MemCat::Scratch));
}

static void t_threads() {
    auto s0 = snap();
    auto& L = MemoryLedger::global();
    L.reset_peaks();
    std::vector<std::thread> th;
    for (int t = 0; t < 8; ++t)
        th.emplace_back([t] {
            ScopedCategory c(MemCat(t % 5));
            for (int i = 0; i < 20000; ++i) { TrackedVector<char> v(64 + (i % 7)); }
        });
    for (auto& x : th) x.join();
    auto s1 = snap();
    CHECK(s1.live_total == s0.live_total);
    for (size_t i = 0; i < kMemCats; ++i) CHECK(s1.live[i] == s0.live[i]);
    CHECK(s1.allocs - s0.allocs == 8u * 20000 && s1.frees - s0.frees == 8u * 20000);
    CHECK(s1.peak_total >= s0.live_total + 64 && s1.peak_total <= s0.live_total + 8 * 80);   // at most one live block per thread
}

static void t_phases() {
    PhaseRecorder rec;
    auto base = snap().live_total;
    TrackedVector<float>* keep = nullptr;
    {
        auto p = rec.phase("forward");
        TrackedVector<float> big(10 * MB / 4);
        TrackedVector<float> small(MB / 4);
        keep = new TrackedVector<float>(MB / 4);            // survives the phase
    }
    {
        auto p = rec.phase("backward");
        TrackedVector<float> g(2 * MB / 4);
    }
    auto r = rec.records();
    CHECK(r.size() == 2 && r[0].name == "forward" && r[1].name == "backward");
    CHECK(r[0].peak_total - base >= 11 * MB + MB / 1 && r[0].peak_total - base <= 12 * MB + 1024);   // 10 + 1 + 1 MB
    CHECK(r[0].live_at_end - base == MB);                   // only `keep` is left
    // the second phase starts from the carried-over live bytes, not from the first phase's peak
    CHECK(r[1].live_at_start - base == MB);
    CHECK(r[1].peak_total - base == 3 * MB);
    CHECK(rec.max_peak("forward") == r[0].peak_total && rec.max_peak("nope") == 0);
    delete keep;
    CHECK(snap().live_total == base);
}

static void t_proc_parse() {
    auto pm = detail::parse_proc_status("Name:\tx\nVmPeak:\t  900000 kB\nVmHWM:\t  123456 kB\nVmRSS:\t  100000 kB\nVmSwap:\t    2048 kB\nThreads:\t4\n");
    CHECK(pm.ok && pm.rss == 100000ull * 1024 && pm.rss_peak == 123456ull * 1024 && pm.swap == 2048ull * 1024);
    CHECK(!detail::parse_proc_status("garbage").ok);
    CHECK(detail::parse_proc_status("VmRSS: 500 kB\n").rss_peak == 500ull * 1024);   // missing HWM: never below rss
    MemFs fs;
    fs.put("/proc/self/status", "VmRSS: 1000 kB\nVmHWM: 2000 kB\n");
    fs.put("/proc/self/smaps_rollup", "00400000-ff000000 ---p 00000000 00:00 0 [rollup]\nRss:    1000 kB\nPss:     640 kB\n");
    auto a = read_process_memory_linux(fs, false), b = read_process_memory_linux(fs, true);
    CHECK(!a.pss && b.pss == 640ull * 1024 && b.ok);
    fs.files.erase("/proc/self/smaps_rollup");
    CHECK(!read_process_memory_linux(fs, true).pss);       // old kernels: PSS unavailable, not an error
    CHECK(!read_process_memory_linux(*std::make_unique<MemFs>(), false).ok);
}

static void t_real_process() {
    auto before = read_process_memory();
    CHECK(before.ok && before.rss > MB && before.rss_peak >= before.rss);
    if (kAsan) return;
    std::vector<char> big(64 * MB);
    for (size_t i = 0; i < big.size(); i += 4096) big[i] = 1;         // touch
    auto after = read_process_memory();
    CHECK(after.rss >= before.rss + 50 * MB);
    CHECK(after.rss_peak >= after.rss);
}

static void t_sampler() {
    std::atomic<int> n{0};
    // spike of 900 in the middle of a quiet series: the sampler must see it
    RssSampler s([&]() -> std::optional<uint64_t> { int i = n++; return i == 5 ? 900 : 100; }, std::chrono::milliseconds(2));
    s.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    CHECK(s.stop() == 900 && s.peak() == 900);
    RssSampler dead([]() -> std::optional<uint64_t> { return std::nullopt; }, std::chrono::milliseconds(1));
    dead.start(); CHECK(dead.stop() == 0);                            // unreadable source: no crash, zero
    RssSampler never([] { return std::optional<uint64_t>(5); }, std::chrono::milliseconds(1));
    CHECK(never.stop() == 0);                                         // stop() before start() is harmless
}

static void t_overhead() {
    // fake RSS that grows exactly 1.3x the bytes touched -> factor ~1.3
    uint64_t fake = 50 * MB;
    int calls = 0;
    // The function reads RSS twice (before/after); emulate growth by returning a value dependent on a global counter.
    auto r = measure_allocator_overhead([&]() -> std::optional<uint64_t> {
        if (calls++ == 0) return fake;
        return fake + uint64_t(1.3 * double(24 * MB));
    });
    CHECK(r.valid && r.requested >= 24 * MB && std::abs(r.factor - 1.3 * 24.0 * MB / r.requested) < 0.01);
    // RSS did not grow (pages reused): reported invalid with the default factor
    calls = 0;
    auto r2 = measure_allocator_overhead([&]() -> std::optional<uint64_t> { calls++; return 50 * MB; });
    CHECK(!r2.valid && r2.factor == 1.25);
    // unreadable RSS
    CHECK(!measure_allocator_overhead([]() -> std::optional<uint64_t> { return std::nullopt; }).valid);
    if (kAsan) return;
    auto real = measure_allocator_overhead([] { auto p = read_process_memory(); return p.ok ? std::optional<uint64_t>(p.rss) : std::nullopt; });
    std::printf("real allocator overhead: valid=%d factor=%.3f requested=%llu MB delta=%llu MB\n", int(real.valid), real.factor,
                (unsigned long long)(real.requested / MB), (unsigned long long)(real.rss_delta / MB));
    CHECK(real.requested >= 24 * MB);
    if (real.valid) CHECK(real.factor >= 1.0 && real.factor < 3.0);
}

static HardwareFingerprint phone_fp(uint64_t total_mb, uint64_t avail_mb) {
    HardwareFingerprint fp;
    fp.memory.total = total_mb * MB; fp.memory.available = avail_mb * MB; fp.memory.effective_available = avail_mb * MB;
    return fp;
}

static void t_estimates() {
    CHECK(tensor_bytes(Dtype::F32, 10) == 40 && tensor_bytes(Dtype::F16, 10) == 20 && tensor_bytes(Dtype::I8, 10) == 10);
    CHECK(tensor_bytes(Dtype::FP8, 64) == 64 + 2 * 4);               // 2 blocks of 32, 4-byte scale each
    CHECK(tensor_bytes(Dtype::FP4, 64) == 32 + 2 * 4);               // packed two per byte
    CHECK(tensor_bytes(Dtype::FP4, 33) == 17 + 2 * 4);               // odd count rounds up, partial block still has a scale
    CHECK(tensor_bytes(Dtype::FP4, 0) == 0);
    CHECK(optimizer_state_bytes(OptimizerKind::SGD, 100) == 0 && optimizer_state_bytes(OptimizerKind::Momentum, 100) == 400 &&
          optimizer_state_bytes(OptimizerKind::Adam, 100) == 800);
    MemoryEstimate e; e[MemCat::Parameters] = 100; e[MemCat::Activations] = 300;
    MemoryEstimate f; f[MemCat::Activations] = 50; f[MemCat::Scratch] = 10;
    e += f;
    CHECK(e.total() == 460 && e[MemCat::Activations] == 350);
}

static void t_budget() {
    ProcessMemory proc; proc.ok = true; proc.rss = 200 * MB;
    auto fp = phone_fp(8192, 3000);
    auto b = plan_budget(fp, proc);                                   // background, defaults
    CHECK(b.reserve_bytes == uint64_t(8192 * MB * 0.08));             // 8% of total beats the 128 MB floor
    CHECK(b.headroom_bytes == 3000 * MB - b.reserve_bytes);
    CHECK(std::abs(b.fraction - 0.40) < 1e-12);
    CHECK(b.max_additional_bytes == uint64_t(double(b.headroom_bytes) * 0.40));
    CHECK(b.tensor_budget_bytes == uint64_t(double(b.max_additional_bytes) / 1.25));
    CHECK(b.max_total_process_bytes == proc.rss + b.max_additional_bytes);
    BudgetPolicy fgp; fgp.mode = RunMode::Foreground;
    auto f = plan_budget(fp, proc, fgp);
    CHECK(f.tensor_budget_bytes > b.tensor_budget_bytes && std::abs(f.fraction - 0.65) < 1e-12);
    // measured overhead shrinks the tensor budget, never the real-bytes budget
    BudgetPolicy hp; hp.overhead_factor = 2.0;
    auto h = plan_budget(fp, proc, hp);
    CHECK(h.max_additional_bytes == b.max_additional_bytes && h.tensor_budget_bytes < b.tensor_budget_bytes);
    BudgetPolicy lo; lo.overhead_factor = 0.2;                         // nonsense below 1 is clamped
    CHECK(plan_budget(fp, proc, lo).overhead_factor == 1.0);
    // monotone in available memory
    uint64_t prev = 0;
    for (uint64_t a : {200, 500, 1000, 2000, 4000, 8000}) {
        auto x = plan_budget(phone_fp(8192, a), proc);
        CHECK(x.tensor_budget_bytes >= prev); prev = x.tensor_budget_bytes;
    }
    // below the reserve: zero
    auto low = plan_budget(phone_fp(8192, 500), proc);
    CHECK(low.headroom_bytes == 0 && low.tensor_budget_bytes == 0);
    // LMK threshold raises the reserve (small-RAM phone, 2 GB: 8% = 164 MB; lmk 80640 pages = 315 MB, x2 = 630 MB)
    auto fpl = phone_fp(2048, 900); fpl.memory.lmk_min_free_bytes = 80640ull * 4096;
    auto bl = plan_budget(fpl, proc);
    CHECK(bl.reserve_bytes == 2 * 80640ull * 4096 && bl.headroom_bytes == 900 * MB - bl.reserve_bytes);
    // pressure
    auto fpp = fp; fpp.memory.psi_some_avg10 = 12.0;
    auto bp = plan_budget(fpp, proc);
    CHECK(bp.pressured && std::abs(bp.fraction - 0.20) < 1e-12);
    fpp.memory.psi_some_avg10 = 30.0;
    auto bz = plan_budget(fpp, proc);
    CHECK(bz.tensor_budget_bytes == 0 && bz.pressured);
    // unknown memory: zero with a note
    HardwareFingerprint none;
    auto bn = plan_budget(none, proc);
    CHECK(bn.tensor_budget_bytes == 0 && !bn.notes.empty());
    // explicit fraction wins over mode
    BudgetPolicy ex; ex.fraction = 0.9;
    CHECK(std::abs(plan_budget(fp, proc, ex).fraction - 0.9) < 1e-12);
    // fit check
    MemoryEstimate est; est[MemCat::Parameters] = b.tensor_budget_bytes / 2; est[MemCat::Activations] = b.tensor_budget_bytes / 4;
    auto fr = check_fit(est, b);
    CHECK(fr.fits && fr.largest == MemCat::Parameters && fr.utilization > 0.7 && fr.utilization < 0.8);
    est[MemCat::Scratch] = b.tensor_budget_bytes;
    CHECK(!check_fit(est, b).fits);
    MemoryEstimate some; some[MemCat::Data] = 1;
    auto fz = check_fit(some, low);
    CHECK(!fz.fits && std::isinf(fz.utilization));
    CHECK(check_fit(MemoryEstimate{}, low).fits);                      // asking for nothing always fits
}

static void t_guard() {
    GuardConfig cfg; cfg.reserve_bytes = 1000;
    MemState st; st.available = 5000;
    MemoryGuard g(cfg, [&] { return st; });
    CHECK(g.check() == MemAction::Ok);
    st.available = 1400;  CHECK(g.check() == MemAction::Shrink);       // < 1.5 * reserve
    st.available = 1600;  CHECK(g.check() == MemAction::Shrink);       // above 1500 but inside the recovery margin (needs 1725)
    st.available = 1800;  CHECK(g.check() == MemAction::Ok);
    st.available = 900;   CHECK(g.check() == MemAction::Stop);         // into the reserve
    st.available = 1050;  CHECK(g.check() == MemAction::Stop);         // not yet 15% clear of the reserve
    st.available = 1200;  CHECK(g.check() == MemAction::Shrink);       // out of Stop, still under 1.5x
    st.available = 5000;  st.psi_some_avg10 = 30;  CHECK(g.check() == MemAction::Stop);   // pressure alone stops
    st.psi_some_avg10 = 22; CHECK(g.check() == MemAction::Stop);       // 22 is not below 25/1.15 = 21.7: still Stop
    st.psi_some_avg10 = 12; CHECK(g.check() == MemAction::Shrink);     // soft zone
    st.psi_some_avg10 = 5;  CHECK(g.check() == MemAction::Ok);
    // reading from a fake /proc through the real code path
    MemFs fs;
    fs.put("/proc/meminfo", "MemTotal: 8000000 kB\nMemAvailable: 100000 kB\n");
    fs.put("/proc/pressure/memory", "some avg10=40.00 avg60=0 avg300=0 total=1\n");
    auto m = read_memory_linux(fs);
    MemoryGuard g2(cfg, [&] { MemState s; s.available = read_memory_linux(fs).effective_available; s.psi_some_avg10 = read_memory_linux(fs).psi_some_avg10; return s; });
    CHECK(m.psi_some_avg10 && g2.check() == MemAction::Stop);
    // the real reader works on this machine
    auto rs = read_mem_state();
    CHECK(rs.available > 0);
}

static void t_report() {
    MemoryReport r;
    r.process = read_process_memory();
    r.ledger = snap();
    PhaseRecorder rec;
    { auto p = rec.phase("forward \"quoted\""); TrackedVector<char> v(100); }
    r.phases = rec.records();
    r.budget = plan_budget(phone_fp(4096, 2000), r.process);
    r.overhead.valid = true;
    auto js = r.to_json();
    std::ofstream("/tmp/mem_report.json") << js;
    CHECK(js.find("\"by_category\"") != std::string::npos && js.find("forward \\\"quoted\\\"") != std::string::npos);
}

int main() {
    t_ledger_basic(); t_allocator(); t_threads(); t_phases(); t_proc_parse(); t_real_process(); t_sampler();
    t_overhead(); t_estimates(); t_budget(); t_guard(); t_report();
    std::printf("%d/%d checks passed\n", checks - fails, checks);
    return fails ? 1 : 0;
}
