#include <cassert>
#include <cstdio>
#include <fstream>
#include <set>
#include "profiler/DeviceBenchmark.hpp"
using namespace tensorf::profile;
using namespace tensorf::profile::kernels;

static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++fails; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_UNDEFINED__)
  constexpr bool kPerfChecks = false;      // sanitised builds distort timing; correctness checks still run
#else
  constexpr bool kPerfChecks = true;
#endif

static BenchConfig qcfg() {
    auto c = BenchConfig::quick();
    if (!kPerfChecks) c.budget_seconds *= 10;          // sanitised kernels are many times slower; keep every stage in play
    return c;
}

static void t_chase() {
    for (size_t bytes : {size_t(512), size_t(4096), size_t(65536), size_t(1u << 20)}) {
        ChaseTable ct(bytes);
        std::set<uint32_t> seen; uint32_t p = 0; size_t steps = 0;
        do { seen.insert(p); p = ct.walk(p, 1); ++steps; } while (p != 0 && steps <= ct.nodes + 1);
        CHECK(p == 0 && steps == ct.nodes && seen.size() == ct.nodes);   // exactly one cycle covering every node
        CHECK(ct.walk(0, int64_t(ct.nodes) * 3) == 0);
    }
}

static void t_gemm() {
    for (auto [M, N, K] : {std::array<int,3>{8, 12, 7}, {6, 5, 9}, {4, 4, 4}, {13, 3, 1}}) {
        GemmCase<float> g(M, N, K);
        std::vector<double> ref(size_t(M) * N, 0.0);
        for (int i = 0; i < M; ++i) for (int k = 0; k < K; ++k) for (int j = 0; j < N; ++j)
            ref[size_t(i) * N + j] += double(g.A[size_t(i) * K + k]) * double(g.B[size_t(k) * N + j]);
        double fl = g.run();
        CHECK(fl == 2.0 * M * N * K);
        double err = 0; for (size_t i = 0; i < ref.size(); ++i) err = std::max(err, std::abs(ref[i] - g.C[i]));
        CHECK(err < 1e-4);
    }
    GemmCase<float> g(64, 64, 64);                     // stays bounded over many runs
    for (int i = 0; i < 400; ++i) g.run();
    bool finite = true; for (float v : g.C) finite = finite && std::isfinite(v);
    CHECK(finite);
}

static void t_stream_fma() {
    StreamCase sc(1 << 16);
    double bytes = sc.run();
    CHECK(bytes == 12.0 * sc.n && sc.a[0] == 2.0f && sc.a[sc.n - 1] == 2.0f);
    std::vector<float> a(kFmaLanes, 1.0f);
    double fl = fma_kernel<float>(10, a.data());
    CHECK(fl == 2.0 * kFmaLanes * kFmaDepth * 10);
    // reference: same recurrence in double, 10*16 steps
    double r = 1.0; for (int i = 0; i < 10 * kFmaDepth; ++i) r = r * double(1.0000001f) + double(1e-7f);
    CHECK(std::abs(a[0] - r) < 1e-4 && std::isfinite(a[kFmaLanes - 1]));
    std::vector<double> b(kFmaLanes, 1.0);
    fma_kernel<double>(10, b.data());
    CHECK(std::abs(b[5] - r) < 1e-4);
}

static void t_graph() {
    Graph g = make_powerlaw_graph(20000);
    bool mono = true, inrange = true;
    for (int i = 0; i < g.n; ++i) mono = mono && g.rowptr[i + 1] > g.rowptr[i];      // every node has >= 1 neighbour
    for (auto c : g.col) inrange = inrange && c < uint32_t(g.n);
    CHECK(mono && inrange && g.rowptr[g.n] == g.col.size());
    double mean = double(g.edges()) / g.n;
    uint32_t mx = 0; for (int i = 0; i < g.n; ++i) mx = std::max(mx, g.rowptr[i + 1] - g.rowptr[i]);
    CHECK(mean > 6 && mean < 20);                      // Pareto(min 4, alpha 1.5) has mean ~12 before the cap
    CHECK(mx >= 50 && mx <= 512);                      // heavy tail, capped
    auto g2 = make_powerlaw_graph(20000);              // deterministic
    CHECK(g2.col == g.col && g2.rowptr == g.rowptr);

    GatherCase gc(3000);
    double edges = gc.run();
    CHECK(edges == double(gc.g.edges()));
    double err = 0;
    for (int i = 0; i < gc.g.n; i += 7)
        for (int f = 0; f < GatherCase::F; ++f) {
            double s = 0;
            for (uint32_t e = gc.g.rowptr[i]; e < gc.g.rowptr[i + 1]; ++e) s += gc.X[size_t(gc.g.col[e]) * GatherCase::F + f];
            err = std::max(err, std::abs(s - gc.Y[size_t(i) * GatherCase::F + f]));
        }
    CHECK(err < 1e-2);
}

static void t_conv() {
    ConvCase c(3, 9, 11);
    double fl = c.run();
    CHECK(fl == 2.0 * 3 * 3 * 9 * 7 * 9);
    double err = 0;
    for (int oc = 0; oc < 3; ++oc) for (int y = 0; y < 7; ++y) for (int x = 0; x < 9; ++x) {
        double s = 0;
        for (int ic = 0; ic < 3; ++ic) for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx)
            s += double(c.w[((size_t(oc) * 3 + ic) * 3 + ky) * 3 + kx]) * c.in[(size_t(ic) * 9 + y + ky) * 11 + x + kx];
        err = std::max(err, std::abs(s - c.out[(size_t(oc) * 7 + y) * 9 + x]));
    }
    CHECK(err < 1e-4);
}

static void t_md() {
    MdCase md(24);
    double inter = md.run();
    CHECK(inter == 24.0 * 24.0);
    double worst = 0;
    for (int i = 0; i < md.n; ++i) {
        double ax = 0, ay = 0, az = 0;
        for (int j = 0; j < md.n; ++j) {
            double dx = double(md.x[j]) - md.x[i], dy = double(md.y[j]) - md.y[i], dz = double(md.z[j]) - md.z[i];
            double r2 = dx * dx + dy * dy + dz * dz + 0.25, i2 = 1 / r2, i6 = i2 * i2 * i2, f = (i6 * i6 - i6) * i2;
            ax += dx * f; ay += dy * f; az += dz * f;
        }
        double scale = std::max({1.0, std::abs(ax), std::abs(ay), std::abs(az)});
        worst = std::max({worst, std::abs(ax - md.fx[i]) / scale, std::abs(ay - md.fy[i]) / scale, std::abs(az - md.fz[i]) / scale});
    }
    CHECK(worst < 1e-3);
    // Newton's third law: net force ~ 0
    double sx = 0, mag = 0; for (int i = 0; i < md.n; ++i) { sx += md.fx[i]; mag += std::abs(md.fx[i]); }
    CHECK(std::abs(sx) < 1e-3 * std::max(1.0, mag));
}

static void t_rnn() {
    RnnCase r(16);
    std::vector<double> h(16, 0.1);
    for (int step = 0; step < 5; ++step) {
        std::vector<double> nh(16);
        for (int j = 0; j < 16; ++j) { nh[j] = r.x[j]; for (int k = 0; k < 16; ++k) nh[j] += double(r.Wt[size_t(k) * 16 + j]) * h[k]; }
        for (int j = 0; j < 16; ++j) h[j] = nh[j] / (1 + std::abs(nh[j]));
        r.step();
    }
    double err = 0; for (int j = 0; j < 16; ++j) err = std::max(err, std::abs(h[j] - r.h[j]));
    CHECK(err < 1e-5);
    for (float v : r.h) CHECK(std::abs(v) < 1.0f);
    CHECK(r.flops_per_step() == 2.0 * 16 * 16 + 3.0 * 16);
}

static void t_pool() {
    BenchPool pool(std::vector<int>(3, -1));
    std::atomic<int> hits[3]; for (auto& h : hits) h = 0;
    std::function<void(int)> fn = [&](int t) { hits[t]++; };
    for (int i = 0; i < 200; ++i) pool.run(fn);
    CHECK(hits[0] == 200 && hits[1] == 200 && hits[2] == 200);
    CHECK(pool.all_pinned());                              // nobody asked for a pin
#if defined(__linux__)
    BenchPool p0({0});
    std::atomic<int> cpu{-2};
    std::function<void(int)> where = [&](int) { cpu = sched_getcpu(); };
    p0.run(where);
    CHECK(p0.all_pinned() && cpu == 0);
    BenchPool bad({9999});                                 // impossible cpu: pin fails and is reported
    CHECK(!bad.all_pinned());
#endif
}

static void t_measure() {
    auto m = detail::measure([] { std::this_thread::sleep_for(std::chrono::milliseconds(1)); return 1.0; }, 0.05);
    CHECK(m.ok && m.windows >= 3);
    CHECK(m.median > 100 && m.median < 1001);              // ~1 call per ms at most
    CHECK(m.best >= m.median && m.rsd >= 0);
    double acc = 0;
    auto m2 = detail::measure([&] { for (int i = 0; i < 1000; ++i) acc += i; kernels::keep(&acc); return 1000.0; }, 0.02);
    CHECK(m2.ok && m2.median > 1e6);
}

static HardwareFingerprint fake_two_class(int cpu_a, int cpu_b) {
    HardwareFingerprint fp;
    fp.cpu.arch = "aarch64";
    fp.cpu.logical_cpus = fp.cpu.online_cpus = 2;
    for (int id : {cpu_a, cpu_b}) { CoreInfo c; c.id = id; fp.cpu.cores.push_back(c); }
    CoreClass l; l.index = 0; l.name = "little"; l.cpus = {cpu_a};
    CoreClass b; b.index = 1; b.name = "big"; b.cpus = {cpu_b};
    CacheInfo ci; ci.level = 3; ci.type = "Unified"; ci.size_bytes = 4u << 20; b.caches.push_back(ci);
    fp.cpu.classes = {l, b};
    fp.usable_threads = 2;
    fp.memory.total = 8ull << 30; fp.memory.effective_available = 4ull << 30;
    return fp;
}

static void t_full_quick() {
    auto fp = collect();
    auto cfg = qcfg();
    auto P = run_device_benchmark(fp, cfg);
    std::printf("---- quick run ----\n%s", P.summary().c_str());
    CHECK(P.fma_f32.ok && P.fma_f64.ok && P.fma_f32.median > 0.05);
    CHECK(P.gemm.count("f32_16") && P.gemm.count("f32_64") && P.gemm.count("f32_256") && P.gemm.count("f32_tall") && P.gemm.count("f64_256"));
    for (auto& kv : P.gemm) CHECK(kv.second.ok && kv.second.median > 0.01 && kv.second.median < 1e5);
    CHECK(!P.latency.empty() && P.latency.size() == P.stream_1t.size());
    for (size_t i = 0; i + 1 < P.latency.size(); ++i) CHECK(P.latency[i].bytes < P.latency[i + 1].bytes);
    CHECK(P.latency.back().bytes <= cfg.max_working_set_bytes);
    CHECK(P.gather_edges_per_s.ok && P.gather_gbps > 0 && P.gather_efficiency > 0);
    CHECK(P.conv_gflops.ok && P.md_ginteractions.ok && P.rnn_steps_per_s.ok && P.rnn_gflops > 0);
    CHECK(P.fork_join_us > 0 && P.fork_join_us < 5000 && P.call_overhead_ns > 0 && P.call_overhead_ns < 5000);
    CHECK(!P.scaling.empty() && P.scaling.front().threads == 1 && P.scaling.back().threads == fp.usable_threads);
    CHECK(P.all_cores.total > 0 && int(P.all_cores.gflops.size()) == fp.usable_threads);
    double sum = 0; for (double g : P.all_cores.gflops) sum += g;
    CHECK(std::abs(sum - P.all_cores.total) < 1e-6 * std::max(1.0, sum));
    CHECK(P.sustained.ran && P.sustained.window_gflops.size() >= 3 && P.sustained.fraction > 0.2 && P.sustained.fraction < 3.0);
    CHECK(P.duration_s < cfg.budget_seconds + 1.0);
    CHECK(P.device_id == fp.id() && P.derived_ok());
    if (kPerfChecks) {
        CHECK(P.build.optimized);
        CHECK(P.latency.back().ns > 1.5 * P.latency.front().ns);       // DRAM is slower than L1
        CHECK(P.stream_1t.front().gbps > P.stream_1t.back().gbps * 0.8); // cache-resident stream is not slower than DRAM
        CHECK(P.fma_f32.median >= 0.9 * P.fma_f64.median);
        if (fp.usable_threads >= 2) CHECK(P.scaling.back().gflops > 0.8 * P.scaling.front().gflops);   // more threads must not collapse throughput (vCPUs may share a core, so no speed-up is demanded)
    }
    std::ofstream("/tmp/bench_real.json") << P.to_json();
}

static void t_controls() {
    auto fp = collect();
    {   // governor stop before anything runs
        auto cfg = qcfg(); cfg.should_stop = [] { return true; };
        auto t0 = std::chrono::steady_clock::now();
        auto P = run_device_benchmark(fp, cfg);
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        CHECK(P.stopped_early && !P.skipped.empty() && el < 0.5 && !P.fma_f32.ok);
    }
    {   // stop mid-run: allow the first two stage checks, then refuse
        auto cfg = qcfg(); int calls = 0; cfg.should_stop = [&] { return ++calls > 3; };
        auto P = run_device_benchmark(fp, cfg);
        CHECK(P.stopped_early && P.fma_f32.ok && !P.sustained.ran);
    }
    {   // tiny budget is honoured and reported
        auto cfg = qcfg(); cfg.budget_seconds = kPerfChecks ? 0.6 : 3.0;
        auto P = run_device_benchmark(fp, cfg);
        bool budget_note = false; for (auto& s : P.skipped) budget_note |= s.find("budget") != std::string::npos;
        CHECK(budget_note);
        CHECK(P.duration_s < cfg.budget_seconds + (kPerfChecks ? 0.8 : 8.0));               // overshoot bounded by one stage
    }
    {   // memory cap is honoured, and a too-small probe is flagged instead of reported as DRAM
        auto cfg = qcfg(); cfg.max_working_set_bytes = 2ull << 20; cfg.sustained_seconds = 0;
        auto fp2 = fp; fp2.cpu.classes[0].caches.push_back(CacheInfo{3, "Unified", 32ull << 20, 64, 16, {}});
        auto P = run_device_benchmark(fp2, cfg);
        CHECK(P.latency.back().bytes <= (2u << 20) && !P.dram_probe_valid);
        bool noted = false; for (auto& n : P.notes) noted |= n.find("optimistic") != std::string::npos;
        CHECK(noted && !P.sustained.ran);
    }
    {   // free memory limits the working set: 20% of what is available
        auto cfg = qcfg(); cfg.sustained_seconds = 0;
        auto fp3 = fp; fp3.memory.effective_available = 20ull << 20;      // 4 MB cap
        auto P = run_device_benchmark(fp3, cfg);
        CHECK(P.latency.back().bytes <= (4u << 20));
    }
    {   // temperature callback is used for the sustained probe
        auto cfg = qcfg(); double t = 40; cfg.read_temp_c = [&]() -> std::optional<double> { t += 5; return t; };
        auto P = run_device_benchmark(fp, cfg);
        CHECK(P.sustained.temp_start_c && P.sustained.temp_end_c && *P.sustained.temp_end_c > *P.sustained.temp_start_c);
    }
}

static void t_classes() {
    auto cfg = qcfg(); cfg.sustained_seconds = 0;
#if defined(__linux__)
    auto P = run_device_benchmark(fake_two_class(0, 1), cfg);       // this machine has cpus 0 and 1
    CHECK(P.classes.size() == 2);
    if (P.classes.size() == 2) {
        CHECK(P.classes[0].pinned && P.classes[1].pinned && P.classes[0].gflops_1t > 0 && P.classes[1].gflops_all > 0);
        CHECK(P.classes[0].cpus == std::vector<int>{0} && P.classes[1].name == "big");
    }
    CHECK(P.all_cores.pinned && P.all_cores.cpus == (std::vector<int>{1, 0}));   // fastest class first
    auto Q = run_device_benchmark(fake_two_class(0, 4000), cfg);    // cpu 4000 does not exist
    bool unpinned = false, note = false;
    for (auto& c : Q.classes) unpinned |= !c.pinned;
    for (auto& n : Q.notes) note |= n.find("could not pin") != std::string::npos;
    CHECK(unpinned && note);
    CHECK(!Q.all_cores.pinned && Q.all_cores.total > 0);              // falls back to an unpinned run, still measured
    std::ofstream("/tmp/bench_fake.json") << Q.to_json();
#endif
}

int main() {
    t_chase(); t_gemm(); t_stream_fma(); t_graph(); t_conv(); t_md(); t_rnn(); t_pool(); t_measure();
    t_full_quick(); t_controls(); t_classes();
    std::printf("%d/%d checks passed\n", checks - fails, checks);
    return fails ? 1 : 0;
}
