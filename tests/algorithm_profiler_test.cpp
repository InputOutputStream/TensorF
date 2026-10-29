#include <cstdio>
#include <fstream>
#include "profiler/AlgorithmProfiler.hpp"
using namespace tensorf::profile;
namespace fwbench = tensorf::fwbench;

static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++fails; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_UNDEFINED__)
  constexpr bool kPerf = false;
#else
  constexpr bool kPerf = true;
#endif
static bool near(double a, double b, double rel = 1e-9) { return std::abs(a - b) <= rel * std::max(1.0, std::max(std::abs(a), std::abs(b))); }
constexpr double KB = 1024.0, MB = 1024.0 * 1024.0;

static Metric M(double v) { Metric m; m.median = m.best = v; m.windows = 5; m.ok = true; return m; }

// A hand-made device: every number is round so expected times can be computed by hand.
static CapabilityProfile synth() {
    CapabilityProfile P;
    P.gemm["f32_16"] = M(2); P.gemm["f32_64"] = M(5); P.gemm["f32_256"] = M(8); P.gemm["f32_512"] = M(7);
    P.gemm["f32_tall"] = M(6); P.gemm["f64_256"] = M(4);
    P.conv_gflops = M(4); P.rnn_gflops = 3; P.md_ginteractions = M(0.5); P.gather_gbps = 4;
    P.stream_1t = {{uint64_t(32 * KB), 0, 40}, {uint64_t(1 * MB), 0, 30}, {uint64_t(32 * MB), 0, 10}, {uint64_t(256 * MB), 0, 8}};
    P.latency = {{uint64_t(32 * KB), 2, 0}, {uint64_t(1 * MB), 8, 0}, {uint64_t(32 * MB), 100, 0}, {uint64_t(256 * MB), 120, 0}};
    P.scaling = {{1, 10, 8, 1, 1}, {2, 18, 14, 0.9, 1.75}, {4, 30, 20, 0.75, 2.5}};
    P.all_cores.gflops = {10, 9, 6, 5};
    P.all_cores.total = 30;
    P.fork_join_us = 3; P.call_overhead_ns = 30;
    P.sustained.ran = true; P.sustained.window_gflops = {10, 10, 9, 8, 8, 8}; P.sustained.fraction = 0.8;
    return P;
}

static StepCosts one(const LayerCost& l) { StepCosts s; s.layers = {l}; s.units = 1; return s; }

static void t_device_view() {
    auto P = synth();
    DeviceView d(P);
    CHECK(d.complete());
    CHECK(d.gemm_gflops(16, 16, 16) == 2 && d.gemm_gflops(64, 64, 64) == 5 && d.gemm_gflops(200, 200, 200) == 8);
    CHECK(d.gemm_gflops(1000, 1000, 1000) == 7 && d.gemm_gflops(8192, 32, 32) == 6);
    CHECK(d.gemm_gflops(4, 4, 4) == 2);                                      // below the smallest case: smallest rate
    auto P2 = P; P2.gemm.erase("f32_512"); P2.gemm.erase("f32_tall");
    DeviceView d2(P2);
    CHECK(d2.gemm_gflops(1000, 1000, 1000) == 8 && d2.gemm_gflops(8192, 32, 32) == 8);   // missing shapes fall back to f32_256
    CHECK(d.speedup_compute(1) == 1.0 && near(d.speedup_compute(2), 1.9) && near(d.speedup_compute(4), 3.0) && near(d.speedup_compute(64), 3.0));
    CHECK(d.speedup_bw(1) == 1.0 && near(d.speedup_bw(2), 1.75) && near(d.speedup_bw(3), 17.0 / 8) && near(d.speedup_bw(4), 2.5) && near(d.speedup_bw(64), 2.5));
    CHECK(d.max_threads() == 4);
    CHECK(near(d.bw_gbps(32 * KB), 40) && near(d.bw_gbps(1 * KB), 40) && near(d.bw_gbps(1e12), 8));          // clamped at both ends
    CHECK(near(d.bw_gbps(std::sqrt(32 * KB * 1 * MB)), 35));                                               // log midpoint
    CHECK(near(d.latency_ns(std::sqrt(32 * KB * 1 * MB)), 5) && near(d.latency_ns(32 * MB), 100));
    CHECK(near(d.sustained_fraction(), 0.8));
    auto P3 = P; P3.sustained.window_gflops = {1, 2};                          // too few windows to trust
    CHECK(DeviceView(P3).sustained_fraction() == 1.0);
    auto P4 = P; P4.sustained.fraction = 0.1;
    CHECK(DeviceView(P4).sustained_fraction() == 0.3);                          // clamp: one bad probe must not zero the device
    // without per-thread rates the scaling curve is used
    auto P5 = P; P5.all_cores.gflops.clear();
    CHECK(near(DeviceView(P5).speedup_compute(2), 1.8) && near(DeviceView(P5).speedup_compute(4), 3.0));
    // gather speed follows the working set once the benchmark's table size is known
    auto P6 = P; P6.gather_ws_bytes = uint64_t(32 * MB);                        // measured on a 32 MB table: 10 GB/s stream level
    DeviceView d6(P6);
    CHECK(near(d6.gather_gbps(32 * MB), 4.0));                                 // same size: unchanged
    CHECK(near(d6.gather_gbps(1 * MB), 4.0 * 3.0));                            // 30/10 = 3x, at the clamp
    CHECK(near(d6.gather_gbps(256 * MB), 4.0 * 0.8));                          // 8/10
    CHECK(near(d.gather_gbps(1 * MB), 4.0));                                   // size unknown (0): no scaling
    CHECK(near(d6.gather_gbps(0), 4.0));
    // empty profile: incomplete, finite floors
    CapabilityProfile E; DeviceView de(E);
    std::vector<std::string> miss;
    CHECK(!de.complete(&miss) && miss.size() == 7);
    CHECK(de.gemm_gflops(64, 64, 64) > 0 && de.bw_gbps(1e6) > 0 && de.latency_ns(1e6) > 0 && de.speedup_compute(4) == 4.0 && de.max_threads() == 1);
}

// The framework's own GEMM (Matrix::matmul through the selected BLAS) replaces the portable loop rate when measured.
static void t_framework_view() {
    auto P = synth();
    DeviceView plain(P);
    CHECK(!plain.has_framework());
    P.fw.available = true;
    auto add = [&](const char* k, double g, bool f64 = false) { fwbench::FwPoint p; p.key = k; p.gflops = g; p.f64 = f64; p.ok = true; P.fw.points.push_back(p); };
    add("f32_16", 20); add("f32_64", 40); add("f32_256", 80); add("f32_512", 70); add("f32_tall", 60); add("f32_skinny", 30);
    for (int t : {1, 2, 4}) { fwbench::FwScale s; s.threads = t; s.gflops = 80.0 * (t == 4 ? 3.0 : t); s.speedup = t == 4 ? 3.0 : t; s.eff = s.speedup / t; P.fw.scaling.push_back(s); }
    DeviceView d(P);
    CHECK(d.has_framework());
    CHECK(d.gemm_gflops(200, 200, 200) == 80 && d.gemm_gflops(1000, 1000, 1000) == 70 && d.gemm_gflops(4, 1024, 1024) == 30);   // framework rates, not 8 / 7
    CHECK(near(d.speedup_gemm(4), 3.0) && near(d.speedup_gemm(3), 2.5) && near(d.speedup_gemm(1), 1.0));
    // a BLAS that does not scale: GEMM gets no speed-up while the loops-based compute curve is untouched
    for (auto& s : P.fw.scaling) { s.speedup = 1.0; s.eff = 1.0 / s.threads; }
    DeviceView flat(P);
    CHECK(flat.speedup_gemm(4) == 1.0 && near(flat.speedup_compute(4), 3.0));
    // the cost model prices a dense layer with the framework rate and says nothing about a missing measurement
    auto P0 = synth();                                                               // DeviceView/CostModel keep a reference: no temporaries
    CostModel cm(P), cm0(P0);
    ExecPlan plan1; plan1.threads = 1; plan1.sustained = false;
    auto dl = layers::dense("d", 256, 512, 512, Dtype::F32);
    auto a = cm.predict(one(dl), plan1), b = cm0.predict(one(dl), plan1);
    CHECK(a.compute_s < b.compute_s / 5.0);                                          // ~70 vs 7 GFLOP/s
    bool warned = false; for (auto& n : a.notes) warned |= n.find("(BLAS) was not measured") != std::string::npos;
    CHECK(!warned);
    warned = false; for (auto& n : b.notes) warned |= n.find("(BLAS) was not measured") != std::string::npos;
    CHECK(warned);                                                                   // portable-only prediction is flagged
}

static void t_builders() {
    auto d = layers::dense("d", 256, 512, 512, Dtype::F32);
    CHECK(d.flops_fwd == 2.0 * 256 * 512 * 512 && d.flops_bwd == 2 * d.flops_fwd);
    CHECK(d.params == 512.0 * 512 + 512 && d.act_bytes == 256.0 * 512 * 4);
    CHECK(d.bytes_fwd == (256.0 * 512 + 512 * 512 + 256 * 512) * 4);
    auto di = layers::dense("d", 256, 512, 512, Dtype::F32, false);
    CHECK(di.flops_bwd == 0 && di.bytes_bwd == 0 && di.act_bytes == 0 && di.dispatches == 1);
    auto d8 = layers::dense("d", 256, 512, 512, Dtype::FP8);
    CHECK(d8.bytes_fwd < d.bytes_fwd / 3.5 && d8.bytes_fwd > d.bytes_fwd / 4.5 && d8.flops_fwd == d.flops_fwd);   // bytes shrink ~4x, flops unchanged
    auto c = layers::conv2d("c", 8, 16, 32, 3, 28, 28, 2, Dtype::F32);
    CHECK(c.flops_fwd == 2.0 * 8 * 32 * 16 * 9 * 14 * 14 && c.params == 32.0 * 16 * 9 + 32 && c.act_bytes == 8.0 * 32 * 14 * 14 * 4);
    auto a = layers::attention("a", 2, 128, 256, 4, Dtype::F32);
    CHECK(a.size() == 2);
    CHECK(a[0].flops_fwd == 2.0 * 256 * 256 * 1024);                                   // 4 projections: rows*seq=256 tokens, 256 -> 1024
    CHECK(a[1].flops_fwd == 4.0 * 2 * 128 * 128 * 256);                                // QK^T and AV
    CHECK(a[1].act_bytes == 2.0 * 4 * 128 * 128 * 4);                                  // attention probabilities kept for backward
    auto g = layers::spmm_aggregate("g", 100, 1000, 5000, 32, Dtype::F32);
    CHECK(g.gather_bytes_fwd == 1000.0 * 32 * 4 && g.gather_bytes_bwd == g.gather_bytes_fwd && g.working_set == 5000.0 * 32 * 4 && g.params == 0);
    auto r = layers::recurrent_cell("r", 8, 64, 256, 32, 4, Dtype::F32);
    CHECK(r.flops_fwd == 32.0 * 2 * 8 * (64 + 256) * 256 * 4 && r.seq_depth == 32 && r.width == 8 && r.dispatches == 96);
    CHECK(r.params == (64.0 + 256) * 256 * 4 + 256 * 4);
    auto e = layers::embedding("e", 1000, 64, 50000, Dtype::F32);
    CHECK(e.params == 50000.0 * 64 && e.gather_bytes_fwd == 1000.0 * 64 * 4);
    auto p = layers::pairwise("p", 1000, 50, 10, Dtype::F32);
    CHECK(p.interactions == 50000 && p.flops_bwd == 0 && p.act_bytes == 0);
    auto m = layers::markov_walk("m", 1000, 10, 4e6);
    CHECK(m.dep_loads == 10000 && m.working_set == 4e6 && m.seq_depth == 10);
    CHECK(tensor_bytes(Dtype::F32, 8) == 32);
}

static void t_cost_exact() {
    auto P = synth();
    CostModel cm(P);
    ExecPlan plan1; plan1.threads = 1; plan1.sustained = false;
    // compute-bound dense: 256x512x512 picks the 512 GEMM rate (7 GFLOP/s)
    auto dl = layers::dense("d", 256, 512, 512, Dtype::F32);
    auto pr = cm.predict(one(dl), plan1);
    const double flops = 6.0 * 256 * 512 * 512;
    const double t_comp = flops / 7e9, t_mem = 6291456.0 / (26e9);              // 2 MB working set: 30 + 0.2*(10-30) = 26 GB/s
    CHECK(near(pr.seconds, std::max(t_comp, t_mem) + 3 * 30e-9, 1e-12));
    CHECK(pr.bound == "compute" && near(pr.compute_s, t_comp, 1e-12) && pr.complete);
    // sustained scaling divides by 0.8
    ExecPlan plan_s; plan_s.threads = 1; plan_s.sustained = true;
    CHECK(near(cm.predict(one(dl), plan_s).seconds, pr.seconds / 0.8, 1e-12));
    // repeat multiplies
    auto dr = dl; dr.repeat = 3;
    CHECK(near(cm.predict(one(dr), plan1).seconds, 3 * pr.seconds, 1e-12));
    // threads: width 256 >= 4 threads -> compute speedup 3.0 and a fork-join per dispatch
    ExecPlan plan4; plan4.threads = 4; plan4.sustained = false;
    auto p4 = cm.predict(one(dl), plan4);
    CHECK(near(p4.seconds, t_comp / 3.0 + 3 * (3e-6 + 30e-9), 1e-9));
    // width limits threads: a layer with width 1 gets no speedup and no fork-join
    auto narrow = dl; narrow.width = 1;
    CHECK(near(cm.predict(one(narrow), plan4).seconds, pr.seconds, 1e-12));
    // width 2 -> 2 threads
    narrow.width = 2;
    CHECK(near(cm.predict(one(narrow), plan4).seconds, t_comp / 1.9 + 3 * (3e-6 + 30e-9), 1e-9));
    // memory-bound elementwise: 64 MB working set -> 8 GB/s
    auto ew = layers::elementwise("e", 16e6, Dtype::F32, 2.0, false);        // 128 MB -> clamped to the largest level? 64 MB ws
    ew.working_set = 64 * MB;
    auto pe = cm.predict(one(ew), plan1);
    const double expect_mem = ew.bytes_fwd / (std::pow(10.0, 0) * ( [&]{ double t=(std::log(64*MB)-std::log(32*MB))/(std::log(256*MB)-std::log(32*MB)); return (10 + t*(8-10)) * 1e9; }() ));
    CHECK(pe.bound == "memory" && near(pe.seconds, expect_mem + 1 * 30e-9, 1e-9));
    // memory-bound scaling uses the bandwidth curve, not the compute curve: 4 threads -> 2.5x, not 3x
    auto pe4 = cm.predict(one(ew), plan4);
    CHECK(near(pe4.seconds, expect_mem / 2.5 + (3e-6 + 30e-9), 1e-9));
    // gather: bytes / gather_gbps, scaled by the bandwidth curve when parallel
    auto gl = layers::spmm_aggregate("g", 10000, 100000, 10000, 32, Dtype::F32, false);
    auto pg = cm.predict(one(gl), plan1);
    const double tg = gl.gather_bytes_fwd / 4e9, tr = gl.bytes_fwd / (30e9);
    CHECK(pg.bound == "irregular" && pg.seconds >= tg && pg.seconds < tg + tr + 1e-5);
    CHECK(near(pg.irregular_s, tg, 1e-12));
    // chase: dependent loads x latency at the table size (32 MB -> 100 ns)
    auto ch = layers::markov_walk("m", 1000, 100, 32 * MB);
    auto pc = cm.predict(one(ch), plan1);
    CHECK(near(pc.irregular_s, 1000.0 * 100 * 100e-9, 1e-12) && pc.bound == "irregular");
    // chase parallelises over walkers with the compute curve
    CHECK(near(cm.predict(one(ch), plan4).irregular_s, 1000.0 * 100 * 100e-9 / 3.0, 1e-12));
    // recurrent: one row is a latency-bound matvec at the rnn rate; many rows reach the gemm rate
    auto r1 = layers::recurrent_cell("r", 1, 0, 256, 10, 1, Dtype::F32);
    auto pr1 = cm.predict(one(r1), plan1);
    CHECK(near(pr1.compute_s, (r1.flops_fwd + r1.flops_bwd) / 3e9, 1e-12));
    auto r64 = layers::recurrent_cell("r", 64, 0, 256, 10, 1, Dtype::F32);
    const double rate64 = std::min(3.0 * 64, cm.device().gemm_gflops(64, 256, 256));
    CHECK(near(cm.predict(one(r64), plan1).compute_s, (r64.flops_fwd + r64.flops_bwd) / (rate64 * 1e9), 1e-12));
    // sequential depth makes parallel dispatch expensive: 30 dispatches x fork-join
    auto r64p = cm.predict(one(r64), plan4);
    CHECK(near(r64p.overhead_s, r64.dispatches * (3e-6 + 30e-9), 1e-12));
    // pairwise: interactions / (0.5e9 per s)
    auto pw = layers::pairwise("p", 10000, 50, 10, Dtype::F32);
    CHECK(near(cm.predict(one(pw), plan1).compute_s, 500000.0 / 0.5e9, 1e-12));
    // core seconds: busy cores x time
    CHECK(near(p4.core_seconds, p4.seconds * 4, 1e-12) && near(pr.core_seconds, pr.seconds, 1e-12));
    // incomplete profile still gives a finite pessimistic answer and says so
    CapabilityProfile E; CostModel ce(E);
    auto pi = ce.predict(one(dl), plan1);
    CHECK(!pi.complete && std::isfinite(pi.seconds) && pi.seconds > 0 && !pi.notes.empty());
}

static void t_models() {
    auto P = synth();
    ExecPlan plan; plan.threads = 4;
    for (const auto& name : algo::names()) {
        auto m = algo::make(name);
        CHECK(m && m->name() == name);
        auto ts = m->tunables();
        CHECK(!ts.empty());
        for (const auto& t : ts) {
            CHECK(!t.values.empty() && std::is_sorted(t.values.begin(), t.values.end()));
            CHECK(t.def >= t.values.front() || t.name == "microbatch" || t.name == "checkpoint");
        }
        auto sc = m->costs(m->defaults());
        CHECK(!sc.layers.empty() && sc.units > 0 && sc.accum >= 1);
        auto pr = CostModel(P).predict(sc, plan);
        CHECK(pr.seconds > 0 && std::isfinite(pr.seconds) && pr.units_per_s > 0 && !pr.bound.empty());
        CHECK(sc.memory.total() > 0);
        if (!m->training()) CHECK(sc.memory[MemCat::Gradients] == 0 && sc.memory[MemCat::OptimizerState] == 0);
        else CHECK(sc.memory[MemCat::Gradients] == sc.memory[MemCat::Parameters] && sc.memory[MemCat::OptimizerState] == 2 * sc.memory[MemCat::Parameters]);
    }
    CHECK(algo::make("nope") == nullptr);

    // MLP memory by category, by hand
    algo::MlpModel mlp({784, 512, 256, 10});
    Config c = mlp.defaults();
    auto sc = mlp.costs(c);
    const double p = 784.0 * 512 + 512 + 512.0 * 256 + 256 + 256.0 * 10 + 10;
    CHECK(sc.memory[MemCat::Parameters] == uint64_t(4 * p) && sc.memory[MemCat::OptimizerState] == uint64_t(8 * p));
    CHECK(sc.memory[MemCat::Data] == 128u * 784 * 4);
    CHECK(sc.batch == 128 && sc.microbatch == 128 && sc.accum == 1);
    // microbatching: same total work, activations scale with the microbatch, accumulation covers the rest
    Config cm4 = c; cm4["microbatch"] = 32;
    auto s4 = mlp.costs(cm4);
    CHECK(s4.accum == 4 && s4.microbatch == 32);
    double f1 = 0, f4 = 0;
    for (auto& l : sc.layers) f1 += (l.flops_fwd + l.flops_bwd) * l.repeat;
    for (auto& l : s4.layers) f4 += (l.flops_fwd + l.flops_bwd) * l.repeat;
    CHECK(near(f1, f4, 1e-12));
    CHECK(near(double(s4.memory[MemCat::Activations]) * 4, double(sc.memory[MemCat::Activations]), 1e-9));
    CHECK(s4.memory[MemCat::Parameters] == sc.memory[MemCat::Parameters]);          // weights do not shrink
    // microbatch larger than batch is clamped
    Config big = c; big["microbatch"] = 1e9;
    CHECK(mlp.costs(big).microbatch == 128);
    // checkpointing: less activation memory, more backward work
    Config ck = c; ck["checkpoint"] = 1;
    auto sk = mlp.costs(ck);
    CHECK(sk.checkpoint && sk.memory[MemCat::Activations] < sc.memory[MemCat::Activations]);
    double b0 = 0, b1 = 0;
    for (auto& l : sc.layers) b0 += l.flops_bwd;
    for (auto& l : sk.layers) b1 += l.flops_bwd;
    CHECK(b1 > b0 && near(b1 - b0, [&] { double x = 0; for (auto& l : sc.layers) x += l.flops_fwd; return x; }(), 1e-12));   // exactly one extra forward
    // precision: FP8 activations cost about a quarter of the bytes
    Config f8 = c; f8["dtype"] = double(int(Dtype::FP8));
    auto s8 = mlp.costs(f8);
    CHECK(s8.memory[MemCat::Activations] < sc.memory[MemCat::Activations] / 3 && s8.memory[MemCat::Parameters] == sc.memory[MemCat::Parameters]);
    // bigger batch -> more activations and more time
    Config b2 = c; b2["batch"] = 512;
    CHECK(mlp.costs(b2).memory[MemCat::Activations] > 3 * sc.memory[MemCat::Activations]);
    CHECK(CostModel(P).predict(mlp.costs(b2), plan).seconds > 3 * CostModel(P).predict(sc, plan).seconds);

    // Transformer parameter count ~ 12 L d^2 + vocab*d
    algo::TransformerModel tr(256, 4, 4, 8000, 4);
    Config tc = tr.defaults();
    auto st = tr.costs(tc);
    double tp = 0; for (auto& l : st.layers) tp += l.params;
    CHECK(std::abs(tp - (12.0 * 4 * 256 * 256 + 8000.0 * 256 * 2)) / tp < 0.05);
    CHECK(st.units == 4 * 128);
    Config t512 = tc; t512["seq"] = 512;
    double a1 = 0, a2 = 0;
    for (auto& l : st.layers) if (l.name.find("scores") != std::string::npos) a1 += l.flops_fwd;
    for (auto& l : tr.costs(t512).layers) if (l.name.find("scores") != std::string::npos) a2 += l.flops_fwd;
    CHECK(near(a2 / a1, 16.0, 1e-9));                                       // attention is quadratic in context at fixed batch (x4 tokens, x4 per token)

    // GNN: more fanout -> bigger frontier, more time, more data; frontier capped by graph size
    algo::GnnModel gnn(128, 128, 2, 16, 1e6);
    Config g5 = gnn.defaults(), g25 = g5; g5["fanout"] = 5; g25["fanout"] = 25;
    auto s5 = gnn.costs(g5), s25 = gnn.costs(g25);
    {   // dense work on the frontier grows with fanout, gather work with fanout squared: total between 5x and 25x
        auto t5 = CostModel(P).predict(s5, plan), t25 = CostModel(P).predict(s25, plan);
        const double r = t25.seconds / t5.seconds;
        CHECK(r > 5.0 && r < 25.0);
        CHECK(t25.irregular_s / t5.irregular_s > 15.0 && t25.irregular_s / t5.irregular_s < 26.0);
    }
    CHECK(gnn.data_bytes(g25, 256) > 10 * gnn.data_bytes(g5, 256));
    algo::GnnModel tiny(128, 128, 3, 16, 5000);
    Config gc = tiny.defaults(); gc["fanout"] = 25; gc["batch"] = 2048;
    CHECK(tiny.data_bytes(gc, 2048) <= 5000u * 128 * 4);                     // cannot need more nodes than exist
    // inference-style models: the particle count is fixed, neighbours drive the time
    algo::ParticleModel pm(20000);
    Config n32 = pm.defaults(), n128 = n32; n32["neighbors"] = 32; n128["neighbors"] = 128;
    CHECK(near(CostModel(P).predict(pm.costs(n128), plan).compute_s / CostModel(P).predict(pm.costs(n32), plan).compute_s, 4.0, 1e-9));
    CHECK(pm.costs(n32).units == 20000);
}

static void t_bounds() {
    auto P = synth();
    ExecPlan plan; plan.threads = 4;
    auto bound = [&](const std::string& name, Config tweak = {}) {
        auto m = algo::make(name);
        Config c = m->defaults();
        for (auto& kv : tweak) c[kv.first] = kv.second;
        return CostModel(P).predict(m->costs(c), plan).bound;
    };
    CHECK(bound("mlp", {{"batch", 1024}}) == "compute");
    CHECK(bound("cnn", {{"batch", 256}}) == "compute");
    CHECK(bound("transformer", {{"batch", 8}}) == "compute");
    {   // a GNN is not simply "irregular": the share depends on feature width; at the default it is a large minority
        algo::GnnModel g; auto pr = CostModel(P).predict(g.costs(g.defaults()), plan);
        CHECK(pr.irregular_s > 0.2 * pr.seconds && pr.irregular_s < 0.8 * pr.seconds);
        // with very narrow features and deep sampling the gather dominates
        algo::GnnModel narrow(8, 8, 3, 4, 1e6); Config c = narrow.defaults(); c["fanout"] = 15;
        auto pn = CostModel(P).predict(narrow.costs(c), plan);
        CHECK(pn.bound == "irregular");
    }
    CHECK(bound("markov") == "irregular");
    CHECK(bound("particles") == "compute");
    // the same algorithm changes character with its knobs: a tiny RNN batch is latency/overhead dominated
    auto rnn = algo::make("rnn");
    Config tiny = rnn->defaults(); tiny["batch"] = 1; tiny["bptt"] = 16;       // one sequence: nothing to split across threads
    ExecPlan one_t; one_t.threads = 1;
    ExecPlan four; four.threads = 4;
    CHECK(CostModel(P).predict(rnn->costs(tiny), four).seconds >= 0.95 * CostModel(P).predict(rnn->costs(tiny), one_t).seconds);   // threads do not help a single sequential chain
    Config eight = tiny; eight["batch"] = 8;                                     // with rows to split, they do
    CHECK(CostModel(P).predict(rnn->costs(eight), four).seconds < 0.6 * CostModel(P).predict(rnn->costs(eight), one_t).seconds);
    // while a wide model scales
    auto mlp = algo::make("mlp");
    Config wide = mlp->defaults(); wide["batch"] = 1024;
    CHECK(CostModel(P).predict(mlp->costs(wide), four).seconds < 0.5 * CostModel(P).predict(mlp->costs(wide), one_t).seconds);
}

static void t_profiler_and_calibration() {
    auto P = synth();
    MemoryBudget B; B.tensor_budget_bytes = 400ull << 20; B.max_additional_bytes = 500ull << 20; B.overhead_factor = 1.25;
    auto model = algo::make("mlp");
    AlgorithmProfiler ap(model, P, B);
    Config c = model->defaults();
    ExecPlan plan; plan.threads = 2;
    auto r = ap.evaluate(c, plan);
    CHECK(r.feasible && r.fit.fits && r.time.seconds > 0 && r.error == 0 && r.real_bytes == double(r.costs.memory.total()) * 1.25);
    // too big for the budget
    Config huge = c; huge["batch"] = 1024; huge["dtype"] = 0;
    MemoryBudget small = B; small.tensor_budget_bytes = 1 << 20;
    ap.set_budget(small);
    auto rs = ap.evaluate(huge, plan);
    CHECK(!rs.feasible && !rs.fit.fits && !rs.notes.empty());
    ap.set_budget(B);
    ExecPlan many; many.threads = 64;
    bool noted = false; for (auto& n : ap.evaluate(c, many).notes) noted |= n.find("more threads") != std::string::npos;
    CHECK(noted);
    // calibration: the "real device" is 2x slower than the model with +-5% noise
    Config c2 = c; c2["batch"] = 256; Config c3 = c; c3["batch"] = 512;
    std::vector<std::pair<Config, ExecPlan>> runs = {{c, plan}, {c2, plan}, {c3, plan}};
    const double mult[3] = {1.9, 2.0, 2.1};
    int calls = 0;
    std::map<double, double> raw;
    ProbeFn probe = [&](const Config& cf, const ExecPlan& pl) {
        const double base = ap.raw_prediction(cf, pl).seconds;
        const int idx = int(std::lround(std::log2(cfg_get(cf, "batch", 128) / 128.0)));      // 128 -> 0, 256 -> 1, 512 -> 2
        ++calls;
        return base * mult[std::clamp(idx, 0, 2)];
    };
    auto fit = ap.calibrate(probe, runs, 3);
    CHECK(fit.valid && fit.samples == 3 && calls == 9);
    CHECK(near(fit.scale, std::cbrt(1.9 * 2.0 * 2.1), 1e-9) && fit.spread > 0.04 && fit.spread < 0.06);
    auto rc = ap.evaluate(c2, plan);
    CHECK(near(rc.time.seconds, ap.raw_prediction(c2, plan).seconds * fit.scale, 1e-12) && near(rc.error, fit.spread, 1e-12));
    CHECK(near(rc.time.units_per_s, rc.costs.units / rc.time.seconds, 1e-12));
    // calibrating again does not compound the scale
    auto fit2 = ap.calibrate(probe, runs, 3);
    CHECK(near(fit2.scale, fit.scale, 1e-9));
    // garbage probes are ignored; no valid sample leaves the model untouched
    const double before = ap.scale();
    auto bad = ap.calibrate([](const Config&, const ExecPlan&) { return std::nan(""); }, runs, 1);
    CHECK(!bad.valid && ap.scale() == before);
    auto zero = ap.calibrate([](const Config&, const ExecPlan&) { return 0.0; }, runs, 1);
    CHECK(!zero.valid && ap.scale() == before);
    // swapping the algorithm needs nothing but a different model object
    for (const auto& name : algo::names()) {
        MemoryBudget big; big.tensor_budget_bytes = 8ull << 30; big.max_additional_bytes = 10ull << 30;
        AlgorithmProfiler other(algo::make(name), P, big);
        auto res = other.evaluate(other.model().defaults(), plan);
        CHECK(res.feasible && res.time.units_per_s > 0);
        auto js = res.to_json();
        CHECK(js.find("\"bound\"") != std::string::npos && js.find("\"tensor_total\"") != std::string::npos);
    }
    std::ofstream("/tmp/algo_profile.json") << ap.evaluate(c, plan).to_json();
}

// Compare the model with the real kernels at shapes the benchmark did NOT use.
static void t_validate_against_kernels() {
    auto fp = collect();
    auto cfg = BenchConfig::quick();
    cfg.sustained_seconds = 0;
    auto P = run_device_benchmark(fp, cfg);
    CostModel cm(P);
    ExecPlan plan; plan.threads = 1; plan.sustained = false;
    auto measure = [&](auto&& unit) { return 1.0 / tensorf::profile::detail::measure(unit, 0.1).median; };    // seconds per call
    struct Row { const char* name; double pred, meas; };
    std::vector<Row> rows;

    // GEMM layers are priced with the framework rate when it was measured, so they are checked against what the
    // framework runs (Matrix::matmul through the selected BLAS, one thread), not against the portable loop.
    const bool via_fw = P.fw.usable();
    fwbench::BlasThreads one_thread(1);
    auto gemm_time = [&](int M, int N, int K) {
        if (via_fw) { fwbench::GemmRunner<float> g(M, N, K); return measure([&] { g.run(); return 1.0; }); }
        kernels::GemmCase<float> g(M, N, K); return measure([&] { g.run(); return 1.0; });
    };
    { double m = gemm_time(192, 300, 500);
      rows.push_back({"gemm 192x300x500", cm.predict(one(layers::dense("d", 192, 500, 300, Dtype::F32, false)), plan).seconds, m}); }
    { double m = gemm_time(1000, 64, 64);
      rows.push_back({"gemm 1000x64x64", cm.predict(one(layers::dense("d", 1000, 64, 64, Dtype::F32, false)), plan).seconds, m}); }
    { kernels::ConvCase c(8, 48, 48);
      double m = measure([&] { c.run(); return 1.0; });
      rows.push_back({"conv 8ch 48x48", cm.predict(one(layers::conv2d("c", 1, 8, 8, 3, 46, 46, 1, Dtype::F32, false)), plan).seconds, m}); }
    { kernels::GatherCase g(50000);
      double m = measure([&] { g.run(); return 1.0; });
      rows.push_back({"gather 50k nodes", cm.predict(one(layers::spmm_aggregate("g", 50000, double(g.g.edges()), 50000, 32, Dtype::F32, false)), plan).seconds, m}); }
    { kernels::RnnCase r(128);
      double m = measure([&] { r.step(); return 1.0; });
      rows.push_back({"rnn H=128 step", cm.predict(one(layers::recurrent_cell("r", 1, 0, 128, 1, 1, Dtype::F32, false)), plan).seconds, m}); }
    { kernels::MdCase md(2048);
      double m = measure([&] { md.forces(); return 1.0; });
      rows.push_back({"md 2048 brute", cm.predict(one(layers::pairwise("p", 2048, 2048, 1e9, Dtype::F32)), plan).seconds, m}); }
    { kernels::ChaseTable ct(32ull << 20); uint32_t pos = 0;
      double m = measure([&] { pos = ct.walk(pos, 20000); kernels::keep(&pos); return 1.0; });
      rows.push_back({"chase 32MB x20000", cm.predict(one(layers::markov_walk("m", 1, 20000, 32.0 * MB)), plan).seconds, m}); }

    std::printf("---- model vs real kernels (1 thread; gemm rows via %s) ----\n", via_fw ? P.fw.backend.c_str() : "portable loop");
    for (auto& r : rows) {
        std::printf("%-22s predicted %9.3f ms  measured %9.3f ms  ratio %.2f\n", r.name, r.pred * 1e3, r.meas * 1e3, r.pred / r.meas);
        CHECK(std::isfinite(r.pred) && r.pred > 0 && r.meas > 0);
        // within ~2.5x at unseen shapes, no wild misses; irregular gather is the least predictable (cache residency), allow 3.5x
        const double hi = std::string(r.name).rfind("gather", 0) == 0 ? 3.5 : 2.8;
        if (kPerf) CHECK(r.pred / r.meas > 0.35 && r.pred / r.meas < hi);
    }
}

int main() {
    t_device_view(); t_framework_view(); t_builders(); t_cost_exact(); t_models(); t_bounds(); t_profiler_and_calibration(); t_validate_against_kernels();
    std::printf("%d/%d checks passed\n", checks - fails, checks);
    return fails ? 1 : 0;
}