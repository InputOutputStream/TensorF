/*
 * profiler_workload_test.cpp
 * ==========================
 * Tests for WorkloadModel.hpp (step 2a of the profiler rewrite).
 *
 *   g++ -std=c++20 -O2 -Wall -Wextra -Iinclude tests/profiler_workload_test.cpp -o bin/profiler_workload_test -pthread
 *   ./bin/profiler_workload_test            # ~5 s: includes one short real capability run
 *   ./bin/profiler_workload_test --quick    # pure tests only
 *
 * Almost everything runs against a SYNTHETIC CapabilityVector with round numbers, so
 * every expected value can be worked out by hand and nothing depends on the host:
 *
 *   f32 10 GFLOP/s (skinny 4, i8 20, conv 8, nbody 12, f64 5)
 *   bandwidth L1 100 / L2 60 / L3 30 / DRAM 10 GB/s, triad 10 -> 18 -> 20 GB/s at 1/2/4 threads
 *   gather 2 GB/s, weak scaling efficiency 1.0 / 0.95 / 0.8 at 1 / 2 / 4 threads
 *   cache 32 KB / 256 KB / 8 MB, tiny op 10 ns, barrier round trip 2 us
 */

#include <cstdio>
#include <cstring>
#include <string>

#include "profiler/tmp/WorkloadModel.hpp"

using namespace tensorf::prof;
using namespace tensorf::prof::wl;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failed; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while (0)
#define CHECK_NEAR(a, b, tol) do { ++g_checks; double _a=(double)(a), _b=(double)(b); if (!(std::fabs(_a-_b) <= (tol))) { ++g_failed; \
    std::printf("  FAIL %s:%d  %s ~ %s  (%g vs %g)\n", __FILE__, __LINE__, #a, #b, _a, _b); } } while (0)
#define CHECK_REL(a, b, rel) do { ++g_checks; double _a=(double)(a), _b=(double)(b); if (!(std::fabs(_a-_b) <= (rel) * std::fabs(_b))) { ++g_failed; \
    std::printf("  FAIL %s:%d  %s ~ %s within %g%%  (%g vs %g)\n", __FILE__, __LINE__, #a, #b, (rel) * 100.0, _a, _b); } } while (0)
#define SECTION(name) std::printf("[%s]\n", name)

// ─── Synthetic device ────────────────────────────────────────────────────────

static CapabilityVector make_cv(bool with_scaling = true) {
    CapabilityVector cv;
    cv.f32_gflops_best = 10; cv.f32_gflops_skinny = 4; cv.i8_gops = 20; cv.conv_gflops = 8; cv.nbody_gflops = 12; cv.f64_gflops = 5;
    cv.bw_l1_gbs = 100; cv.bw_l2_gbs = 60; cv.bw_l3_gbs = 30; cv.bw_dram_gbs = 10;
    cv.triad_1t_gbs = 10; cv.triad_max_gbs = 20;
    cv.triad_sweep = {{1, 10.0}, {2, 18.0}, {4, 20.0}};
    cv.dram_saturation_threads = 4;
    cv.gather_gbs = 2.0;
    cv.tiny_op_ns = 10; cv.barrier_roundtrip_us = 2.0; cv.thread_spawn_us = 50;
    cv.recommended_class = CoreClass::UNIFORM;
    if (with_scaling) {
        ClusterScaling sc; sc.cls = CoreClass::UNIFORM; sc.cpus = 4;
        sc.independent.points = {{1, 10.0, 1.0}, {2, 19.0, 0.95}, {4, 32.0, 0.8}};
        sc.independent.ok = true;
        cv.scaling.push_back(sc);
        cv.recommended_threads = 4;
    }
    return cv;
}

static DeviceFingerprint make_fp(uint64_t budget = 4ull << 30) {
    DeviceFingerprint fp;
    CoreCluster c; c.cpus = {0, 1, 2, 3}; c.physical_cores = 4;
    c.l1d_bytes = 32 << 10; c.l2_bytes = 256 << 10; c.l3_bytes = 8ull << 20;
    fp.clusters.push_back(c);
    fp.logical_cpus = 4;
    fp.memory.total_bytes = budget * 2; fp.memory.available_bytes = budget; fp.memory.safe_budget_bytes = budget;
    return fp;
}

static Workload one_layer(const LayerCost& l, double units = 1.0) {
    Workload w; w.name = "t"; w.layers = {l}; w.units_per_micro_step = units; return w;
}

static RunSpec infer_spec() { RunSpec r; r.mode = Mode::INFER; return r; }
static RunSpec train_spec() { RunSpec r; r.mode = Mode::TRAIN; r.opt_states = 2; return r; }

// ═════════════════════════════════════════════════════════════════════════════
// Layer library
// ═════════════════════════════════════════════════════════════════════════════

static void test_dense() {
    SECTION("dense: flops, bytes, memory");
    const double M = 100, K = 200, N = 50;
    LayerCost inf = dense(M, K, N, infer_spec(), "d", true);
    CHECK_NEAR(inf.flops_fwd, 2.0 * M * K * N, 1e-6);
    CHECK_NEAR(inf.flops_bwd, 0.0, 1e-12);
    CHECK_NEAR(inf.bytes_fwd, (M * K + K * N + M * N) * 4.0, 1e-6);
    CHECK_NEAR(inf.mem.params, (K * N + N) * 4.0, 1e-6);
    CHECK_NEAR(inf.mem.grads, 0.0, 1e-12);                  // inference holds no gradients
    CHECK_NEAR(inf.mem.optimizer, 0.0, 1e-12);

    RunSpec t = train_spec();
    LayerCost tr = dense(M, K, N, t);
    CHECK_NEAR(tr.flops_bwd, 4.0 * M * K * N, 1e-6);        // dW + dX
    CHECK_NEAR(tr.flops(), 6.0 * M * K * N, 1e-6);          // the textbook 3x forward
    CHECK_NEAR(tr.mem.grads, (K * N + N) * 4.0, 1e-6);
    CHECK_NEAR(tr.mem.optimizer, (K * N + N) * 4.0 * 2.0, 1e-6);   // Adam: m and v
    CHECK_NEAR(tr.mem.activations, M * K * 4.0, 1e-6);      // the input is kept for dW
    CHECK(tr.gemm_m == M);

    t.need_input_grad = false;                              // first layer: no dX
    CHECK_NEAR(dense(M, K, N, t).flops_bwd, 2.0 * M * K * N, 1e-6);
    t.need_input_grad = true; t.trainable = false;          // frozen weights: dX only, no grads or optimizer state
    LayerCost fr = dense(M, K, N, t);
    CHECK_NEAR(fr.flops_bwd, 2.0 * M * K * N, 1e-6);
    CHECK_NEAR(fr.mem.grads, 0.0, 1e-12);
    CHECK_NEAR(fr.mem.optimizer, 0.0, 1e-12);

    t.trainable = true; t.opt_states = 0;                   // SGD: no optimizer state
    CHECK_NEAR(dense(M, K, N, t).mem.optimizer, 0.0, 1e-12);

    RunSpec h = train_spec(); h.dtype = Dtype::F16;         // storage halves
    CHECK_NEAR(dense(M, K, N, h).mem.params, (K * N + N) * 2.0, 1e-6);
    CHECK_NEAR(dense(M, K, N, h).mem.optimizer, (K * N + N) * 4.0 * 2.0, 1e-6);   // optimizer state stays f32
}

static void test_transformer_layer_shape() {
    SECTION("a transformer layer is ~12*E^2 parameters (the old advisor assumed 2*E^2)");
    const double E = 256, T = 128, B = 4, M = B * T;
    RunSpec r = infer_spec();
    double params = 0, flops = 0;
    for (int i = 0; i < 4; ++i) { LayerCost c = dense(M, E, E, r, "qkvo", false); params += c.mem.params / 4.0; flops += c.flops_fwd; }
    { LayerCost c = dense(M, E, 4 * E, r, "up", false);   params += c.mem.params / 4.0; flops += c.flops_fwd; }
    { LayerCost c = dense(M, 4 * E, E, r, "down", false); params += c.mem.params / 4.0; flops += c.flops_fwd; }
    CHECK_REL(params, 12.0 * E * E, 1e-9);
    // matmul flops of the layer: 2 * M * 12E^2, six times what the old estimate counted (2 * 2 * M * E^2)
    CHECK_REL(flops, 2.0 * M * 12.0 * E * E, 1e-9);
    LayerCost att = attention(B, 8, T, T, E, r);
    CHECK(att.flops_fwd > 0);
    CHECK_NEAR(att.flops_fwd, 4.0 * B * T * T * E + 5.0 * B * 8 * T * T, 1e-3);
}

static void test_attention() {
    SECTION("attention: quadratic scores, flash-style, cached decode");
    RunSpec r = train_spec();
    LayerCost a1 = attention(2, 4, 64, 64, 128, r);
    LayerCost a2 = attention(2, 4, 128, 128, 128, r);
    CHECK_REL(a2.mem.scratch / a1.mem.scratch, 4.0, 1e-9);            // score matrix grows with T^2
    CHECK(a2.flops_fwd / a1.flops_fwd > 3.9);

    LayerCost fl = attention(2, 4, 128, 128, 128, r, true, /*materialise_scores=*/false);
    CHECK_NEAR(fl.mem.scratch, 0.0, 1e-12);                            // flash-style: nothing quadratic is stored
    CHECK(fl.mem.activations < a2.mem.activations);
    CHECK(fl.bytes_fwd < a2.bytes_fwd);

    // Cached decode: one query against a long cache is far less arithmetic-dense than prefill.
    LayerCost pre = attention(1, 8, 512, 512, 512, infer_spec(), true, false);
    LayerCost dec = attention(1, 8, 1, 512, 512, infer_spec(), true, false);
    CHECK(dec.intensity() < pre.intensity() * 0.1);
    CHECK(dec.parallel_width < pre.parallel_width);
    CHECK_NEAR(dec.flops_bwd, 0.0, 1e-12);
}

static void test_spmm() {
    SECTION("spmm: gather is irregular");
    LayerCost s = spmm_aggregate(10000, 80000, 64, train_spec());
    CHECK_NEAR(s.flops_fwd, 2.0 * 80000 * 64, 1e-6);
    CHECK(s.irregular_frac > 0.7 && s.irregular_frac < 1.0);
    CHECK_NEAR(s.flops_bwd, s.flops_fwd, 1e-6);                        // symmetric adjacency: same SpMM backward
    RunSpec first = train_spec(); first.need_input_grad = false;
    CHECK_NEAR(spmm_aggregate(10000, 80000, 64, first).flops_bwd, 0.0, 1e-12);
    CHECK(s.intensity() < 1.0);                                        // memory-hungry, not compute-dense
}

static void test_recurrent() {
    SECTION("recurrent: dependent chain");
    const double B = 8, in = 32, h = 64, steps = 100;
    LayerCost lstm = recurrent_cell(B, in, h, steps, 4, train_spec());
    CHECK_NEAR(lstm.seq_depth, steps, 1e-12);
    CHECK_NEAR(lstm.mem.params, ((in + h) * 4 * h + 4 * h) * 4.0, 1e-6);
    LayerCost gru = recurrent_cell(B, in, h, steps, 3, train_spec());
    CHECK(gru.flops_fwd < lstm.flops_fwd);
    CHECK_NEAR(lstm.flops_bwd, 2.0 * lstm.flops_fwd, 1e-6);
    CHECK_NEAR(lstm.parallel_width, B * h, 1e-12);
    LayerCost longer = recurrent_cell(B, in, h, 2 * steps, 4, train_spec());
    CHECK_REL(longer.mem.activations / lstm.mem.activations, 2.0, 1e-9);   // BPTT keeps every step
}

static void test_particles() {
    SECTION("particles: all pairs vs neighbour list");
    LayerCost all = particle_interaction(1000, 0, 20, 10, infer_spec());
    CHECK_NEAR(all.flops_fwd, 10.0 * 1000 * 1000 * 20, 1e-3);
    CHECK_NEAR(all.irregular_frac, 0.0, 1e-12);
    CHECK_NEAR(all.seq_depth, 10, 1e-12);
    LayerCost nl = particle_interaction(1000, 30, 20, 10, infer_spec());
    CHECK(nl.flops_fwd < all.flops_fwd);
    CHECK(nl.irregular_frac > 0.0);
    CHECK(all.intensity() > nl.intensity());
}

static void test_embedding_norm_elementwise_optimizer() {
    SECTION("embedding, norm, elementwise, optimizer");
    RunSpec t = train_spec();
    LayerCost dg = embedding(512, 50000, 128, t, true);
    LayerCost sg = embedding(512, 50000, 128, t, false);
    CHECK_NEAR(dg.mem.grads, 50000.0 * 128 * 4, 1e-6);                 // dense gradient table
    CHECK_NEAR(sg.mem.grads, 512.0 * 128 * 4, 1e-6);                   // sparse rows only
    CHECK(dg.bytes_bwd > sg.bytes_bwd);
    CHECK_NEAR(dg.flops_fwd, 0.0, 1e-12);

    LayerCost ln = norm(1024, 256, t, false), rms = norm(1024, 256, t, true);
    CHECK(rms.flops_fwd < ln.flops_fwd);
    CHECK_NEAR(ln.mem.params, 2.0 * 256 * 4, 1e-6);
    CHECK_NEAR(rms.mem.params, 256.0 * 4, 1e-6);

    LayerCost ew = elementwise(1e6, 1, 3, infer_spec());
    CHECK_NEAR(ew.bytes_fwd, 1e6 * 4 * 3, 1e-6);

    LayerCost adam = optimizer_update(1e6, t);
    LayerCost sgd = optimizer_update(1e6, [] { RunSpec r; r.opt_states = 0; return r; }());
    CHECK(adam.once_per_step && sgd.once_per_step);
    CHECK(adam.bytes_fwd > sgd.bytes_fwd);
    CHECK_NEAR(adam.bytes_fwd, 1e6 * 4 * (2 + 2 * 2), 1e-6);
}

// ═════════════════════════════════════════════════════════════════════════════
// Memory, checkpointing, totals
// ═════════════════════════════════════════════════════════════════════════════

static Workload stack_of_dense(int n, double M = 64, double D = 64, Mode mode = Mode::TRAIN) {
    RunSpec r; r.mode = mode; r.opt_states = 2;
    Workload w; w.name = "stack"; w.units_per_micro_step = M;
    for (int i = 0; i < n; ++i) w.layers.push_back(dense(M, D, D, r, "d" + std::to_string(i)));
    return w;
}

static void test_memory_estimate() {
    SECTION("memory: categories add, scratch is the max");
    Workload w = stack_of_dense(4);
    MemoryEstimate m = estimate_memory(w, 1.0);
    const double P = (64.0 * 64 + 64) * 4;
    CHECK_NEAR(m.params, 4 * P, 1e-6);
    CHECK_NEAR(m.grads, 4 * P, 1e-6);
    CHECK_NEAR(m.optimizer, 8 * P, 1e-6);
    CHECK_NEAR(m.activations, 4.0 * 64 * 64 * 4, 1e-6);
    CHECK_NEAR(m.scratch, 64.0 * 64 * 4, 1e-6);                         // the largest one, not the sum
    CHECK_NEAR(m.tensor_total, m.params + m.grads + m.optimizer + m.activations + m.scratch, 1e-6);
    CHECK_NEAR(estimate_memory(w, 1.5).process_total, 1.5 * m.tensor_total, 1e-6);
    // Inference needs less than training for the same layers. With activations dominating
    // (small shapes) the gap is modest; with parameters dominating (many weights, few rows) it is ~4x
    // (training adds grads + two Adam moments = 3 extra copies of the parameters).
    CHECK(estimate_memory(stack_of_dense(4, 64, 64, Mode::INFER), 1.0).tensor_total < m.tensor_total * 0.5);
    const double inf_p = estimate_memory(stack_of_dense(4, 4, 512, Mode::INFER), 1.0).tensor_total;
    const double trn_p = estimate_memory(stack_of_dense(4, 4, 512, Mode::TRAIN), 1.0).tensor_total;
    CHECK(trn_p / inf_p > 3.5 && trn_p / inf_p < 4.1);
}

static void test_checkpointing() {
    SECTION("checkpointing: less activation memory, more flops");
    Workload w = stack_of_dense(8);
    const double base_act = estimate_memory(w, 1.0).activations;
    const double base_fl = totals(w).flops;
    Workload same = with_checkpointing(w, 1);
    CHECK_NEAR(estimate_memory(same, 1.0).activations, base_act, 1e-6);
    CHECK_NEAR(totals(same).flops, base_fl, 1e-3);

    for (int s : {2, 3}) {
        Workload c = with_checkpointing(w, s);
        CHECK(estimate_memory(c, 1.0).activations < base_act);
        CHECK(totals(c).flops > base_fl);
        CHECK(totals(c).flops < base_fl * 1.4);                         // costs about one extra forward, not more
        CHECK(c.name.find("ckpt") != std::string::npos);
        // persistent memory is untouched
        CHECK_NEAR(estimate_memory(c, 1.0).params, estimate_memory(w, 1.0).params, 1e-6);
    }
    // 2 segments of 4 equal layers: boundary(1) + largest segment(4) = 5 units of 8
    const double unit = 64.0 * 64 * 4;
    CHECK_NEAR(estimate_memory(with_checkpointing(w, 2), 1.0).activations, 5 * unit, 1e-6);
    // empty and oversized segment counts are safe
    CHECK_NEAR(estimate_memory(with_checkpointing(Workload{}, 4), 1.0).tensor_total, 0.0, 1e-12);
    CHECK(estimate_memory(with_checkpointing(w, 1000), 1.0).activations > 0);
}

static void test_totals_and_micro_steps() {
    SECTION("totals: micro-steps repeat layers, not the optimizer");
    RunSpec r = train_spec();
    Workload w; w.units_per_micro_step = 8;
    w.layers = {dense(8, 64, 64, r), optimizer_update(64.0 * 64, r)};
    const double one = totals(w).flops;
    w.micro_steps = 4;
    const double four = totals(w).flops;
    const double dense_fl = w.layers[0].flops(), opt_fl = w.layers[1].flops();
    CHECK_NEAR(one, dense_fl + opt_fl, 1e-6);
    CHECK_NEAR(four, 4 * dense_fl + opt_fl, 1e-6);
    CHECK_NEAR(w.units_per_step(), 32, 1e-12);
}

// ═════════════════════════════════════════════════════════════════════════════
// Calibration
// ═════════════════════════════════════════════════════════════════════════════

static void test_calibration() {
    SECTION("calibration");
    const int D = (int)OpKind::DENSE, S = (int)OpKind::SPMM;
    Calibration c;
    CHECK_NEAR(c.factor_for(OpKind::DENSE), 1.0, 1e-12);
    CHECK_NEAR(c.error_for(OpKind::DENSE), Calibration::kUncalibratedError, 1e-12);

    // Exact 2x miss: factor 2, no residual error.
    c.fit({{D, 1.0, 2.0}, {D, 2.0, 4.0}, {D, 0.5, 1.0}});
    CHECK_REL(c.factor_for(OpKind::DENSE), 2.0, 1e-9);
    CHECK_NEAR(c.error_for(OpKind::DENSE), 0.0, 1e-9);
    CHECK(c.calibrated(D));
    CHECK(!c.calibrated(S));
    CHECK_NEAR(c.factor_for(OpKind::SPMM), 1.0, 1e-12);                 // untouched kind: identity

    // One outlier does not move the median factor.
    Calibration o;
    o.fit({{D, 1, 1.5}, {D, 1, 1.5}, {D, 1, 1.5}, {D, 1, 1.5}, {D, 1, 60.0}});
    CHECK_REL(o.factor_for(OpKind::DENSE), 1.5, 1e-9);

    // Noisy but unbiased: factor near truth, error reflects the spread.
    Calibration n;
    std::vector<Observation> ob;
    const double ratios[] = {0.8, 1.25, 0.9, 1.1, 1.0, 1.2, 0.85};
    for (double r : ratios) ob.push_back({D, 1.0, 3.0 * r});
    n.fit(ob);
    CHECK_REL(n.factor_for(OpKind::DENSE), 3.0, 0.05);
    CHECK(n.error_for(OpKind::DENSE) > 0.05 && n.error_for(OpKind::DENSE) < 0.35);

    // Too few samples: still fits, but refuses to claim a tight error.
    Calibration f; f.fit({{D, 1.0, 2.0}});
    CHECK_REL(f.factor_for(OpKind::DENSE), 2.0, 1e-9);
    CHECK(f.error_for(OpKind::DENSE) >= 0.30);

    // Garbage observations are ignored.
    Calibration g;
    g.fit({{D, 0.0, 1.0}, {D, 1.0, 0.0}, {D, -1.0, 1.0}, {D, std::numeric_limits<double>::quiet_NaN(), 1.0},
           {D, 1.0, std::numeric_limits<double>::infinity()}});
    CHECK(!g.calibrated(D));

    // Factors are clamped so one broken measurement cannot produce absurd predictions.
    Calibration k; k.fit({{D, 1.0, 1e6}, {D, 1.0, 1e6}, {D, 1.0, 1e6}});
    CHECK_NEAR(k.factor_for(OpKind::DENSE), Calibration::kMaxFactor, 1e-12);
    Calibration k2; k2.fit({{D, 1.0, 1e-6}, {D, 1.0, 1e-6}, {D, 1.0, 1e-6}});
    CHECK_NEAR(k2.factor_for(OpKind::DENSE), Calibration::kMinFactor, 1e-12);

    // Whole-step (global) observations are the fallback for kinds without their own.
    Calibration gl; gl.fit({{kOpKinds, 1.0, 1.8}, {kOpKinds, 1.0, 1.8}, {D, 1.0, 3.0}});
    CHECK_REL(gl.factor_for(OpKind::DENSE), 3.0, 1e-9);                 // own wins
    CHECK_REL(gl.factor_for(OpKind::CONV2D), 1.8, 1e-9);                // falls back to global
}

// ═════════════════════════════════════════════════════════════════════════════
// Prediction (synthetic device, hand-checkable)
// ═════════════════════════════════════════════════════════════════════════════

static void test_predict_compute_bound() {
    SECTION("predict: compute-bound dense");
    CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    const double n = 1024;
    LayerCost d = dense(n, n, n, infer_spec(), "big", false);
    Prediction p = predict(one_layer(d, n), fp, cv);
    CHECK(p.valid);
    CHECK_REL(p.step_s, d.flops_fwd / 10e9, 0.01);                      // 2.147e9 / 10e9
    CHECK(p.dominant == Bound::COMPUTE);
    CHECK(p.layers.size() == 1 && p.layers[0].bound == Bound::COMPUTE);
    CHECK_REL(p.units_per_s, n / p.step_s, 1e-9);

    // 2x the rows: 2x the time (no hidden constant)
    Prediction p2 = predict(one_layer(dense(2 * n, n, n, infer_spec(), "big2", false), 2 * n), fp, cv);
    CHECK_REL(p2.step_s / p.step_s, 2.0, 0.02);

    // f64 runs at the f64 rate (half here); int8 at the int8 rate (twice).
    RunSpec r64 = infer_spec(); r64.dtype = Dtype::F64;
    RunSpec r8 = infer_spec();  r8.dtype = Dtype::I8;
    Prediction pf = predict(one_layer(dense(n, n, n, r64, "f64", false)), fp, cv);
    Prediction pi = predict(one_layer(dense(n, n, n, r8, "i8", false)), fp, cv);
    CHECK_REL(pf.step_s / p.step_s, 2.0, 0.05);
    CHECK_REL(pi.step_s / p.step_s, 0.5, 0.05);
    // f16 is storage only: same compute time as f32 for a compute-bound layer
    RunSpec r16 = infer_spec(); r16.dtype = Dtype::F16;
    Prediction ph = predict(one_layer(dense(n, n, n, r16, "f16", false)), fp, cv);
    CHECK_REL(ph.step_s / p.step_s, 1.0, 0.02);
}

static void test_predict_memory_bound() {
    SECTION("predict: memory-bound elementwise");
    CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    LayerCost e = elementwise(1e8, 1, 3, infer_spec(), "big_add");       // 1.2 GB over a 10 GB/s DRAM
    Prediction p = predict(one_layer(e), fp, cv);
    CHECK(p.dominant == Bound::MEMORY);
    CHECK_REL(p.step_s, e.bytes_fwd / 10e9, 0.01);

    // Halve DRAM bandwidth -> time doubles.
    CapabilityVector slow = cv;
    slow.bw_dram_gbs = 5; slow.triad_max_gbs = 10; slow.triad_1t_gbs = 5;
    for (auto& pt : slow.triad_sweep) pt.second *= 0.5;
    Prediction ps = predict(one_layer(e), fp, slow);
    CHECK_REL(ps.step_s / p.step_s, 2.0, 0.02);

    // A small tensor that fits L1 is served from L1, much faster per byte.
    LayerCost small = elementwise(2000, 1, 3, infer_spec(), "small");    // 24 KB working set < 0.75 * 32 KB
    Prediction pl = predict(one_layer(small), fp, cv);
    CHECK(pl.layers[0].t_memory < small.bytes_fwd / 50e9);

    // Lower-precision storage cuts the memory-bound time proportionally.
    RunSpec h = infer_spec(); h.dtype = Dtype::F16;
    Prediction ph = predict(one_layer(elementwise(1e8, 1, 3, h, "half")), fp, cv);
    CHECK_REL(ph.step_s / p.step_s, 0.5, 0.02);
}

static void test_predict_irregular() {
    SECTION("predict: irregular access uses the gather rate");
    CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    LayerCost s = spmm_aggregate(200000, 1600000, 128, infer_spec());    // large: DRAM-sized
    Prediction p = predict(one_layer(s), fp, cv);
    const double irr = s.bytes_fwd * s.irregular_frac, reg = s.bytes_fwd - irr;
    CHECK_REL(p.layers[0].t_memory, reg / 10e9 + irr / 2e9, 0.01);
    CHECK(p.dominant == Bound::MEMORY);
    // The same bytes treated as streaming would have been several times faster: that is the point.
    CHECK(p.layers[0].t_memory > 2.0 * s.bytes_fwd / 10e9);
}

static void test_predict_threads() {
    SECTION("predict: threads, width cap, assumed scaling");
    CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    LayerCost d = dense(1024, 1024, 1024, infer_spec(), "big", false);
    PredictOptions o1, o4; o1.threads = 1; o4.threads = 4;
    Prediction p1 = predict(one_layer(d), fp, cv, o1), p4 = predict(one_layer(d), fp, cv, o4);
    CHECK_REL(p1.step_s / p4.step_s, 4 * 0.8, 0.03);                     // 4 threads * 0.8 measured efficiency
    PredictOptions o2; o2.threads = 2;
    Prediction p2 = predict(one_layer(d), fp, cv, o2);
    CHECK_REL(p1.step_s / p2.step_s, 2 * 0.95, 0.03);
    PredictOptions o3; o3.threads = 3;                                   // between measured points: interpolated
    Prediction p3 = predict(one_layer(d), fp, cv, o3);
    CHECK(p3.step_s < p2.step_s && p3.step_s > p4.step_s);
    CHECK(p1.notes.empty() || p1.notes[0].find("assumed") == std::string::npos);

    // A one-row layer cannot use more threads than it has rows.
    LayerCost narrow = dense(1, 4096, 4096, infer_spec(), "row", false);
    Prediction n1 = predict(one_layer(narrow), fp, cv, o1), n4 = predict(one_layer(narrow), fp, cv, o4);
    CHECK_REL(n4.step_s / n1.step_s, 1.0, 0.10);

    // Skinny GEMM uses the skinny rate, not the square-GEMM rate.
    LayerCost skinny = dense(8, 1024, 1024, infer_spec(), "skinny", false);
    LayerCost square = dense(64, 1024, 1024, infer_spec(), "sq", false);
    Prediction ps = predict(one_layer(skinny), fp, cv), pq = predict(one_layer(square), fp, cv);
    CHECK(ps.layers[0].t_compute / skinny.flops_fwd > 2.0 * pq.layers[0].t_compute / square.flops_fwd);

    // No scaling data: the prediction assumes an efficiency and SAYS so.
    CapabilityVector noscale = make_cv(false);
    Prediction pa = predict(one_layer(d), fp, noscale, o4);
    bool said = false; for (const auto& n : pa.notes) if (n.find("assumed") != std::string::npos) said = true;
    CHECK(said);
    CHECK_REL(p1.step_s / pa.step_s, 1.0 + 3 * 0.75, 0.03);

    // Overhead: many tiny ops on many threads are overhead-bound.
    LayerCost tiny = elementwise(64, 1, 3, infer_spec(), "tiny"); tiny.op_calls = 500; tiny.parallel_width = 8;
    Prediction pt = predict(one_layer(tiny), fp, cv, o4);
    CHECK(pt.dominant == Bound::OVERHEAD);
    CHECK_REL(pt.layers[0].t_overhead, 500 * (10e-9 + 2e-6), 0.01);
    Prediction pt1 = predict(one_layer(tiny), fp, cv, o1);               // one thread: no fork/join cost
    CHECK(pt1.layers[0].t_overhead < pt.layers[0].t_overhead / 10);
}

static void test_predict_sequential() {
    SECTION("predict: dependent chains pay a barrier per step");
    CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    LayerCost rnn = recurrent_cell(32, 64, 128, 200, 4, infer_spec());
    PredictOptions o4; o4.threads = 4;
    Prediction p = predict(one_layer(rnn), fp, cv, o4);
    CHECK_REL(p.layers[0].t_sequential, 200 * 2e-6, 1e-9);
    PredictOptions o1; o1.threads = 1;
    CHECK_NEAR(predict(one_layer(rnn), fp, cv, o1).layers[0].t_sequential, 0.0, 1e-15);
}

static void test_predict_memory_fit() {
    SECTION("predict: memory fit");
    CapabilityVector cv = make_cv();
    Workload big = one_layer(dense(4096, 16384, 16384, train_spec(), "huge"));
    Prediction fit = predict(big, make_fp(64ull << 30), cv);
    CHECK(fit.memory_fits);
    Prediction nofit = predict(big, make_fp(256ull << 20), cv);
    CHECK(!nofit.memory_fits);
    bool said = false; for (const auto& n : nofit.notes) if (n.find("memory") != std::string::npos) said = true;
    CHECK(said);
    PredictOptions half; half.memory_fraction = 0.5;
    Workload mid = one_layer(dense(64, 2048, 2048, infer_spec(), "mid"));
    const double need = estimate_memory(mid, 1.25).process_total;
    DeviceFingerprint f = make_fp((uint64_t)(need * 1.2));
    CHECK(predict(mid, f, cv).memory_fits);
    CHECK(!predict(mid, f, cv, half).memory_fits);                      // the governor's smaller cap flips it
    DeviceFingerprint unknown = make_fp(0);
    Prediction pu = predict(mid, unknown, cv);
    CHECK(pu.memory_fits);                                              // unknown is "not known to fail"...
    bool said2 = false; for (const auto& n : pu.notes) if (n.find("unknown") != std::string::npos) said2 = true;
    CHECK(said2);                                                       // ...and it says so
}

static void test_predict_calibration_and_micro() {
    SECTION("predict: calibration and micro-steps");
    CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    RunSpec r = train_spec();
    Workload w; w.units_per_micro_step = 256;
    w.layers = {dense(256, 512, 512, r, "d"), optimizer_update(512.0 * 512, r)};
    Prediction base = predict(w, fp, cv);
    CHECK(!base.calibrated);
    CHECK_NEAR(base.rel_error, Calibration::kUncalibratedError, 1e-12);
    CHECK_REL(base.high_s() / base.low_s(), (1 + base.rel_error) * (1 + base.rel_error), 1e-9);

    Calibration cal; cal.fit({{(int)OpKind::DENSE, 1, 2}, {(int)OpKind::DENSE, 1, 2}, {(int)OpKind::DENSE, 1, 2}});
    Prediction pc = predict(w, fp, cv, {}, &cal);
    CHECK(pc.calibrated);
    CHECK_REL(pc.layers[0].seconds / base.layers[0].seconds, 2.0, 1e-9);         // dense scaled
    CHECK_REL(pc.layers[1].seconds / base.layers[1].seconds, 1.0, 1e-9);         // optimizer untouched
    CHECK(pc.rel_error < base.rel_error);                                        // calibrated layers lower the error

    Workload acc = w; acc.micro_steps = 4;
    Prediction pa = predict(acc, fp, cv);
    CHECK_REL(pa.layers[0].seconds / base.layers[0].seconds, 4.0, 1e-9);
    CHECK_REL(pa.layers[1].seconds / base.layers[1].seconds, 1.0, 1e-9);         // optimizer once
    CHECK_REL(pa.units_per_s, 4.0 * 256 / pa.step_s, 1e-9);
}

static void test_predict_degenerate() {
    SECTION("predict: degenerate inputs");
    DeviceFingerprint fp = make_fp();
    CapabilityVector empty;                                              // nothing measured
    Prediction p = predict(one_layer(dense(64, 64, 64, infer_spec())), fp, empty);
    CHECK(!p.valid);
    CHECK(!std::isfinite(p.step_s));
    CHECK(!p.notes.empty());
    CHECK(p.units_per_s == 0.0);

    Prediction e = predict(Workload{}, fp, make_cv());                   // no layers
    CHECK(!e.valid);

    PredictOptions weird; weird.threads = 0; weird.memory_fraction = 7.0;   // clamped, no crash
    Prediction w = predict(one_layer(dense(64, 64, 64, infer_spec())), fp, make_cv(), weird);
    CHECK(w.valid);

    CapabilityVector cut = make_cv(); cut.stopped_reason = "thermal: hot";
    Prediction c = predict(one_layer(dense(64, 64, 64, infer_spec())), fp, cut);
    bool said = false; for (const auto& n : c.notes) if (n.find("cut short") != std::string::npos) said = true;
    CHECK(said);
}

static void test_speedup_function() {
    SECTION("parallel_speedup");
    CapabilityVector cv = make_cv();
    bool assumed = true;
    CHECK_NEAR(parallel_speedup(cv, 1, 0.75, &assumed), 1.0, 1e-12);
    CHECK(!assumed);
    CHECK_NEAR(parallel_speedup(cv, 4, 0.75, &assumed), 3.2, 1e-9);
    CHECK(!assumed);
    parallel_speedup(cv, 8, 0.75, &assumed);                             // beyond the measured range
    CHECK(assumed);
    CHECK(parallel_speedup(cv, 8, 0.75) >= 4.0);                         // never claims LESS than the last measured point
    CapabilityVector one = make_cv(); one.scaling[0].independent.points.resize(1);
    parallel_speedup(one, 4, 0.75, &assumed);
    CHECK(assumed);                                                      // a single point is not a curve
    CHECK_NEAR(parallel_speedup(make_cv(false), 3, 0.5, nullptr), 1.0 + 2 * 0.5, 1e-9);
}

// ═════════════════════════════════════════════════════════════════════════════
// Algorithm interface
// ═════════════════════════════════════════════════════════════════════════════

static Algorithm make_mlp_algorithm() {
    Algorithm a; a.name = "mlp";
    a.tunables = {
        {"micro_batch", {8, 16, 32, 64}, true, "step"},
        {"threads", {1, 2, 4}, true, "step"},
        {"precision_bits", {32, 16}, false, "step"},
        {"batch", {64, 128, 256}, false, "step"},
    };
    a.build = [](const Config& c) {
        const double mb = c.at("micro_batch"), batch = c.at("batch");
        RunSpec r = train_spec();
        r.dtype = c.count("precision_bits") && c.at("precision_bits") == 16 ? Dtype::F16 : Dtype::F32;
        Workload w; w.name = "mlp"; w.unit = "sample"; w.units_per_micro_step = mb;
        w.micro_steps = (int)std::max(1.0, std::round(batch / mb));
        RunSpec first = r; first.need_input_grad = false;
        w.layers.push_back(dense(mb, 784, 256, first, "fc1"));
        w.layers.push_back(elementwise(mb * 256, 1, 2, r, "relu"));
        w.layers.push_back(dense(mb, 256, 10, r, "fc2"));
        w.layers.push_back(optimizer_update(784.0 * 256 + 256.0 * 10, r));
        return w;
    };
    return a;
}

static void test_algorithm_interface() {
    SECTION("Algorithm: tunables and rebuild per config");
    Algorithm a = make_mlp_algorithm();
    CHECK(a.preserving().size() == 2);
    CHECK(a.altering().size() == 2);                                     // precision and batch change the maths
    for (const Tunable* t : a.preserving()) CHECK(t->preserves_semantics);

    Config c = {{"micro_batch", 16}, {"batch", 64}, {"precision_bits", 32}, {"threads", 1}};
    Workload w = a.build(c);
    CHECK(w.micro_steps == 4);
    CHECK_NEAR(w.units_per_step(), 64, 1e-12);
    Config c2 = c; c2["micro_batch"] = 64;
    CHECK(a.build(c2).micro_steps == 1);

    // Smaller micro-batches need less activation memory for the same effective batch.
    CHECK(estimate_memory(a.build(c), 1.0).activations < estimate_memory(a.build(c2), 1.0).activations);
    // Half precision halves parameter storage.
    Config c3 = c; c3["precision_bits"] = 16;
    CHECK_REL(estimate_memory(a.build(c3), 1.0).params, 0.5 * estimate_memory(a.build(c), 1.0).params, 1e-9);

    // The advisor's search loop is just: build -> predict -> rank.
    CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    double best = 0; double best_mb = 0;
    for (double mb : a.tunables[0].values) {
        Config cc = c; cc["micro_batch"] = mb;
        Prediction p = predict(a.build(cc), fp, cv);
        CHECK(p.valid);
        if (p.units_per_s > best) { best = p.units_per_s; best_mb = mb; }
    }
    CHECK(best > 0 && best_mb > 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// Algorithm independence: a GNN, a CNN and an RNN go through the same predictor
// ═════════════════════════════════════════════════════════════════════════════

static void test_algorithm_independence() {
    SECTION("same predictor, different algorithms, different bottlenecks");
    CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    RunSpec r = train_spec();

    Workload gnn; gnn.name = "gcn"; gnn.units_per_micro_step = 100000;
    RunSpec first = r; first.need_input_grad = false;
    gnn.layers = {spmm_aggregate(100000, 800000, 128, first), dense(100000, 128, 64, r, "w1"),
                  spmm_aggregate(100000, 800000, 64, r), dense(100000, 64, 7, r, "w2")};

    Workload cnn; cnn.name = "cnn"; cnn.units_per_micro_step = 32;
    cnn.layers = {conv2d(32, 3, 64, 32, 32, 3, 1, 1, first), conv2d(32, 64, 128, 32, 32, 3, 1, 1, r),
                  conv2d(32, 128, 256, 16, 16, 3, 1, 1, r)};

    Workload rnn; rnn.name = "lstm"; rnn.units_per_micro_step = 16 * 100;
    rnn.layers = {recurrent_cell(16, 64, 128, 100, 4, r)};

    Workload nb; nb.name = "nbody"; nb.units_per_micro_step = 2000;
    nb.layers = {particle_interaction(2000, 0, 20, 4, infer_spec())};

    PredictOptions o; o.threads = 1;
    Prediction pg = predict(gnn, fp, cv, o), pc = predict(cnn, fp, cv, o), pr = predict(rnn, fp, cv, o), pn = predict(nb, fp, cv, o);
    CHECK(pg.valid && pc.valid && pr.valid && pn.valid);
    CHECK(pg.layers[0].bound == Bound::MEMORY);          // aggregation is gather-bound
    CHECK(pc.dominant == Bound::COMPUTE);                // convolutions are compute-bound
    CHECK(pn.dominant == Bound::COMPUTE);                // all-pairs n-body is compute-bound
    PredictOptions o4; o4.threads = 4;
    CHECK(predict(rnn, fp, cv, o4).layers[0].t_sequential > 0);   // the RNN pays per-step synchronisation when threaded
    // GNN aggregation is NOT sped up by tripling the compute rate: it is memory-bound.
    CapabilityVector fast = cv; fast.f32_gflops_best *= 3; fast.f32_gflops_skinny *= 3;
    CHECK_REL(predict(gnn, fp, fast, o).layers[0].seconds, pg.layers[0].seconds, 0.02);
    // The CNN is: ~3x faster.
    CapabilityVector fastc = cv; fastc.conv_gflops *= 3;
    CHECK_REL(pc.step_s / predict(cnn, fp, fastc, o).step_s, 3.0, 0.1);
}

// ═════════════════════════════════════════════════════════════════════════════
// JSON
// ═════════════════════════════════════════════════════════════════════════════

static bool balanced_clean(const std::string& s) {
    int brace = 0, brack = 0; bool in_str = false, esc = false;
    for (char ch : s) {
        if (in_str) { if (esc) esc = false; else if (ch == '\\') esc = true; else if (ch == '"') in_str = false; continue; }
        if (ch == '"') in_str = true;
        else if (ch == '{') ++brace; else if (ch == '}') --brace;
        else if (ch == '[') ++brack; else if (ch == ']') --brack;
        if (brace < 0 || brack < 0) return false;
    }
    // NaN / inf as VALUES (a bare substring test would trip on the key "dominant").
    for (const char* bad : {":nan", ":-nan", ":inf", ":-inf", ",nan", ",inf", "[nan", "[inf"})
        if (to_lower(s).find(bad) != std::string::npos) return false;
    return !in_str && brace == 0 && brack == 0;
}

static void test_json() {
    SECTION("prediction JSON");
    CapabilityVector cv = make_cv(); DeviceFingerprint fp = make_fp();
    Workload w = make_mlp_algorithm().build({{"micro_batch", 16}, {"batch", 64}});
    Prediction p = predict(w, fp, cv);
    Json j; j.begin_object(); to_json(j, p); j.end_object();
    const std::string s = j.finish();
    CHECK(balanced_clean(s));
    CHECK(s.find("\"layers\"") != std::string::npos && s.find("\"fc1\"") != std::string::npos);
    CHECK(s.find("\"dominant\"") != std::string::npos);

    Prediction bad = predict(Workload{}, fp, CapabilityVector{});        // step_s is NaN: must still be valid JSON
    Json j2; j2.begin_object(); to_json(j2, bad); j2.end_object();
    const std::string s2 = j2.finish();
    CHECK(balanced_clean(s2));
    CHECK(s2.find("\"step_s\":null") != std::string::npos);                // NaN is written as null, not as NaN
}

// ═════════════════════════════════════════════════════════════════════════════
// Live smoke: predict with a REAL capability vector
// ═════════════════════════════════════════════════════════════════════════════

static void test_live(bool quick) {
    SECTION("live: real capability vector");
    if (quick) { std::printf("  (skipped: --quick)\n"); return; }
    DeviceFingerprint fp = detect_device();
    SuiteConfig cfg; cfg.time_budget_s = 14.0; cfg.run_sustained = false;   // a slow 1-core VM needs ~9 s for the other phases
    CapabilityVector cv = CapabilitySuite(cfg).run(fp);
    if (!cv.stopped_reason.empty()) { std::printf("  (capability run stopped: %s; skipping)\n", cv.stopped_reason.c_str()); return; }
    Workload w = make_mlp_algorithm().build({{"micro_batch", 32}, {"batch", 64}});
    Prediction p = predict(w, fp, cv);
    CHECK(p.valid);
    CHECK(p.step_s > 0 && p.step_s < 10.0);
    CHECK(p.units_per_s > 0);
    CHECK(p.memory_fits);
    CHECK(!p.calibrated);                                                // nothing has been measured against it yet
    std::printf("  (info) mlp batch 64 on this device: %.2f ms/step predicted (+/- %.0f%%), dominant %s\n",
                p.step_s * 1e3, p.rel_error * 100, to_string(p.dominant));
}

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    test_dense();
    test_transformer_layer_shape();
    test_attention();
    test_spmm();
    test_recurrent();
    test_particles();
    test_embedding_norm_elementwise_optimizer();
    test_memory_estimate();
    test_checkpointing();
    test_totals_and_micro_steps();
    test_calibration();
    test_predict_compute_bound();
    test_predict_memory_bound();
    test_predict_irregular();
    test_predict_threads();
    test_predict_sequential();
    test_predict_memory_fit();
    test_predict_calibration_and_micro();
    test_predict_degenerate();
    test_speedup_function();
    test_algorithm_interface();
    test_algorithm_independence();
    test_json();
    test_live(quick);
    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
