#pragma once

/*
 * WorkloadModel.hpp
 * =================
 * The swappable, per-algorithm half of the profiler (design doc, section 3).
 *
 * An ALGORITHM is a list of layers plus its tunables. Switching from a CNN to a GNN
 * means supplying a different list; nothing in the capability suite, the predictor
 * or the calibration code knows which algorithm it is looking at.
 *
 *   layer cost library   dense, conv2d, attention, spmm_aggregate, recurrent_cell,
 *                        elementwise, norm, embedding, particle_interaction, optimizer
 *   what a layer returns flops (fwd / bwd), bytes moved, irregularity, parallel width,
 *                        sequential depth, op-call count, memory by category
 *   Workload             layers + the unit of work (tokens, nodes, particles ...)
 *   Algorithm            a factory  Config -> Workload  and the tunables it declares
 *   predict()            roofline-style: max(compute, memory) + sequential + per-op
 *                        overhead, per layer, from the CapabilityVector
 *   Calibration          multiplicative correction per layer kind, fitted from measured
 *                        runs, with an error estimate that travels with the prediction
 *
 * What this file does NOT do:
 *   - run anything (no kernels, no Matrix/Tensor). The probe that measures real ops and
 *     feeds Calibration is WorkloadProbe.hpp.
 *   - choose a model size. Factories take their sizes as Config values; the advisor only
 *     searches the tunables an algorithm declares.
 *   - predict energy. That needs the governor's joules-per-op calibration (separate
 *     document); Prediction carries FLOPs and bytes so energy can be derived later.
 *
 * Honesty notes on the analytic model:
 *   - Byte counts are an "ideal traffic" estimate: each tensor read or written once per
 *     pass. Real kernels with poor blocking move more; Calibration absorbs that.
 *   - Capability rates are single-thread; thread speedup comes from the measured scaling
 *     curve, and prediction says when it had to assume one (Prediction::notes).
 *   - Low precision: FP16/BF16/FP8/FP4 are STORAGE formats here (the design doc: no phone
 *     CPU executes FP8/FP4 natively), so they cut bytes but run at the f32 compute rate.
 *     INT8 and INT4 use the measured int8 rate.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "CapabilitySuite.hpp"
#include "DeviceFingerprint.hpp"
#include "ProfilerCommon.hpp"

namespace tensorf::prof::wl {

// ─── Basic vocabulary ────────────────────────────────────────────────────────

enum class Dtype { F32, F64, F16, BF16, FP8, FP4, I8, I4 };

inline double bytes_of(Dtype d) {
    switch (d) {
        case Dtype::F64: return 8.0;
        case Dtype::F32: return 4.0;
        case Dtype::F16: case Dtype::BF16: return 2.0;
        case Dtype::FP8: case Dtype::I8: return 1.0;
        case Dtype::FP4: case Dtype::I4: return 0.5;
    }
    return 4.0;
}
inline const char* to_string(Dtype d) {
    switch (d) {
        case Dtype::F32: return "f32";  case Dtype::F64: return "f64";
        case Dtype::F16: return "f16";  case Dtype::BF16: return "bf16";
        case Dtype::FP8: return "fp8";  case Dtype::FP4: return "fp4";
        case Dtype::I8: return "i8";    case Dtype::I4: return "i4";
    }
    return "?";
}
// Integer formats run on the int8 pipeline; every float format runs at the f32 rate.
inline bool uses_int_pipeline(Dtype d) { return d == Dtype::I8 || d == Dtype::I4; }

enum class OpKind { DENSE, CONV2D, ATTENTION, SPMM, RECURRENT, ELEMENTWISE, NORM, EMBEDDING, PARTICLE, OPTIMIZER, COUNT };
constexpr int kOpKinds = (int)OpKind::COUNT;

inline const char* to_string(OpKind k) {
    switch (k) {
        case OpKind::DENSE: return "dense";           case OpKind::CONV2D: return "conv2d";
        case OpKind::ATTENTION: return "attention";   case OpKind::SPMM: return "spmm";
        case OpKind::RECURRENT: return "recurrent";   case OpKind::ELEMENTWISE: return "elementwise";
        case OpKind::NORM: return "norm";             case OpKind::EMBEDDING: return "embedding";
        case OpKind::PARTICLE: return "particle";     case OpKind::OPTIMIZER: return "optimizer";
        default: return "?";
    }
}

enum class Bound { COMPUTE, MEMORY, OVERHEAD, SEQUENTIAL };
inline const char* to_string(Bound b) {
    switch (b) {
        case Bound::COMPUTE: return "compute";   case Bound::MEMORY: return "memory";
        case Bound::OVERHEAD: return "overhead"; case Bound::SEQUENTIAL: return "sequential";
    }
    return "?";
}

enum class Mode { INFER, TRAIN };

// How a layer is being run. Shared by every layer builder.
struct RunSpec {
    Mode  mode            = Mode::TRAIN;
    Dtype dtype           = Dtype::F32;   // storage / compute format of weights + activations
    Dtype grad_dtype      = Dtype::F32;
    int   opt_states      = 2;            // optimizer state tensors per parameter: SGD 0, momentum 1, Adam 2
    bool  need_input_grad = true;         // false for the first layer of a network (no dX needed)
    bool  trainable       = true;         // false: frozen weights (LoRA base, embeddings frozen ...)
    bool  train() const { return mode == Mode::TRAIN; }
};

// ─── Cost of one layer ───────────────────────────────────────────────────────

// Memory by category, in BYTES. `activations` are tensors kept alive for backward
// (train) or the peak transient output (infer); `scratch` is transient working space.
struct MemByCategory {
    double params = 0, grads = 0, optimizer = 0, activations = 0, scratch = 0;
    double persistent() const { return params + grads + optimizer; }
    MemByCategory& operator+=(const MemByCategory& o) {
        params += o.params; grads += o.grads; optimizer += o.optimizer; activations += o.activations; scratch += o.scratch;
        return *this;
    }
};

struct LayerCost {
    OpKind      kind = OpKind::ELEMENTWISE;
    std::string name;

    double flops_fwd = 0, flops_bwd = 0;     // per execution of the layer
    double bytes_fwd = 0, bytes_bwd = 0;     // ideal DRAM-level traffic
    double irregular_frac = 0;               // share of bytes that are data-dependent gathers/scatters (0..1)
    double parallel_width = 1;               // independent work items per op: more threads than this cannot help
    double seq_depth = 1;                    // length of the dependent chain inside the layer (RNN steps ...)
    double op_calls = 1;                     // framework op invocations per execution (fwd + bwd); each pays dispatch overhead
    double working_set = 0;                  // bytes the layer wants hot at once (decides which cache level feeds it)
    double act_out_bytes = 0;                // size of the layer OUTPUT (used by activation checkpointing)
    double gemm_m = 0;                       // rows of the GEMM-like op (0 = not a GEMM); small values use the skinny rate
    Dtype  dtype = Dtype::F32;
    bool   once_per_step = false;            // optimizer-style: runs once per full step, not once per micro-step
    MemByCategory mem;

    double flops() const { return flops_fwd + flops_bwd; }
    double bytes() const { return bytes_fwd + bytes_bwd; }
    // Arithmetic intensity, flop per byte. Compare with CapabilityVector::ridge_flop_per_byte.
    double intensity() const { return bytes() > 0 ? flops() / bytes() : 0.0; }
};

namespace detail {
inline double pos(double x) { return x > 0 ? x : 0.0; }
// Training adds gradient + optimizer state for trainable parameters.
inline void add_param_mem(LayerCost& c, double n_params, const RunSpec& r) {
    c.mem.params += n_params * bytes_of(r.dtype);
    if (r.train() && r.trainable) {
        c.mem.grads     += n_params * bytes_of(r.grad_dtype);
        c.mem.optimizer += n_params * bytes_of(Dtype::F32) * r.opt_states;   // optimizer state kept in f32
    }
}
}  // namespace detail

// ─── Layer library ───────────────────────────────────────────────────────────

// Y[M,N] = X[M,K] * W[K,N] (+ b[N]).
// fwd 2MKN. bwd: dW = X^T dY (2MKN) and, unless it is the first layer, dX = dY W^T (2MKN).
inline LayerCost dense(double M, double K, double N, const RunSpec& r, std::string name = "dense", bool bias = true) {
    LayerCost c; c.kind = OpKind::DENSE; c.name = std::move(name); c.dtype = r.dtype;
    const double e = bytes_of(r.dtype);
    c.flops_fwd = 2.0 * M * K * N;
    c.bytes_fwd = (M * K + K * N + M * N) * e;
    c.gemm_m = M;
    c.parallel_width = std::max(1.0, M);
    c.working_set = (K * N + M * K + M * N) * e;
    c.act_out_bytes = M * N * e;
    c.op_calls = bias ? 2 : 1;
    detail::add_param_mem(c, K * N + (bias ? N : 0), r);
    if (r.train()) {
        const double gradw = r.trainable ? 2.0 * M * K * N : 0.0;
        const double gradx = r.need_input_grad ? 2.0 * M * K * N : 0.0;
        c.flops_bwd = gradw + gradx;
        c.bytes_bwd = (r.trainable ? (M * N + M * K + K * N) * e : 0.0) + (r.need_input_grad ? (M * N + K * N + M * K) * e : 0.0);
        c.op_calls += (r.trainable ? 2 : 0) + (r.need_input_grad ? 1 : 0) + (bias ? 1 : 0);
        c.mem.activations = M * K * e;                 // the input is kept for dW
    } else {
        c.mem.activations = M * N * e;
    }
    c.mem.scratch = M * N * e;
    return c;
}

// Direct convolution, NCHW. Output spatial size from stride + padding.
inline LayerCost conv2d(double B, double Cin, double Cout, double H, double W, double k, double stride, double pad,
                        const RunSpec& r, std::string name = "conv2d") {
    LayerCost c; c.kind = OpKind::CONV2D; c.name = std::move(name); c.dtype = r.dtype;
    const double e = bytes_of(r.dtype);
    stride = std::max(1.0, stride);
    const double Ho = std::floor((H + 2 * pad - k) / stride) + 1, Wo = std::floor((W + 2 * pad - k) / stride) + 1;
    const double macs = B * Cout * Ho * Wo * Cin * k * k;
    const double in_b = B * Cin * H * W * e, out_b = B * Cout * Ho * Wo * e, w_b = Cout * Cin * k * k * e;
    c.flops_fwd = 2.0 * macs;
    c.bytes_fwd = in_b + out_b + w_b;
    c.parallel_width = std::max(1.0, B * Cout * Ho);
    c.working_set = w_b + Cin * k * W * e * 2.0;       // weights + a few input rows
    c.act_out_bytes = out_b;
    c.op_calls = 2;
    detail::add_param_mem(c, Cout * Cin * k * k + Cout, r);
    if (r.train()) {
        c.flops_bwd = (r.trainable ? 2.0 * macs : 0.0) + (r.need_input_grad ? 2.0 * macs : 0.0);
        c.bytes_bwd = (r.trainable ? in_b + out_b + w_b : 0.0) + (r.need_input_grad ? in_b + out_b + w_b : 0.0);
        c.op_calls += 3;
        c.mem.activations = in_b;
    } else {
        c.mem.activations = out_b;
    }
    c.mem.scratch = out_b;
    return c;
}

// Attention CORE only (QK^T, softmax, AV); the Q/K/V/O projections are separate dense()
// layers. q_len x kv_len covers training/prefill (equal) and cached decode (q_len = 1,
// which is memory-bound on reading K and V). `materialise_scores` = false models a
// flash-style fused kernel that never writes the score matrix.
inline LayerCost attention(double B, double heads, double q_len, double kv_len, double embed, const RunSpec& r,
                           bool causal = true, bool materialise_scores = true, std::string name = "attention") {
    LayerCost c; c.kind = OpKind::ATTENTION; c.name = std::move(name); c.dtype = r.dtype;
    const double e = bytes_of(r.dtype);
    heads = std::max(1.0, heads);
    // A plain implementation computes the full matrix even when it is then masked.
    // 4*B*Tq*Tk*E = QK^T (2*..) + AV (2*..); softmax ~5 flop per score.
    const double scores = B * heads * q_len * kv_len;
    c.flops_fwd = 4.0 * B * q_len * kv_len * embed + 5.0 * scores;
    const double kv_b = 2.0 * B * kv_len * embed * e, q_b = B * q_len * embed * e, o_b = q_b;
    const double score_b = materialise_scores ? scores * e * 2.0 : 0.0;     // written then read back
    c.bytes_fwd = q_b + kv_b + o_b + score_b;
    c.parallel_width = std::max(1.0, B * heads * q_len);
    c.working_set = materialise_scores ? q_len * kv_len * e * heads + kv_b / std::max(1.0, B) : kv_b / std::max(1.0, B);
    c.act_out_bytes = o_b;
    c.op_calls = materialise_scores ? 6 : 2;
    (void)causal;                                           // kept for callers; naive kernels do not skip masked work
    if (r.train()) {
        c.flops_bwd = 2.5 * c.flops_fwd;                    // dScores, dV, dQ, dK plus softmax backward
        c.bytes_bwd = 2.0 * c.bytes_fwd;
        c.op_calls *= 2.5;
        // Without flash-style recomputation the probabilities are kept for backward.
        c.mem.activations = q_b + kv_b + (materialise_scores ? scores * e : 0.0);
    } else {
        c.mem.activations = o_b;
    }
    c.mem.scratch = materialise_scores ? scores * e : 0.0;
    return c;
}

// Graph aggregation  H' = A_hat * X  with a CSR adjacency of `nnz` entries. The feature
// gather is data dependent, so most of its bytes are marked irregular and are served at
// the measured gather rate, not the streaming rate.
inline LayerCost spmm_aggregate(double nodes, double nnz, double feat, const RunSpec& r, std::string name = "spmm") {
    LayerCost c; c.kind = OpKind::SPMM; c.name = std::move(name); c.dtype = r.dtype;
    const double e = bytes_of(r.dtype);
    c.flops_fwd = 2.0 * nnz * feat;
    const double index_b = nnz * (4.0 + e);                 // col index + value
    const double gather_b = nnz * feat * e;                  // one feature row per edge
    const double write_b = nodes * feat * e;
    c.bytes_fwd = index_b + gather_b + write_b;
    c.irregular_frac = c.bytes_fwd > 0 ? gather_b / c.bytes_fwd : 0.0;
    c.parallel_width = std::max(1.0, nodes);
    c.working_set = nodes * feat * e;                        // the feature matrix is what gets hit at random
    c.act_out_bytes = write_b;
    c.op_calls = 1;
    if (r.train()) {
        // Symmetric normalised adjacency: the backward pass is the same SpMM on dY.
        c.flops_bwd = r.need_input_grad ? c.flops_fwd : 0.0;
        c.bytes_bwd = r.need_input_grad ? c.bytes_fwd : 0.0;
        c.op_calls += r.need_input_grad ? 1 : 0;
        c.mem.activations = 0;                               // dX needs no saved input; dW is in the dense layer
    } else {
        c.mem.activations = write_b;
    }
    c.mem.scratch = write_b;
    return c;
}

// Recurrent cell unrolled over `steps`. gates: 1 vanilla RNN, 3 GRU, 4 LSTM.
// seq_depth = steps: step t needs step t-1, so only the batch and hidden dimensions
// are parallel inside a step.
inline LayerCost recurrent_cell(double B, double in_dim, double hidden, double steps, double gates, const RunSpec& r,
                                std::string name = "recurrent") {
    LayerCost c; c.kind = OpKind::RECURRENT; c.name = std::move(name); c.dtype = r.dtype;
    const double e = bytes_of(r.dtype);
    const double per_step_macs = B * (in_dim + hidden) * gates * hidden;
    const double ew_flops = B * gates * hidden * 8.0;        // gate non-linearities + state update
    c.flops_fwd = steps * (2.0 * per_step_macs + ew_flops);
    const double w_b = (in_dim + hidden) * gates * hidden * e;
    c.bytes_fwd = steps * (w_b + B * (in_dim + 2 * hidden) * e + B * gates * hidden * e * 2.0);   // weights re-read every step
    c.parallel_width = std::max(1.0, B * hidden);
    c.seq_depth = std::max(1.0, steps);
    c.working_set = w_b + B * gates * hidden * e;
    c.act_out_bytes = steps * B * hidden * e;
    c.op_calls = steps * 6.0;
    detail::add_param_mem(c, (in_dim + hidden) * gates * hidden + gates * hidden, r);
    if (r.train()) {
        c.flops_bwd = 2.0 * c.flops_fwd;                     // BPTT: same chain again, ~2x the work
        c.bytes_bwd = 2.0 * c.bytes_fwd;
        c.op_calls *= 3.0;
        c.mem.activations = steps * B * (gates + 1.0) * hidden * e;
    } else {
        c.mem.activations = B * hidden * e * 2.0;
    }
    c.mem.scratch = B * gates * hidden * e;
    return c;
}

// n elements, `flops_per_elem` arithmetic, `streams` tensors touched (read + write).
inline LayerCost elementwise(double n, double flops_per_elem, double streams, const RunSpec& r, std::string name = "elementwise") {
    LayerCost c; c.kind = OpKind::ELEMENTWISE; c.name = std::move(name); c.dtype = r.dtype;
    const double e = bytes_of(r.dtype);
    c.flops_fwd = n * flops_per_elem;
    c.bytes_fwd = n * e * streams;
    c.parallel_width = std::max(1.0, n / 64.0);
    c.working_set = n * e * streams;
    c.act_out_bytes = n * e;
    c.op_calls = 1;
    if (r.train()) {
        c.flops_bwd = c.flops_fwd;
        c.bytes_bwd = c.bytes_fwd;
        c.op_calls += 1;
        c.mem.activations = n * e;
    } else {
        c.mem.activations = n * e;
    }
    c.mem.scratch = n * e;
    return c;
}

// LayerNorm / RMSNorm over rows x cols. `rms` skips the mean pass.
inline LayerCost norm(double rows, double cols, const RunSpec& r, bool rms = false, std::string name = "norm") {
    LayerCost c; c.kind = OpKind::NORM; c.name = std::move(name); c.dtype = r.dtype;
    const double e = bytes_of(r.dtype);
    const double n = rows * cols;
    c.flops_fwd = n * (rms ? 5.0 : 8.0);
    c.bytes_fwd = n * e * 3.0;                               // two reads (stats, normalise) + one write
    c.parallel_width = std::max(1.0, rows);
    c.working_set = cols * e * 4.0;
    c.act_out_bytes = n * e;
    c.op_calls = rms ? 4 : 6;
    detail::add_param_mem(c, cols * (rms ? 1.0 : 2.0), r);
    if (r.train()) {
        c.flops_bwd = 2.0 * c.flops_fwd;
        c.bytes_bwd = 2.0 * c.bytes_fwd;
        c.op_calls *= 2.0;
        c.mem.activations = n * e;
    } else {
        c.mem.activations = n * e;
    }
    c.mem.scratch = n * e;
    return c;
}

// Table lookup. No arithmetic; the cost is random row reads. `dense_grad` models a
// framework that allocates a full-size gradient for the table (TensorF's Matrix
// gradients are dense), `false` models sparse row updates.
inline LayerCost embedding(double tokens, double vocab, double dim, const RunSpec& r, bool dense_grad = true,
                           std::string name = "embedding") {
    LayerCost c; c.kind = OpKind::EMBEDDING; c.name = std::move(name); c.dtype = r.dtype;
    const double e = bytes_of(r.dtype);
    c.flops_fwd = 0.0;
    c.bytes_fwd = tokens * dim * e * 2.0;                    // random row read + sequential write
    c.irregular_frac = 0.5;
    c.parallel_width = std::max(1.0, tokens);
    c.working_set = vocab * dim * e;
    c.act_out_bytes = tokens * dim * e;
    c.op_calls = 1;
    c.mem.params = vocab * dim * e;
    if (r.train() && r.trainable) {
        c.mem.grads = (dense_grad ? vocab * dim : tokens * dim) * bytes_of(r.grad_dtype);
        c.mem.optimizer = vocab * dim * 4.0 * r.opt_states;
        c.bytes_bwd = dense_grad ? vocab * dim * bytes_of(r.grad_dtype) + tokens * dim * e : tokens * dim * e * 2.0;   // zero + scatter-add
        c.flops_bwd = tokens * dim;
        c.irregular_frac = 0.5;
        c.op_calls += 1;
    }
    c.mem.activations = tokens * dim * e;
    return c;
}

// N particles with pair interactions. neighbours == 0 means all pairs (N^2, regular,
// compute-bound); otherwise each particle visits `neighbours` partners through a list
// (irregular loads). `flops_per_pair` ~ 20 for a softened gravity or LJ force.
inline LayerCost particle_interaction(double n, double neighbours, double flops_per_pair, double substeps, const RunSpec& r,
                                      std::string name = "particles") {
    LayerCost c; c.kind = OpKind::PARTICLE; c.name = std::move(name); c.dtype = r.dtype;
    const double e = bytes_of(r.dtype);
    const bool all_pairs = neighbours <= 0;
    const double pairs = all_pairs ? n * n : n * neighbours;
    c.flops_fwd = substeps * pairs * flops_per_pair;
    const double state_b = n * 7.0 * e;                      // x y z m + a(xyz) streamed per substep
    c.bytes_fwd = substeps * (state_b * 2.0 + (all_pairs ? 0.0 : pairs * (4.0 + 4.0 * e)));
    c.irregular_frac = all_pairs ? 0.0 : (c.bytes_fwd > 0 ? substeps * pairs * 4.0 * e / c.bytes_fwd : 0.0);
    c.parallel_width = std::max(1.0, n);
    c.seq_depth = std::max(1.0, substeps);                   // substeps are a dependent chain
    c.working_set = state_b;
    c.act_out_bytes = n * 6.0 * e;
    c.op_calls = substeps * 3.0;
    c.mem.params = 0;
    c.mem.activations = state_b;
    c.mem.scratch = n * 3.0 * e;
    (void)r;
    return c;
}

// Parameter update. Runs once per full step regardless of micro-batching.
// Adam reads p, g, m, v and writes p, m, v: ~(2 + 2*states) reads+writes of n floats.
inline LayerCost optimizer_update(double n_params, const RunSpec& r, std::string name = "optimizer") {
    LayerCost c; c.kind = OpKind::OPTIMIZER; c.name = std::move(name); c.dtype = Dtype::F32;
    const double e = 4.0;
    c.flops_fwd = n_params * (r.opt_states >= 2 ? 12.0 : r.opt_states == 1 ? 4.0 : 2.0);
    c.bytes_fwd = n_params * e * (2.0 + 2.0 * r.opt_states);
    c.parallel_width = std::max(1.0, n_params / 1024.0);
    c.working_set = n_params * e * (2.0 + r.opt_states);
    c.op_calls = 6.0;
    c.once_per_step = true;
    c.mem.scratch = 0;
    return c;
}

// ─── Workload, tunables, algorithms ──────────────────────────────────────────

using Config = std::map<std::string, double>;

struct Tunable {
    std::string         name;
    std::vector<double> values;          // the discrete candidates the advisor may try, ascending
    bool                preserves_semantics = true;   // false: changes the maths (batch, precision, particle count)
    std::string        applied_at = "step";           // "step" | "epoch"
};

struct Workload {
    std::string name;
    std::vector<LayerCost> layers;
    std::string unit = "sample";         // what one unit of throughput is: "token", "node", "particle-step" ...
    double units_per_micro_step = 1;     // samples processed by ONE pass through `layers`
    int    micro_steps = 1;              // gradient-accumulation count; layers (except once_per_step) run this many times

    double units_per_step() const { return units_per_micro_step * micro_steps; }
};

struct Algorithm {
    std::string name;
    std::vector<Tunable> tunables;
    std::function<Workload(const Config&)> build;
    // Result-preserving tunables are exhausted before result-changing ones (governor doc).
    std::vector<const Tunable*> preserving() const { std::vector<const Tunable*> v; for (const auto& t : tunables) if (t.preserves_semantics) v.push_back(&t); return v; }
    std::vector<const Tunable*> altering()   const { std::vector<const Tunable*> v; for (const auto& t : tunables) if (!t.preserves_semantics) v.push_back(&t); return v; }
};

// ─── Totals + memory ─────────────────────────────────────────────────────────

struct Totals {
    double flops = 0, bytes = 0;
    double irregular_bytes = 0;
    double intensity() const { return bytes > 0 ? flops / bytes : 0.0; }
};

inline Totals totals(const Workload& w) {
    Totals t;
    for (const auto& l : w.layers) {
        const double reps = l.once_per_step ? 1.0 : (double)w.micro_steps;
        t.flops += l.flops() * reps;
        t.bytes += l.bytes() * reps;
        t.irregular_bytes += l.bytes() * l.irregular_frac * reps;
    }
    return t;
}

struct MemoryEstimate {
    double params = 0, grads = 0, optimizer = 0, activations = 0, scratch = 0;
    double tensor_total = 0;             // sum of the categories, what tensor accounting should read
    double process_total = 0;            // tensor_total * overhead_factor: what the OS will charge us
    double overhead_factor = 1.0;
};

// Activations of all layers are alive together at the end of the forward pass (that is
// what backward needs), so they add; scratch is transient, so the LARGEST one counts.
inline MemoryEstimate estimate_memory(const Workload& w, double overhead_factor = 1.25) {
    MemoryEstimate m; m.overhead_factor = overhead_factor;
    double max_scratch = 0;
    for (const auto& l : w.layers) {
        m.params += l.mem.params; m.grads += l.mem.grads; m.optimizer += l.mem.optimizer; m.activations += l.mem.activations;
        max_scratch = std::max(max_scratch, l.mem.scratch);
    }
    m.scratch = max_scratch;
    m.tensor_total = m.params + m.grads + m.optimizer + m.activations + m.scratch;
    m.process_total = m.tensor_total * overhead_factor;
    return m;
}

// Activation checkpointing: keep only the inputs of `segments` equal groups of layers
// and recompute each group's forward pass during backward. Saves memory, costs one extra
// forward pass over everything except the last segment. Result-preserving.
// Layers' `mem.activations` are replaced by: the boundary outputs + the largest segment.
inline Workload with_checkpointing(const Workload& w, int segments) {
    Workload out = w;
    const int n = (int)w.layers.size();
    segments = std::clamp(segments, 1, std::max(1, n));
    if (n == 0 || segments <= 1) return out;
    const int per = (n + segments - 1) / segments;
    double boundary = 0, max_seg = 0;
    std::vector<double> seg_act;
    for (int s = 0; s < segments; ++s) {
        double a = 0; int lo = s * per, hi = std::min(n, lo + per);
        if (lo >= n) break;
        for (int i = lo; i < hi; ++i) a += w.layers[(size_t)i].mem.activations;
        seg_act.push_back(a);
        max_seg = std::max(max_seg, a);
        if (hi < n) boundary += w.layers[(size_t)(hi - 1)].act_out_bytes;
    }
    const int nseg = (int)seg_act.size();
    // Recompute every segment except the last (the last one's activations are still live).
    const int last_lo = (nseg - 1) * per;
    for (int i = 0; i < last_lo; ++i) {
        out.layers[(size_t)i].flops_bwd += w.layers[(size_t)i].flops_fwd;
        out.layers[(size_t)i].bytes_bwd += w.layers[(size_t)i].bytes_fwd;
        out.layers[(size_t)i].op_calls  += std::max(1.0, w.layers[(size_t)i].op_calls * 0.4);
    }
    // Put the reduced activation total on the first layer and zero the rest, so the
    // sum in estimate_memory() equals boundary + max_seg.
    for (auto& l : out.layers) l.mem.activations = 0;
    out.layers[0].mem.activations = boundary + max_seg;
    out.name += "+ckpt" + std::to_string(segments);
    return out;
}

// ─── Calibration ─────────────────────────────────────────────────────────────

struct Observation {
    int    kind = kOpKinds;              // OpKind index, or kOpKinds for a whole-step (global) observation
    double predicted_s = 0;
    double measured_s = 0;
};

// Multiplicative correction per layer kind + a global factor, with an error estimate.
// factor = exp(median(log(measured/predicted))): robust to one bad sample.
// error  = exp(median |log ratio - log factor|) - 1: the typical relative miss AFTER correction.
struct Calibration {
    std::array<double, kOpKinds + 1> factor;
    std::array<double, kOpKinds + 1> error;
    std::array<int,    kOpKinds + 1> samples;
    static constexpr double kUncalibratedError = 0.60;     // honest default: analytic roofline is often 2x off on phones
    static constexpr double kMinFactor = 0.1, kMaxFactor = 10.0;

    Calibration() { factor.fill(1.0); error.fill(kUncalibratedError); samples.fill(0); }

    bool calibrated(int idx) const { return samples[(size_t)idx] > 0; }

    void fit(const std::vector<Observation>& obs) {
        std::array<std::vector<double>, kOpKinds + 1> logs;
        for (const auto& o : obs) {
            if (!(o.predicted_s > 0) || !(o.measured_s > 0) || !std::isfinite(o.predicted_s) || !std::isfinite(o.measured_s)) continue;
            const int k = (o.kind >= 0 && o.kind <= kOpKinds) ? o.kind : kOpKinds;
            logs[(size_t)k].push_back(std::log(o.measured_s / o.predicted_s));
        }
        for (int k = 0; k <= kOpKinds; ++k) {
            auto& v = logs[(size_t)k];
            if (v.empty()) continue;
            const double lf = median(v);
            factor[(size_t)k] = std::clamp(std::exp(lf), kMinFactor, kMaxFactor);
            samples[(size_t)k] = (int)v.size();
            if (v.size() >= 3) {
                std::vector<double> dev; dev.reserve(v.size());
                for (double x : v) dev.push_back(std::fabs(x - lf));
                error[(size_t)k] = std::exp(median(dev)) - 1.0;
            } else {
                error[(size_t)k] = std::max(0.30, kUncalibratedError * 0.5);   // too few samples to trust a spread
            }
        }
    }

    // Factor used for a layer kind: its own if fitted, else the global one, else 1.
    double factor_for(OpKind k) const {
        if (calibrated((int)k)) return factor[(size_t)k];
        if (calibrated(kOpKinds)) return factor[(size_t)kOpKinds];
        return 1.0;
    }
    double error_for(OpKind k) const {
        if (calibrated((int)k)) return error[(size_t)k];
        if (calibrated(kOpKinds)) return error[(size_t)kOpKinds];
        return kUncalibratedError;
    }
};

// ─── Prediction ──────────────────────────────────────────────────────────────

struct PredictOptions {
    int    threads = 1;
    double memory_fraction = 1.0;        // share of fp.memory.safe_budget_bytes this job may use (the governor's cap)
    double mem_overhead_factor = 1.25;   // measured RSS / tensor bytes (step 3 replaces the default with a measurement)
    double assumed_thread_efficiency = 0.75;   // used only when no scaling curve was measured
};

struct LayerPrediction {
    std::string name;
    OpKind kind = OpKind::ELEMENTWISE;
    double seconds = 0;                  // per full step, includes micro-step repeats and calibration
    double t_compute = 0, t_memory = 0, t_overhead = 0, t_sequential = 0;
    Bound  bound = Bound::COMPUTE;
};

struct Prediction {
    bool   valid = false;                // false: the capability vector had no usable compute rate
    double step_s = std::numeric_limits<double>::quiet_NaN();
    double units_per_s = 0;
    double rel_error = Calibration::kUncalibratedError;   // typical relative miss; the advisor reports this as the error range
    Bound  dominant = Bound::COMPUTE;
    std::vector<LayerPrediction> layers;
    MemoryEstimate memory;
    double memory_budget = 0;            // bytes; 0 = unknown
    bool   memory_fits = true;           // unknown budget counts as "not known to fail"
    double flops = 0, bytes = 0;         // step totals, for later energy accounting
    bool   calibrated = false;
    std::vector<std::string> notes;
    double low_s()  const { return step_s / (1.0 + rel_error); }
    double high_s() const { return step_s * (1.0 + rel_error); }
};

// Parallel speed-up of independent work on `threads` threads, from the measured weak
// scaling curve of the recommended core class. `assumed` is set when no curve exists.
inline double parallel_speedup(const CapabilityVector& cv, int threads, double assumed_eff, bool* assumed = nullptr) {
    if (assumed) *assumed = false;
    if (threads <= 1) return 1.0;
    const ClusterScaling* best = nullptr;
    for (const auto& s : cv.scaling) if (s.cls == cv.recommended_class && !s.independent.points.empty()) best = &s;
    if (!best) for (const auto& s : cv.scaling) if (!s.independent.points.empty()) { best = &s; break; }
    if (!best || best->independent.points.size() < 2) {      // one point (or none) says nothing about scaling
        if (assumed) *assumed = true;
        return 1.0 + (double)(threads - 1) * assumed_eff;
    }
    const auto& pts = best->independent.points;
    double lo_t = 0, lo_e = 1.0, hi_t = 0, hi_e = 1.0; bool have_lo = false, have_hi = false;
    for (const auto& p : pts) {
        if (p.threads <= threads) { lo_t = p.threads; lo_e = p.efficiency; have_lo = true; }
        if (p.threads >= threads && !have_hi) { hi_t = p.threads; hi_e = p.efficiency; have_hi = true; }
    }
    double eff;
    if (have_lo && have_hi && hi_t > lo_t) eff = lo_e + (hi_e - lo_e) * ((double)threads - lo_t) / (hi_t - lo_t);
    else if (have_lo) { eff = lo_e; if (lo_t < threads && assumed) *assumed = true; }   // beyond the measured range: hold the last efficiency
    else eff = hi_e;
    eff = std::clamp(eff, 0.05, 1.0);
    return std::max(1.0, (double)threads * eff);
}

// Streaming bandwidth (GB/s) available to a working set of `ws` bytes on `threads` threads.
inline double bandwidth_gbs(const CapabilityVector& cv, const DeviceFingerprint& fp, double ws, int threads, double speedup) {
    const double l1 = (double)fp.l1d_bytes(), l2 = (double)fp.l2_bytes(), l3 = (double)fp.l3_bytes();
    const double per_thread_ws = ws / std::max(1, threads);
    if (l1 > 0 && per_thread_ws <= l1 * 0.75 && cv.bw_l1_gbs > 0) return cv.bw_l1_gbs * speedup;
    if (l2 > 0 && per_thread_ws <= l2 * 0.75 && cv.bw_l2_gbs > 0) return cv.bw_l2_gbs * speedup;
    if (l3 > 0 && ws <= l3 * 0.5 && cv.bw_l3_gbs > 0)             return cv.bw_l3_gbs * std::min(speedup, 4.0);   // shared: it saturates early
    // DRAM: take the measured triad at this thread count when we have it.
    double best = 0;
    for (const auto& p : cv.triad_sweep) if (p.first <= threads) best = std::max(best, p.second);
    if (best > 0) return best;
    if (cv.bw_dram_gbs > 0) return cv.bw_dram_gbs * std::min(speedup, (double)std::max(1, cv.dram_saturation_threads));
    return cv.triad_max_gbs;
}

namespace detail {
inline double compute_rate(const CapabilityVector& cv, const LayerCost& l) {
    double g = 0;                                           // GFLOP/s or GOP/s, single thread
    switch (l.kind) {
        case OpKind::CONV2D:   g = cv.conv_gflops; break;
        case OpKind::PARTICLE: g = cv.nbody_gflops; break;
        default: break;
    }
    if (g <= 0) {
        if (uses_int_pipeline(l.dtype) && cv.i8_gops > 0)        g = cv.i8_gops;
        else if (l.dtype == Dtype::F64 && cv.f64_gflops > 0)     g = cv.f64_gflops;
        else if (l.gemm_m > 0 && l.gemm_m <= 16 && cv.f32_gflops_skinny > 0) g = cv.f32_gflops_skinny;   // decode / small batch
        else g = cv.f32_gflops_best;
    }
    return g * 1e9;
}
}  // namespace detail

inline Prediction predict(const Workload& w, const DeviceFingerprint& fp, const CapabilityVector& cv,
                          const PredictOptions& opt = {}, const Calibration* cal = nullptr) {
    Prediction p;
    p.memory = estimate_memory(w, opt.mem_overhead_factor);
    if (fp.memory.safe_budget_bytes > 0) {
        p.memory_budget = (double)fp.memory.safe_budget_bytes * std::clamp(opt.memory_fraction, 0.0, 1.0);
        p.memory_fits = p.memory.process_total <= p.memory_budget;
        if (!p.memory_fits) p.notes.push_back("memory: estimated " + std::to_string((long long)(p.memory.process_total / 1048576.0)) +
                                              " MB exceeds the budget of " + std::to_string((long long)(p.memory_budget / 1048576.0)) + " MB");
    } else {
        p.notes.push_back("memory budget unknown: fit not checked");
    }
    if (!(cv.f32_gflops_best > 0)) {
        p.notes.push_back("capability vector has no compute rate: prediction unavailable");
        return p;
    }
    const int threads = std::max(1, opt.threads);
    bool assumed = false;
    const double speedup = parallel_speedup(cv, threads, opt.assumed_thread_efficiency, &assumed);
    if (assumed) p.notes.push_back("thread scaling not measured on this device: assumed efficiency " + std::to_string(opt.assumed_thread_efficiency));
    if (cv.bw_dram_gbs <= 0 && cv.triad_max_gbs <= 0) p.notes.push_back("no bandwidth measurement: memory time treated as zero");
    if (!cv.stopped_reason.empty()) p.notes.push_back("capability run was cut short (" + cv.stopped_reason + "): some rates may be missing");

    const double tiny_s = cv.tiny_op_ns * 1e-9;
    const double fork_s = threads > 1 ? cv.barrier_roundtrip_us * 1e-6 : 0.0;
    double step = 0, weighted_err = 0;
    double dom_t[4] = {0, 0, 0, 0};

    for (const auto& l : w.layers) {
        LayerPrediction lp; lp.name = l.name; lp.kind = l.kind;
        const double reps = l.once_per_step ? 1.0 : (double)w.micro_steps;

        // usable parallelism is limited by the layer's own width
        const double width_cap = std::max(1.0, std::min((double)threads, l.parallel_width));
        const double sp_c = std::min(speedup, parallel_speedup(cv, (int)width_cap, opt.assumed_thread_efficiency, nullptr));
        const double rate = detail::compute_rate(cv, l) * sp_c;
        lp.t_compute = l.flops() / rate;

        // memory: regular bytes stream, irregular bytes run at the measured gather rate
        const double bw = bandwidth_gbs(cv, fp, l.working_set, (int)width_cap, sp_c) * 1e9;
        const double irr_b = l.bytes() * l.irregular_frac, reg_b = l.bytes() - irr_b;
        double t_mem = bw > 0 ? reg_b / bw : 0.0;
        if (irr_b > 0) {
            const double gbw = cv.gather_gbs > 0 ? cv.gather_gbs * 1e9 * std::min(sp_c, std::max(1.0, (double)cv.dram_saturation_threads)) : bw;
            t_mem += gbw > 0 ? irr_b / gbw : 0.0;
        }
        lp.t_memory = t_mem;

        // per-op dispatch + fork/join, and the dependent chain inside the layer
        lp.t_overhead = l.op_calls * (tiny_s + fork_s * (width_cap > 1 ? 1.0 : 0.0));
        lp.t_sequential = l.seq_depth > 1 ? l.seq_depth * fork_s * (width_cap > 1 ? 1.0 : 0.0) : 0.0;

        const double core = std::max(lp.t_compute, lp.t_memory);
        double t = core + lp.t_overhead + lp.t_sequential;
        const double f = cal ? cal->factor_for(l.kind) : 1.0;
        t *= f;
        lp.seconds = t * reps;

        const double tc = lp.t_compute, tm = lp.t_memory, to = lp.t_overhead, ts = lp.t_sequential;
        if (ts >= tc && ts >= tm && ts >= to && ts > 0) lp.bound = Bound::SEQUENTIAL;
        else if (to >= tc && to >= tm) lp.bound = Bound::OVERHEAD;
        else lp.bound = tc >= tm ? Bound::COMPUTE : Bound::MEMORY;

        dom_t[(int)lp.bound] += lp.seconds;
        step += lp.seconds;
        weighted_err += lp.seconds * (cal ? cal->error_for(l.kind) : Calibration::kUncalibratedError);
        p.layers.push_back(std::move(lp));
    }
    const Totals tt = totals(w);
    p.flops = tt.flops; p.bytes = tt.bytes;
    p.step_s = step;
    p.valid = step > 0 && std::isfinite(step);
    p.units_per_s = p.valid ? w.units_per_step() / step : 0.0;
    p.rel_error = step > 0 ? weighted_err / step : Calibration::kUncalibratedError;
    p.calibrated = cal && [&] { for (const auto& l : w.layers) if (cal->calibrated((int)l.kind) || cal->calibrated(kOpKinds)) return true; return false; }();
    int best = 0; for (int b = 1; b < 4; ++b) if (dom_t[b] > dom_t[best]) best = b;
    p.dominant = (Bound)best;
    return p;
}

// ─── JSON ────────────────────────────────────────────────────────────────────

inline void to_json(Json& j, const Prediction& p, const char* key = "prediction") {
    j.begin_object(key);
    j.boolean("valid", p.valid).num("step_s", p.step_s).num("units_per_s", p.units_per_s)
     .num("rel_error", p.rel_error).boolean("calibrated", p.calibrated).str("dominant", to_string(p.dominant))
     .num("flops", p.flops).num("bytes", p.bytes);
    j.begin_object("memory")
     .num("params", p.memory.params).num("grads", p.memory.grads).num("optimizer", p.memory.optimizer)
     .num("activations", p.memory.activations).num("scratch", p.memory.scratch)
     .num("tensor_total", p.memory.tensor_total).num("process_total", p.memory.process_total)
     .num("budget", p.memory_budget).boolean("fits", p.memory_fits).end_object();
    j.begin_array("layers");
    for (const auto& l : p.layers)
        j.begin_object().str("name", l.name).str("kind", to_string(l.kind)).num("seconds", l.seconds)
         .num("t_compute", l.t_compute).num("t_memory", l.t_memory).num("t_overhead", l.t_overhead)
         .num("t_sequential", l.t_sequential).str("bound", to_string(l.bound)).end_object();
    j.end_array();
    j.begin_array("notes");
    for (const auto& n : p.notes) { j.begin_object().str("note", n).end_object(); }
    j.end_array();
    j.end_object();
}

}  // namespace tensorf::prof::wl
