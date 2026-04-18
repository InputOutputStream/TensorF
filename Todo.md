Let me think through each layer carefully for your specific codebase.

---

## Layer 2 — Framework Primitives

### Module / Parameter system

Right now your `Tensor` is just data + graph. You need a `Module` class that owns parameters and knows how to traverse them. The key insight is that every layer (Linear, Conv, etc.) **is-a** Module, and Modules can contain other Modules — this is the composite pattern.

A Module needs to do exactly four things:
- return all its `Tensor` parameters (for the optimizer to update)
- call `zero_grad()` on all of them
- define a `forward()` method
- recursively do all of the above for any child Modules it contains

A `Linear` layer then just holds a weight tensor and bias tensor, implements forward as `x * W.T + b`, and registers them as parameters. Nothing more.

### Optimizer

An optimizer holds a **reference** to the parameters (not a copy), and after `backward()` is called it walks those parameters and applies the update rule. SGD is just:

```
param->data = param->data - lr * param->grad
```

Adam requires storing two extra matrices per parameter (first and second moment estimates `m` and `v`), plus a timestep counter. The update is more involved but still just matrix arithmetic on `data` and `grad`. The tricky part is that `m` and `v` must persist **across steps**, so the optimizer must own them — they're not part of the Tensor itself.

### Loss functions

These compose entirely from your primitives once the graph works. MSE is subtract + power + mean. Cross-entropy is log + multiply + sum + negate. You don't need special Operation classes for these — they're just functions that take tensors and return tensors, using your existing ops. The graph builds itself automatically.

### DataLoader

This is pure CPU bookkeeping — a class that holds your dataset, shuffles indices each epoch, and yields batches as pairs of `(Tensor_t<T> X, Tensor_t<T> y)`. No autograd involvement at all.

---

## Layer 3 — GPU with SYCL

### The architecture decision

The cleanest approach is to make `Matrix` backend-agnostic by introducing a storage abstraction. Your `Matrix` currently owns a `std::vector<T> data` directly. Instead, you want something like a `Storage` that can be either CPU or GPU-backed, with the Matrix not caring which.

In practice for a learning project, a simpler approach works: add a device flag to Matrix (`CPU` or `GPU`), and have each operation check the flag and dispatch to either the CPU implementation or a SYCL kernel. Ugly but transparent.

### SYCL setup

oneAPI uses `sycl::queue` as the execution context. You create one queue per device. The memory model uses `sycl::buffer` + `sycl::accessor`, or the simpler Unified Shared Memory (USM) model where you allocate with `sycl::malloc_device` and pass raw pointers to kernels — much closer to CUDA mentally.

For your matmul kernel, the naive GPU version is each output element computed by one work-item. Then you optimize with tiling into local (shared) memory — each work-group loads a tile of A and a tile of B into fast local memory, computes partial products, then moves to the next tile. This is the fundamental GPU matmul optimization and the pattern is identical in CUDA and SYCL.

The ops you need to port first, in order of importance: elementwise multiply, elementwise add, matmul, exp, sum-reduction. Everything else composes from those.

### What stays on CPU

The graph traversal, backward pass logic, optimizer step — all of these stay on CPU. Only the heavy arithmetic (the actual tensor math inside forward and backward) goes to GPU. This is exactly what PyTorch does — the autograd engine is CPU C++, only kernels run on device.

---

## Layer 4 — Optimizations

### CPU matmul: tiling for cache locality

Your current matmul iterates `i, j, k` in order. The problem is that `B[k][j]` access pattern walks down a column, which in row-major storage means striding through memory — a cache miss on every access for large matrices. The fix is loop tiling: process a `TILE × TILE` block at a time so both A and B tiles fit in L1/L2 cache before moving on. Typical tile sizes are 32 or 64. This alone can give 4–10× speedup on CPU without any parallelism.

### Loop unrolling and SIMD

`#pragma GCC unroll N` before an inner loop asks the compiler to unroll N iterations. More importantly, if your inner loop is clean (no branches, contiguous memory, fixed stride), the compiler will auto-vectorize it using SIMD (AVX2 on modern Intel = 4 doubles per instruction). You help the compiler by keeping the inner loop simple and adding `#pragma GCC ivdep` to assert there are no loop-carried dependencies.

### Fused operations

Instead of computing `relu(matmul(x, W) + b)` as three separate passes over memory, a fused kernel does all three in one pass — you write the output once instead of three times. This is a memory bandwidth optimization. On GPU it matters enormously. You implement it by writing a single SYCL kernel that computes the matmul, adds bias, and applies activation in the same work-item, never writing the intermediate result to global memory.

### Operator fusion at the graph level

More advanced: instead of executing the graph node by node, you inspect sequences of nodes and replace compatible chains with a single fused node before execution. This is what XLA and TVM do. For your project, manually fusing the most common patterns (linear + bias + activation) is enough and much simpler than a general fusion pass.

---

The key dependency chain is: **broadcasting correctness → Module system → training loop → then GPU**. Don't touch GPU until you've trained a small network successfully on CPU. Everything built on a broken foundation just hides the bugs deeper.
