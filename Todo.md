

cpp
// loss.hpp — just free functions
template<typename T> Tensor_t<T> mse(Tensor_t<T> pred, Tensor_t<T> target);
template<typename T> Tensor_t<T> bce(Tensor_t<T> pred, Tensor_t<T> target);
DataLoader
Three responsibilities, keep them separate in your head:

1. Storage — holds the full dataset, nothing fancy, just two tensors (X and y)

2. Shuffling — shuffle an index vector, not the data itself. Never copy the data around.

3. Batching — slice the shuffled indices, return a pair of tensors for that batch

The tricky part for you will be slicing — you'll need a Matrix::slice(start, end, axis) method that returns a subset of rows. That's the only new primitive you need. Everything else is bookkeeping.

Design decision to make upfront: does your DataLoader own the data or reference it? I'd recommend it owns it (takes tensors by value at construction), keeps things simple and avoids dangling references.

Order I'd tackle it
LogOperation → bce loss → Matrix::slice → DataLoader → test with XOR batched

Don't skip the slice — without it your DataLoader can only return the full dataset, which defeats the purpose.


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

Intel integrated graphics (520/630/4400) are exactly what oneAPI/SYCL was built for. Here's my honest advice:

Reality check first
Intel iGPUs are memory-bandwidth constrained — they share RAM with the CPU. For small tensors (like your XOR net) you'll actually be slower on GPU than CPU because the data transfer overhead dominates. The speedup only shows up at larger batch sizes and bigger weight matrices. Set your expectations accordingly when benchmarking.

Setup advice
Use Intel oneAPI Base Toolkit — it's free, includes the DPC++ compiler (icpx), and has drivers for exactly your target hardware (Gen9 = 520/630, Xe = 4400). Don't use the open-source SYCL compilers for Intel iGPU, stick with Intel's own toolchain.

Use USM (Unified Shared Memory) not buffers. For a learning project, USM with sycl::malloc_shared is the right choice — memory is accessible from both CPU and GPU without explicit transfers. It fits your architecture because your CPU graph traversal can read tensor values naturally. Buffers with accessors would require restructuring your entire Matrix class.

The device flag approach
The document suggests adding a device flag to Matrix. That's the right call for your codebase. Concretely:

cpp
enum Device { CPU, GPU };

template<typename T>
class Matrix {
public:
    std::vector<T> data;   // CPU path — unchanged
    T* gpu_data = nullptr; // USM pointer — GPU path
    Device device = CPU;
    sycl::queue* q = nullptr;
};
Each operation then dispatches:

cpp
Matrix<T> operator+(const Matrix<T>& rhs) {
    if (this->device == GPU) return this->add_gpu(rhs);
    return this->add_cpu(rhs);  // your existing code
}
Your existing CPU code stays completely untouched. You're only adding a parallel path.

Port order
The document's order is right. I'd be more specific:

Elementwise add and multiply — simplest kernels, good for validating your setup works
exp — needed for sigmoid and softmax
sum reduction — hardest to get right on GPU, parallel reductions are non-trivial
matmul — the one that actually matters for performance, do naive first then tiled
Don't attempt tiled matmul until the naive version works and you've verified correctness against your CPU path.

The one thing the document undersells
Synchronization. SYCL kernels are asynchronous. After a kernel launches, your CPU code continues before the GPU is done. When your backward pass reads data to compute gradients, you need the forward kernel to have finished. You'll need explicit q.wait() calls at the right points, or your gradients will be computed from uninitialized memory. This is the most common bug when starting out with GPU compute.

What I'd actually do in your position
Get the toolkit installed, write a standalone matmul SYCL program that works on your iGPU, benchmark it against your CPU matmul with a 512x512 matrix. That single experiment will tell you whether the integration is worth it for your use case before you restructure Matrix at all.


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



# Layer 4 — Optimizations
Do tiling first, everything else second. It's the highest ROI change and it's pure algorithmic — no compiler magic, no new dependencies. You'll feel the difference immediately on anything above 256×256.

The pragma hints are worth adding but don't chase them. Modern GCC/Clang auto-vectorizes clean loops already. Write clean code, check the assembly once with -O2 -mavx2 -S, and only add pragmas if the compiler missed something obvious. Don't guess.

Fused operations — the graph-level fusion the document describes is genuinely complex. For your project, the manual approach is enough: write linear_relu_forward as a single function that doesn't materialize intermediates. You don't need a general fusion pass. The general pass is a research project on its own.

One thing the document doesn't mention: profile before you optimize. On your Intel iGPU target, the bottleneck might not be matmul at all — it could be memory allocation, the graph traversal, or something else entirely. Use perf or Intel VTune (free with oneAPI) before deciding what to tile.

Distributed System
This is a large jump in complexity. Be precise about what you actually mean because "distributed" covers very different things:

Data parallelism — split the batch across machines, each has a full model copy, gradients are averaged (AllReduce) after each backward pass. This is what PyTorch DDP does. Fits your architecture most naturally.

Model parallelism — split the model itself across machines because it doesn't fit on one. You don't need this at your scale.

Parameter server — workers send gradients to a central server that updates weights and broadcasts them back. Simpler to reason about than AllReduce but the server becomes a bottleneck.

For a learning project I'd recommend data parallelism with MPI — specifically MPI_Allreduce on the gradient vectors after each backward pass. Your existing training loop barely changes. Each process runs a full forward+backward, then you average gradients across processes before the optimizer step.

The one thing that will force a real redesign: your Optimizer::step() currently reads p->grad directly. For distributed training you need to intercept that gradient, communicate it, and replace it with the averaged value before the step. That's the only structural change.

Don't use raw sockets. Use MPI. It handles topology, buffering, collective ops, and process management. Writing your own communication layer is a semester-long project that teaches you networking, not ML systems.

The honest priority order

train something real (MNIST, not just XOR)
↓
CPU tiling
↓
SYCL on iGPU
↓
fused ops
↓
distributed (MPI data parallel)
Don't skip the "train something real" step. MNIST will expose bugs that XOR never will — larger batches, multi-class output, longer training runs. If your gradients have subtle numerical issues, XOR will still converge (it's too forgiving), MNIST won't.


