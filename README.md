# TensorF

A C++20 tensor and autograd library written from scratch: N-dimensional `Matrix`, reverse-mode `Tensor`, neural-network modules, low-precision types (float16, FP8, FP4), a device profiler, and a REST + WebSocket gateway with a browser dashboard. The library is header-only.

| Read this | When you want to |
|---|---|
| **this file** | build the framework, use `include/`, run the gateway |
| [README_ANDROID.md](README_ANDROID.md) | build and run on a phone, profile a device |
| [README_LLM.md](README_LLM.md) | run GPT-2 / SmolLM, train or fine-tune, federated training |

## Quick start

```bash
sudo apt-get install -y build-essential cmake pkg-config libopenblas-dev nlohmann-json3-dev
git clone https://github.com/InputOutputStream/TensorF.git && cd TensorF
make tests                       # builds every test into bin/ (does not run them)
./bin/matrix_tests && ./bin/backprop_test && ./bin/basic_tests
```

CMake works too, and finds OpenBLAS the same way:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/bin/gcntest
```

### Which BLAS am I actually using?

Both builds auto-detect OpenBLAS and fall back to a slow portable shim (`include/tools/cblas_shim.h`) with a warning if it is missing. Check it, do not assume:

```bash
make blas-check            # or: ./build/bin/blas_check --expect=openblas
# BLAS backend : openblas [OpenBLAS 0.3.26 ... core=Cooperlake, threads=8]
# sgemm 512^3  :  110 GFLOP/s        (the shim does about 4 on the same machine)
```

| | Makefile | CMake |
|---|---|---|
| auto (default) | `make` | `-DTENSORF_BLAS=AUTO` |
| require OpenBLAS | `make BLAS=openblas` | `-DTENSORF_BLAS=OPENBLAS` |
| portable shim | `make BLAS=shim` | `-DTENSORF_BLAS=SHIM` |
| no BLAS (own loops) | `make BLAS=none` | `-DTENSORF_BLAS=NONE` |
| OpenBLAS outside system paths | `OPENBLAS_ROOT=/opt/openblas` | `-DTENSORF_OPENBLAS_ROOT=/opt/openblas` |

The selection reaches the code as `-DTENSORF_USE_OPENBLAS`; see `include/tools/tensorf_blas.h`. Compiling by hand: add that define plus `-lopenblas` (and `-I` to its header dir), otherwise you get the shim.

`ARCH` sets `-march` (default `native`: fastest on this machine, but the binaries will not run on an older CPU). Use `make ARCH=` for a portable build. Changing `BLAS=` or `ARCH=` rebuilds everything; stale binaries are not reused.

Needs: a C++20 compiler (g++ 13 tested), pthreads, a BLAS (OpenBLAS recommended; the shim works but is ~25x slower on GEMM). `nlohmann/json` is needed by anything that includes the GPT-2 tokenizer, Boost (Beast + Asio, header-only) only by the gateway. macOS: `brew install cmake openblas nlohmann-json boost`. Windows: use WSL2; the code uses POSIX APIs.

### Use it from another project

```bash
sudo make install                                   # or: PREFIX=$HOME/.local make install   (no sudo)
g++ -std=c++20 app.cpp $(pkg-config --cflags --libs tensorf) -o app
# or by hand: -I$PREFIX/include/tensorf/{.,core,nn,data,net,profiler,tools} -L$PREFIX/lib -ltensorf
```

`make install` builds `libtensorf.a` (static) and copies the headers and `tensorf.pc`. The library is header-only except for a tiny `src/tensorf_lib.cpp` (version and configured-BLAS strings), so the `.a` is small; the speed comes from the headers being compiled into your program. `tensorf.pc` carries the BLAS choice (`-DTENSORF_USE_OPENBLAS`, `-lopenblas`) and `-march` of the build that installed it, so build your program with the same compiler flags. For a prefix outside the system paths add `PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig`. `make uninstall` removes it. Install honours `PREFIX`, `LIBDIR`, `INCDIR` and `DESTDIR`.

## Using the library

```cpp
#include "DataStructures/Tensor.hpp"
#include "Modules/Linear.hpp"
#include "Modules/Optimizer.hpp"

Linear<float> layer(/*out=*/1, /*in=*/3);
Optimizer<float> opt(layer.parameters(), 1e-2f, ADAM, true);

auto x = make_tensor<float>(Matrix<float>::randn({8, 3}));
auto y = make_tensor<float>(Matrix<float>::randn({8, 1}));

for (int i = 0; i < 100; ++i) {
    opt.zero_grad();
    auto loss = Tensor<float>::mse(y, layer.forward(x));
    loss->backward(make_tensor<float>((float)1.0));
    opt.step();
    loss->reset_graph();          // always: frees the graph, otherwise every iteration leaks
}
```

Three rules cover most mistakes: call `zero_grad()` before `backward()`, call `reset_graph()` after each iteration, and remember that `+=` on a `Tensor_t` builds a new node instead of mutating.

## What is in `include/`

```
include/
├── core/
│   ├── Types/            float16, fp8, fp4, scalar traits, shape_t
│   ├── DataStructures/   Matrix, Tensor, Graph (+CSR), Broadcast
│   ├── Operations/       one differentiable op per file (Add ... spMM)
│   └── Overloads/        operators for std::vector and Tensor_t
├── nn/
│   ├── Modules/          Linear, LoRALinear, Embedding, LayerNorm, RMSNorm, GCNLayer, Optimizer ...
│   │   └── Transformer/  GPT/, Llama/
│   ├── ModelLoader/      GGUF -> GPT / Llama weights
│   └── Tokenizer/        GPT-2 BPE, Llama
├── data/                 Dataset/DataLoader, CSV, images, text, .npy, GGUF, Cora, quantizer, serializer
├── net/                  federated Client/Server/ThreadPool, profiler wire protocol
├── profiler/             hardware fingerprint, device benchmark, framework benchmark, memory profiler, cost model
│   └── tmp/              second profiler API (tensorf::prof): fingerprint, capability suite, workload model, probe, advisor, suite
├── parallel/             placeholders
└── tools/
    ├── tensorf_blas.h    picks OpenBLAS or the shim (the only place that decides)
    └── cblas_shim.h      portable BLAS shim
```

| Piece | One line |
|---|---|
| `Matrix<T>` | NumPy-style N-D array: broadcasting, axis reductions, batched `matmul`, any `transpose`, `concat`/`stack`, seeded RNGs. Works for float, double, int, bool, float16, FP8, FP4. |
| `Tensor_t<T>` | `shared_ptr` graph node holding value, gradient and the op that made it. Backward is a topological pass, so shared subgraphs (residuals) are linear time. |
| Operations | 21 `Operation<T>` subclasses with `forward`/`backward`. To add one, copy a neighbour and add it to the differential matrix in `tests/lowprec_tests.cpp`. |
| `Graph<T>` / `GCNLayer` | graph construction (`connect_by_similarity`), CSR adjacency, sparse SpMM with autograd. |
| `Module<T>` | base class: `register_parameter`, `register_module`, `parameters()`, `save`/`load`. |
| Data | `Dataset`/`DataLoader`, `Tabular` (CSV/TSV), `ImageLoader`, `TextDataset`, `NpyLoader`, `GGUF`, `load_linqs` (Cora), `quant::` block quantizer, `Serializer`. |
| Profiler | see below. |

Header lookup: every folder above is an include root, so `#include "DataStructures/Matrix.hpp"` and `#include "Modules/Linear.hpp"` work wherever the file lives. The Makefile passes `-I. -Iinclude -Iinclude/core -Iinclude/nn -Iinclude/data -Iinclude/net -Iinclude/profiler -Iinclude/parallel -Iinclude/tools`.

Useful defines: `TENSORF_USE_OPENBLAS` (real OpenBLAS; set by the Makefile/CMake), `MATRIX_NO_BLAS` (portable loops, no cblas), `MATRIX_IEEE_DIVISION`, `TENSORF_FORCE_SOFT_FLOAT16`.

## The profiler in one screen

The profiler measures the framework as you built it. `BLAS=openblas` profiles OpenBLAS, `BLAS=shim` profiles the shim, `BLAS=none` profiles the plain loops, because it times `Matrix<T>::matmul` itself and records the backend name next to every number. There are two stacks, both documented in [PROFILER.md](PROFILER.md).

**`tensorf::profile`** (`include/profiler/`), five headers plus the framework benchmark:

1. `HardwareFingerprint`: what the device is (core classes, caches, memory limits, thermal, battery).
2. `DeviceBenchmark`: what it can do. Portable micro-kernels (bandwidth, latency, gather, conv, n-body, RNN, throttling) *and* the framework stage below.
3. `FrameworkBench` (namespace `tensorf::fwbench`): `Matrix::matmul` f32/f64 at 16..512, tall, skinny and the GNN shapes; BLAS thread scaling; a per-core-class pinned rate; sustained load.
4. `MemoryProfiler`: what it can hold (ledger, process RSS, `plan_budget`, `MemoryGuard`).
5. `AlgorithmProfiler`: predicts step time for MLP, CNN, GNN, RNN, Transformer, Markov, particles. Layers that are GEMM-shaped are priced with the framework rate, the rest with the portable loop rate. `algo::make("gnn")` switches algorithm.
6. `HyperparamAdvisor`: picks batch/microbatch/checkpointing/threads under memory, thread and step-time limits; moves a setting that changes the maths only if you opt in by name.

```cpp
auto fp  = collect();
auto cap = run_device_benchmark(fp, BenchConfig::quick());     // includes the framework stage
auto bud = plan_budget(fp, read_process_memory(), BudgetPolicy{});
AlgorithmProfiler prof(algo::make("cnn"), cap, bud);
auto advice = HyperparamAdvisor(prof).advise({}, OptIn{{"batch", {}}});
std::cout << advice.summary();
```

Two things to know when reading the numbers:

- **Portable versus framework.** Portable kernels follow your compile flags (without `-march=native` the f32 GEMM kernel is about 4 GFLOP/s). The framework number is whatever your BLAS achieves with its own runtime kernel selection. They differ on purpose; the planner uses the framework one for GEMM work.
- **More BLAS threads are not always faster.** The thread-scaling curve sets `threads_effective` only if 8 threads beat 1 by at least 1.25x. If not, the planner assigns one thread. If the GNN-shaped GEMM gets no speedup while the process burns much more CPU than wall time, the report says OpenBLAS threads are spinning; try `OPENBLAS_NUM_THREADS=1` or `tensorf::blas::ScopedThreads(1)` around small products (a likely cause when `coratrain` is slower with OpenBLAS than with the shim; confirm by running it with that variable set).

**`tensorf::prof`** (`include/profiler/tmp/`, JSON schema `tensorf.profile/2`) is the newer suite: device fingerprint, capability vector (with `blas_*` fields mirroring the framework stage), workload model, a probe that runs real ops and reports leave-one-out error, tensor accounting, advisor, and `ProfilerSuite` to run them under one time budget. It does not interoperate with the first stack; they are meant to be merged, and which one stays is still open. On a phone see [README_ANDROID.md](README_ANDROID.md).

## Gateway and dashboard

`gateway/` starts and watches the TensorF binaries as jobs and streams their output.

```bash
sudo apt-get install -y libboost-dev
make gateway
TENSORF_BIN_DIR=./bin ./bin/tensorf-gateway --port 8080     # or: PORT=8080 make run-gateway
python3 -m http.server 3000 --directory web                 # dashboard at http://localhost:3000
```

| Route | Does |
|---|---|
| `GET /api/health` | `{"status":"ok","ws_clients":N}` |
| `GET /api/jobs`, `GET /api/jobs/{id}` | list / detail with the last 500 log lines |
| `POST /api/jobs/{benchmark,tests,gpt2,smollm,server,client}` | launch `bin/<name>`; body may carry `{"args":[...]}` |
| `POST /api/jobs/{id}/kill` | SIGTERM |
| `GET /ws` | events `job_started`, `log`, `job_exited` |

The gateway is single-threaded on purpose (it `fork`s jobs). Port from `TENSORF_GATEWAY_PORT` or `--port`, binary directory from `TENSORF_BIN_DIR` or `--bin-dir`. In the dashboard, set the gateway URL in the top bar (default `http://localhost:8080`).

Known gaps: there is no `make benchmark` rule, so the `benchmark` job has no binary; the dashboard presets in `web/js/param-schema.js` use flags the server and client do not parse (`--fed fedprox`, `--mu`, `--device cuda`).

## Tests

```bash
make tests
for t in matrix_tests support_tests lowprec_tests backprop_test spmmtest gcntest basic_tests \
         hw_fingerprint_test device_benchmark_test framework_bench_test memory_profiler_test \
         algorithm_profiler_test hyperparam_advisor_test \
         profiler_capability_test profiler_workload_test profiler_probe_test profiler_memory_test \
         profiler_memory_nohook_test profiler_advisor_test profiler_suite_test profiler_suite_nohook_test; do ./bin/$t || break; done
```

Each prints `N/M checks passed` and returns non-zero on failure. `basic_tests` needs the MNIST CSVs (`python3 Datasets/MNIST_CSV/generate_mnist_csv.py`), `coratrain` needs `./scripts/download_cora.sh`. Sanitizer run: `DEBUG=1 make tests`. Header changes trigger a rebuild of the tests. `make help` lists every target.

## More

Profiler design and numbers: `PROFILER.md`. Static docs: open `docs/index.html`. Roadmap: `docs/ROADMAP.md`. Build/run logs from the author's PC and phone: `REPORT.TXT`, `REPORT2.TXT`.

## License

Apache License 2.0, see `LICENSE`.
