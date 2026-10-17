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

No OpenBLAS? Use CMake, which defaults to the bundled shim `include/tools/cblas.h`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/bin/gcntest
```

Needs: a C++20 compiler (g++ 13 tested), pthreads, a BLAS (OpenBLAS or the shim). `nlohmann/json` is needed by anything that includes the GPT-2 tokenizer, Boost (Beast + Asio, header-only) only by the gateway. macOS: `brew install cmake openblas nlohmann-json boost`. Windows: use WSL2; the code uses POSIX APIs.

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
├── profiler/             hardware fingerprint, device benchmark, memory profiler, cost model
├── parallel/             placeholders
└── tools/cblas.h         portable BLAS shim
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

Useful defines: `MATRIX_NO_BLAS` (portable loops, no cblas), `MATRIX_IEEE_DIVISION`, `TENSORF_FORCE_SOFT_FLOAT16`.

## The profiler in one screen

Five headers in `include/profiler/`, namespace `tensorf::profile`, each with a test:

1. `HardwareFingerprint` — what the device is (core classes, caches, memory limits, thermal, battery).
2. `DeviceBenchmark` — what it can do (GEMM, bandwidth, latency, gather, conv, n-body, RNN, thread scaling, throttling).
3. `MemoryProfiler` — what it can hold (ledger, process RSS, `plan_budget`, `MemoryGuard`).
4. `AlgorithmProfiler` — predicts step time for MLP, CNN, GNN, RNN, Transformer, Markov, particles. `algo::make("gnn")` switches algorithm.
5. `HyperparamAdvisor` — picks batch/microbatch/checkpointing/threads under memory, thread and step-time limits; moves a setting that changes the maths only if you opt in by name.

```cpp
auto fp  = collect();
auto cap = run_device_benchmark(fp, BenchConfig::quick());
auto bud = plan_budget(fp, read_process_memory(), BudgetPolicy{});
AlgorithmProfiler prof(algo::make("cnn"), cap, bud);
auto advice = HyperparamAdvisor(prof).advise({}, OptIn{{"batch", {}}});
std::cout << advice.summary();
```

Status of this part of the repo: `HyperparamAdvisor.hpp`, its test and the newer `AlgorithmProfiler.hpp` are still to be copied into `include/profiler/` and `tests/`; `include/profiler/tmp/` holds an older second API (`tensorf::prof`) that does not interoperate with the five headers. On a phone see [README_ANDROID.md](README_ANDROID.md).

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
         hw_fingerprint_test device_benchmark_test memory_profiler_test algorithm_profiler_test; do ./bin/$t || break; done
```

Each prints `N/M checks passed` and returns non-zero on failure. `basic_tests` needs the MNIST CSVs (`python3 Datasets/MNIST_CSV/generate_mnist_csv.py`), `coratrain` needs `./scripts/download_cora.sh`. Sanitizer run: `DEBUG=1 make tests`.

## More

Static docs: open `docs/index.html`. Roadmap: `docs/ROADMAP.md`. Build/run logs from the author's PC and phone: `REPORT.TXT`, `REPORT2.TXT`.

## License

Apache License 2.0, see `LICENSE`.
