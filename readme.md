# TensorF

A header-only C++20 tensor/autograd library with an integrated device profiler, hardware-fingerprint collector, hyperparameter advisor, federated learning stack, a REST + WebSocket gateway, a browser control-room dashboard, and a compact binary profiler wire protocol. It provides N-dimensional `Matrix` and `Tensor` types with reverse-mode autodiff, neural-network modules (Linear, GCN, Transformer/GPT/Llama, tokenizers), GGUF model loading, a low-precision type system (FP8/FP4/float16), a data-loading layer (CSV/TSV, images, text, NumPy, GGUF), a quantizer, a serializer, a portable CBLAS shim, and a profiler subsystem that measures a device and predicts/optimizes how any algorithm will run on it.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Technologies and Dependencies](#technologies-and-dependencies)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Detailed File and Module Documentation](#detailed-file-and-module-documentation)
  - [Build system](#build-system)
  - [Gateway (C++)](#gateway-c)
  - [Web dashboard](#web-dashboard)
  - [Examples](#examples)
  - [Core types](#core-types)
  - [Data structures (Matrix, Tensor, Graph, Broadcast, MatrixOps)](#data-structures-matrix-tensor-graph-broadcast-matrixops)
  - [Autograd operations](#autograd-operations)
  - [Overloads (vector operators, vecmath, tensor_overloads)](#overloads-vector-operators-vecmath-tensor_overloads)
  - [Data loading](#data-loading)
  - [NN modules](#nn-modules)
  - [Transformer stacks (GPT, Llama)](#transformer-stacks-gpt-llama)
  - [Model loaders (GGUF, GPT, Llama)](#model-loaders-gguf-gpt-llama)
  - [Tokenizers](#tokenizers)
  - [Federated learning](#federated-learning)
  - [Profiler wire protocol](#profiler-wire-protocol)
  - [Profiler (old and new APIs)](#profiler-old-and-new-apis)
  - [CBLAS shim](#cblas-shim)
  - [Scripts](#scripts)
  - [Tests](#tests)
  - [Reports](#reports)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Known Limitations / TODO](#known-limitations--todo)

## Overview

TensorF is a C++20 tensor library, profiling stack, and federated learning framework for running training and inference workloads on heterogeneous devices (desktops, servers, and especially ARM big.LITTLE phones). It solves three problems:

1. **A portable tensor/autograd core.** `Matrix<T>` gives NumPy-like semantics (broadcasting, axis reductions, batched/broadcast `matmul`, arbitrary `transpose` permutations, `concat`/`stack`, slicing, `tril`/`triu`, `one_hot`, random generators, `where`, `maximum`, `elemsAt`, `expand_dims`). `Tensor_t<T>` wraps it with reverse-mode autodiff using an ordered (topological) backward pass that handles shared subgraphs without exponential blow-up. Each differentiable op is a subclass of `Operation<T>` in `include/core/Operations/`. Neural-network modules, tokenizers, and GGUF loading build on top. A vendored CBLAS shim (`include/tools/cblas.h`) lets the library build on platforms without OpenBLAS.

2. **A device-aware profiler, advisor, and federated stack.** A hardware fingerprint collector (`HardwareFingerprint.hpp`, `DeviceFingerprint.hpp`) reads what the device *is* without shelling out. A capability suite (`CapabilityKernels.hpp`, `CapabilitySuite.hpp`, `DeviceBenchmark.hpp`) measures what the device *can do* with self-contained micro-kernels. A workload model (`WorkloadModel.hpp`, `AlgorithmProfiler.hpp`) prices an algorithm's layers against those measurements. A probe (`WorkloadProbe.hpp`) validates the model against real framework ops and fits a calibration. A hyperparameter advisor (`HyperparamAdvisor.hpp`) searches the algorithm's tunables. A federated stack (`Client.hpp`, `Server.hpp`, `io_utils.hpp`) runs FedAvg and FedDistill rounds with chunked, low-RAM weight exchange. A compact binary wire protocol (`MessageProtocol.hpp`, `MessageTransport.hpp`) ships profiler results to a server over TCP, Unix sockets, or UDP.

3. **A gateway and dashboard.** A Boost.Beast HTTP + WebSocket gateway (`gateway/src/main.cpp`) launches TensorF binaries as child processes (`JobManager.hpp`), captures their stdout/stderr line by line, and broadcasts job events over WebSocket (`WsHub.hpp`). A browser control-room dashboard (`web/`) consumes the REST + WebSocket API, parses the binaries' structured log lines live (`js/parser.js`), and renders per-job metrics, federated round telemetry, hardware fingerprints, benchmark results, memory pressure, and a raw event console.

Key features:
- N-dimensional `Matrix<T>` for `float`, `double`, `int`, `bool`, `float16`, FP8, FP4.
- Ordered backprop on shared subgraphs; one `Operation<T>` subclass per differentiable op.
- `Graph<T>` / `CSR<T>` for graph construction and `connect_by_similarity`.
- **NN modules** (`Module<T>` base): `Linear`, `LoRALinear`, `Embedding`, `PositionalEncoding`, `RotationalPositionalEncoding`, `LayerNorm`, `RMSNorm`, `Relu`, `Softmax`, `GCNLayer`, `FeedForward`, `NeuralLayer`, `Optimizer` (SGD/Adam/AdamW/Adafactor), `GPT<T, LinearT>` and `Llama<T, LinearT>` stacks.
- Low-precision codecs: `float16`, `fp8_e4m3`, `fp8_e5m2`, `fp8_e3m4`, `fp4_e2m1`, `FP4<E,M>`.
- **Data loading**: `Dataset`/`DataLoader`, `Tabular` (CSV/TSV), `ImageLoader` (stb_image), `TextDataset`, `NpyLoader` (.npy), `GGUF`, `LazyDataLoader`, `GraphLoader` (LINQS Cora/Citeseer).
- **Quantization**: `quant::` block-wise absmax quantization, `QNTF` format, `Module<T>` bridges.
- **Serialization**: `Serializer<T>` with a `TNSF` binary format.
- **Federated learning**: `Client<T>`, `Server<T, LinearT>`, `ThreadPool`, `io_utils.hpp`.
- **Model loaders**: `GGUFLoader<T>`, `GPTGGUFLoader<T, LinearT>`, `LlamaGGUFLoader<T, LinearT>` (GQA-aware).
- **Tokenizers**: `Tokenizer`, `GPT2Tokenizer`, `LlamaTokenizer`.
- **Gateway**: REST API (`/api/health`, `/api/jobs`, `/api/jobs/{id}`, `/api/jobs/{id}/kill`, `POST /api/jobs/{benchmark,tests,gpt2,smollm,server,client}`), WebSocket (`/ws`) with `job_started`/`log`/`job_exited` events, CORS, single-threaded `io_context` (safe with `fork`).
- **Dashboard**: 9 views (Dashboard, Launch job, Jobs, FL rounds, Hardware profiler, Benchmark, Tests, Memory & checkpoints, Live console, Documentation), dynamic job launcher cards driven by `FLAG_HELP`, live log parsing into metrics, Chart.js visualisations, toast notifications.
- **Profiler wire protocol**: `MessageProtocol.hpp` (TFPR binary format with CRC32/xxHash32) and `MessageTransport.hpp` (TCP/Unix/UDP).
- **Build system**: `Makefile` (master) and `CMakeLists.txt` (Android NDK support).
- Portable CBLAS shim (`include/tools/cblas.h`).
- **Build scripts**: `scripts/download_cora.sh`, `scripts/download_llms.sh`, `scripts/run_android.sh`.

## Project Structure

```
TensorF/
├── CMakeLists.txt
├── Makefile
├── LICENSE
├── Apache License.txt
├── README.md
├── REPORT.TXT
├── REPORT2.TXT
├── Datasets/
│   ├── client1/{shard_ag,shard_ah,shard_ai,train_split.txt,val_split.txt,vocab.txt}
│   ├── client2/{shard_ad,shard_ae,shard_af,train_split.txt,val_split.txt,vocab.txt}
│   ├── client3/{shard_aa,shard_ab,shard_ac,train_split.txt,val_split.txt,vocab.txt}
│   ├── cora/{cora.cites,cora.content}
│   ├── MNIST_CSV/{generate_mnist_csv.py,mnist_test.csv,mnist_train.csv,readme.md}
│   ├── shakespeare.txt
│   ├── shard_aj
│   ├── train_split.txt
│   ├── val_split.txt
│   └── vocab.txt
├── docs/
│   ├── api/{dataloader,gateway,matrix,modelloader,modules,operations,overloads,profiler,tensor,tokenizer,transformers,types}.html
│   ├── architecture.html
│   ├── assets/style.css
│   ├── examples.html
│   ├── federated.html
│   ├── index.html
│   ├── install.html
│   ├── known-issues.html
│   ├── quickstart.html
│   ├── ROADMAP.md
│   └── transformer.txt
├── examples/{GPT2.cpp,SmollLLM.cpp,transformer.cpp}
├── gateway/
│   ├── include/{JobManager.hpp,WsHub.hpp}
│   └── src/main.cpp
├── include/
│   ├── core/
│   │   ├── DataStructures/{Broadcast,Graph,Matrix,MatrixOps,Tensor}.hpp
│   │   ├── Operations/{Add,Concat,Division,Dot,Embedding,Exponent,Index,Log,Matmul,Multiply,Operation,Power,Relu,Reshape,Sigmoid,Softmax,spMM,Subtract,SumAxis,Sum,Transpose,Vdot}Operation.hpp
│   │   ├── Overloads/{Overload,tensor_overloads,VectorMath}.hpp
│   │   └── Types/{float16,fp4,fp8,scalar,shape,types}.hpp
│   ├── data/{DataLoading,DataLoading_Report.docx,Dataset,GGUF,GraphLoader,ImageLoader,ImageLoader_Report.docx,LazyLoader,NPYLoader,Quantizer,Serializer,stb_image.h,Tabular,TextDataset}
│   ├── net/
│   │   ├── Network/{Client,io_utils,Server,ThreadPool}.hpp
│   │   └── Protocol/{MessageProtocol,MessageTransport}.hpp
│   ├── nn/
│   │   ├── ModelLoader/{GGUFLoader,GPTLoader,LlamaLoader}.hpp
│   │   ├── Modules/{Embedding,FeedForward,GCNLayer,LayerNorm,Linear,LoRALinear,Module,nn,Optimizer,PositionalEncoding,Relu,RMSNorm,RotationalPositionalEncoding,Softmax}.hpp
│   │   ├── Transformer/
│   │   │   ├── GPT/{Block,FeedForward,GPT,Head,MultiHeadAttention}.hpp
│   │   │   └── Llama/{Block,FeedForward,Head,Llama,MultiHeadAttention}.hpp
│   │   └── Tokenizer/{GPT2Tokenizer,LlamaTokenizer,Tokenizer}.hpp
│   ├── parallel/{arm_arallel,Parallel,x86_parallel}.hpp
│   ├── profiler/
│   │   ├── AlgorithmProfiler.hpp
│   │   ├── DeviceBenchmark.hpp
│   │   ├── HardwareFingerprint.hpp
│   │   ├── HyperparamAdvisor.hpp
│   │   ├── MemoryProfiler.hpp
│   │   └── tmp/{CapabilityKernels,CapabilitySuite,DeviceFingerprint,ProfilerCommon,WorkloadModel,WorkloadProbe}.hpp
│   └── tools/cblas.h
├── Models/model.hge
├── screencaptures/*.png
├── scripts/{download_cora.sh,download_llms.sh,run_android.sh}
├── SLM/
│   ├── gpt2-small-{f32,q4,q8}.gguf
│   ├── gpt2-tokenizer/{merges.txt,vocab.json}
│   ├── lSmolLM2-135M-Instruct-f16.gguf
│   ├── mSmolLM2-135M-Instruct-f16.gguf
│   ├── qwen2.5-0.5b-instruct-q8_0.gguf
│   ├── SmolLM-135M-Instruct.Q8_0.gguf
│   └── SmolLM2-135M-Instruct.Q4_K_M.gguf
├── tests/
│   ├── algorithm_profiler_test.cpp
│   ├── backprop_test.cpp
│   ├── basic_tests.cpp
│   ├── BUILD.txt
│   ├── contract.cpp
│   ├── coratrain.cpp
│   ├── device_benchmark_test.cpp
│   ├── diff_ops.cpp
│   ├── gcntest.cpp
│   ├── GPT/{Client.cpp,Modes.md,Server.cpp,Trainner.hpp}
│   ├── hw_fingerprint_test.cpp
│   ├── lowprec_tests.cpp
│   ├── matrix_tests.cpp
│   ├── memory_profiler_test.cpp
│   ├── profiler_capability_test.cpp
│   ├── profiler_probe_test.cpp
│   ├── profiler_workload_test.cpp
│   ├── scope_test.cpp
│   ├── spmmtest.cpp
│   ├── support_tests.cpp
│   └── tmp/
└── web/
    ├── css/{base,charts,components,layout}.css
    ├── index.html
    └── js/
        ├── {api,charts,docs-data,jobs,main,param-schema,parser,state,toast,utils}.js
        └── views/{benchmark-view,console-view,docs,federated,jobs-view,launch,memory,overview,profiler,tests-view}.js
```

### File-by-file reference

| File / Directory | Purpose |
|---|---|
| `CMakeLists.txt` | CMake build: C++20, `tensorf_deps` interface target, `TENSORF_USE_OPENBLAS` option (default OFF → shim), verifies `include/tools/cblas.h` is the shim, builds 11 test targets, `Release` default. |
| `Makefile` | Master Makefile: builds `gpt2`, `smollm`, `transformer`, `benchmark`, `tests` (many targets), `gateway`, `server`/`client`, `net`, `clean`, `install-deps`, `check-deps`, `help`. Uses `pkg-config` for OpenBLAS and nlohmann/json. `DEBUG=1` switches to `-O0 -g3 -fsanitize=address,undefined`. |
| `LICENSE`, `Apache License.txt` | Apache License 2.0. |
| `README.md`, `REPORT.TXT`, `REPORT2.TXT` | Existing project docs (reports contain build/run logs — see [Reports](#reports)). |
| `include/core/Types/shape.hpp` | `using shape_t = std::vector<size_t>;`. |
| `include/core/Types/scalar.hpp` | Element traits, `acc<T>`, `to_acc`/`from_acc`, math wrappers. Defines `float16`/`float16_is_native`. |
| `include/core/Types/fp8.hpp` | `FP8<E,M>` codec (IEEE-style E4M3, no subnormals, half-away-from-zero). |
| `include/core/Types/fp4.hpp` | `FP4<E,M>` codec (E≥2 required; E2M1 max finite 3.0; NaN→max finite). |
| `include/core/Types/float16.hpp` | `soft_float16` software binary16 with exhaustive encode/decode. |
| `include/core/Types/types.hpp` | Integer/float aliases, FP8/FP4 aliases, legacy `is_fp8`/`is_fp4`, forward declarations, `Tensor_t<T>`, `Operation_t<T>`. |
| `include/core/DataStructures/Matrix.hpp` | N-D row-major `Matrix<T>` with NumPy-like ops. Build switches `MATRIX_NO_BLAS`, `MATRIX_IEEE_DIVISION`, `MATRIX_NO_VECTOR_OPERATORS`. |
| `include/core/DataStructures/Tensor.hpp` | `Tensor<T>` autograd wrapper: `topo_order`, `accumulate`, `backward`, `zero_grad`, `reset_graph`, op factory methods. |
| `include/core/DataStructures/Graph.hpp` | `Graph<T>`, `CSR<T>`, `connect_by_similarity`, `build`, `adjacency`, `dense_adjacency`. Needs `-pthread`. |
| `include/core/DataStructures/Broadcast.hpp` | `Broadcast<T>` helpers, `mxd::broadcast_shapes`, `mxd::gather`, `mxd::numel`, `mxd::strides_of`, `mxd::shape_str`. |
| `include/core/DataStructures/MatrixOps.hpp` | Free scalar operators, comparison operators (returning `Matrix<bool>`), compound assignment, `mxd::swap_last2`, `sumGradForBroadcast`. |
| `include/core/Operations/Operation.hpp` | Abstract `Operation<T>` base. |
| `include/core/Operations/*Operation.hpp` | One file per differentiable op (21 files). |
| `include/core/Overloads/Overload.hpp` | Opt-in global operators on `std::vector<T>` (numeric types only). |
| `include/core/Overloads/VectorMath.hpp` | `vecmath::` named functions on `std::vector<T>` (no global operators). |
| `include/core/Overloads/tensor_overloads.hpp` | `make_tensor<T>` overloads and Tensor-level operators. |
| `include/data/Dataset.hpp` | `Dataset<T>` (X/Y pair) and `DataLoader<T>` (mini-batches, optional shuffle). |
| `include/data/Tabular.hpp` | `Tabular<T>` CSV/TSV loader; XLS/XLSX stubs. |
| `include/data/Serializer.hpp` | `Serializer<T>` `TNSF` binary save/load. |
| `include/data/DataLoading.hpp` | Umbrella header. |
| `include/data/NPYLoader.hpp` | `NpyLoader<T>` for `.npy`; `.npz` stub. |
| `include/data/GGUF.hpp` | `GGUF` container parser + F32/F16/Q8_0/Q4_0 tensor loading. |
| `include/data/ImageLoader.hpp` | `ImageLoader<T>` using stb_image (HWC/NHWC; ZERO_ONE/MEAN_STD/NONE; bilinear resize). |
| `include/data/TextDataset.hpp` | `TextDataset<T>` character-level LM dataset. |
| `include/data/LazyLoader.hpp` | `LazyDataLoader<T>` wrapper over `TextDataset`. |
| `include/data/GraphLoader.hpp` | `load_linqs`, `GraphDataset<T>`, `Split`, `make_split`. |
| `include/data/Quantizer.hpp` | `quant::` block-wise absmax quantization, `QNTF` format, `Module<T>` bridges. |
| `include/data/stb_image.h` | Vendored stb image decoder. |
| `include/nn/Modules/Module.hpp` | `Module<T>` base class. |
| `include/nn/Modules/Linear.hpp` | Dense layer with Glorot init and optional bias. |
| `include/nn/Modules/LoRALinear.hpp` | Low-rank adapter: frozen `weight`/`bias`, trained `A`/`B`, `scale = alpha/rank`. |
| `include/nn/Modules/Embedding.hpp` | Lookup table with Glorot init. |
| `include/nn/Modules/PositionalEncoding.hpp` | Learned position embeddings. |
| `include/nn/Modules/RotationalPositionalEncoding.hpp` | RoPE (no parameters). |
| `include/nn/Modules/LayerNorm.hpp` | `gamma`/`beta` layer norm. |
| `include/nn/Modules/RMSNorm.hpp` | `gamma`-only RMS norm. |
| `include/nn/Modules/Relu.hpp`, `Softmax.hpp` | Simple activations. |
| `include/nn/Modules/GCNLayer.hpp` | Graph convolution layer (dense and sparse paths). |
| `include/nn/Modules/FeedForward.hpp` | Generic 2-layer `FeedForward<T, LinearT>` (ReLU). |
| `include/nn/Modules/NeuralLayer.hpp` | Sequential container. |
| `include/nn/Modules/Optimizer.hpp` | `Optimizer<T>` with SGD, Adam, AdamW, Adafactor. |
| `include/nn/Transformer/GPT/*.hpp` | GPT stack: LayerNorm blocks, GeLU FFN, learned positional embeddings, MHA. |
| `include/nn/Transformer/Llama/*.hpp` | Llama stack: RMSNorm blocks, SwiGLU FFN, RoPE, MHA. |
| `include/nn/ModelLoader/GGUFLoader.hpp` | `GGUFLoader<T>` base: `load_raw`, `copy_into`, `open`, `inspect`, `report_params`. |
| `include/nn/ModelLoader/GPTLoader.hpp` | `GPTGGUFLoader<T, LinearT>` (asserts `LinearT=Linear`) + `GPT2HyperParams`. |
| `include/nn/ModelLoader/LlamaLoader.hpp` | `LlamaGGUFLoader<T, LinearT>` (GQA-aware) + `LlamaHyperParams`. |
| `include/nn/Tokenizer/Tokenizer.hpp` | Abstract byte-level BPE base. |
| `include/nn/Tokenizer/GPT2Tokenizer.hpp` | GPT-2 tokenizer (`load`, `load_from_arrays`). |
| `include/nn/Tokenizer/LlamaTokenizer.hpp` | Llama tokenizer (`load_from_gguf`), special-token-aware. |
| `include/net/Network/Server.hpp` | `Server<T, LinearT>` base: TCP listener, FedAvg/FedDistill round state, chunked aggregation. |
| `include/net/Network/Client.hpp` | `Client<T>` base: connect, flat-tensor protocol, chunked protocol, logits protocol, LoRA handshake. |
| `include/net/Network/ThreadPool.hpp` | Simple `ThreadPool`. |
| `include/net/Network/io_utils.hpp` | `read_exact`, `write_exact`, `send_chunked`, `recv_chunked`, per-fd bandwidth counters. |
| `include/net/Protocol/MessageProtocol.hpp` | Binary profiler wire protocol (`TFPR`): header, body, storage entries, CRC32/xxHash32. |
| `include/net/Protocol/MessageTransport.hpp` | TCP / Unix / UDP transports + `send_profile` helper. |
| `include/profiler/HardwareFingerprint.hpp` | Old fingerprint (namespace `tensorf::profile`). |
| `include/profiler/DeviceBenchmark.hpp` | Old capability suite (namespace `tensorf::profile`). |
| `include/profiler/MemoryProfiler.hpp` | `MemoryLedger`, `plan_budget`, `MemoryGuard`. |
| `include/profiler/AlgorithmProfiler.hpp` | `AlgorithmProfiler`, `CostModel`, `algo::make`, `DeviceView`. |
| `include/profiler/HyperparamAdvisor.hpp` | `HyperparamAdvisor`, `OptIn`, `Constraints`, `SearchOptions`, `Advice`. |
| `include/profiler/tmp/*.hpp` | Rewritten profiler (namespace `tensorf::prof`). |
| `include/parallel/*.hpp` | Architecture-specific parallelism. |
| `include/tools/cblas.h` | Portable CBLAS shim (row-major only). |
| `gateway/src/main.cpp` | Boost.Beast HTTP + WebSocket gateway: REST routes for jobs, `/ws` for realtime events. Single-threaded `io_context` (safe with `fork()`). |
| `gateway/include/JobManager.hpp` | Spawns TensorF binaries via `fork`/`execv`, captures stdout/stderr line-by-line, keeps an in-memory registry with rolling log buffer (max 5000 lines). |
| `gateway/include/WsHub.hpp` | Tracks connected WebSocket sessions and broadcasts JSON messages to all of them. |
| `examples/GPT2.cpp` | GPT-2 inference: loads `SLM/gpt2-small-f32.gguf`, tokenizes a prompt, generates 30 tokens. |
| `examples/SmollLLM.cpp` | SmolLM2-135M inference with a chat template, generates 30 tokens. |
| `examples/transformer.cpp` | Character-level Llama training on `Datasets/`, saves `Models/Llama.hge`, generates text. |
| `scripts/download_cora.sh` | Downloads Cora from the pygcn repo. |
| `scripts/download_llms.sh` | Downloads GPT-2 GGUF files from Hugging Face into `SLM/`. |
| `scripts/run_android.sh` | Builds with the Android NDK and runs the tests on a connected device. |
| `tests/*.cpp` | Test programs (see [Testing](#testing)). |
| `tests/GPT/{Client,Server}.cpp`, `Trainner.hpp`, `Modes.md` | Federated GPT training harness. |
| `web/index.html` | Dashboard shell: topbar, sidebar, 10 views. |
| `web/css/{base,layout,components,charts}.css` | Instrument-panel design tokens, shell layout, component styles, canvas containment. |
| `web/js/state.js` | Shared mutable store (`ws`, `jobs`, `jobLogs`, `metrics`, `selectedJobId`, `pollTimer`) + `gwBase()`. |
| `web/js/utils.js` | `escapeHtml`, `escapeAttr`, `statusClass`, `statusDotColor`. |
| `web/js/toast.js` | Bottom-right notification popups. |
| `web/js/charts.js` | Chart.js helpers: `safeChart`, `CHART_COLORS`, `chartBaseOptions`, `chartBarOptions`, `enableChartInteractions`, `closeZoomModal`. |
| `web/js/param-schema.js` | `PRESETS` and `FLAG_HELP` for the dynamic launcher. |
| `web/js/parser.js` | Regex-based log parser + benchmark/tests parsers. |
| `web/js/docs-data.js` | Documentation link table + `REPO` / `RAW_DOCS_BASE`. |
| `web/js/api.js` | REST calls + WebSocket client + `pollJobs`, `killJob`, `launchJob`, `launchJobRaw`, `fetchJobDetail`. |
| `web/js/jobs.js` | `upsertJob` (merges REST/WS into state, fans bulk log through the parser) + `goToJob`. |
| `web/js/main.js` | Nav router, `VIEW_RENDERERS`, `boot()`. |
| `web/js/views/*.js` | One renderer per view. |
| `docs/**` | HTML API docs and design documents. |

## Technologies and Dependencies

- **Language**: C++20 (core, tests, gateway), JavaScript ES2020+ (dashboard), HTML/CSS.
- **Build**: CMake ≥ 3.18 (`CMakeLists.txt`) and GNU Make (`Makefile`).
- **BLAS**: OpenBLAS (`libopenblas-dev`, detected via `pkg-config`) or the vendored shim `include/tools/cblas.h`. CMake option `TENSORF_USE_OPENBLAS` (default `OFF`); the CMake configure step **fails** if `include/tools/cblas.h` is not the TensorF shim (it checks for the `tensorf_cblas_shim` string) unless OpenBLAS is enabled.
- **JSON**: `nlohmann/json` 3.11.3 (checked by `make check-deps`), used by `GPT2Tokenizer::load()` and the gateway.
- **xxHash**: `libxxhash-dev` (installed by `make install-deps`; `MessageProtocol.hpp` ships its own `xxhash32_impl` and also links the system lib).
- **Threads**: `Threads::Threads` (CMake) / `-lpthread` (Make).
- **Boost.Beast / Boost.Asio**: required by the gateway (`gateway/src/main.cpp`, `WsHub.hpp`). Header-only.
- **stb_image**: vendored (`include/data/stb_image.h`).
- **Python 3**: for `Datasets/MNIST_CSV/generate_mnist_csv.py`.
- **Android NDK** + `adb`: for `scripts/run_android.sh`.
- **Tested toolchain**: `g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`, OpenBLAS 0.3.26, Clang 21.0.0 (Android NDK 29.0.14206865).
- **Web**: Chart.js 4.4.4 (loaded from jsDelivr CDN), Google Fonts (JetBrains Mono + Inter via `@import` in `base.css`).
- **Platform APIs**: Linux/Android `/proc` and `/sys`; Apple `sysctl`/`mach`; Windows Win32.

No version numbers for the compilers or libraries are stated in the uploaded files beyond the ones above.

## Prerequisites

- A C++20-capable compiler (GCC 13+ or Clang 21+ as tested).
- CMake ≥ 3.18 and/or GNU Make.
- OpenBLAS (`libopenblas-dev`) **or** the shim.
- `nlohmann-json3-dev`.
- `libxxhash-dev`.
- POSIX threads.
- **Boost** (Beast + Asio, header-only) for the gateway.
- Optional: `pkg-config`, Python 3, `curl`, `wget`.
- Optional: Android NDK + `adb`.
- For the dashboard: a modern browser with ES2020 support (all evergreen browsers).

On Ubuntu/Debian, `make install-deps` runs:

```bash
sudo apt-get install -y --no-install-recommends \
    build-essential g++ cmake pkg-config \
    libopenblas-dev nlohmann-json3-dev libxxhash-dev \
    wget curl
```

## Installation

### Native build (Make)

```bash
make install-deps     # installs system dependencies
make check-deps       # verifies g++, nlohmann/json, OpenBLAS, pthreads
make                  # builds gpt2, smollm, transformer, benchmark, tests, gateway
make clean            # removes bin/
```

Useful targets:

```bash
make gpt2             # examples/GPT2.cpp → bin/gpt2
make smollm           # examples/SmollLLM.cpp → bin/smollm
make transformer      # examples/transformer.cpp → bin/transformer
make tests            # builds (and, per-target, runs) the test suite
make gateway          # gateway/src/main.cpp → bin/tensorf-gateway
make run-gateway      # builds + runs the gateway (PORT=8080 by default)
make net              # client + server
make server           # tests/GPT/Server.cpp → bin/server
make client           # tests/GPT/Client.cpp → bin/client
```

With sanitizers:

```bash
DEBUG=1 make <target>   # -O0 -g3 -fsanitize=address,undefined
```

Override the server/client source:

```bash
SERVER_SRC=mon_serveur.cpp make server
CLIENT_SRC=mon_client.cpp make client
```

Override the gateway port:

```bash
PORT=9090 make run-gateway
```

### Native build (CMake)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# With OpenBLAS instead of the shim
cmake -S . -B build -DTENSORF_USE_OPENBLAS=ON
```

The CMake target list is: `gcntest`, `spmmtest`, `coratrain`, `basic_tests`, `algorithm_profiler_test`, `memory_profiler_test`, `device_benchmark_test`, `hw_fingerprint_test`, `profiler_capability_test`, `profiler_probe_test`, `profiler_workload_test`. Binaries go to `${CMAKE_BINARY_DIR}/bin/`.

### Android build (`scripts/run_android.sh`)

```bash
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/29.0.14206865
export PATH=$HOME/Android/Sdk/platform-tools:$PATH

./scripts/run_android.sh                   # arm64-v8a (real phones)
ABI=x86_64 ./scripts/run_android.sh        # x86_64 emulator
./scripts/run_android.sh -DTENSORF_USE_OPENBLAS=ON   # extra args go to cmake
```

Variables: `ABI` (default `arm64-v8a`), `API` (default `24`), `CORA` (default `1`), `EPOCHS` (default `200`). The script builds the CMake target list, pushes the binaries to `/data/local/tmp/tensorf`, and runs a subset. It requires `CMakeLists.txt` to exist at the repo root spelled exactly `CMakeLists.txt`.

### Downloading datasets and models

```bash
./scripts/download_cora.sh    # → Datasets/cora/
./scripts/download_llms.sh    # → SLM/
```

`download_cora.sh` fetches `cora.content` (2708 lines) and `cora.cites` (5429 lines) from `https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora`.

`download_llms.sh` fetches GPT-2 Small (f32/q8_0/q4_k_s) from `https://huggingface.co/ddh0`. Note: the Medium/Large/XL lines are commented out and the Medium commands have the `-O` flag commented out (the file is saved under its remote name).

### Web dashboard

The dashboard is static HTML/JS/CSS. Serve it over HTTP (not `file://`, because of CORS and module loading) and open `web/index.html`:

```bash
cd web
python3 -m http.server 8000
# then open http://localhost:8000/
```

The gateway base URL defaults to `http://localhost:8080` and is editable in the topbar. Start the gateway with `make run-gateway` first.

## Configuration

### Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `TENSORF_GATEWAY_PORT` | gateway | Default HTTP port (default 8080). |
| `TENSORF_BIN_DIR` | gateway | Directory holding the TensorF binaries (default `../bin`, relative to `gateway/build`). |
| `ANDROID_NDK_HOME` | `run_android.sh` | Required; path to the NDK. |
| `ABI`, `API`, `CORA`, `EPOCHS` | `run_android.sh` | See [Installation](#installation). |
| `PORT`, `TENSORF_BIN_DIR` | `make run-gateway` | Override the gateway port and bin dir. |
| `SERVER_SRC`, `CLIENT_SRC` | `Makefile` | Override the federated server/client source. |
| `DEBUG` | `Makefile` | `DEBUG=1` enables ASan+UBSan and `-O0 -g3`. |

### Build-time defines

| Define | Where | Meaning |
|---|---|---|
| `MATRIX_NO_BLAS` | `Matrix.hpp` | Do not use cblas; portable loops. |
| `MATRIX_USE_BLAS` | `Matrix.hpp` | Set to 1 to include `<cblas.h>`. Default 0. |
| `TENSORF_USE_OPENBLAS` | `CMakeLists.txt` | CMake option to link OpenBLAS instead of the shim. |
| `TENSORF_CBLAS_SHIM_H` | `include/tools/cblas.h` include guard | Checked by `WorkloadProbe.hpp`. |
| `MATRIX_IEEE_DIVISION` | `VectorMath.hpp` | Floating-point division by zero follows IEEE. |
| `MATRIX_NO_VECTOR_OPERATORS` | `Matrix.hpp` | Do not include `Overloads/Overload.hpp`. |
| `TENSORF_FORCE_SOFT_FLOAT16` | `scalar.hpp` | Use `soft_float16` even if `_Float16` exists. |
| `ORIG` | `tests/diff_ops.cpp` | Include `Matrix.orig.hpp`. |

### Gateway REST API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | `{"status":"ok","ws_clients":N}`. |
| `GET` | `/api/jobs` | List of jobs (summary). |
| `GET` | `/api/jobs/{id}` | Single job detail + 500-line log tail. |
| `POST` | `/api/jobs/{id}/kill` | SIGTERM the job. |
| `POST` | `/api/jobs/benchmark` | Launch `bin/benchmark`. |
| `POST` | `/api/jobs/tests` | Launch `bin/basic_tests`. |
| `POST` | `/api/jobs/gpt2` | Launch `bin/gpt2`. |
| `POST` | `/api/jobs/smollm` | Launch `bin/smollm`. |
| `POST` | `/api/jobs/server` | `{"port":N}` → launch `bin/server`. |
| `POST` | `/api/jobs/client` | `{"host":"...","port":N}` → launch `bin/client`. |
| `OPTIONS` | any | CORS preflight (204, no body). |

POST bodies may also pass `{"args": ["...", "..."]}` to override the default arguments. All responses are JSON with `access-control-allow-origin: *`.

### Gateway WebSocket

`GET /ws` upgrades to WebSocket and streams JSON events:

```
{"type":"job_started","job":{...}}
{"type":"log","job_id":"...","line":"..."}
{"type":"job_exited","job":{...}}
```

The `job_exited` event is documented in `main.cpp` as "not yet implemented" — the dashboard polls `/api/jobs` every 6 s to catch status transitions the WS stream does not announce.

### Profiler wire protocol

- `PROTO_MAGIC = 0x54465052` ("TFPR"), `PROTO_VERSION = 0x01`.
- `MsgType`: `PROFILE_REPORT`, `PROFILE_ACK`, `PING`, `PONG`.
- `ChecksumType`: `NONE`, `CRC32`, `XXHASH`.
- `ProtoFlags`: `NONE`, `HAS_MEMORY_PHASES`, `WANTS_OVERRIDE`, `IS_RETRY`, `INTERNET_MODE`.
- Default ports: TCP 9731, UDP 9732; Unix socket `/tmp/tensorf_profiler.sock`.
- `MAX_STORAGE_DEVICES = 16`.

### Federated learning

- `Server(model_path, hyper, min_clients_per_round, chunk_mb, model_args...)`.
- `Client::sendLoraConfig(lora_enabled, rank, alpha)` / `receiveLoraAck(accepted, reason)` handshake.
- Chunked protocol: sender and receiver choose their own `chunk_elems` independently.
- Logits protocol for FedDistill: `[n_examples][vocab_size][total_bytes][T × N]`.

### Dashboard configuration

- **Gateway URL**: editable in the topbar (`#gwUrl`), default `http://localhost:8080`. The sidebar footer mirrors it.
- **Launch presets**: `PRESETS` in `param-schema.js` (server:fedavg/fedprox/fedasync, client:default/gpu, benchmark:quick/full).
- **Launch flags**: `FLAG_HELP` in `param-schema.js` — per-kind flag definitions (name, type, default, help, options). The launcher cards build their inputs from this table.
- **Charts**: `js/charts.js` (`chartBaseOptions`, `chartBarOptions`, `CHART_COLORS`).
- **Log parser**: `js/parser.js` (`RX` regex table, `parseMetricLine`, `parseBenchmark`, `parseTests`).

No `.env`, YAML, TOML, or JSON config files were found (other than the tokenizer `vocab.json` read at runtime).

## Usage

### Building and running the gateway

```bash
make gateway                                    # → bin/tensorf-gateway
PORT=9090 make run-gateway                      # build + run on port 9090
TENSORF_BIN_DIR=/path/to/bin ./bin/tensorf-gateway --port 8080
```

The gateway launches `bin/benchmark`, `bin/basic_tests`, `bin/gpt2`, `bin/smollm`, `bin/server`, `bin/client` (relative to `TENSORF_BIN_DIR`). It uses a **single-threaded** `io_context` because it calls `fork()` + `execv()` to launch jobs, which is unsafe with a multithreaded Asio event loop. The main loop wraps `ioc.run()` in a `try`/`catch` so a single malformed request cannot kill every in-flight job's connection.

### Federated learning

```bash
make server    # builds tests/GPT/Server.cpp → bin/server
make client    # builds tests/GPT/Client.cpp → bin/client
make net       # both
```

The server/client sources are `tests/GPT/Server.cpp` and `tests/GPT/Client.cpp` by default; override with `SERVER_SRC` / `CLIENT_SRC`.

### Running the examples

```bash
./bin/gpt2          # GPT-2 inference (needs SLM/gpt2-small-f32.gguf)
./bin/smollm        # SmolLM2 inference (needs SLM/lSmolLM2-135M-Instruct-f16.gguf)
./bin/transformer   # character-level training on Datasets/ (needs Datasets/*.txt)
```

`examples/transformer.cpp` trains a small Llama (`n_heads=4`, `n_layers=12`, `input_dim=128`, `block_size=4`, `batch_size=4`, `ffn_hidden=5`) on `Datasets/`, saves `Models/Llama.hge`, and generates 50 characters from the prompt `"the data type is int"`.

### Running the tests

```bash
make tests                            # builds (and, per-target, runs) the test suite
./bin/basic_tests                     # Matrix/NN/MNIST/serialize
./bin/gcntest                         # synthetic two-cluster GCN training
./bin/spmmtest                        # SpMM gradients + sparse-vs-dense speed
./bin/coratrain Datasets/cora/cora.content Datasets/cora/cora.cites 200 42
./bin/matrix_tests                    # Matrix<T> for all supported types
./bin/lowprec_tests                   # FP8/FP4/float16
./bin/backprop_test                   # ordered backprop
./bin/algorithm_profiler_test         # AlgorithmProfiler + CostModel
./bin/profiler_capability_test [--quick]
./bin/profiler_workload_test [--quick]
./bin/profiler_probe_test [--quick]
./bin/device_benchmark_test
./bin/hw_fingerprint_test
./bin/memory_profiler_test
```

The profiler tests accept `--quick` to skip real-machine runs.

### Android

```bash
./scripts/run_android.sh                    # arm64-v8a
ABI=x86_64 ./scripts/run_android.sh         # x86_64 emulator
CORA=0 ./scripts/run_android.sh             # skip Cora
EPOCHS=50 ./scripts/run_android.sh          # shorter Cora run
./scripts/run_android.sh -DTENSORF_USE_OPENBLAS=ON
```

### Web dashboard

1. Start the gateway: `make run-gateway` (or `PORT=9090 make run-gateway`).
2. Serve the dashboard: `cd web && python3 -m http.server 8000`.
3. Open `http://localhost:8000/`.
4. Set the gateway URL in the topbar if it is not `http://localhost:8080`.
5. Use the sidebar to navigate:
   - **Dashboard** — live job/round/loss counters, loss curves, recent activity, jobs table.
   - **Launch job** — dynamic launcher cards (server, client, benchmark, tests, gpt2, smollm) with per-flag inputs, plus a preset selector.
   - **Jobs** — full jobs table + per-job detail (log console + extracted metrics chart).
   - **FL rounds** — per-instance federated round telemetry (timing, clients, bandwidth, rounds table).
   - **Hardware profiler** — per-client CPU/ISA, cache, RAM, benchmark probes, derived hyperparameters, memory by stage.
   - **Benchmark** — parsed benchmark results (matmul, mem BW, latency) + advisor box + memory box.
   - **Tests** — parsed test sections (PASS/FAIL counts) + loss chart.
   - **Memory & checkpoints** — memory pressure per running job + free/kill controls, and an explainer card on server-side checkpointing flags.
   - **Live console** — the unfiltered `/ws` firehose, colour-coded, with autoscroll and clear.
   - **Documentation** — links to the project's own docs site and GitHub.

### Profiler wire protocol

```cpp
#include "MessageProtocol.hpp"
#include "MessageTransport.hpp"

WireBuffer wb = serialize(fp, bench, cfg, mem, client_id,
                          ChecksumType::CRC32,
                          ProtoFlags::HAS_MEMORY_PHASES);

TransportConfig tcfg;
tcfg.server_host = "192.168.1.10";
tcfg.internet_mode = true;   // use XXHASH

send_profile(wb, tcfg);

// Server side
TCPTransport tt;
WireBuffer received;
tt.listen_once(DEFAULT_TCP_PORT, received,
    [](const ParseResult& r, const std::string& peer) {
        if (r.ok) { /* r.body, r.storage, r.client_id, r.timestamp_us */ }
    });

print_wire(received);
```

**Note**: `MessageProtocol.hpp` includes `Profiler/HardwareFingerprint.hpp`, `Profiler/BenchmarkRunner.hpp`, `Profiler/HyperparamAdvisor.hpp`, and `Profiler/MemoryProfiler.hpp`, and uses types `BenchmarkResult`, `HyperparamConfig`, `CPUGen`, `ISA`, `CPUVendor`, `RAMType`, `StorageType`, `QuantPolicy`. `BenchmarkRunner.hpp` was **not uploaded**, and the other names are not in the uploaded profiler headers (which use `tensorf::profile::HardwareFingerprint` and `tensorf::prof::DeviceFingerprint`). This file as uploaded would not compile against the uploaded headers; it is documented here as it appears.

## Detailed File and Module Documentation

### Build system

#### `CMakeLists.txt`

- `cmake_minimum_required(VERSION 3.18)`, `project(TensorF LANGUAGES CXX)`.
- `CMAKE_CXX_STANDARD 20`, `CMAKE_CXX_STANDARD_REQUIRED ON`, `CMAKE_CXX_EXTENSIONS OFF`.
- Defaults `CMAKE_BUILD_TYPE` to `Release`.
- `TENSORF_INCLUDES` lists the same include roots as the Makefile.
- `option(TENSORF_USE_OPENBLAS "Link a real OpenBLAS instead of the shim" OFF)`.
- `add_library(tensorf_deps INTERFACE)` with `target_include_directories(... BEFORE INTERFACE ${TENSORF_INCLUDES})`.
- If `TENSORF_USE_OPENBLAS`: `find_library(OPENBLAS_LIB openblas REQUIRED)` and link it.
- Else: reads `include/tools/cblas.h`, checks for the string `tensorf_cblas_shim`; if not found, `message(FATAL_ERROR ...)`.
- `find_package(Threads REQUIRED)` and link `Threads::Threads`.
- Builds 11 test executables in a `foreach` loop: `gcntest spmmtest coratrain basic_tests algorithm_profiler_test memory_profiler_test device_benchmark_test hw_fingerprint_test profiler_capability_test profiler_probe_test profiler_workload_test`. Each links `tensorf_deps`, gets `-Wall -Wextra -Wno-unused-parameter`, and outputs to `${CMAKE_BINARY_DIR}/bin`.

#### `Makefile`

- `CXX := g++`, `STD := -std=c++20`, `OPT := -O2`, `WARN := -Wall -Wextra -Wno-unused-parameter`, `DEBUG ?= 0`. `DEBUG=1` switches to `-O0 -g3 -fsanitize=address,undefined`.
- `INC` lists the same include roots as CMake.
- `LIBS := -lpthread`, plus OpenBLAS via `pkg-config --libs openblas` (fallback `-lblas`).
- `JSON_INC := $(shell pkg-config --cflags nlohmann_json 2>/dev/null)`.
- `mkdir -p bin` at parse time.
- `.PHONY` targets: `all gpt2 llama smollm transformer benchmark tests gateway run-gateway clean install-deps check-deps help`.
- `all: check-deps gpt2 smollm transformer benchmark tests gateway`.
- Per-target rules for every test and example.
- `tests: <each target>` — each test is appended to the `tests` phony target so `make tests` builds them all.
- `net: client server`, `examples: gpt2 smollm transformer`.
- `run-gateway: gateway` runs `TENSORF_BIN_DIR=$(TENSORF_BIN_DIR) $(BIN)/tensorf-gateway --port $(PORT)`.
- `clean` removes `bin/`.
- `install-deps` runs `apt-get install -y --no-install-recommends build-essential g++ cmake pkg-config libopenblas-dev nlohmann-json3-dev libxxhash-dev wget curl`.
- `check-deps` prints the compiler version, checks `pkg-config --modversion nlohmann_json` (fallback `/usr/include/nlohmann/json.hpp`), checks `pkg-config --modversion openblas`, and compiles a tiny `int main(){}` with `-pthread` to verify pthreads.
- `help` prints the list of targets.

### Gateway (C++)

#### `gateway/src/main.cpp`

Boost.Beast HTTP + WebSocket server. Namespaces: `beast`, `http`, `websocket`, `net`, `tcp`, `json` (nlohmann).

- `static std::string BIN_DIR = "../bin"` — overridable via `TENSORF_BIN_DIR` or `--bin-dir`.
- `bin_path(name)` → `BIN_DIR + "/" + name`.
- `on_job_line(job_id, line)` — builds `{"type":"log","job_id":...,"line":...}` and broadcasts.
- `launch_and_announce(kind, binary, args)` — `JobManager::instance().launch(...)`, gets the job, broadcasts `{"type":"job_started","job":...}`, returns the job JSON.
- `json_response(req, status, body)` — sets `server: tensorf-gateway`, CORS headers, and `keep_alive(req.keep_alive())`. For 1xx/204/304, leaves the body and content-type unset (RFC 9110) — the comment explains that the old OPTIONS handler dumped `"{}"` into a 204 body, which made Beast's serializer throw `std::invalid_argument` and terminated the process.
- `parse_body(req)` — parses JSON or returns `json::object()`.
- `route(req)` — strips the query string from `req.target()`, handles `OPTIONS` → 204, `GET /api/health`, `GET /api/jobs`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/kill`, and `POST /api/jobs/{benchmark,tests,gpt2,smollm,server,client}`. POST bodies may contain `args` (array) to override the default arguments. Returns 404 JSON otherwise.
- `HttpSession` — reads an HTTP request; if it is a WebSocket upgrade, hands the socket and the already-read request to a new `WsSession`; otherwise routes and writes the response.
- `Listener` — opens, binds, and listens on the endpoint; accepts into a new `HttpSession` on a strand.
- `main(argc, argv)` — reads `TENSORF_GATEWAY_PORT` and `TENSORF_BIN_DIR` from the environment, then `--port` / `--bin-dir` from the command line. Creates a **single-threaded** `io_context{1}` (safe with `fork()`), starts the listener, and runs `ioc.run()` in a `try`/`catch` loop that logs and continues on exception.

#### `gateway/include/JobManager.hpp`

- `enum class JobStatus { Running, Exited, Failed, Killed }` + `to_string`.
- `struct Job` — `id`, `kind`, `command`, `pid`, `running` (atomic bool), `status`, `exit_code`, `started_at`, `ended_at`, `log_lines` (deque), `log_seq`, `log_mutex`, `MAX_LOG_LINES = 5000`, `on_line` callback, `push_line(line)`.
- `class JobManager` — singleton via `instance()`.
  - `launch(kind, binary_path, args, on_line)` — creates a pipe, `fork()`s, in the child `dup2`s the pipe to stdout/stderr and `execv`s the binary (exits 127 on failure), in the parent closes the write end, sets the read end non-blocking, creates a `Job`, stores it, and starts a detached reader thread that reads chunks, splits lines, calls `job->push_line`, then `waitpid`s and sets the final status.
  - `kill_job(id)` — `::kill(pid, SIGTERM)`.
  - `get(id)`, `list()`.
  - `job_to_json(job, log_tail = 200)` — id, kind, command, pid, status, running, exit_code, log_seq, and the last `log_tail` log lines.
  - `generate_id()` — `hex(now) + "-" + counter`.

#### `gateway/include/WsHub.hpp`

- `class WsHub` — singleton via `instance()`. `join(session)`, `leave(session)`, `broadcast(msg)`, `count()`. A `std::mutex` guards a `std::set<std::shared_ptr<WsSession>>`.
- `class WsSession : public std::enable_shared_from_this<WsSession>` — owns a `websocket::stream<tcp::socket>`. `run(req)` sets the timeout to `suggested(server)`, sets a `server: tensorf-gateway` decorator, `async_accept`s the already-read upgrade request, joins the hub, and starts reading. `send(payload)` posts to the WebSocket executor, pushes onto `outgoing_`, and starts `do_write()` if it was empty. `do_read()` consumes the buffer and loops. `do_write()` sends the front of `outgoing_`, pops it, and continues.
- `WsHub::broadcast(msg)` — serialises to a string, locks, and calls `send` on every session.

### Web dashboard

#### `web/index.html`

The dashboard shell. Loads CSS (`base`, `layout`, `components`, `charts`), Chart.js 4.4.4 from jsDelivr, and the JS modules in dependency order (state → utils → toast → charts → param-schema → parser → docs-data → api → jobs → views → main). Contains:

- **Topbar**: brand, WebSocket status dot + label, gateway URL input, Reconnect button, Docs link.
- **Sidebar**: 5 groups (Overview, Compute, Federated, Analysis, System) with nav items (`overview`, `launch`, `jobs`, `federated`, `profiler`, `benchmark`, `tests`, `memory`, `console`, `docs`), a jobs badge, and a footer with the gateway URL and job count.
- **10 views** (`view-overview`, `view-launch`, `view-jobs`, `view-federated`, `view-profiler`, `view-benchmark`, `view-tests`, `view-memory`, `view-console`, `view-docs`).
- **Chart zoom modal** (`#chartZoomModal`) — click a chart to open it full-size.
- **Toasts container** (`#toasts`).

#### `web/css/base.css`

Design tokens, reset, typography. The comment describes the design language as an "instrument panel" — the profiler fingerprints ISA/cache/RAM down to the byte, and the UI should feel like the readout of that same probe. Imports JetBrains Mono + Inter from Google Fonts. Defines CSS variables for `--bg`, `--bg-panel`, `--bg-raised`, `--line`, `--line-soft`, `--amber`, `--amber-dim`, `--cyan`, `--cyan-dim`, `--green`, `--red`, `--red-dim`, `--text`, `--text-dim`, `--text-faint`, `--mono`, `--sans`. Includes a `body` radial-gradient background, custom scrollbars, `::selection`, and a `.section-sub` helper.

#### `web/css/layout.css`

Shell layout: `.shell` is a CSS grid `240px 1fr` × `56px 1fr`. `.topbar` spans both columns; `.sidebar` is column 1 / row 2; `.main` is column 2 / row 2. `.view` is hidden by default and `.view.active` is shown with a fade-in. Also `.brand`, `.ws-status`, `.gw-input`, `.docs-link`, `.nav-item`, `.nav-badge`, `.sidebar-footer`, `.view-header`, `.view-title`, `.view-desc`, `.view-actions`, `.two-col`, `.job-detail-grid`.

#### `web/css/components.css`

Buttons (`.btn`, `.primary`, `.danger`, `.cyan`, `.sm`, `.disabled`), grids (`.grid.cols-4/3/2`), cards (`.card`, `.card-title`), stat tiles (`.stat`, `.stat-label`, `.stat-value`, `.stat-sub`, `.stat.ok/warn/bad`), launch cards (`.launch-grid`, `.launch-card`, `.lc-kind`, `.lc-name`, `.lc-desc`, `.lc-run`), field rows (`.field-row`, `.field`), launch-page-specific styles (`.preset-bar`, `.job-card`, `.flag-grid`, `.flag-row`, `.launch-btn`, `.add-buttons`), jobs table (`.jobs-table`, `.job-kind`, `.job-status`, `.job-id`, `.job-cmd`, `.empty-row`), log console (`.console`, `.ln`, `.ln.hl-loss/round/err/ok`, `.console-empty`), pills (`.pill-row`, `.pill`), profiler readout (`.probe-table`, `.isa-badges`, `.isa-badge`), toasts (`.toasts`, `.toast`, `.toast.err/ok`), empty state (`.empty-state`), memory bars (`.mem-bar-track`, `.mem-bar-fill`, `.warn`, `.bad`), docs (`.doc-grid`, `.doc-card`), and a few misc helpers (`.chart-wrap`, `.spinner`, `.checkbox-row`).

#### `web/css/charts.css`

Only `canvas{max-width:100%;}` — Chart.js theming happens in JS.

#### `web/js/state.js`

Single shared mutable store. Exports `state` with `ws`, `jobs`, `jobLogs`, `metrics`, `selectedJobId`, `pollTimer`, and `gwBase()` (reads the topbar input, strips a trailing slash).

#### `web/js/utils.js`

`escapeHtml`, `escapeAttr`, `statusClass(job)`, `statusDotColor(cls)`.

#### `web/js/toast.js`

`toast(msg, type = 'info')` — appends a `.toast` to `#toasts`, fades it out after 4.2 s.

#### `web/js/charts.js`

- `safeChart(ctx, config)` — checks for `Chart`, catches construction errors, shows a `.chart-error` message on failure.
- `CHART_COLORS` — 6-colour palette.
- `chartBaseOptions(xlabel, ylabel)` — line chart options with a linear x-axis.
- `chartBarOptions()` — bar chart options.
- `enableChartInteractions(canvasId, chartInstance)` — click to open the zoom modal, plus an injected "Print" button.
- `closeZoomModal()`.

#### `web/js/param-schema.js`

- `PRESETS` — named launch presets (`server:fedavg`, `server:fedprox`, `server:fedasync`, `client:default`, `client:gpu`, `benchmark:quick`, `benchmark:full`).
- `FLAG_HELP` — per-kind flag definitions for the dynamic launcher (`server`, `client`, `benchmark`, `tests`, `gpt2`, `smollm`). Each flag has `flag`, `type` (`number`, `text`, `select`, `boolean`), `default`, `help`, and optionally `options`.
- Exposes both on `window`.

#### `web/js/parser.js`

Regex-based log parser. The header comment explains that the gateway only ships raw stdout/stderr per job, so the dashboard parses the binaries' structured printf/cout lines live. The `RX` table contains ~30 patterns covering training/distillation loss, client/server rounds, send/receive timings, aggregate/round-total times, RSS, CPU/ISA/RAM/L3, matmul/mem-BW/latency, storage, recommended batch/block/quant, param RAM, physical cores, max freq, total/available RAM, theoretical BW, and PASS/FAIL lines.

- `emptyMetrics()` — `{ loss: [], rounds: [], profiler: null, bench: null, hyper: null }`.
- `lastRound(m)` — the last round, creating a placeholder if empty.
- `parseMetricLine(jobId, line)` — applies every regex and pushes into `state.metrics[jobId]`.
- `ingestLogLine(jobId, line, fromBulk = false)` — appends to `state.jobLogs[jobId]` (capped at 5000 lines) and calls `parseMetricLine`. `fromBulk` avoids re-pushing lines that were already seen.
- `classifyLogLine(line)` — returns `hl-loss`, `hl-round`, `hl-err`, `hl-ok`, or `''`.
- `renderLogLine(line)` — wraps in a `.ln` div with the classification class.
- `parseBenchmark(log)` — extracts `[Results]`, `HYPERPARAMETER ADVISOR REPORT`, and `MEMORY PROFILE REPORT` blocks into `{ results, advisor, memory }`.
- `parseTests(log)` — splits on `=== Section ===`, counts `[PASS]`/`[FAIL]`, and extracts `iter: X loss: [Y]` lines.

#### `web/js/docs-data.js`

`REPO` (GitHub), `RAW_DOCS_BASE`, and a `DOCS` array of `{ cat, title, desc, href }` entries pointing at the project's own `docs/` site.

#### `web/js/api.js`

- `api(path, opts)` — fetch wrapper with JSON headers and error handling.
- `connectWs()` — opens a WebSocket to `<gwBase>/ws`, sets the status dot, reconnects after 4 s on close, parses JSON messages, dispatches to `handleWsMessage`.
- `setWsStatus(s)` — updates `#wsDot` and `#wsLabel`.
- `handleWsMessage(msg)` — appends to the global console; on `job_started`, `upsertJob` + toast + refresh; on `log`, `ingestLogLine` + (if selected) `appendJobConsole` + refresh.
- `pollJobs()` — REST fallback every 6 s (`GET /api/jobs`).
- `killJob(id)` — `POST /api/jobs/{id}/kill`.
- `launchJob(kind, body)` — `POST /api/jobs/{kind}`.
- `fetchJobDetail(id)` — `GET /api/jobs/{id}`.
- `launchJobRaw(kind, args)` — `launchJob(kind, { args })`.

#### `web/js/jobs.js`

- `upsertJob(job)` — merges the job into `state.jobs`, initialises `jobLogs`/`metrics`, fans the bulk log through `ingestLogLine(..., fromBulk=true)`, and updates the two job-count labels.
- `goToJob(id)` — switches to the Jobs view and selects the job.

#### `web/js/main.js`

- `VIEW_RENDERERS` — maps view name → renderer. `benchmark` and `tests` are inline lambdas that pick the latest matching job.
- `refreshCurrentView(light = false)` — re-renders the active view; if `light` and the Jobs view, re-renders the detail metrics only.
- `initNav()` — nav item click handlers.
- `initTopbar()` — Reconnect button + gateway URL change.
- `boot()` — `renderDocs`, `initNav`, `initTopbar`, `initLaunchView`, `initConsoleView`, sets the sidebar gateway label, starts the 6 s poll timer, calls `pollJobs`, `connectWs`, `renderOverview`.
- `DOMContentLoaded` → `boot`.

#### `web/js/views/overview.js`

`renderOverview()` — 4 stat tiles (running, failed, FL rounds, loss samples), the last 8 jobs table, `renderLossChart`, `renderActivityFeed`. `renderLossChart` builds one dataset per job with loss samples and colours them from `CHART_COLORS`. `renderActivityFeed` shows the last log line of the 10 most recently active jobs. `refreshOverview` = `renderOverview`.

#### `web/js/views/launch.js`

- `addJobCard(kind)` — creates a card with per-flag inputs from `FLAG_HELP[kind]` (select, checkbox, or text/number), a Launch button that collects `--flag value` pairs (or `--flag` for checked booleans) and calls `launchJobRaw`, and a remove button.
- `initLaunchView()` — injects the preset bar, `#jobCards` container, and the "+ Add Server/Client/Benchmark/Tests/GPT-2/SmolLM" buttons; wires the preset selector; adds a default `server` card.

#### `web/js/views/jobs-view.js`

- `renderJobRows(tbody, jobs)` — renders the jobs table rows with View/Kill buttons and row click → `selectJob`.
- `renderJobsTable()` — sorts jobs by `started_at` descending and renders into `#jobsTable`.
- `selectJob(id)` — sets `state.selectedJobId`, fetches the detail, renders the pills (kind, status, pid, log lines, Kill button), fills the log console with `renderLogLine`, adds a Copy Log button, and calls `renderJobDetailMetrics`.
- `renderJobDetailMetrics(id)` — draws the per-job loss chart or shows the empty state.
- `appendJobConsole(id, line)` — appends a log line if the job is selected.

#### `web/js/views/federated.js`

- `getFlInstances()` — ids with non-empty `rounds`.
- `renderInstanceCards()` — one card per FL instance with status, kind, last round, last loss, and a View button.
- `renderFederatedDetail(instanceId)` — 4 stat tiles (latest round, server rounds, client rounds, clients last) + `renderTimingChart`, `renderClientsChart`, `renderBandwidthChart`, `renderRoundsTable`.
- `renderTimingChart(serverRounds)` — aggregate ms and round-total ms per round.
- `renderClientsChart(serverRounds)` — clients per round.
- `renderBandwidthChart(clientRounds)` — send-deltas and recv-weights MB/s per round.
- `renderRoundsTable(allRounds)` — last 40 rounds, newest first.
- `renderFederated()` — injects the whole view markup on first call, then `renderInstanceCards()` + `renderFederatedDetail(selectedFederatedInstance)`.

#### `web/js/views/profiler.js`

- `ISA_LEVELS` — `['BASELINE','AVX','AVX2','AVX512','AVX512_VNNI','AMX']`.
- `renderProfiler()` — populates the job selector with jobs that have profiler data, shows/hides the empty state, and calls `renderProfilerContent`.
- `renderProfilerContent(id)` — fills the CPU/ISA, cache, RAM, hyperparameter tables; renders the ISA badges; calls `renderBenchChart` and `renderMemStageChart`. If the job is a `benchmark`, delegates to `renderBenchmark`.
- `renderBenchChart(b)` — matmul L3 vs RAM GFLOP/s.
- `renderMemStageChart(p, h, m)` — RSS by stage (baseline, loaded, train, infer).

#### `web/js/views/memory.js`

`renderMemory()` — one card per running job with the last known RSS, a memory bar (green/amber/red by percentage), the command, and Free/kill + View logs buttons. Shows an empty state when nothing is running.

#### `web/js/views/benchmark-view.js`

`renderBenchmark(job)` — parses the job log with `parseBenchmark`, renders a results grid (matmul peak/L3, mem BW read/write, L3/RAM latency), an advisor box (hardware score, config fit, n_embed, n_heads, n_layers, block_size, batch_size, quant, threads, params, train/infer peak), a memory box (baseline/loaded/infer RSS + heap, peak RSS, recommended free), and a bar chart.

#### `web/js/views/tests-view.js`

`renderTests(job)` — parses the job log with `parseTests`, renders per-section PASS/FAIL counts, a loss chart if losses were found, and a copy-log console.

#### `web/js/views/docs.js`

`renderDocs()` — one card per `DOCS` entry + a GitHub card.

#### `web/js/views/console-view.js`

- `appendGlobalConsole(msg)` — appends a colour-coded line to `#globalConsole` (`▶ job_started ...` for starts, `[<id>] <line>` for logs, raw JSON otherwise), autoscrolls, and trims to 2000 children.
- `initConsoleView()` — wires the Clear button.

### Examples

#### `examples/GPT2.cpp`

Loads `SLM/gpt2-small-f32.gguf` with `GPT2HyperParams{50257, 768, 1024, 12, 12}`, builds a `GPTGGUFLoader<float, Linear>`, loads the model and tokenizer, encodes the prompt `"The data type is int"`, prints each token id, builds a `{1, seq_len}` context tensor, calls `model.generate(context, 30, 0.8f, 30)`, and decodes only the generated tokens.

#### `examples/SmollLLM.cpp`

Loads `SLM/lSmolLM2-135M-Instruct-f16.gguf` with `LlamaHyperParams{49152, 576, 512, 9, 3, 30, 1536}`, builds a `LlamaGGUFLoader<float, Linear>`, loads the model and a `LlamaTokenizer`, tests a special-token prompt, then formats a chat template with a system message, encodes it, builds a `{1, seq_len}` context tensor, calls `model.generate(context, 30, 0.6f, 40)`, and decodes only the generated part.

#### `examples/transformer.cpp`

A minimal character-level Llama training script:
- `TextDataset<float> ds("Datasets", block_size=4, batch_size=4); ds.load();`
- `Llama<float, Linear> model(vocab_size, input_dim=128, block_size=4, n_heads=4, n_layers=12, ffn_hidden=5);`
- `model.train(get_batch, iters=1, evals=1, true);`
- `model.save("Models/Llama.hge");`
- Encodes the prompt `"the data type is int"`, builds a `{1, seq_len}` context, calls `model.generate(context, 50)`, decodes only the generated part.

### Core types

#### `include/core/Types/shape.hpp`

`using shape_t = std::vector<size_t>;` — the single definition, included by every other header.

#### `include/core/Types/scalar.hpp`

Element traits, accumulator type, and math wrappers. Includes `fp8.hpp`, `fp4.hpp`, `float16.hpp`, `shape.hpp`.

- `float16` is `_Float16` when available (and `TENSORF_FORCE_SOFT_FLOAT16` is not defined), otherwise `soft_float16`. `float16_is_native` reports which.
- Provides a `std::numeric_limits<_Float16>` specialisation for libstdc++ versions that ship none.
- Traits: `is_fp8_v<T>`, `is_fp4_v<T>`, `is_low_precision_v<T>`.
- `acc<T>::type` / `acc_t<T>`: low-precision types accumulate in `float`.
- Conversions: `to_acc<T>(v)`, `from_acc<T>(v)`.
- Math wrappers: `exp`, `log`, `sqrt`, `cbrt`, `sin`, `cos`, `tan`, `pow`, `abs`, `isnan`, `isinf`.

#### `include/core/Types/fp8.hpp`

`FP8<E,M>` — 8-bit float, layout `[sign | E exponent | M mantissa]` MSB-first. Documented decisions:
- E4M3 here is IEEE-style (exponent all-ones encodes inf/NaN, max finite 240). **Not** the OCP MX E4M3 format.
- No subnormals: anything below `2^(1-bias)` flushes to signed 0.
- Rounding is half-away-from-zero.
- `bits` is zero-initialised.
- Encode/decode use `frexp`/`ldexp`.
- `numeric_limits<FP8<E,M>>` is specialised as a class.

#### `include/core/Types/fp4.hpp`

`FP4<E,M>` — 4-bit float. Documented decisions:
- E2M1 max finite is 3.0.
- No subnormals; anything below `2^(1-bias)` flushes to signed 0.
- Rounding is half-away-from-zero.
- `FP4<1,2>` is degenerate, so `static_assert(E >= 2, ...)` rejects it. `fp4_e1m2` stays declared but fails when instantiated.
- `uint4_t` is retained for ABI compatibility only.
- NaN maps to max finite. Signed zero is preserved.

#### `include/core/Types/float16.hpp`

`soft_float16` — software IEEE-754 binary16, `[1 sign | 5 exponent | 10 mantissa]`, bias 15. Always available. `scalar.hpp` aliases `float16` to `_Float16` when there is one, otherwise to this type.

- Arithmetic is done in `float` and rounded back once.
- Conversions are round-to-nearest-even and handle subnormals, ±0, ±inf, and NaN exactly like hardware `_Float16` (verified bit-for-bit against `_Float16` in `lowprec_tests`).
- `encode(float)` / `decode(uint16_t)` are the codec; `from_bits` / `to_bits` are the bit accessors.
- `std::numeric_limits<soft_float16>` is specialised.

#### `include/core/Types/types.hpp`

Central type aliases and forward declarations. Pulls in `shape.hpp` and `scalar.hpp`.

- Integer typedefs `int8`/`int16`/`int32`/`int64` are (misleadingly) **unsigned**; kept for API compatibility. Correctly-named aliases `i8`/`i16`/`i32`/`i64`/`u8`/`u16`/`u32`/`u64` are provided.
- `float32` = `float`, `float64` = `double`.
- FP8 aliases: `fp8_e3m4 = FP8<3,4>`, `fp8_e4m3 = FP8<4,3>`, `fp8_e5m2 = FP8<5,2>`.
- FP4 aliases: `fp4_e2m1 = FP4<2,1>`, `fp4_e1m2 = FP4<1,2>` (declared but fails when instantiated).
- Legacy `is_fp8<T>` / `is_fp4<T>` traits.
- Forward declarations: `Tensor`, `Graph`, `CSR`, `GGUFLoader`, `GPTGGUFLoader`, `LlamaGGUFLoader`, `Operation`, `Module`.
- `Operation_t<T> = std::shared_ptr<Operation<T>>`, `Tensor_t<T> = std::shared_ptr<Tensor<T>>`.

### Data structures (Matrix, Tensor, Graph, Broadcast, MatrixOps)

#### `include/core/DataStructures/Matrix.hpp`

Dense row-major n-D tensor. Include this file only: it pulls in `Broadcast.hpp` first and `MatrixOps.hpp` last.

Build switches: `MATRIX_NO_BLAS`, `MATRIX_IEEE_DIVISION`, `MATRIX_NO_VECTOR_OPERATORS`.

Invariants (from the header comment): `data.size() == prod(shape)` always. 0-D results have shape `{}` and exactly one element. Not thread-safe: the random generator is one shared global.

Namespace `mxd` helpers: `dependent_false_v`, `is_float_like_v`, `store_t`, `calc_t`, `rng`, `CallRng`, `draw_uniform`, `draw_normal`, `printable`, `blas_int`, `gemm_loops`, `gemm`, `dot_flat`.

`Matrix<T>` members: `data`, `shape`, `gpu`. Constructors from scalar, pointer, copy/move, `std::vector<T>`, `std::vector<T> + shape_t`, `std::vector<std::vector<T>>`, initializer lists, 3-D initializer lists. `from<K>` builds from array-like inputs.

Arithmetic: `+`, `-`, `*`, `/` (broadcasting), unary `-`, `==` (both `Matrix` and scalar), `!=` (scalar), `pow` (scalar and `Matrix`), `exponent`, `sqrt`, `cbrt`, `ln` (clamped at 1e-9 for float-like `T`).

Reductions: `sum()`, `sum(axis)`, `mean()`, `mean(axis)`, `variance()`, `std()`, `var(axis, ddof0)`, `std(axis, ddof0)`.

Accessors: `get_size`, `get_ndims`, `strides`, `numElementsSeen`, `col`, `row`, `at(shape_t)`, `at(initializer_list)`, `at(Matrix<bool>)`, `slice_row`, `slice_cols`, `slice_axis`, `flatten`, `reshape`, `get_data`, `elemsAt`.

Static factories: `inf`, `nan`, `ravel`, `expand_dims`, `any`, `hasNaN`, `concat`, `where`, `stack`, `arrange`/`arange`, `zeros`, `ones`, `choice`, `random`/`randu`, `manual_seed`, `randn`/`randomn`, `he`, `eye`, `tril`/`triup`/`triu`, `one_hot`.

In-place: `ones`, `zeros`, `copy_from`, `maximum`, `clear`.

Transpose: `transpose()` (reverse axes; 1-D → `{n,1}`) and `transpose(shape_t perm)`.

Products: `matmul` (1-D promotion, batch broadcast, last-two-axes contraction), `dot` (NumPy semantics), `vdot` (flattened inner product).

Printing: `print(std::ostream&)`, `operator<<` via `MatrixOps.hpp`.

Documented behaviour changes (from the header):
- `dot` for two 2-D operands is the **flattened** inner product (`np.sum(a*b)`), not a matrix product.
- `random()` is uniform [0,1) from the shared generator (was `std::rand()`).
- `log()` clamps exactly like `ln()`.
- `maximum(a)` returns `a` for values below `a` (was 0); identical for `a == 0`.

#### `include/core/DataStructures/Tensor.hpp`

`Tensor<T>` — autograd wrapper around `Matrix<T>`. Members: `val`, `grad`, `shape`, `ndims`, `backOp` (`Operation_t<T>`), `requires_grad`.

Key methods:
- `topo_order()` — iterative DFS returning every tensor reachable through `backOp->inputs()`, inputs before outputs, `this` last. Each node appears once; input order is fixed, so the order is deterministic.
- `accumulate(const Matrix<T>& g)` — adds a gradient contribution; throws on shape mismatch; returns early if `!requires_grad`.
- `backward(Matrix<T> seed)` — clears the gradients of non-leaf nodes, accumulates the seed into `this`, then iterates `topo_order()` in reverse calling `backOp->backward(n->grad)` on each node with a non-empty gradient. Includes a NaN gradient check that prints the op and the gradient and throws `std::runtime_error`.
- `backward(Tensor_t<T> ingrad)` — calls `backward(ingrad->val)`.
- `zero_grad()` — clears the gradient of `this` and every tensor in `topo_order()`.
- `reset_graph()` — calls `backOp->reset_graph()` and nulls `backOp`.
- `operator=` copies `val`, `grad`, `backOp`, `shape`, `ndims`, `requires_grad`.
- Op factory methods: `matmul`, `transpose` (with and without permutation), `softmax`, `relu`, `sum(axis)`, `sum()`, `sqrt`, `power`, `ln`, `exp`, `sigmoid`, `reshape`, `embed`, `dot`, `bool_index`, `cbrt`, `mean(axis)`, `mean()`, `var(axis)`, `std(axis)`.
- Static functions: `concat`, `cross_entropy`, `masked_cross_entropy`, `binary_cross_entropy`, `mse`, `rms_norm`, `transpose`, `zeros`, `ones`, `GeLU`, `SiLU`, `randn`, `random`, `eye`, `from`.
- `maximum(int)`, `at(initializer_list)`, `at(Tensor_t<bool>)`, `size()`.

#### `include/core/DataStructures/Graph.hpp`

Plain data container + graph construction. No autograd. Needs `-pthread` (`connect_by_similarity` spawns `std::thread` workers).

Lifecycle (from the header comment):
1. `add_node(...)` / `add_edge(...)` — raw, mutable.
2. `connect_by_similarity(threshold, ...)` — optional, adds edges from features.
3. `build()` — symmetrize, dedupe, self-loops, `D^-1/2 (A+I) D^-1/2` as CSR.
4. `adjacency()`, `features()`, `labels()` — read-only views for the model.

`CSR<T>` members: `n`, `row_ptr`, `col_idx`, `values`, `symmetric`, `nnz()`.

`Graph<T>` members: `add_node`, `add_edge`, `connect_by_similarity`, `build`, `num_nodes`, `feature_dim`, `num_raw_edges`, `features`, `labels`, `adjacency`, `adjacency_shared`, `dense_adjacency`. `connect_by_similarity` accumulates in `double`, so norms/dots are computed identically for float / double / FP8 / FP4 / float16. It throws if more than `max_edges` edges are found.

#### `include/core/DataStructures/Broadcast.hpp`

`mxd::numel`, `mxd::strides_of`, `mxd::shape_str`, `mxd::broadcast_shapes` (right-aligned; 0 broadcasts with 1 gives 0), `mxd::gather` (materialises a source into an output shape using per-dimension element strides; stride 0 repeats an element).

`Broadcast<T>` (all members static): `computeShapes`, `assertBroadcast`, `computeBroadcastResultShape`, `broadcast`, `broadcastTo` (numpy rules: right-aligned, size-1 dims repeat; throws on incompatible shapes or when the source data vector is shorter than its shape claims).

#### `include/core/DataStructures/MatrixOps.hpp`

Free operators for `Matrix<T>`. Scalar parameters are non-deduced (`std::type_identity_t<T>`), so `2*m`, `m+1`, `m+=1` work for `Matrix<float>`.

- `operator<<(std::ostream&, const Matrix<E>&)` — prints via `m.print`.
- Scalar arithmetic: `*`, `/`, `+`, `-` (both orders) and `pow` (scalar and `Matrix`).
- Comparisons returning `Matrix<bool>`: `<`, `>`, `<=`, `>=` (both orders). `Matrix<T> * Matrix<bool>` treats the mask as 0/1 (needed because comparisons now return `Matrix<bool>`, while `ReluOperation` multiplies `grad * (x > 0)`).
- Compound assignment via the `MATRIX_COMPOUND_OP` macro: `+=`, `-=`, `*=`, `/=` for both `Matrix` (same shape in place, otherwise broadcast) and scalar.
- `mxd::swap_last2` — swaps the last two axes (identity for rank < 2).
- `sumGradForBroadcast(grad, originalShape)` — reduces a broadcast gradient back to the original shape.

### Autograd operations

#### `include/core/Operations/Operation.hpp`

Abstract base class for every differentiable op.

```cpp
template <typename T>
class Operation : public std::enable_shared_from_this<Operation<T>> {
public:
    Tensor_t<T> t1, t2;
    virtual std::vector<Tensor_t<T>> inputs() const = 0;
    virtual void backward(Matrix<T> grad) = 0;
    virtual void to_string() = 0;
    virtual void zero_grad() = 0;
    virtual void reset_graph() = 0;
    virtual Tensor_t<T> forward() = 0;
    virtual ~Operation() = default;
};
```

`Tensor<T>` and `Matrix<T>` are forward-declared. Every op inherits `enable_shared_from_this` so `forward()` can pass `shared_from_this()` as the backward link of the output tensor.

#### The `Operations/*.hpp` classes

Every op is a `template <typename T> class <Name>Operation : public Operation<T>` with the same interface: `inputs()`, `forward()`, `backward(grad)`, `zero_grad()`, `reset_graph()`, `to_string()`.

| Op | Forward | Backward |
|---|---|---|
| `AddOperation` | `x + y` | `grad` to both, broadcast-reduced. |
| `SubtractOperation` | `x - y` | `grad` to `x`, `-grad` to `y`, broadcast-reduced. |
| `MultiplyOperation` | `x * y` | `grad * y` to `x`, `grad * x` to `y`, broadcast-reduced. |
| `DivisionOperation` | `x / y` | `grad / y` to `x`, `-grad * x / y²` to `y`, broadcast-reduced. |
| `MatmulOperation` | `A @ B` | `matmul_grads(A, B, grad)` (1-D promotion, batch broadcast, last-two-axes transpose). |
| `DotOperation` | `dot(A, B)` | 1-D inner product, matmul-compatible, or general N-D contraction via `dot_rhs_perm`. |
| `VdotOperation` | `vdot(A, B)` | `B * grad[0]` to `A`, `A * grad[0]` to `B`, reshaped. |
| `PowerOperation` | `x^y` | `grad * y * x^(y-1)` to `x`. |
| `LogOperation` | `ln(x)` | `1/x` where `x > 1e-9`, else `0` (float-like `T`). |
| `ExponentOperation` | `exp(x)` | `grad * exp(x)`. |
| `ReluOperation` | `max(x, 0)` | `grad * (x > 0)`. |
| `SigmoidOperation` | `1/(1+exp(-x))` | `grad * s * (1 - s)`. |
| `SoftmaxOperation` | row-wise softmax over the last axis | `s * (g - dot(g, s))`. |
| `SumOperation` | `sum(x)` | `grad[0]` broadcast to all elements. |
| `SumAxisOperation` | `sum(x, axis)` (kept as size-1) | `grad` broadcast along `axis`. |
| `TransposeOperation` | `transpose(x)` or `transpose(x, perm)` | inverse permutation applied to `grad`. |
| `ReshapeOperation` | `reshape(x, shape)` | `grad` reshaped back to the original shape. |
| `ConcatOperation` | `concat({x0, x1, …}, axis)` | `grad` sliced along `axis` at the recorded boundaries. |
| `IndexOperation` | `x.at(mask)` (bool mask) | `grad * mask`. |
| `EmbeddingOperation` | `table.elemsAt(idx)` | scatter-add `grad` rows into the table gradient. |
| `SpMMOperation` | `A @ X` (CSR × dense) | `Aᵀ @ dY` (same row-gather when `A` is symmetric, else scatter). |

Details worth noting:

- `MatmulOperation` exposes a static `matmul_grads(A, B, G)` used by `DotOperation` too. It reshapes 1-D operands, restores the squeezed output axes, and uses `mxd::swap_last2`.
- `DotOperation::backward` has three cases: 1-D × 1-D inner product; `rb == 1 || (ra == 2 && rb == 2)` delegates to `MatmulOperation::matmul_grads`; otherwise general N-D contraction using `Matrix<T>::dot_rhs_perm(rb)`.
- `PowerOperation::backward` only accumulates into `t1`, even though `zero_grad()` and `reset_graph()` handle both.
- `LogOperation::backward` is written element-by-element so the `1e-9` clamp is applied consistently for FP8/FP4/float16.
- `SoftmaxOperation::forward` handles any N-D input, treats the last dim as the class axis, uses max-subtraction for numerical stability, and accumulates `exp` in `scalar::acc_t<T>`.
- `SumAxisOperation` keeps the reduced axis as size 1 so it broadcasts with the original shape.
- `IndexOperation::backward` throws `std::invalid_argument` if the gradient size does not match the mask.
- `EmbeddingOperation::backward` rounds indices the same way as `elemsAt` and throws `std::out_of_range` if an index is out of bounds.
- `SpMMOperation` has a static `gather_rows` and a `run_gather` that splits rows across `num_threads` threads; it uses 1 thread when `nnz * F < 2^16`. Its constructor throws `std::invalid_argument` if `A` is null or `X` is not 2-D with `A->n` rows. `backward` uses the row-gather when `A->symmetric`, else a serial scatter form.

### Overloads (vector operators, vecmath, tensor_overloads)

#### `include/core/Overloads/VectorMath.hpp`

Element-wise math on `std::vector<T>` as named functions in namespace `vecmath`. **No operators and nothing in the global namespace here**, so including this header can never change the meaning of an expression in unrelated code. The operator syntax lives in the opt-in `Overload.hpp`.

- `vecmath::Element` concept: arithmetic or low-precision types.
- `impl::div_check` (throws `std::runtime_error("Division by zero")` unless `MATRIX_IEEE_DIVISION` and the type is floating), `impl::same_size` (throws `std::invalid_argument` on size mismatch).
- Vector-vector: `add`, `sub`, `mul`, `div`.
- Vector-scalar and scalar-vector: `add_s`, `mul_s`, `sub_s`, `s_sub`, `div_s`, `s_div`, `neg`, `mod_s`, `s_mod`.
- In-place: `add_inplace`, `sub_inplace`, `mul_inplace`, `div_inplace`, `add_inplace_s`, `sub_inplace_s`, `mul_inplace_s`, `div_inplace_s`.
- Comparisons (returning `std::vector<uint8_t>`): `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge` (both orders).
- Equality: `equal`, `eq_s`, `ne_s`.
- Math: `exponent`, `pow_s` (with fast paths for 2, 3, 4), `pow_v` (single-element `b` is a documented scalar broadcast), `has_nan_or_inf`.

#### `include/core/Overloads/Overload.hpp`

Opt-in global operators on `std::vector<T>` for numeric element types only. Forwards every operation to `vecmath::`. Also provides the backwards-compatible `detail::` names.

- Vector-vector: `+`, `-`, `*`, `/`.
- Vector-scalar and scalar-vector: `+`, `-`, `*`, `/`, `%`, unary `-`.
- Comparisons returning `std::vector<T>` (kept for API compat; Matrix uses the `uint8_t` versions): `<`, `>`, `<=`, `>=`.
- Equality: `operator==` for vector-vector; `operator==`/`!=` for vector-scalar and scalar-vector returning `std::vector<uint8_t>`.
- Compound assignment: `+=`, `-=`, `*=`, `/=` for vector-scalar and vector-vector.
- Helper functions: `exponent`, `pow` (scalar and vector), `has_nan_or_inf`, `check_nan`.
- Stream operator: numeric element types only; `uint8_t`/`int8_t` print as numbers; `size_t` prints with `(...)`; every element is followed by a comma.

#### `include/core/Overloads/tensor_overloads.hpp`

`make_tensor<T>` overloads (empty, from constant, from vector, from tensor, from initializer lists, from `Matrix<T>` by value/pointer/reference, with and without an `Operation_t<T>`), and the Tensor-level operators that build the corresponding `Operation<T>` and call `forward()`:

- `+`, `*`, `-`, `/` between `Tensor_t<T>`.
- `pow` between `Tensor_t<T>`.
- `==` between `Tensor_t<T>` (compares `val`).
- `!=` and `==` between `Tensor_t<E>` and a scalar `U` (returns `Tensor_t<bool>`).
- `+=`, `-=`, `*=`, `/=` between `Tensor_t<T>` and between `Tensor_t<T>` and a scalar `T` (note: these return a new tensor, they do not mutate `left`).
- Scalar operators: `+`, `-`, `*`, `/` (both orders) and `pow`.
- Unary `-` (implemented as `ten * -1`).
- Comparisons `<`, `>`, `<=`, `>=` between `Tensor_t<E>` and a scalar `E` (returns `Tensor_t<bool>`).
- `spmm<T>(shared_ptr<const CSR<T>>, Tensor_t<T>, size_t num_threads)` — convenience wrapper for `SpMMOperation<T>`.

### Data loading

#### `include/data/Dataset.hpp`

`Dataset<T>` — wraps a feature matrix `X` (`[n_samples, n_features]`) and a label matrix `Y` (`[n_samples, n_outputs]`). The constructor throws `std::runtime_error` if `X.shape[0] != Y.shape[0]`. `size()` returns `X.shape[0]`. `get(idx)` returns a pair of `{1, n_feat}` and `{1, n_out}` `Matrix<T>` rows, throwing `std::out_of_range` if `idx >= size()`.

`DataLoader<T>` — iterates a `Dataset<T>` in mini-batches. Constructor `(ds, batch_size, shuffle = true, seed = 42)` fills `indices` with `0..N-1`. `on_epoch_start()` shuffles `indices` when `shuffle` is true. `begin()` calls `on_epoch_start()` and returns an `Iterator(*this, 0)`; `end()` returns `Iterator(*this, dataset.size())`. `Iterator::operator*` builds `{batch, n_feat}` and `{batch, n_out}` `Matrix<T>` from the rows of the current batch. `num_batches()` returns `(N + batch_size - 1) / batch_size`.

#### `include/data/Tabular.hpp`

`Tabular<T>` — CSV/TSV loader. `TYPE` enum: `CSV`, `TSV`, `XLS`, `XLSX`. `getFileType(path)` detects by extension (lowercased). `split_line(line, delim)` strips whitespace and carriage returns. `load_delimited(path, delim)` reads the file, optionally consumes a header row, parses each field with `std::stod` (non-numeric fields become `NaN`), checks column consistency, and returns a `Matrix<T>` of shape `{row_count, col_count}`. `load_csv` / `load_tsv` call `load_delimited` with `,` / `\t`. `load_xls` and `load_xlsx` are stubs that throw with instructions to link `libxls` / `OpenXLSX`.

Public: `Tabular(bool header = false)`, `load(path)`, `rows()`, `cols()`, `get_headers()`, `train_test_split(data, test_ratio = 0.2f, shuffle = true)` (returns `{train, test}` `Matrix<T>`).

#### `include/data/Serializer.hpp`

`Serializer<T>` — binary save/load of a `std::vector<Tensor_t<T>>`. File format (`TNSF`):
- Magic `"TNSF"` (4 bytes), version `uint32_t` (1), number of tensors `uint64_t`.
- Per tensor: rank `uint32_t`, dims `int64_t × rank`, number of elements `uint64_t`, raw data `T × n_elements`.

`save(model, path)` writes the file and prints the tensor count. `load(path)` validates the magic and version, reads each tensor, builds a `Matrix<T>`, wraps it in a `Tensor<T>`, and returns the vector. Both throw `std::runtime_error` on I/O or format errors.

#### `include/data/DataLoading.hpp`

Umbrella header: `Tabular.hpp`, `Serializer.hpp`, `Dataset.hpp`, `NPYLoader.hpp`, `GGUF.hpp`, `ImageLoader.hpp`, `TextDataset.hpp`.

#### `include/data/NPYLoader.hpp`

`NpyLoader<T>` — loads a single NumPy `.npy` file into a `Matrix<T>`. `NPY_MAGIC` is `"\x93NUMPY"`. `parse_header(f)` reads the 6-byte magic, 1-byte major and minor version, the 2-byte (v1) or 4-byte (v2) header length, and the Python-dict header string. From the header it extracts `descr` (dtype, e.g. `"<f4"`), `fortran_order` (warns if true), and `shape` (parsed from the tuple). `total_elements(shape)` computes the element count. `load_npy(path)` reads `n * sizeof(T)` bytes into a `std::vector<T>` and returns `Matrix<T>(data, h.shape)`. `load_npz(path)` is a stub that throws with integration steps for `miniz` or `zlib`.

#### `include/data/GGUF.hpp`

`GGUF` class — parses GGUF files and loads tensors.

- `TensorInfo { name, dimensions, ggml_type, offset }`.
- `GGUFValueType` enum: `UINT8`=0 … `FLOAT64`=12.
- `GGMLType` enum: `F32`=0, `F16`=1, `Q4_0`=2, `Q4_1`=3, `Q8_0`=8, `Q6_K`=14, `Q4_K`=12, `Q5_K`=13.
- `file` (`std::ifstream`), `tensors` (`std::vector<TensorInfo>`), `data_start_offset`, `array_metadata` (string arrays), `scalar_metadata` (strings).
- `get_array(key)`, `get_string(key)`.
- `read_binary<T>(file)`, `read_string(file)`, `skip_kv_value(file, val_type)`.
- `fp16_to_float(uint16_t)` — bit-exact FP16 decode.
- `dequant_q8_to_fp8<E,M>(file, n_elem)` and `dequant_q4_to_fp4<E,M>(file, n_elem)` — block-wise dequantizers (32 elements per block, float scale for Q8_0, FP16 scale for Q4_0).
- `parse_gguf(filepath)` — validates the `"GGUF"` magic, reads version/tensor_count/kv_count, iterates the KV pairs (honouring `general.alignment`, storing string scalars and string arrays, skipping other types), reads the tensor infos, and computes `data_start_offset` after alignment padding. The error message on a bad magic mentions that files written by this project's own `quant::save_quantized` / `save_module` start with `"QNTF"`, not `"GGUF"`, and must be loaded with `quant::load_module_into<QT>` instead.
- `load_tensor<T>(file, info, data_start)` — reverses GGUF's dimension order, seeks to `data_start + info.offset`, and reads the data. Supports `F32`, `F16`, `Q8_0`, `Q4_0`. For `Q8_0` and `Q4_0` it uses the FP8/FP4 dequantizers when `T` is an FP8/FP4 type, otherwise a plain float path. Throws on unsupported `ggml_type`.

#### `include/data/ImageLoader.hpp`

`ImageLoader<T>` — loads images into `Matrix<T>` / `Tensor_t<T>` using stb_image.

- `NormMode`: `ZERO_ONE` (`/255`), `MEAN_STD` (`(x/255 - mean)/std` per channel; ImageNet defaults `{0.485, 0.456, 0.406}` / `{0.229, 0.224, 0.225}`), `NONE` (raw uint8 cast to `T`).
- `ChannelMode`: `KEEP_ORIGINAL`=0, `GRAY`=1, `GRAY_ALPHA`=2, `RGB`=3, `RGBA`=4.
- `bilinear_resize` — resizes a flat HWC uint8 buffer.
- `normalize(raw, ch)` — applies the norm mode.
- `load_raw(path)` — calls `stbi_load`, optionally resizes, normalizes, and returns `{ data, h, w, c }`.
- Public: `ImageLoader(NormMode, ChannelMode, resize_h, resize_w)`, `set_mean_std(mean, std)`, `load(path)` → `{H,W,C}`, `load_tensor(path)` → `Tensor_t<T>`, `load_batch(paths)` → `{N,H,W,C}` (throws on mismatched shapes unless resize is set), `load_batch_tensor(paths)`, `load_directory(dir, extension = ".jpg")` (requires `target_h`/`target_w`; sorts paths for determinism), `shape_of(path)`.

#### `include/data/TextDataset.hpp`

`TextDataset<T>` — character-level language-model dataset.

- Constructor `(folder, block_size, batch_size, seed = 42)` sets `train_path`/`val_path`/`vocab_path` inside `folder`.
- `scan_txt_files()` lists `.txt` (and extensionless) files in the folder, rejecting likely-binary files by checking the fraction of non-printable bytes in the first 512 bytes.
- `read_file(path)`, `is_allowed_char(c)` (printable ASCII + `\n` + `\t`), `encode(text)` (skips disallowed chars).
- `load()` reads all files, builds the vocabulary from allowed chars, splits 90/10 by character count, writes `train_split.txt`, `val_split.txt`, `vocab.txt`, and encodes the splits into `train_data`/`val_data`.
- `get_batch("train"|"val")` draws `batch_size` random windows of `block_size` from the split and returns `{x, y}` as `Tensor_t<T>` of shape `{batch_size, block_size}` (y is shifted by one).
- Accessors: `vocab_size()`, `train_tokens()`, `val_tokens()`, `decode(id)`, `encode_char(c)`, `decode_sequence(ids)`.

#### `include/data/LazyLoader.hpp`

`LazyDataLoader<T>` — thin wrapper over `TextDataset<T>`. Constructor `(path, block_size, batch_size)` calls `ds.load()`. `getNextBatch(label = "train")` returns `ds.get_batch(label)`.

#### `include/data/GraphLoader.hpp`

LINQS-format loader for Cora/Citeseer.

- `GraphDataset<T>` holds a `Graph<T>`, `class_names`, `skipped_edges`, and `num_classes()`.
- `load_linqs<T>(content_path, cites_path, row_normalize = true)`:
  1. Reads `.content` line by line, parsing the paper id, the numeric features, and the class label.
  2. Builds an alphabetical class-name-to-index map.
  3. Adds nodes to a `Graph<T>`, optionally row-normalizing each feature row by its sum.
  4. Reads `.cites`, adding edges for known ids and counting skipped ones.
  5. Calls `graph.build()`.
- `Split { train, val, test, n_train, n_val, n_test }`.
- `make_split(labels, num_classes, per_class = 20, n_val = 500, n_test = 1000, seed = 42)` — Planetoid-style split: `per_class` training nodes per class, then `n_val` validation and `n_test` test nodes from the rest.

#### `include/data/Quantizer.hpp`

`quant::` — block-wise absmax quantization between a full-precision compute type (`float`/`double`) and a narrow software float storage type (`FP8<E,M>` / `FP4<E,M>`).

Convention (from the header comment): every contiguous run of `group` elements (default 32, same as GGUF) shares one `float` absmax scale. `encode: code = QT(value / scale)`; `decode: value = float(code) * scale`. The scale is picked so the block's largest-magnitude value lands exactly on `QT`'s largest finite magnitude.

- `qmax_value<QT>()` — largest finite magnitude representable by `QT`.
- `QuantTensor<QT> { codes, scales, shape, group }` — in-memory representation; one `QT` value per element (FP4 is **not** bit-packed in memory, only on save).
- `quantize<QT, SrcT>(m, group = 32)` — `Matrix<SrcT>` → `QuantTensor<QT>`.
- `dequantize<QT, DstT = float>(q)` — `QuantTensor<QT>` → `Matrix<DstT>`.
- `save_quantized<QT, SrcT>(params, path, group = 32)` — writes a `QNTF` file: magic `"QNTF"`, version, format tag (0 = dense 1B/elem, 1 = packed FP4 nibbles), group size, tensor count; per tensor: rank, dims, `n_elem`, `n_blocks`, `scales` (float × n_blocks), codes (packed per format tag). Prints original vs packed sizes.
- `load_quantized<QT, DstT = float>(path)` — validates magic, version, and format tag against `QT`, then reads and dequantizes each tensor. Throws on truncation or format mismatch.
- `save_module<QT>(Module<T>& m, path, group = 32)` — calls `save_quantized<QT, T>(m.parameters(), path, group)`.
- `load_module_into<QT>(Module<T>& m, path)` — loads and dequantizes, then checks the tensor count against `m.parameters()` and copies each dequantized matrix into `params[i]->val`. Throws on count mismatch.

### NN modules

#### `include/nn/Modules/Module.hpp`

`Module<T>` base class. A module owns a list of parameters (`params`) and a list of submodules (`submodules`), plus a `Serializer<T> Ser`.

- `register_parameter(Tensor_t<T> p)` — pushes into `params`.
- `register_module(Module<T>* m)` — pushes into `submodules`.
- `parameters()` — returns `params` followed by the recursive `parameters()` of every submodule.
- `zero_grad()` — calls `zero_grad()` on every parameter and every submodule.
- `save(path)` — `Ser.save(this->parameters(), path)`.
- `load(path)` — `Ser.load(path)`, checks the tensor count against `this->parameters()`, and copies each loaded `Matrix<T>` into the corresponding `params[i]->val`. Throws on count mismatch.
- `to_string()`, `reset_graph()`, `forward()` — declared but not defined in this header.

`Parameter<T>` — a named tensor wrapper. Not used by the uploaded modules.

#### `include/nn/Modules/Linear.hpp`

`Linear<T>` — dense layer. Members: `weight` (`{out_features, in_features}`), `bias` (`{out_features}`), `sbias`.

- Constructor: Glorot uniform init `[-limit, limit]` with `limit = sqrt(6/(in+out))`; registers `weight`, always registers `bias` only when `sbias` is true.
- `forward(x)` — `x->matmul(weight->transpose())` plus `bias` when `sbias`.
- `load_pretrained(Linear<T>& src)` — copies `weight` and (if both have a bias) `bias`.
- Friend declarations for `GGUFLoader<T>`, `GPTGGUFLoader`, `LlamaGGUFLoader`.

#### `include/nn/Modules/LoRALinear.hpp`

`LoRALinear<T>` — low-rank adapter. Members: `weight` (`{out, in}`, frozen, initialised to zeros), `bias` (`{out}`, frozen), `A` (`{in, rank}`, randn), `B` (`{rank, out}`, zeros), `scale = alpha / rank`.

- Constructor registers only `A` and `B` as parameters (`weight`/`bias` are frozen).
- `forward(x)`: `base = x @ weightᵀ (+ bias)`; `lora = x @ A @ B`; returns `base + scale * lora`.
- `load_pretrained(Linear<T>& src)` — copies `src.weight` (and `src.bias` if both have one) into the frozen path; `A`/`B` are deliberately left alone.
- Friend declarations for the GGUF loaders.

#### `include/nn/Modules/Embedding.hpp`

`Embedding<T>` — lookup table. Members: `embeddings` (`{vocab_size, input_dim}`), `vocab_size`, `input_dim`.

- Constructor: Glorot uniform init; registers `embeddings`.
- `forward(indices)` — `embeddings->embed(indices)`.
- Friend declarations for the GGUF loaders.

#### `include/nn/Modules/PositionalEncoding.hpp`

`PositionalEncoding<T>` — learned position embeddings. Members: `input_dim`, `max_sequence_length`, `weight` (`{max_sequence_length, input_dim}`).

- Constructor: Glorot uniform init; registers `weight`.
- Move constructor re-registers `weight` at its new address; copy constructor is deleted.
- `forward(indices)` — `weight->embed(indices)`.
- Friend declarations for the GGUF loaders.

#### `include/nn/Modules/RotationalPositionalEncoding.hpp`

`RotationalPositionalEncoding<T>` — RoPE. No parameters.

- Constructor stores `input_dim` and `max_sequence_length`.
- `forward(index)` computes `even_i = arrange(0, input_dim, 2)`, `denominator = 100000^(even_i/input_dim)`, `position = index->reshape({last_dim, 1})`, `even_PE = sin(position/denominator)`, `odd_PE = cos(position/denominator)`, `stacked = concat({even_PE, odd_PE}, 1)`, and returns `stacked.reshape({index->val.get_size(), input_dim})`.
- Not a `Module<T>` subclass (no parameters).

#### `include/nn/Modules/LayerNorm.hpp`

`LayerNorm<T>` — layer normalisation over the last axis. Members: `eps`, `tol`, `normalized_shape`, `gamma` (ones), `beta` (zeros).

- Two constructors (`shape_t` and `initializer_list<size_t>`); both register `gamma` and `beta`.
- `forward(x)`:
  - `axis = x->ndims - 1`
  - `mu = x->mean(axis)` → `{B,T,1}`
  - `var = x->var(axis)` → `{B,T,1}`
  - `stdev = (var + eps)->sqrt()`
  - `xhat = (x - mu) / stdev`
  - returns `gamma * xhat + beta`
- Friend declarations for the GGUF loaders.

#### `include/nn/Modules/RMSNorm.hpp`

`RMSNorm<T>` — root-mean-square normalisation. Members: `eps`, `normalized_shape`, `gamma` (ones).

- Two constructors; both register `gamma`.
- Move constructor re-registers `gamma`; copy constructor is deleted.
- `forward(x)`:
  - `axis = x->ndims - 1`
  - `x_sq = x * x`
  - `ms = x_sq->mean(axis)` → `{B,T,1}`
  - `rms = (ms + eps)->sqrt()`
  - `x_hat = x / rms`
  - returns `gamma * x_hat`
- Friend declarations for the GGUF loaders.

#### `include/nn/Modules/Relu.hpp` and `Softmax.hpp`

Simple activation modules: `Relu<T>::forward(x)` returns `x->relu()`; `Softmax<T>::forward(x)` returns `x->softmax()`.

#### `include/nn/Modules/GCNLayer.hpp`

`GCNLayer<T>` plus three free helpers:

- `features_tensor(g)` — `Matrix<T>(g.features(), {N, F})`, `requires_grad = false`.
- `dense_adjacency_tensor(g)` — `Matrix<T>(g.dense_adjacency(), {N, N})`, `requires_grad = false`.
- `onehot_targets(g, num_classes, train_mask)` — one-hot targets, all-zero rows for unlabeled or masked-out nodes, `requires_grad = false`.

`GCNLayer<T>` members: `weight` (`{in, out}`, Glorot uniform), `bias` (`{out}`, zeros), `use_bias`.

- Dense `forward(adj, x)`: `h = adj @ (x @ weight)` (multiply `X*W` first, cheaper than `A*X`); returns `h + bias` when `use_bias`.
- Sparse `forward(shared_ptr<const CSR<T>> adj, x, num_threads = 1)`: `h = spmm(adj, x @ weight, num_threads)`; returns `h + bias` when `use_bias`.

#### `include/nn/Modules/FeedForward.hpp` (generic)

`FeedForward<T, LinearT>` — two-layer feed-forward with ReLU.

- Constructor `(in_features, hidden, out_features)` builds `up(hidden, in_features, true)` and `down(out_features, hidden, true)`, and registers both.
- Move constructor re-registers `up`/`down` at their new addresses; copy constructor is deleted.
- `get_up()`, `get_down()`, `load_pretrained(FeedForward<T, Linear>& src)`.
- `forward(x)` — `down.forward(up.forward(x)->relu())`.
- Friend declarations for the GGUF loaders.

Note: this header is included by the GPT `Block.hpp`, not the Llama one.

#### `include/nn/Modules/NeuralLayer.hpp`

`NeuralLayer<T>` — sequential container. Inherits `Module<T>` protected.

- Constructor takes an `initializer_list<Module<T>*>` and registers each as a submodule.
- `forward(x)` — runs `x` through every submodule in order and returns the result.

#### `include/nn/Modules/Optimizer.hpp`

`Optimizer<T>` — SGD / Adam / AdamW / Adafactor. Protected state: `optimizer`, `lr`, `parameters`, `requires_grad`, `t`, `initialized`, `m`, `v`, `b1`, `b2`, `eps`, `lambda`, and the Adafactor accumulators (`af_row`, `af_col`, `af_v`, `af_decay`, `af_eps1`, `af_eps2`, `af_clip_threshold`).

- Three constructors: `(params, lr, optim, requires_grad)`, `(params, lr, optim, beta1, beta2, eps, requires_grad)`, `(params, lr, optim, beta1, beta2, eps, lambda, requires_grad)`.
- `sgd()` — `p -= lr * grad` (cblas `saxpy`/`daxpy` for float/double, `Matrix` arithmetic otherwise).
- `Adam()` — standard Adam with bias correction; uses cblas `sscal`/`saxpy`/`dscal`/`daxpy` for float/double.
- `AdamW()` — Adam with decoupled weight decay `p *= (1 - lr*lambda)` before the Adam step.
- `adafactor()` — Adafactor with factored second moment for 2-D parameters (`R_t`, `C_t`, rank-1 reconstruction), per-element second moment for other ranks, relative step size from the parameter RMS, and RMS clipping.
- `set_adafactor_params(decay, eps1, eps2, clip_threshold)`.
- Public: `zero_grad()`, `clip_grad_norm(max_norm = 1.0)` (global L2 clip via cblas `sdot`/`ddot`/`sscal`/`dscal`), `step()` (calls `clip_grad_norm(1.0)` unless the optimizer is Adafactor, then dispatches to the selected algorithm).

### Transformer stacks (GPT, Llama)

#### `include/nn/Transformer/GPT/`

- `Head.hpp` — `Head<T, LinearT>`: Q/K/V projections, `set_mask(seq)` (lower-triangular mask with `-inf` on the masked positions, rebuilt only when `seq` changes), `scaled_dot_product_attention(q, k, v, apply_mask)` (scales by `1/sqrt(d_k)`, adds the mask, softmax, multiplies by `v`), `load_pretrained(Head<T, Linear>&)`, `forward(x, apply_mask)`.
- `MultiHeadAttention.hpp` — `MultiHeadAttention<T, LinearT>`: `n_heads` `Head`s plus a `proJ` projection; `forward` concatenates the head outputs on the last axis and projects; `load_pretrained`.
- `FeedForward.hpp` — `FeedForward<T, LinearT>`: two `LinearT`s (`up`, `down`) with GeLU between them (this is the GPT-specific FFN; it differs from the generic `FeedForward.hpp` in `Modules/`).
- `Block.hpp` — `Block<T, LinearT>`: `ln1` → `mha` → residual, `ln2` → `ffwd` → residual. `checked_head_size(input_dim, n_heads)` throws if `input_dim % n_heads != 0`. `load_pretrained`.
- `GPT.hpp` — `GPT<T, LinearT>`: `position_embedding_table`, `embedding_table`, `decoder_blocks`, `ln_f`, `lm_head`. `forward(index, targets, apply_mask)` returns the logits when `targets == nullptr` and the cross-entropy loss otherwise. `topk_softmax`, `generate`, `train_step`, `train`, `eval_step`. `load_backbone_from(pretrained)` copies positional/embedding/ln_f params and calls `load_pretrained` on each block; `load_head_from(pretrained)` copies the `lm_head` weight. `lora_parameters()` returns `lm_head.parameters()`; `backbone_parameters()` returns the rest. `GPTDense<T>` = `GPT<T, Linear>`, `GPTLoRA<T>` = `GPT<T, LoRALinear>`.

#### `include/nn/Transformer/Llama/`

- `Head.hpp` — `Head<T, LinearT>`: Q/K/V projections, a `RotationalPositionalEncoding<T> rope`, `set_mask`, `scaled_dot_product_attention`, `load_pretrained`, `forward(x, apply_mask)`. Applies RoPE to `q` and `k` before the attention.
- `MultiHeadAttention.hpp` — same structure as the GPT one, with `Head<T, LinearT>` from the Llama directory.
- `FeedForward.hpp` — three `LinearT`s (`gate_proj`, `up_proj`, `down_proj`). `make_linear` forces `bias = false` for `LinearT`s that accept a bias flag (via `if constexpr (std::is_constructible_v<LinearT<T>, size_t, size_t, bool>)`), and passes the args through for `LinearT`s that do not. `forward(x)` computes `silu(gate_proj(x)) * up_proj(x)` then `down_proj`.
- `Block.hpp` — `rms1` → `mha` → residual, `rms2` → `ffwd` → residual. `checked_head_size`. `load_pretrained`.
- `Llama.hpp` — `Llama<T, LinearT>`: `embedding_table`, `decoder_blocks`, `rms`, `lm_head`. The constructor forwards its extra args to every `LinearT` built inside. `forward(index, targets, apply_mask)` returns the logits when `targets == nullptr` and the cross-entropy loss otherwise. `topk_softmax`, `generate` (with EOS check for `<|im_end|>` = 2), `train_step`, `train`, `eval_step`. `load_backbone_from(pretrained)` / `load_head_from(pretrained)`. `lora_parameters()` / `backbone_parameters()`. `LlamaDense<T>` = `Llama<T, Linear>`, `LlamaLoRA<T>` = `Llama<T, LoRALinear>`. The header comment explicitly notes that the `Args...` pack is shared across all `LinearT` instances, so passing `false` disables bias everywhere, including `lm_head` and attention Q/K/V/proj.

### Model loaders (GGUF, GPT, Llama)

#### `include/nn/ModelLoader/GGUFLoader.hpp`

`GGUFLoader<T>` base class. Documents the weight-shape contract:
- GGUF stores dimensions with the fastest-varying (last PyTorch) dim first. PyTorch weight `[out, in]` → GGUF `info.dimensions = [in, out]` → `load_tensor()` reverses → `Matrix{out, in}`.
- New `Linear` stores weight as `{out, in}` and computes `x @ weight.T`. Therefore `load_raw()` already delivers the correct `{out, in}` shape, and **no `.transpose()` is ever needed for a Linear weight**.

Members: `GGUF gguf`, `map<string, size_t> tensor_index`.

Protected methods:
- `build_index()` — builds `tensor_index` from `gguf.tensors`.
- `load_raw(name, required = true)` — looks up the tensor, prints its shape and `ggml_type`, and calls `gguf.load_tensor<T>`. Returns an empty `Matrix<T>` (and prints `[GGUFLoader] MISSING: name`) when not found and `required` is false.
- `copy_into(dst, src, debug_name)` — checks the element count, prints `[GGUFLoader] Size mismatch 'name': GGUF=N model=M → skipped` on mismatch, else copies `src` into `dst->val` and sets `dst->shape`.
- `open(path)` — prints `[GGUFLoader] Parsing: path`, calls `gguf.parse_gguf(path)`, `build_index()`.

Public methods:
- `get_metadata_array(key)`, `get_metadata_string(key)`.
- `inspect(path)` — prints the full tensor inventory.
- `report_params(model)` — prints the total parameter count.
- `virtual ~GGUFLoader() = default`.

#### `include/nn/ModelLoader/GPTLoader.hpp`

`GPTGGUFLoader<T, LinearT>` (GPT-2 family). `static_assert` restricts `LinearT` to `Linear` with a message explaining how to fine-tune with LoRA (load dense, construct a separate `GPT<T, LoRALinear>` student, call `load_backbone_from`/`load_head_from`).

- `GPT2HyperParams { vocab_size = 50257, d_model = 768, block_size = 1024, n_layer = 12, n_head = 12 }`.
- `load_block(model, hp, n)` — loads `blk.N.attn_norm.{weight,bias}`, `blk.N.ffn_norm.{weight,bias}`, `blk.N.attn_qkv.{weight,bias}` (sliced per head into `Head::Q/K/V`), `blk.N.attn_output.{weight,bias}`, `blk.N.ffn_up.{weight,bias}`, `blk.N.ffn_down.{weight,bias}`.
- `load_weights(model, hp)` — `token_embd.weight`, `position_embd.weight`, `output_norm.{weight,bias}`, `output.weight` (with tied-weight fallback to `token_embd.weight`), then every block.
- `load_model(path, hp)` — `open(path)`, builds the `GPT<T, LinearT>` skeleton, `load_weights`, `report_params`, returns the model.
- `load_tokenizer()` — reads `tokenizer.ggml.tokens` and `tokenizer.ggml.merges` from the GGUF metadata and builds a `GPT2Tokenizer`. Warns and returns an empty tokenizer if the metadata is missing.
- The header documents the full GGUF → GPT-2 weight mapping in a comment.

#### `include/nn/ModelLoader/LlamaLoader.hpp`

`LlamaGGUFLoader<T, LinearT>` (Llama/SmolLM/Mistral family), with GQA support.

- `LlamaHyperParams { vocab_size, input_dim, block_size, n_heads, n_kv_heads, n_layer, ffn_hidden }`.
- `load_attention_heads(mha, hp, pfx)` — loads `attn_q.weight` (`{nH*hs, D}`), `attn_k.weight` (`{nKV*hs, D}`), `attn_v.weight` (`{nKV*hs, D}`), then slices per head: `Q_full.slice_row(h*hs, (h+1)*hs)` and `K/V_full.slice_row(kv_h*hs, (kv_h+1)*hs)` with `kv_h = h / kv_ratio`. This handles GQA (many Q heads share one KV head).
- `load_block(model, hp, blk_idx)` — loads `blk.N.attn_norm.weight`, `blk.N.ffn_norm.weight`, `blk.N.attn_output.weight`, the Q/K/V weights, and `blk.N.ffn_{gate,up,down}.weight`.
- `load_weights(model, hp)` — `token_embd.weight`, `output_norm.weight`, `output.weight` (with tied-weight fallback), then every block.
- `load_model(path, hp)` — `open(path)`, builds the `Llama<T, LinearT>` skeleton, `load_weights`, `report_params`, returns the model.
- `load_tokenizer()` — reads `tokenizer.ggml.tokens` and `tokenizer.ggml.merges`, converts the merges to pairs, and builds a `LlamaTokenizer`. Warns and returns an empty tokenizer if the metadata is missing.
- The header documents the full GGUF → Llama weight mapping in a comment.

### Tokenizers

#### `include/nn/Tokenizer/Tokenizer.hpp`

Abstract base class for byte-level BPE tokenizers. Every GPT-2-style tokenizer (GPT-2, LLaMA/SmolLM/Mistral, …) shares the vocabulary storage, merge-rank table, byte↔unicode mapping, BPE merge algorithm, and GPT-2 pre-tokenization regex. Only `encode()`/`decode()` differ.

Members:
- `encoder` (`unordered_map<string, int>`), `decoder` (`unordered_map<int, string>`).
- `bpe_ranks` (`map<pair<string,string>, int>`).
- `byte_encoder` (`unordered_map<uint8_t, char32_t>`), `byte_decoder` (`unordered_map<char32_t, uint8_t>`).

Methods:
- `build_byte_encoder()` — the GPT-2 / HF `bytes_to_unicode` mapping. Visible ASCII (33–126) and two Latin-1 ranges (161–172, 174–255) are kept as-is; everything else is remapped to codepoints starting at 256 (space → `Ġ` U+0120, newline → `Ċ` U+010A).
- `codepoint_to_utf8(char32_t)` — static; encodes a codepoint to a UTF-8 string.
- `for_each_codepoint(const string&, Fn)` — static; walks a UTF-8 string and calls `cb(codepoint)` for each codepoint.
- `split_utf8_chars(const string&)` — static; splits a token's raw bytes into UTF-8 characters.
- `bpe(const string&)` — repeatedly merges the lowest-rank adjacent pair until none remain.
- `pre_tokenize(const string&)` — the GPT-2 regex (`'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+`).
- `set_vocab(const vector<string>&)` — builds `encoder`/`decoder` from a token list.
- `set_merges(const vector<pair<string,string>>&)` — builds `bpe_ranks` from pairs.
- `set_merges(const vector<string>&)` — builds `bpe_ranks` from `"left right"` strings.
- `vocab_size()`, `encode_special(const string&)`.

Pure virtual: `encode(const string&)`, `decode(const vector<int>&)`.

#### `include/nn/Tokenizer/GPT2Tokenizer.hpp`

`GPT2Tokenizer : public Tokenizer`.

- `load_from_arrays(vocab, merges)` — calls `build_byte_encoder()`, `set_vocab(vocab)`, `set_merges(merges)`.
- `load(vocab_path, merges_path)` — calls `build_byte_encoder()`, reads `vocab.json` via `nlohmann::json` (filling `encoder`/`decoder`), then reads `merges.txt` (skipping the `#version` header) into `bpe_ranks`.
- `encode(text)` — pre-tokenizes, maps each word's bytes through `byte_encoder` and re-encodes as UTF-8, applies `bpe`, looks each piece up in `encoder`. Prints `[GPT2Tokenizer] Unknown token: "..."` on a miss.
- `decode(ids)` — for each id, looks up the token, walks its codepoints, and maps each through `byte_decoder` to a byte (`Ġ` → space, `Ċ` → newline). Skips unmappable codepoints.

#### `include/nn/Tokenizer/LlamaTokenizer.hpp`

`LlamaTokenizer : public Tokenizer`.

- `load_from_gguf(tokens, merges)` — calls `build_byte_encoder()`, `set_vocab(tokens)`, `set_merges(merges)`, then collects special tokens (`<...>`) into `special_tokens_` and a length-descending `special_tokens_list_`.
- `encode(text)` — scans for special tokens first (longest match wins), then encodes the plain text between them with `encode_plain`. Prints `[LlamaTokenizer] Unknown token: "..."` on a miss.
- `decode(ids)` — maps `▁` (U+2581) and `Ġ` (U+0120) to space, `Ċ` (U+010A) to newline, otherwise through `byte_decoder`.
- `encode_plain(text)` — standard byte-level BPE encoding for non-special text.

### Federated learning

#### `include/net/Network/Server.hpp`

`Server<T, LinearT>` — base class for federated learning servers. Owns the `GPT<T, LinearT>` global model, a `Trainer<GPT<T, LinearT>, T>` (server-side lr = 0, so the optimizer is inert), a TCP listener socket, a `ThreadPool`, the FedAvg round state, and the FedDistill round state.

Protected members:
- `GPT2HyperParams hp`, `GPTGGUFLoader<T, Linear> loader` (always dense), `GPT2Tokenizer tokenizer`.
- `int server_fd = -1`, `std::unique_ptr<ThreadPool> pool` (sized `min_clients + 4`).
- FedAvg: `std::mutex locker`, `std::condition_variable round_cv`, `std::vector<T> round_accum`, `size_t clients_done`, `std::vector<int> client_sockets`, `size_t min_clients = 2`, `size_t chunk_elems`.
- Round barrier: `std::condition_variable round_done_cv`, `std::atomic<size_t> round_generation{0}`.
- FedDistill: `std::vector<std::vector<T>> logit_updates`, `uint64_t proxy_n_examples`, `uint64_t proxy_vocab_size`.
- Parameter layout: `std::vector<size_t> param_sizes`, `size_t total_param_elems`.

Key methods:
- `build_param_layout()` — caches each parameter tensor's element count and the total.
- `broadcast()` — streams the global model to every connected client in `chunk_elems`-sized pieces via `send_chunked`. Uses a fresh `FlatParamCursor` per fd. Returns elapsed milliseconds.
- `close_client(sock)` — `net_forget_fd(sock)` then `::close(sock)`.
- `struct FlatParamCursor` — incremental flat-offset ↔ tensor mapping with `seek`, `read_into`. Construct a fresh one per stream.
- `handleClient(sock)` — persistent per-client FedAvg loop: `recv_chunked` accumulating directly into `round_accum` (guarded by `locker`), then block on `round_done_cv` until `round_generation > my_round` (or `server_fd < 0`), then loop back.
- `handleClientLogits(sock)` — persistent per-client FedDistill loop: read `[n_examples][vocab_size][total_bytes][T × N]`, push into `logit_updates`, block on `round_done_cv`, loop back.
- `broadcastLogits(consensus)` — sends `[n_examples][vocab_size][total_bytes][T × N]` to every connected client.
- `closeClientSocks()` — closes and clears all client sockets, resets `clients_done`, zeroes `round_accum`.
- `clearLogitUpdates()` — clears the FedDistill round state.

Public members:
- `GPT<T, LinearT> model`, `Trainer<GPT<T, LinearT>, T> trainer` (declared in this order because `trainer` holds a reference to `model`).
- `Server(model_path, hyper, min_clients_per_round, chunk_mb, model_args...)` — loads the dense GGUF checkpoint and copies it into `model` via `load_backbone_from` / `load_head_from`, sizes the `ThreadPool`, computes `chunk_elems`, loads the tokenizer, builds the parameter layout, and allocates `round_accum`.
- `stop()` — `closeClientSocks()`, close `server_fd`, `round_done_cv.notify_all()`.

#### `include/net/Network/Client.hpp`

`Client<T>` — lightweight networking base used by `FederatedClient` (client.cpp, not uploaded). Owns the socket and implements three wire protocols.

- `connect_to_server(ip, port)`, `disconnect()`, `fd()`.
- **LoRA handshake**: `sendLoraConfig(lora_enabled, rank, alpha)` (`[uint8 enabled][uint32 rank][float alpha]`), `receiveLoraAck(accepted, reason)` (`[uint8 ok][uint32 reason_len][char × reason_len]`).
- **Flat-tensor protocol**: `send(flat)` (`[uint64 total_bytes][T × N]`), `receive(flat)`.
- **Batch protocol** (server → client): `receiveNextBatch()` (`[uint64 batch_size][uint64 block_size][T × batch*block inputs][T × batch*block targets]`).
- **Chunked protocol**: `sendChunked(total_elems, chunk_elems, get_chunk)`, `receiveChunked(expected_total, on_chunk)` — forwards to `io_utils.hpp`'s generic `send_chunked`/`recv_chunked`.
- **Logits protocol** (federated distillation): `sendLogits(flat_logits, n_examples, vocab_size)` (`[uint64 n_examples][uint64 vocab_size][uint64 total_bytes][T × N]`), `receiveLogits(flat_logits, n_examples, vocab_size)`.

#### `include/net/Network/ThreadPool.hpp`

Simple `ThreadPool`. Constructor takes a worker count. `enqueue(std::function<void()>)` pushes a task and notifies one worker. Destructor sets `stop`, notifies all, and joins.

#### `include/net/Network/io_utils.hpp`

- **Bandwidth counters**: `NetIoCounters { sent, received }`, `g_net_bytes_sent`, `g_net_bytes_received`, `g_net_per_fd` (a map fd → counters). Accessors: `net_bytes_sent()`, `net_bytes_received()`, `net_reset_counters()`, `net_per_fd(fd)`, `net_per_fd_snapshot()`, `net_forget_fd(fd)`.
- **`read_exact(fd, buf, len)`** / **`write_exact(fd, buf, len)`** — retry on partial reads/writes; return false on connection drop; update the byte counters on success.
- **`send_chunked<T>(fd, total_elems, chunk_elems, get_chunk)`** — wire format `[uint64 total_elems]` then repeated `[uint64 chunk_len][T × chunk_len]`. Sender and receiver choose their own `chunk_elems` independently.
- **`recv_chunked<T>(fd, expected_total, on_chunk)`** — verifies the advertised total against `expected_total` (throws on mismatch), then reads each chunk and calls `on_chunk(offset, data, len)`.

### Profiler wire protocol

#### `include/net/Protocol/MessageProtocol.hpp`

Binary profiler wire protocol (`TFPR` magic). Design goals (from the header): zero allocation on the read path (direct cast to struct), zero parsing, transport-agnostic, integrity (CRC32/xxHash32), extensible (VERSION + FLAGS + RESERVED).

- Constants: `PROTO_MAGIC`, `PROTO_VERSION`, `MsgType`, `ChecksumType`, `ProtoFlags` (+ `operator|` and `has_flag`), `ISACode`, `CPUGenCode`, `RAMTypeCode`, `StorageCode`, `QuantCode`.
- `ProfilerHeader` (64 bytes, `__attribute__((packed))`, `static_assert(sizeof == 64)`): magic, version, flags, msg_type, num_storage, client_id, timestamp_us, body_len, checksum_type, `_pad[3]`, checksum, `reserved[28]`.
- `ProfilerBody` (`__attribute__((packed))`): CPU (16 B), cache (12 B), RAM (16 B), benchmarks (48 B), memory profiling (32 B), recommended config (32 B). All float values are encoded as `uint16_t` fixed-point (`f2u16(v, scale)`).
- `StorageEntry` (32 bytes, `static_assert(sizeof == 32)`): `dev_name[12]`, `storage_type`, `_pad[3]`, `size_bytes`, `seq_read_mbs`, `seq_write_mbs`, `_pad2`.
- `MAX_STORAGE_DEVICES = 16`, `MAX_MSG_SIZE`.
- `WireBuffer` — flat byte buffer with `header()`, `body()`, `storage_entries()`, `num_storage()`.
- `crc32_impl::compute`, `xxhash32_impl::compute`, `compute_checksum(type, data, len)`.
- `serialize(fp, bench, cfg, mem, client_id, chk_type, flags)` — fills a `WireBuffer`. Writes the body and storage entries first, then the header (so the checksum can be computed over body + storage entries). Timestamp from `clock_gettime(CLOCK_REALTIME)`.
- `ParseResult`, `deserialize(data, len)` — validates magic, version, msg_type, body length, and checksum. Returns `body`, `storage`, `num_storage`, `client_id`, `timestamp_us`.
- `print_wire(wb)` — debug printer.

**Note**: `MessageProtocol.hpp` includes `Profiler/HardwareFingerprint.hpp`, `Profiler/BenchmarkRunner.hpp`, `Profiler/HyperparamAdvisor.hpp`, and `Profiler/MemoryProfiler.hpp`, and uses types `HardwareFingerprint`, `BenchmarkResult`, `HyperparamConfig`, `MemoryReport`, `CPUGen`, `ISA`, `CPUVendor`, `RAMType`, `StorageType`, `QuantPolicy`. `BenchmarkRunner.hpp` was **not uploaded**, and the other names are not in the uploaded profiler headers (which use `tensorf::profile::HardwareFingerprint` and `tensorf::prof::DeviceFingerprint`). This file as uploaded would not compile against the uploaded headers; it is documented here as it appears.

#### `include/net/Protocol/MessageTransport.hpp`

Three transports, one interface (`bool send(const WireBuffer&)`, `bool recv(WireBuffer&)` on the server side).

- `DEFAULT_TCP_PORT = 9731`, `DEFAULT_UDP_PORT = 9732`, `DEFAULT_UNIX_PATH = "/tmp/tensorf_profiler.sock"`.
- `send_all(fd, buf, len)`, `recv_all(fd, buf, len, timeout_ms = 5000)` (uses `poll` for the timeout).
- **`TCPTransport`** — `connect(host, port)` (sets `TCP_NODELAY`), `send(wb)` (4-byte length prefix + payload), `listen_once(port, out, on_recv)`, `close()`.
- **`UnixTransport`** — `connect(path)`, `send(wb)`, `listen_once(path, out, on_recv)` (unlinks any stale socket first), `close()`.
- **`UDPTransport`** — `setup_client(host, port)`, `send(wb)` (single `sendto`, no length prefix), `listen_once(port, out, on_recv)`, `close()`.
- **`TransportConfig`** — `server_host`, `tcp_port`, `udp_port`, `unix_path`, `try_unix`, `try_tcp`, `try_udp`, `internet_mode`.
- **`send_profile(wb, cfg)`** — tries Unix → TCP → UDP in order and prints which one succeeded.

### Profiler (old and new APIs)

#### Old API (namespace `tensorf::profile`)

- `HardwareFingerprint.hpp` — `collect()`, `collect_linux`, `DeviceFingerprint`, `MemFs`, parsers.
- `DeviceBenchmark.hpp` — `run_device_benchmark`, `BenchConfig`, `CapabilityProfile`, kernel cases.
- `MemoryProfiler.hpp` — `MemoryLedger`, `TrackedVector`, `plan_budget`, `MemoryGuard`, `RssSampler`.
- `AlgorithmProfiler.hpp` — `AlgorithmProfiler`, `CostModel`, `algo::make`, `DeviceView`, algorithm models.
- `HyperparamAdvisor.hpp` — `HyperparamAdvisor`, `OptIn`, `Constraints`, `SearchOptions`, `Advice`.

#### New API (namespace `tensorf::prof`)

- `ProfilerCommon.hpp` — clocks, stats, parsing helpers, `Json` writer.
- `DeviceFingerprint.hpp` — `detect_device`, `ThermalProbe`, parsers. Replaces `HardwareFingerprint.hpp` (namespace `tensorf::profile`).
- `CapabilityKernels.hpp` — micro-kernels: GEMM, stream, chase, gather, conv, RNN, nbody, overheads, scaling, sustained.
- `CapabilitySuite.hpp` — `CapabilitySuite`, `CapabilityVector`, `SuiteConfig`, thermal guard.
- `WorkloadModel.hpp` — layer cost library, `Algorithm`, `predict`, `Calibration`.
- `WorkloadProbe.hpp` — `measure_cases`, `reference_cases`, `run_probe`, `ProbeReport`.

The two APIs are documented in detail in the previous sections of this README.

### CBLAS shim

#### `include/tools/cblas.h`

A minimal CBLAS replacement for platforms without OpenBLAS (Android NDK builds, quick ports). Implements exactly what TensorF calls: `saxpy`/`daxpy`, `sscal`/`dscal`, `sdot`/`ddot`, `sgemm`/`dgemm`, row-major only.

- Include guard `TENSORF_CBLAS_SHIM_H` (checked by `WorkloadProbe.hpp`).
- Enums `CBLAS_ORDER` and `CBLAS_TRANSPOSE`.
- Internal namespace `tensorf_cblas_shim` with templates `axpy`, `scal`, `dot`, `gemm`.
- `gemm` aborts with `cblas shim: only row-major supported` if `order != CblasRowMajor`. Uses i-k-j order with a contiguous inner loop over `j`. Skips work when `a == 0`. Handles `beta == 0` (zeroes C), `beta == 1` (no scaling), and other `beta` values (scales C).
- Public CBLAS-compatible entry points: `cblas_saxpy`, `cblas_daxpy`, `cblas_sscal`, `cblas_dscal`, `cblas_sdot`, `cblas_ddot`, `cblas_sgemm`, `cblas_dgemm`.
- Not implemented: column-major order, `CblasConjTrans` conjugation semantics (treated as transpose), integer or half-precision types.

The header comment states: "This is a portability shim, not a fast BLAS. For speed, build with OpenBLAS (`-DTENSORF_USE_OPENBLAS=ON`) or replace `gemm` below with a tiled/NEON kernel."

### Scripts

#### `scripts/download_cora.sh`

Downloads `cora.content` and `cora.cites` from `https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora` into `Datasets/cora/`. Skips files that already exist and are non-empty. Prints the line counts at the end (expects 2708 content lines and 5429 cites lines).

#### `scripts/download_llms.sh`

Downloads GPT-2 GGUF files from `https://huggingface.co/ddh0` into `SLM/`:
- `gpt2-small-f32.gguf` (654 MB, full F32 precision).
- `gpt2-small-q8.gguf` (178 MB, Q8_0).
- `gpt2-small-q4.gguf` (135 MB, Q4_K_S).
- Medium/Large/XL variants are present but commented out. The Medium commands have a subtle bug: the `-O` flag is commented out, so the file is saved under its remote name (the line continuation `\` is followed by a comment, which the shell ignores).

#### `scripts/run_android.sh`

Builds a fixed set of targets with the Android NDK and runs them on a connected device/emulator.

- Requires `ANDROID_NDK_HOME` (defaults to `$HOME/Android/Sdk/ndk/29.0.14206865` in the script; set it yourself before running).
- Adds `$HOME/Android/Sdk/platform-tools` to `PATH`.
- Variables: `ABI` (default `arm64-v8a`), `API` (default `24`), `CORA` (default `1`), `EPOCHS` (default `200`).
- Asserts `CMakeLists.txt` exists at the repo root (spelled exactly `CMakeLists.txt`).
- Runs `cmake -S "$ROOT" -B "$BUILD" -G "Unix Makefiles"` with the Android toolchain, then `cmake --build` for the fixed target list.
- Checks `adb get-state`, prints the device ABI, pushes the binaries to `/data/local/tmp/tensorf`, `chmod +x` them, and runs a subset: `spmmtest`, `gcntest`, `algorithm_profiler_test`, `profiler_workload_test`, `profiler_probe_test`, plus `coratrain` on Cora if `CORA=1`. The other tests are present but commented out in the run block (they are still built and pushed).

### Tests

See the [Testing](#testing) table.

### Reports

`REPORT.TXT` and `REPORT2.TXT` are build and run logs from the author's machine. They show:

- Successful native builds with `g++ -std=c++20 -O2 ... -I. -I./include ... -lopenblas` of the examples and tests, and of the gateway (`g++ ... -Igateway/include gateway/src/main.cpp -o bin/tensorf-gateway ...`).
- `make` runs `check-deps` (g++ 13.3.0, nlohmann/json 3.11.3, OpenBLAS 0.3.26, pthreads OK) and builds/runs every test.
- `./scripts/run_android.sh` builds with Clang 21.0.0 (Android NDK 29.0.14206865) for `arm64-v8a`, pushes the binaries, and runs them on a Xiaomi 23053RN02A (Android 15, MT6769V/CZ, 8 cores: 2× Cortex-A75 @ 2.0 GHz, 6× Cortex-A55 @ 1.8 GHz, 7.6 GB RAM).
- `spmmtest`, `gcntest`, `algorithm_profiler_test`, `profiler_workload_test`, and `profiler_probe_test` all pass.
- `memory_profiler_test` fails one check (`after.rss >= before.rss + 50 * MB`) on the phone; the allocator-overhead measurement reports `factor=1.000, requested=24 MB, delta=23 MB`.
- `hw_fingerprint_test` fails one check (`fp.storage.known && ...`) on the phone; the notes say cache topology is unreadable, no thermal zones, and no `power_supply` data.
- `profiler_capability_test` fails one check (`cv.bw_l1_gbs > cv.bw_dram_gbs`) on the phone (reported `f32 4.0 GFLOP/s, DRAM read 8.7 GB/s, noise 11%`).
- `device_benchmark_test` fails two checks (`P.sustained.ran ...` and `P.sustained.temp_start_c ...`) on the phone when the sustained probe is skipped for lack of time budget.
- The `profiler_probe_test` real run reports `blas: shim` and a leave-one-out median miss of 43%.
- `run_android.sh` initially failed with `gmake: *** No rule to make target 'backprop_test'. Stop.` — the uploaded `CMakeLists.txt` now includes `backprop_test` in its target list, and the Makefile has a `backprop_test` rule, but the script's `cmake --build ... --target ...` list does not mention `backprop_test`, so the failure must have come from an earlier version of the script or CMakeLists.

## API Reference

### Gateway REST API

See [Configuration → Gateway REST API](#gateway-rest-api). All responses are JSON with `access-control-allow-origin: *`.

### Gateway WebSocket

See [Configuration → Gateway WebSocket](#gateway-websocket).

### Web dashboard JS API

| Module | Exports |
|---|---|
| `state.js` | `state`, `gwBase()` |
| `utils.js` | `escapeHtml`, `escapeAttr`, `statusClass`, `statusDotColor` |
| `toast.js` | `toast` |
| `charts.js` | `safeChart`, `CHART_COLORS`, `chartBaseOptions`, `chartBarOptions`, `enableChartInteractions`, `closeZoomModal` |
| `param-schema.js` | `PRESETS`, `FLAG_HELP` (on `window`) |
| `parser.js` | `RX`, `emptyMetrics`, `lastRound`, `parseMetricLine`, `ingestLogLine`, `classifyLogLine`, `renderLogLine`, `parseBenchmark`, `parseTests` |
| `docs-data.js` | `REPO`, `RAW_DOCS_BASE`, `DOCS` |
| `api.js` | `api`, `connectWs`, `setWsStatus`, `handleWsMessage`, `pollJobs`, `killJob`, `launchJob`, `fetchJobDetail`, `launchJobRaw` |
| `jobs.js` | `upsertJob`, `goToJob` |
| `main.js` | `VIEW_RENDERERS`, `refreshCurrentView`, `initNav`, `initTopbar`, `boot` |
| `views/overview.js` | `renderOverview`, `refreshOverview` |
| `views/launch.js` | `addJobCard`, `initLaunchView` |
| `views/jobs-view.js` | `renderJobRows`, `renderJobsTable`, `selectJob`, `renderJobDetailMetrics`, `appendJobConsole` |
| `views/federated.js` | `getFlInstances`, `renderInstanceCards`, `renderFederatedDetail`, `renderTimingChart`, `renderClientsChart`, `renderBandwidthChart`, `renderRoundsTable`, `renderFederated` |
| `views/profiler.js` | `renderProfiler`, `renderProfilerContent`, `renderBenchChart`, `renderMemStageChart` |
| `views/memory.js` | `renderMemory` |
| `views/benchmark-view.js` | `renderBenchmark` |
| `views/tests-view.js` | `renderTests` |
| `views/docs.js` | `renderDocs` |
| `views/console-view.js` | `appendGlobalConsole`, `initConsoleView` |

### C++ API

- **Autograd operations**: `Operation<T>` base + one subclass per op.
- **Core types**: `Matrix<T>`, `Tensor<T>`, `Tensor_t<T>`, `Operation_t<T>`, `shape_t`, `Graph<T>`, `CSR<T>`, `FP8<E,M>`, `FP4<E,M>`, `soft_float16`, `float16`, `scalar::`, `vecmath::`, `mxd::`.
- **NN modules**: `Module<T>`, `Linear<T>`, `LoRALinear<T>`, `Embedding<T>`, `PositionalEncoding<T>`, `RotationalPositionalEncoding<T>`, `LayerNorm<T>`, `RMSNorm<T>`, `Relu<T>`, `Softmax<T>`, `GCNLayer<T>`, `FeedForward<T, LinearT>` (generic/GPT/Llama), `NeuralLayer<T>`, `Optimizer<T>`, `GPT<T, LinearT>`, `Llama<T, LinearT>`.
- **Model loaders**: `GGUFLoader<T>`, `GPTGGUFLoader<T, LinearT>`, `LlamaGGUFLoader<T, LinearT>`.
- **Tokenizers**: `Tokenizer`, `GPT2Tokenizer`, `LlamaTokenizer`.
- **Data loading**: `Dataset<T>`, `DataLoader<T>`, `Tabular<T>`, `NpyLoader<T>`, `GGUF`, `ImageLoader<T>`, `TextDataset<T>`, `LazyDataLoader<T>`, `Serializer<T>`, `quant::`, `load_linqs<T>`, `GraphDataset<T>`, `make_split`.
- **Networking**: `Client<T>`, `Server<T, LinearT>`, `ThreadPool`, `read_exact`, `write_exact`, `send_chunked`, `recv_chunked`, `net_bytes_sent`, `net_bytes_received`, `net_per_fd`, `net_per_fd_snapshot`, `net_forget_fd`.
- **Profiler wire protocol**: `ProfilerHeader`, `ProfilerBody`, `StorageEntry`, `WireBuffer`, `serialize`, `deserialize`, `print_wire`, `compute_checksum`, `TCPTransport`, `UnixTransport`, `UDPTransport`, `TransportConfig`, `send_profile`.
- **Profiler (old API, `tensorf::profile`)**: `collect()`, `run_device_benchmark`, `AlgorithmProfiler`, `HyperparamAdvisor`, `MemoryLedger`, `plan_budget`, `MemoryGuard`.
- **Profiler (new API, `tensorf::prof`)**: `detect_device()`, `CapabilitySuite::run()`, `wl::predict()`, `wl::run_probe()`, `ThermalProbe`, `Deadline`, `Json`.
- **CBLAS shim**: `cblas_saxpy`, `cblas_daxpy`, `cblas_sscal`, `cblas_dscal`, `cblas_sdot`, `cblas_ddot`, `cblas_sgemm`, `cblas_dgemm`. Row-major only.

## Testing

Build and run per `tests/BUILD.txt`, the CMake target list, or the Makefile. All tests print a `N/M checks passed` summary and return non-zero on failure.

| Test file | Coverage |
|---|---|
| `matrix_tests.cpp` | `Matrix<T>` for float/double/int/bool/FP8/FP4/float16 |
| `support_tests.cpp` | Vector overloads, scalar traits, FP8/FP4/float16 |
| `lowprec_tests.cpp` | FP8/FP4/float16 codec, `scalar::`, Matrix, Tensor, Graph, every autograd op |
| `backprop_test.cpp` | Ordered backprop on shared subgraphs |
| `contract.cpp` | Compile-time `Matrix` call forms used by Tensor/Graph |
| `diff_ops.cpp` | Old-vs-new `Matrix.hpp` output diff |
| `scope_test.cpp` | Operator overload scope (numeric types only) |
| `spmmtest.cpp` | SpMM gradients, sparse vs dense GCN, speed |
| `gcntest.cpp` | Synthetic GCN training |
| `coratrain.cpp` | Cora GCN training |
| `basic_tests.cpp` | Matrix/NN/MNIST/serialize functional tests |
| `algorithm_profiler_test.cpp` | `AlgorithmProfiler`, `CostModel`, calibration |
| `hyperparam_advisor_test.cpp` | `HyperparamAdvisor` (referenced by the Makefile; contents not uploaded) |
| `device_benchmark_test.cpp` | `DeviceBenchmark` kernels and controls |
| `hw_fingerprint_test.cpp` | `HardwareFingerprint` parsers and fixtures |
| `memory_profiler_test.cpp` | `MemoryLedger`, `plan_budget`, `MemoryGuard` |
| `profiler_capability_test.cpp` | `CapabilitySuite`, `DeviceFingerprint` |
| `profiler_workload_test.cpp` | `WorkloadModel` layers, memory, calibration, prediction |
| `profiler_probe_test.cpp` | `WorkloadProbe` fitting, guards, real ops |

The profiler tests accept `--quick` to skip real-machine runs. `tests/GPT/` contains a federated GPT training harness. `scripts/run_android.sh` runs a subset on a connected Android device/emulator.

## Contributing

No `CONTRIBUTING.md` was uploaded, so formal guidelines are not available. The following conventions are visible in the test sources, the headers, the build files, and the dashboard:

- Use C++20 for the core and gateway.
- Keep tests plain-assert and self-contained where possible.
- New `Matrix` call forms should be added to `tests/contract.cpp`.
- New `Matrix` behaviour should be validated against NumPy semantics (`tests/matrix_tests.cpp`) and against the previous implementation via `tests/diff_ops.cpp`.
- New autograd ops should follow the `Operation<T>` interface exactly and be added to the differential matrix in `tests/lowprec_tests.cpp`.
- New NN modules should inherit `Module<T>` and call `register_parameter` / `register_module` in their constructors; move constructors must re-register their submodules at the new address (copy constructors are deleted).
- New tokenizers should inherit `Tokenizer` and implement `encode()`/`decode()`.
- New model loaders should inherit `GGUFLoader<T>`.
- BLAS-related changes should keep `include/tools/cblas.h` row-major only and cover the four calls TensorF makes.
- Data-loader changes should keep the stubs (`load_xls`, `load_xlsx`, `load_npz`) explicit about what is missing.
- Profiler changes should keep the "honesty notes" in the headers accurate.
- Gateway changes should keep the single-threaded `io_context` (safe with `fork()`), the `try`/`catch` around `ioc.run()`, and the no-body rule for 1xx/204/304 responses.
- Dashboard changes should keep the script load order in `index.html` (state → utils → toast → charts → param-schema → parser → docs-data → api → jobs → views → main) and the "every render function is global" convention.
- New dashboard views should add a `render*` function in `js/views/`, register it in `VIEW_RENDERERS` (`main.js`), and add a nav item + `view-*` container in `index.html`.
- New log-line patterns should be added to `RX` in `parser.js` and matched against the actual printf/cout calls in the C++ sources.
- Build-system changes should keep `CMakeLists.txt` and the `Makefile` in sync (same include roots, same test target list).
- A TODO in `tests/basic_tests.cpp` reads: `// TODO intergrate the gtest framework to the tsting chain`.

## License

Apache License 2.0. The full text is in `Apache License.txt` (and, per the project tree, `LICENSE`). The Appendix boilerplate is included in the uploaded license file, with the copyright placeholder `[yyyy] [name of copyright owner]` unfilled.

## Known Limitations / TODO

- **Files not inspected**: `LICENSE`, `README.md`, `docs/**`, `tests/GPT/{Client,Server}.cpp`, `tests/GPT/Trainner.hpp`, `tests/GPT/Modes.md`, `tests/tmp/`, `include/profiler/BenchmarkRunner.hpp`, `tests/hyperparam_advisor_test.cpp`, `include/nn/ModelLoader/*.hpp` (documented from earlier uploads), `include/nn/Tokenizer/*.hpp` (documented from earlier uploads), `include/net/**` (documented from earlier uploads), and the binary assets in `Datasets/`, `Models/`, `SLM/`, `screencaptures/`. Their internals are not documented because their contents were not uploaded.
- **`make benchmark` target**: listed in `help` and in `.PHONY`, but no explicit rule was found in the uploaded `Makefile`.
- **`make llama` target**: listed in `help` and in `.PHONY`, but no explicit rule was found (the `smollm` rule builds `examples/SmollLLM.cpp`).
- **`make all` includes `benchmark`** but not `server`/`client`/`net` (the `server`/`client` lines are commented out).
- **CMake target list vs Makefile**: CMake builds 11 test targets; the Makefile also builds `contract`, `diff_ops`, `scope_test`, `support_tests`, `lowprec_tests`, `backprop_test`, and `hyperparam_advisor_test`. CMake does **not** build `tensorf-gateway`, `gpt2`, `smollm`, or `transformer`.
- **Boost dependency**: the gateway needs Boost.Beast + Boost.Asio (header-only), but neither the Makefile nor the CMake target list mentions Boost. `make install-deps` does not install it.
- **`MessageProtocol.hpp` includes missing headers**: `Profiler/BenchmarkRunner.hpp` was not uploaded, and the other profiler headers it includes use different namespaces/types. As uploaded, `MessageProtocol.hpp` would not compile against the uploaded headers.
- **`download_llms.sh` Medium/Large/XL lines**: commented out; the Medium commands have the `-O` flag commented out, so `wget -c <url> \` followed by a comment saves the file under its remote name.
- **`memory_profiler_test` failure on Android**: `after.rss >= before.rss + 50 * MB` fails on the Xiaomi phone; the allocator-overhead measurement reports `factor=1.000, requested=24 MB, delta=23 MB`, so the 64 MB allocation in `t_real_process()` may be served from already-resident pages (the test is skipped under ASan only).
- **`hw_fingerprint_test` failure on Android**: `fp.storage.known && ...` fails; the notes say cache topology, thermal zones, and `power_supply` are all unreadable (the app shell must supply battery/thermal via the OS framework).
- **`profiler_capability_test` failure on Android**: `cv.bw_l1_gbs > cv.bw_dram_gbs` fails (reported `f32 4.0 GFLOP/s, DRAM read 8.7 GB/s, noise 11%`).
- **`device_benchmark_test` failures on Android**: the sustained probe is skipped for lack of time budget, so `P.sustained.ran` and the temperature checks fail.
- **Two profiler APIs coexist**: the old one in `tensorf::profile` and the rewritten one in `tensorf::prof`. The headers do not show them interoperating.
- **`PowerOperation` only propagates to `t1`**.
- **`LogOperation` clamp** at 1e-9 for float-like `T`.
- **`TransposeOperation` 1-D case** reshapes to the input shape.
- **`SpMMOperation` threading** only when `nnz * F >= 2^16`; the symmetric flag must be set correctly.
- **CBLAS shim limits**: row-major only; aborts on `CblasColMajor`; no conjugation; no integer/half types; "not a fast BLAS".
- **`Matrix<T>::dot` for two 2-D operands is the flattened inner product**, not a matrix product.
- **`Matrix<T>::random()` is uniform [0,1) from the shared generator** (was `std::rand()`).
- **`Matrix<T>::log()` clamps exactly like `ln()`**.
- **`Matrix<T>::maximum(a)` returns `a` for values below `a`** (was 0).
- **`Tabular::load_xls` / `load_xlsx` are stubs**.
- **`NpyLoader::load_npz` is a stub**; `.npy` is implemented but does not byte-swap and reads `fortran_order` as-is.
- **`GGUF::load_tensor` supports F32/F16/Q8_0/Q4_0 only**.
- **`GGUF::parse_gguf` only stores string scalars and string arrays**.
- **`TextDataset::load()` writes `train_split.txt`, `val_split.txt`, `vocab.txt` into the folder it is given**.
- **`ImageLoader::load_directory` requires `target_h`/`target_w`**.
- **`ImageLoader::shape_of` loads the full image** (the vendored stb stub does not expose `stbi_info`).
- **`Dataset::get` / `DataLoader::Iterator::operator*` allocate per call**.
- **`quant::QuantTensor` stores one `QT` struct per element** (FP4 is not bit-packed in memory).
- **`quant::load_quantized` checks the format tag against `QT`**; **`load_module_into` checks the tensor count**.
- **`quant::` uses `is_fp4<QT>::value`** (the legacy trait), not `scalar::is_fp4_v`.
- **`Llama::forward` / `GPT::forward`** return the output when `targets == nullptr`, the loss otherwise.
- **`Llama::generate` checks EOS only for the first batch element** and uses `eos_token_id = 2`.
- **`Llama`'s shared `Args...` pack** disables bias everywhere when `bias = false` is passed, including `lm_head` and attention Q/K/V/proj (documented in the header comment).
- **`Llama::FeedForward::make_linear`** forces `bias = false` for `LinearT`s that accept a bias flag.
- **`GPT::train` / `Llama::train`** construct an `Optimizer<T>` with `ADAMw` and `1e-4` lr internally; neither exposes the optimizer type or hyperparameters.
- **`GPT::lora_parameters()` / `Llama::lora_parameters()`** return `lm_head.parameters()` regardless of `LinearT`.
- **`Optimizer::clip_grad_norm` is always called with `max_norm = 1.0`** from `step()` (except for Adafactor).
- **`Optimizer.hpp` includes `<cblas.h>` unconditionally**.
- **`Module<T>::to_string()`, `reset_graph()`, `forward()`** are declared but not defined in `Module.hpp`.
- **`Parameter<T>`** is declared in `Module.hpp` but not used by any uploaded module.
- **Gateway `job_exited` event**: documented in `main.cpp` as "not yet implemented"; the dashboard polls `/api/jobs` every 6 s to catch status transitions.
- **Gateway OPTIONS preflight**: returns 204 with no body (the comment in `main.cpp` explains that the old handler dumped `"{}"` into a 204 body, which made Beast's serializer throw `std::invalid_argument` and terminated the process).
- **Gateway single-threaded**: the `io_context` is `io_context{1}` because the gateway calls `fork()` + `execv()`, which is unsafe with a multithreaded Asio event loop.
- **Dashboard launch API mismatch**: `param-schema.js`'s `FLAG_HELP` and `PRESETS` define flags (`--fed`, `--rounds`, `--mu`, `--min_clients`, `--max_clients`, `--client_id`, `--device`, `--iterations`, `--epochs`) that the gateway's `POST /api/jobs/*` handlers do not document in `main.cpp` (they only forward `args` to the binary). The gateway will pass them through, but the binaries must accept them.
- **Dashboard parser coupling**: `parser.js`'s regexes are matched against the binaries' `printf`/`cout` calls; if those strings change, the dashboard's metrics stop updating. The header comment says only `parser.js` and `ingestLogLine()`'s caller need to change if the gateway ever gains a real metrics endpoint.
- **Dashboard docs links**: `docs-data.js` points at `docs/*.html` on the main branch of `https://github.com/InputOutputStream/TensorF`, which were not uploaded.
- **`tests/tmp/`** is listed in the tree but its contents were not uploaded.
- **`datasets/MNIST_CSV/readme.md`** and the two `*_Report.docx` files in `include/data/` were not uploaded.
- **`tests/BUILD.txt`** mentions `stub/cblas.o` and `Matrix.orig.hpp` (via `-DORIG`); neither was uploaded.
- **`docs/ROADMAP.md`** and **`docs/known-issues.html`** exist in the tree but were not uploaded.
- **Platform adapters**: the Apple and Windows adapters are implemented but "NOT compiled or run here". iOS thermal/battery and Android 8+ battery need the OS framework.
- **Probe scope**: `WorkloadProbe.hpp` states that every probe case runs single-threaded (`threads_validated` = 1), `conv2d` and `particle_interaction` are not covered, and `blas` records which BLAS was linked.
- **Workload model scope**: `WorkloadModel.hpp` does not run anything, does not choose a model size, and does not predict energy.
- **Cost model assumptions**: `AlgorithmProfiler.hpp` states low precision is storage-only; layers run sequentially; time = max(compute, regular memory) + irregular + dependent loads + overhead.
- **Capability suite safety**: `CapabilitySuite.hpp` states its thermal limits are "STARTING POINTS, unverified; calibrate per device".
- **A TODO in `tests/basic_tests.cpp`**: `// TODO intergrate the gtest framework to the tsting chain`.
- **No environment variables or config files** were found beyond the ones listed in [Configuration](#configuration).
- **No HTTP/RPC API** was found other than the gateway's REST + WebSocket API; the `docs/api/*.html` files were not uploaded.
- **Android build**: the `run_android.sh` target list does not mention `backprop_test`, and an earlier run of the script failed with `gmake: *** No rule to make target 'backprop_test'. Stop.` — the uploaded `CMakeLists.txt` now includes it, but the script's `cmake --build ... --target ...` list still does not.