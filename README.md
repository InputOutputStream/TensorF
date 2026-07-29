# TensorF

A small tensor / deep learning framework built from scratch in C++, with support for federated training over a custom client-server protocol.

TensorF includes:
- A core tensor/matrix engine with autograd-style operations (`include/core`)
- Neural network building blocks — layers, attention, optimizers, transformers (`include/nn`)
- Dataset loaders for images, text, tabular data, GGUF/NPY formats (`include/data`)
- A federated learning networking layer — client, server, thread pool, message protocol (`include/net`)
- Profiling, benchmarking, and hyperparameter tooling (`include/tools`)
- Example model implementations: GPT-2 and a small LLM (`examples`)

## Project structure

```
TensorF/
├── include/
│   ├── core/             # Tensor/Matrix engine
│   │   ├── Operations/
│   │   ├── Overloads/
│   │   ├── Types/
│   │   └── DataStructures/
│   ├── nn/                # Neural network layers & models
│   │   ├── Modules/
│   │   ├── ModelLoader/
│   │   └── Tokenizer/
│   ├── data/               # Dataset & data loading utilities
│   │   └── DataLoader/
│   ├── net/                # Federated learning networking
│   │   ├── Network/
│   │   └── Protocol/
│   └── tools/               # Profiling & benchmarking
│       └── Profiler/
├── examples/                # Example model drivers (GPT2.cpp, SmollLLM.cpp)
├── tests/                   # Test suite & benchmark harness
├── scripts/                  # Utility scripts (e.g. download.sh)
├── docs/                     # Design notes and roadmap
├── documentation/            # Static HTML docs (architecture, quickstart, API)
└── Makefile
```

## Building

Requires a C++20 compiler and BLAS (e.g. OpenBLAS).

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

make tests       # build and run the test suite
make gpt2        # build the GPT-2 example
make smollm      # build the small LLM example
make benchmark   # build the benchmark harness
```

## Federated learning

The `net/` module provides a client-server protocol for distributed/federated training.

```bash
make server       # build bin/server from $(SERVER_SRC), default include/net/Network/Server.cpp
make client        # build bin/client from $(CLIENT_SRC), default include/net/Network/Client.cpp

# Override the source file if you have a custom entry point:
SERVER_SRC=my_server.cpp make server
```

See `include/net/Network/Modes.md` and `documentation/federated.html` for protocol details.

## Documentation

Static docs live in `documentation/` — open `documentation/index.html`, or see:
- `documentation/architecture.html` — framework architecture
- `documentation/quickstart.html` — getting started
- `documentation/federated.html` — federated learning protocol
- `documentation/install.html` — installation
- `documentation/api/` — API reference

## Roadmap

See `docs/ROADMAP.md` for planned work (KV cache, fused attention, GPU/SYCL backend, optimizations).

## Known issues

There are currently signature mismatches between `Optimizer` and callers in `include/nn/Modules/FeedForward.hpp` / `tests/basic_tests.cpp` that block a full `make tests` build — see `documentation/known-issues.html`.

## License

Apache-2.0 — see `LICENSE`.
