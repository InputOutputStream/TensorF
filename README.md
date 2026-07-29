# TensorF

A small tensor / deep learning framework built from scratch in C++, with support for federated training over a custom client-server protocol — plus a web dashboard for managing training and federated jobs.

TensorF includes:
- A core tensor/matrix engine with autograd-style operations (`include/core`)
- Neural network building blocks — layers, attention, optimizers, transformers (`include/nn`)
- Dataset loaders for images, text, tabular data, GGUF/NPY formats (`include/data`)
- A federated learning networking layer — client, server, thread pool, message protocol (`include/net`)
- Profiling, benchmarking, and hyperparameter tooling (`include/tools`)
- Example model implementations: GPT-2 and a small LLM (`examples`)
- A REST + WebSocket gateway for managing jobs from a web UI (`gateway`)

## Project structure

```
TensorF/
├── include/
│   ├── core/               # Tensor/Matrix engine
│   │   ├── Operations/
│   │   ├── Overloads/
│   │   ├── Types/
│   │   └── DataStructures/
│   ├── nn/                 # Neural network layers & models
│   │   ├── Modules/
│   │   ├── ModelLoader/
│   │   └── Tokenizer/
│   ├── data/                # Dataset & data loading utilities
│   │   └── DataLoader/
│   ├── net/                 # Federated learning networking
│   │   ├── Network/
│   │   └── Protocol/
│   └── tools/                # Profiling & benchmarking
│       └── Profiler/
├── examples/                 # Example model drivers (GPT2.cpp, SmollLLM.cpp)
├── tests/                    # Test suite & benchmark harness
├── scripts/                   # Utility scripts (e.g. download.sh)
├── gateway/                   # Web management API (REST + WebSocket) for the dashboard
│   ├── include/               # JobManager (process spawn/log capture), WsHub (broadcast)
│   └── src/main.cpp           # Boost.Beast HTTP server + routes (built by the root Makefile)
├── docs/                       # Static HTML documentation site + ROADMAP.md
│   ├── index.html, install.html, quickstart.html, architecture.html, federated.html, known-issues.html
│   └── api/                    # Per-module API reference, including api/gateway.html
└── Makefile
```

Headers resolve via `-Iinclude -Iinclude/core -Iinclude/nn -Iinclude/data -Iinclude/net -Iinclude/tools`, so any header can be included by its bare subsystem path (e.g. `#include "Types/types.hpp"`) regardless of which group it lives under.

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

`make tests` currently compiles and runs successfully (XOR training sample converges); the only remaining failure is a missing local dataset file (`Datasets/MNIST_CSV/mnist_train.csv`), not a code issue.

## Federated learning

The `net/` module provides a client-server protocol for distributed/federated training.

```bash
make server       # build bin/server from $(SERVER_SRC), default include/net/Network/Server.cpp
make client        # build bin/client from $(CLIENT_SRC), default include/net/Network/Client.cpp

# Override the source file if you have a custom entry point:
SERVER_SRC=my_server.cpp make server
```

See `include/net/Network/Modes.md` and `docs/federated.html` for protocol details.

## Web dashboard / management API

`gateway/` is a standalone Boost.Beast HTTP + WebSocket server that launches and monitors the binaries above (`server`, `client`, `benchmark`, `gpt2`, `smollm`, `tests`) as jobs, without touching TensorF's own code. It builds from the root `Makefile` alongside everything else.

```bash
make gateway              # builds bin/tensorf-gateway
make run-gateway          # build + run, defaults to PORT=8080, TENSORF_BIN_DIR=./bin

# equivalent by hand:
TENSORF_BIN_DIR=./bin ./bin/tensorf-gateway --port 8080
```

- REST API under `/api` — list/launch/kill jobs, health check.
- WebSocket at `/ws` — live `job_started` / `log` events as jobs run.

Full route table and job payload shape: `docs/api/gateway.html`. (The frontend dashboard itself is in progress.)

## Documentation

Static docs live in `docs/` — open `docs/index.html`, or see:
- `docs/architecture.html` — framework architecture
- `docs/quickstart.html` — getting started
- `docs/federated.html` — federated learning protocol
- `docs/install.html` — installation, directory layout, build targets, gateway setup
- `docs/api/` — API reference, including `docs/api/gateway.html` for the management API
- `docs/known-issues.html` — tracked bugs and design inconsistencies from code review

## License

Apache-2.0 — see `LICENSE`.
