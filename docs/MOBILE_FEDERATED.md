# Federated learning on phones — `include/net/Mobile/`

A self-contained, model-agnostic layer for federated averaging over **unreliable
clients** (phones on Wi-Fi/LTE, battery- and thermal-limited). It aggregates a flat
`float` vector; what the vector holds (LoRA adapters, full weights, a GCN) is up to
the caller. It does not depend on BLAS, GGUF or the rest of TensorF, so it builds
anywhere `g++`/the NDK builds.

| File | Role |
|---|---|
| `FedWire.hpp` | Wire protocol: 8-byte header (magic/version/type/length), little-endian, HELLO/WELCOME, READY→GO/SKIP, chunked fp32/fp16 streams |
| `FedCoordinator.hpp` | Server: sessions, cohorts, deadlines, exact dropout handling, JSON events |
| `MobileClient.hpp` | Client: connect/backoff, SYNC, participation gate, fp16 upload, keeps undelivered updates |
| `tests/fed_mobile_sim.cpp` | 35 loopback checks (also run clean under TSAN and ASAN+UBSAN) |
| `tests/fed_mobile_server.cpp`, `fed_mobile_client.cpp` | CLI demo / phone smoke test |

## What it fixes compared with the legacy `Server.hpp`/`Client.hpp`

* **Rounds have a fixed cohort.** A client that is late or arrives mid-round is told `SKIP`
  (or waits for the next round); it can never write into a round in progress.
* **Dropout is exact.** Chunks arrive in order, so a client lost after `k` elements contributed to
  `[0,k)`. Per element the divisor is `W_completed + Σ weight(dropped clients with prefix > i)`.
* **Deadlines and timeouts everywhere:** `ready_timeout_ms`, `upload_deadline_ms`, per-socket
  `SO_RCVTIMEO/SO_SNDTIMEO`, TCP keepalive. A silent phone is cut off instead of blocking the round.
* **Weighted by samples** (`n_samples` from READY), optional staleness decay.
* **Everyone starts from the same model:** clients `SYNC` on first connect.
* **Thread-per-session with limits**, not a fixed pool that silently starves client N+1; handshake runs
  on the session thread so a slow client cannot block `accept()`.
* **Safe I/O:** `MSG_NOSIGNAL` (no SIGPIPE kill), `EINTR` retry, bounded message/chunk sizes, size checked before allocating, optional shared-secret token.
* **Bandwidth:** fp16 upload (negotiated, ~2 bytes/element) — about half the cellular traffic.
* **Observability:** the coordinator prints `@@TFEVT {json}` lines (`fed_round_started/done/aborted`,
  `fed_client_joined/left/dropped/rejected`); the gateway forwards them as `{"type":"fed_event"}` over `/ws`.

## Protocol summary (version 1)

```
header  : "TFFD" | u8 version | u8 type | u16 body_len        (body_len <= 1024)
HELLO   : id, token, param_count, fingerprint, dtype_mask, chunk_kb, ram_mb, platform, flags
WELCOME : accepted, reason, global_round, up_dtype, down_dtype, chunk_kb, io/ready/upload timeouts
READY   : n_samples, base_round, flags        ->  GO(round, dtype) | SKIP(round, retry_ms, flags, reason)
GO      : client streams [u64 total][ (u32 len, len elems)* ]     (fp32 or fp16, little-endian)
RESULT  : round, participants, status, dtype  +  global stream     (also the reply to SYNC_REQ)
```
`SKIP` flags: bit0 = "stale, SYNC first", bit1 = "training finished".

## Architecture: the transport knows nothing about the algorithm

```
 client:  ClientTask (GCN | GraphSAGE | GAT | RGCN | MLP | LoRA ...) + FedProx pull
 server:  ServerTaskInfo (param layout + initial vector)             + post_aggregate hook
                 │  flat float vector, algo name, fingerprint
 MobileClient (reconnect, gate, fp16, resume)  <── FedWire ──>  FedCoordinator (cohorts, deadlines, weighted mean)
```

| Layer | Files | Knows about |
|---|---|---|
| Transport | `FedWire`, `FedCoordinator`, `MobileClient` | a flat float vector, an algo name, a fingerprint |
| Task contract | `FedTask.hpp` (`ClientTask`, `TaskSpec`, `TaskRegistry`) | the 3 callbacks + registry; no tensors |
| Strategies | `FedStrategies.hpp` | FedAvg (default), FedAvgM, FedAdam (server), FedProx (client) |
| Tensor bridge | `TensorParams.hpp` (`ParamView`) | `Tensor_t<T>` lists to flat vector + shape fingerprint, for any model |
| Graph tasks | `graph/GraphPartition.hpp`, `graph/NodeClassifierTask.hpp` | graphs, partitioning, a `Model` policy |
| Built-ins | `BuiltinTasks.hpp` | `synthetic`, `gcn` |

No LLM code is involved anywhere in this stack; LLM/LoRA parity with the new architecture is a later, separate step (it will be one more `ClientTask`).

### Adding an algorithm (GraphSAGE, GAT, RGCN, GNN, ...)

For node classification, write one `Model` policy and register it. The server, the client binaries, the protocol and the gateway do not change:

```cpp
struct GraphSAGE2 {
    static const char* name() { return "sage"; }
    GraphSAGE2(size_t F, size_t C, const TaskOptions& o, unsigned seed);
    Tensor_t<float> forward(std::shared_ptr<const CSR<float>> adj, Tensor_t<float> x, bool training, std::mt19937& rng);
    std::vector<Tensor_t<float>> parameters();            // exactly what gets averaged
    Tensor_t<float> regularizer(float weight_decay);
};
// in register_builtin_tasks():
r.add("sage", graph::node_classifier_factory<GraphSAGE2>("GraphSAGE node classification"));
```
The fingerprint is derived from `name()` and every tensor shape, so a `gcn` client can never join a `sage` server, and `hidden=32` vs `16` is caught at connect time with a readable message.
For anything that is not node classification (graph classification, link prediction, RGCN with a relation dimension, ...) implement `ClientTask` directly: `train()`, `get_params()`, `set_params()`.

How each planned method maps onto the transport:

* **GraphSAGE / GAT / GNN / RGCN**: parameters are fixed by the architecture config (not by graph size), so plain FedAvg of `parameters()` applies. RGCN's relation count is part of the config and of the fingerprint.
* **GraphSAINT**: a *local training procedure* (subgraph/random-walk sampling), not a different model. It lives inside `train()` of the task (sample a subgraph per step instead of full-batch), which also bounds per-step memory on a phone.
* **Other payloads** (embeddings, logits for distillation, per-layer subsets): the coordinator averages any fixed-length vector, so expose it through `get_params/set_params` (or select a subset in `ParamView`).
* **Not handled by the transport:** cross-device edges. Each device sees only its induced subgraph; `cut_fraction()` measures the loss (18 % for `bfs`, 75 % for `random` on Cora with 4 devices). Methods that exchange neighbour information between devices (FedSage+-style) would need a new message type.

### Distribution methods (strategies)

`--strategy fedavg` (default) | `fedavgm:beta[:lr]` | `fedadam:lr[:b1:b2:tau]` on the server; `--opt prox_mu=0.01` (FedProx) on GCN clients;
`--weighting samples|uniform`; `--staleness-decay`. Write your own as a `post_aggregate` function.

## Federated GCN on Cora: measured here (host CPU, BLAS shim, 4 device processes over real sockets)

`./scripts/fed_gcn_demo.sh 4 30 5 fedavg bfs` (K=4 devices, 30 rounds x 5 local epochs, standard 140/500/1000 split; accuracy on the 1000 full-graph test nodes):

| Setting | Test acc. |
|---|---|
| Centralized (one machine, 150 epochs) | 78.0 % |
| **Federated GCN, BFS partition (non-IID, 18 % edges cut)** | **70.4 %** |
| Federated GCN, random partition (about IID, 75 % edges cut) | 74.2 % |
| Local only, no communication (mean of 4 devices) | 33.2 % |
| Federated, BFS, FedAvgM beta=0.9 lr=1 (untuned) | 43.0 % |

Reading: federation recovers most of the gap to centralized training; the non-IID label skew of the BFS partition costs more than the lost edges.
FedAvgM with beta=0.9 and lr=1 is too aggressive for 30 rounds: the hook works (unit-tested), the hyper-parameters are not tuned. One run each, seeds not averaged: treat differences of a few points as noise.
Traffic per device (BFS run): 1.38 MB up (fp16, 23 040 parameters x 30 rounds), 2.86 MB down (fp32 + initial sync).

## Data shards and heterogeneity-aware partitioning (protocol v2)

**When to use it.** Shards are for *distributed training with a central data owner* (compute offload to edge devices): the server holds the graph, partitions it, and each phone receives only its own part. The server therefore sees the full dataset. If the data originates on the devices (classic federated learning) you do not use shards.

### What changed in the protocol (v1 → v2)

* `HELLO` now carries the device capability: `score` (0–100) and `mem_budget_mb`.
* New messages `ShardReq` (also *enrols* the device), `ShardHdr` (metadata + blob descriptors, followed by one chunked stream per blob: f32, f16-on-the-wire or raw u32), `ShardWait` (retry / denied).
* The coordinator closes enrollment once `enroll_min` devices (default `min_clients`) have asked and `enroll_grace_ms` passed without a new one, calls the `ShardProvider` **once**, then serves shards. A device that arrives after enrollment closed is refused with a clear message (no repartitioning yet).
* The model layout does not depend on the data (`features`, `classes`, hidden size), so a phone can connect and pass the fingerprint check *before* it has any data.
* Shards are generic (`FedShard.hpp`: string metadata + named blobs). The graph task ships `x` (features), `y` (labels), `flags` (train/val/test bits), `edges` (local pairs, each undirected edge once). Edges to other devices' nodes are dropped.

### Capacity policy (profiler score → share of the graph)

`CapacityPolicy` is independent of the partitioner: `c(s) = clamp(s, floor, hi)` (defaults `floor=5`, `hi=70`), shares `t_i = c(s_i) / Σ c(s_j)`.
Below `hi` the share is proportional to the score; at or above it every strong device gets the same maximum share (set `hi=100` for plain proportional). Example 95/80/40/15 → 35.9 % / 35.9 % / 20.5 % / 7.7 %.
Vertex weights default to `1 + degree` (aggregation cost of a GNN layer), clamped at 8× the median so hubs do not wreck the balance (METIS is known to struggle with very large weight ratios).
`mem_budget_mb` is a **hard** cap, enforced after partitioning by `enforce_caps` (METIS' `ubvec` is only a tolerance, not a limit); infeasible caps raise an error before anything is sent.

### What each device receives: partition, halo or the whole graph

Each device picks (`--shard auto|partition|halo[:K]|full` on the client), the server may impose one for everybody (`--opt shard_mode=...`), and a request that does not fit the device's memory budget is downgraded (full → halo → partition) with the reason recorded in the shard (`reason=memory budget`).

| Mode | The device gets | Trains on | Cross-device edges |
|---|---|---|---|
| `partition` (default) | its owned nodes | its owned train nodes | dropped |
| `halo:K` | owned nodes + their K-hop neighbourhood (context only) | its owned train nodes | kept up to K hops (K = number of GNN layers gives exact outputs) |
| `full` | the whole graph | its owned train nodes | all kept, but every step runs on the whole graph |

In every mode ownership partitions the graph exactly (each training label is used by exactly one device); non-owned nodes carry no train/val/test flag. Shards are built on demand, not cached.

Cora, 4 devices, METIS, 20 rounds × 5 local epochs, server-side test accuracy (single runs, same seed): `partition` 55.9 %, `halo:2` 58.4 %, `full` 57.7 %. The differences are within what one run can show; the clear effect is cost: `full` was several times slower per round than `halo:2` (the benchmark machine was also compiling in the background, so only the order of magnitude is meaningful).

### Datasets

`./scripts/download_datasets.sh list` | `cora` | `ogbn-arxiv` | `ogbn-products` (`--from file.zip` to use an archive you already have, `OGB_BASE=` for a mirror, `DATASETS_DIR=` to choose the target).
OGB archives are converted once to a compact `graph.tfg` (`tools/ogb2tfg.cpp`, streams `gzip -dc`, no dependency); the loader reads it by streaming and uses the **official OGB split**. `--opt data=Datasets/ogbn-arxiv` works everywhere a dataset is accepted. Not supported: ogbn-proteins (multi-label), ogbn-mag (heterogeneous; RGCN later), ogbn-papers100M (needs an out-of-core loader).
Verified here: the converter, loader and script against a synthetic archive with the exact OGB layout (`scripts/make_fake_ogb.py`; 100 000 nodes convert in under a second). **Not verified:** the real download from `snap.stanford.edu` (blocked from the build environment) and ogbn-products (whole graph is held in memory: expect several GB; start with arxiv).

### Partitioning backends: all connected, test them one by one

Do not write a multilevel partitioner from scratch. Four backends sit behind one function (`--opt partitioner=auto|metis|kaminpar|scotch|builtin`, auto = METIS, then Scotch, then KaMinPar, then builtin):

| Backend | License | How heterogeneous targets are expressed | Build |
|---|---|---|---|
| **METIS 5** | Apache-2.0 | `tpwgts` per part + `ubvec` | `apt install libmetis-dev` (auto-detected) |
| **KaMinPar** | MIT | relative max **and min** block weights | `KAMINPAR_ROOT=/path/to/KaMinPar make ...` after building its `KaMinPar` target (needs TBB, numa) |
| **Scotch** | CeCILL-C | weighted complete target architecture (capacity per device) | `apt install libscotch-dev` (auto-detected) |
| builtin | — | targets in region growing + greedy refinement | always |

KaHyPar (GPL, hypergraphs) is not applicable. Partitioning runs on the server, so none of this is cross-compiled for Android.
`./bin/fed_partition_bench --data Datasets/cora --scores 95,80,40,15 [--only metis,scotch] [--synthetic N,deg]` runs every compiled-in backend on the same input and prints cut, communication volume, imbalance and time.

Measured here (1 CPU core, 1 run, vertex weights 1 + degree, tolerance 3 %; imbalance = max weight / target; cut = % of edges between devices):

| Input | builtin | METIS | KaMinPar | Scotch |
|---|---|---|---|---|
| Cora, 4 devices | 11.0 % / 0.8 ms | 4.5 % / 8 ms | **4.5 %** / 7 ms | 7.6–9.5 % / 16–22 ms |
| synthetic 170k nodes, 6 devices | 26.1 % / 0.27 s | **16.7 %** / 2.5 s | 19.2 % / 0.6 s | 16.7 % / 1.7 s |
| synthetic 1M nodes, 6 devices | 24.0 % / 4.0 s | **16.7 %** / 25.9 s | 20.1 % / 3.6 s | 18.2 % / 135 s |

All backends respect the capacity-proportional targets (imbalance ≤ 1.03; KaMinPar needed explicit lower bounds to stop under-filling the weakest device). Reading: METIS gives the lowest cut on the larger graphs, KaMinPar is 4–7× faster for a few points more cut (so it is the one to use when partitioning time matters), Scotch is accurate on targets but slow at scale, the builtin fallback is fast but clearly worse. The synthetic graphs are planted-community random graphs, not real data; only Cora is real. KaMinPar was run with 1 thread; its advantage grows with cores.

End to end (`scripts/fed_gcn_hetero_demo.sh`, 4 simulated phones with speeds 1.0/0.8/0.4/0.15, clients run in an empty directory with no access to the dataset, 20 rounds): capacity-aware shards took 23.5 s against 44.9 s for equal shares (≈1.9× faster), at 55.9 % vs 51.4 % test accuracy (single run; do not over-read the accuracy gap). Speeds are simulated by time dilation.

### Limits

* Enrollment is one-shot: no repartitioning when devices join, leave or change speed; late joiners are refused.
* Single objective (cut or volume); memory is a post-hoc hard cap, not a METIS constraint.
* Scores and memory budgets are reported by the device and not authenticated.
* With shards the server holds the whole dataset (compute offload, not data-stays-on-device FL).
* Partitioner timings above are from a single-core sandbox and synthetic graphs for the large cases; re-measure on your hardware and on real OGB graphs.

## Build, test, run

```bash
make fed-mobile-test                      # transport simulation (35 checks) + task/strategy/partition tests
make fed-mobile                           # fed_mobile_sim/_server/_client, fed_task_test, fed_gcn_baseline in bin/
./bin/fed_mobile_server --list            # registered algorithms
./bin/fed_mobile_server --algo gcn --opt data=Datasets/cora --min-clients 3 --rounds 30
./bin/fed_mobile_client --algo gcn --id phoneA --opt data=Datasets/cora --opt parts=3 --opt part=0 --opt eval_full=1
./scripts/download_datasets.sh cora ogbn-arxiv && ./scripts/fed_gcn_demo.sh 4 30 5 fedavg bfs      # full demo + baselines
./scripts/fed_gcn_hetero_demo.sh capacity 20 5       # shards + METIS + heterogeneous devices (vs: equal)
sudo apt install libmetis-dev libscotch-dev           # optional partitioners; the Makefile auto-detects them
./bin/fed_partition_bench --data Datasets/cora --scores 95,80,40,15      # compare every compiled-in backend
```
Android: `./scripts/run_android.sh` now also pushes and runs `fed_mobile_sim` on the device.
`FED=1 ./scripts/run_android.sh` additionally runs a real phone↔host federation over USB
(`adb reverse tcp:9740 tcp:9740`, no Wi-Fi needed). Gateway: `POST /api/jobs/fedserver` and `/api/jobs/fedclient` with `{"algo":"gcn","opts":{...},"strategy":"..."}`.

Security: no TLS. The token is an access check, not encryption — use a VPN/SSH tunnel off the LAN.
The gateway now listens on 127.0.0.1 by default and refuses a non-loopback bind without `TENSORF_GATEWAY_TOKEN`.

## Status and limits (read before relying on it)

* Verified here: host build with g++ 13, 35/35 transport checks + 2 736 task/partition checks, ThreadSanitizer and ASan+UBSan clean on the transport, gateway end-to-end
  (REST + WebSocket events), federated GCN end-to-end (table above). **Not verified:** NDK cross-build, any real phone, CMake configure of the full tree, and the 3 timing-based tests under heavy CPU load.
* LLM parity is deferred: the legacy GPT `tests/GPT/Server.cpp`/`Client.cpp` still cannot build (missing `Profiler/Profiler.hpp`) and are **not**
  ported. Porting = one `ClientTask` over `Trainer` using `ParamView` on the LoRA adapters.
* Two GCN client modes: `source=partition` (simulation: loads the full graph, keeps its part) and `source=shard` (the phone receives only its shard from the server; no dataset access). The `gcn` task has not been run on ARM/Android; only the transport simulation is wired into `run_android.sh`.
* Aggregation runs under the round lock; `post_aggregate` and the weighted mean are O(P) per round, fine for GNN-sized models, to be revisited for very large vectors.
* Known issues in the legacy path (not fixed): `GPT::lora_parameters()` returns only `lm_head` adapters, so block adapters are never trained;
  `parameters()` sent on the wire includes non-adapter tensors; FedDistill averages logits computed on each client's *different* local batch.
* fp16 upload halves bandwidth but rounds each client's weights (≈1e-3 relative); keep `fp16_download=false` unless needed.
* Single coordinator, no persistence: restart loses the round counter (clients re-SYNC and continue).
