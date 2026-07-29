# TensorF Federated — CLI Test Modes

Quick reference for every client/server combination, updated for the
checkpointing flags (`--save-path`, `--load-path`, `--save-every`,
`--quantize`, the client's `--rounds`/`--client-id`), and now LoRA
(`--lora`, `--lora-rank`, `--lora-alpha`).

**Defaults worth knowing before you run anything:**
- Client checkpoints default to `checkpoints/client_<id>/student.tnsf`, namespaced
  by `--client-id` (default `"default"`). **Set `--client-id` explicitly whenever
  you run more than one client on the same machine**, or they'll overwrite each
  other's checkpoint.
- Server checkpoints default to `checkpoints/global/model.tnsf`.
- Both save every round by default (`--save-every 1`). Pass `--save-every 0` to
  disable — handy for quick smoke tests so you don't litter `checkpoints/`.
- `--quantize fp8` / `--quantize fp4` writes an *additional*, smaller checkpoint
  (`<save-path>.fp8` / `<save-path>.fp4`) alongside the plain float one. The
  float one is what you resume training from; the quantized one is the
  smaller deliverable.
- **`--lora` must match between server and every client.** The backbone is
  still loaded dense from `--model` either way — only the adapter
  ({A,B}) weights are trained/exchanged when `--lora` is set. On connect,
  the client sends its `--lora`/`--lora-rank`/`--lora-alpha` and the server
  accepts or rejects the connection based on whether it matches its own
  config; a rejected client logs the mismatch and falls back to local-only
  training instead of sending weight deltas that won't fit the server's
  parameter layout.

---

## Local modes (no server)

```bash
# Local finetune only — no server, no federated coordination
./bin/client \
  --model  SLM/gpt2-small-f32.gguf \
  --dataset Dataset \
  --iters  100 \
  --eval   10 \
  --batch  4 \
  --block  512 \
  --lr     1e-4 \
  --prompt "the data type is int" \
  --tokens 40 \
  --no-federated
```

```bash
# Local finetune, quiet (no per-iteration output), skip hardware profiling
./bin/client \
  --model  SLM/gpt2-small-f32.gguf \
  --dataset Dataset \
  --iters  200 \
  --batch  8 \
  --block  1024 \
  --lr     5e-5 \
  --no-federated \
  --no-profile \
  --quiet
```

```bash
# Local finetune with a defined end — saves a checkpoint after 5 rounds,
# plus an fp8 copy ~4x smaller on disk
./bin/client \
  --model  SLM/gpt2-small-f32.gguf \
  --dataset Dataset \
  --iters  100 --batch 4 --block 512 --lr 1e-4 \
  --no-federated \
  --rounds 5 \
  --save-path checkpoints/local/student.tnsf \
  --quantize fp8
```

```bash
# Local LoRA finetune — only the adapter weights train, backbone stays frozen
# at whatever --model's dense GGUF weights are. rank/alpha only matter when
# --lora is passed.
./bin/client \
  --model  SLM/gpt2-small-f32.gguf \
  --dataset Dataset \
  --iters  100 --batch 4 --block 512 --lr 1e-4 \
  --no-federated \
  --lora \
  --lora-rank 8 \
  --lora-alpha 16.0 \
  --save-path checkpoints/local/lora_student.tnsf
```

---

## FedAvg — server + clients (default mode)

```bash
# Terminal 1 — server, wait for 2 clients per round, run indefinitely.
# Saves checkpoints/global/model.tnsf after every round (default --save-every 1).
./bin/server \
  --model   SLM/gpt2-small-f32.gguf \
  --port    8080 \
  --clients 2 \
  --prompt  "the data type is int" \
  --tokens  30
```

```bash
# Terminal 2 — client A
./bin/client \
  --server    127.0.0.1 \
  --port      8080 \
  --model     SLM/gpt2-small-f32.gguf \
  --dataset   Dataset/client_A \
  --iters     50 \
  --eval      10 \
  --batch     4 \
  --block     512 \
  --lr        1e-4 \
  --client-id client_A
```

```bash
# Terminal 3 — client B
./bin/client \
  --server    127.0.0.1 \
  --port      8080 \
  --model     SLM/gpt2-small-f32.gguf \
  --dataset   Dataset/client_B \
  --iters     50 \
  --eval      10 \
  --batch     4 \
  --block     512 \
  --lr        1e-4 \
  --client-id client_B
```

---

## FedAvg — fixed number of rounds, then stop

```bash
# Server shuts down cleanly after 10 rounds, with profiler JSON
./bin/server \
  --model   SLM/gpt2-small-f32.gguf \
  --port    8080 \
  --clients 3 \
  --rounds  10 \
  --prompt  "the data type is int" \
  --json-log
```

```bash
# Client — capped at the same round count so it exits (and saves its final
# checkpoint) cleanly instead of looping forever waiting on a closed socket
./bin/client \
  --server    127.0.0.1 \
  --port      8080 \
  --model     SLM/gpt2-small-f32.gguf \
  --dataset   Dataset/client_A \
  --iters     50 \
  --batch     4 \
  --block     512 \
  --lr        1e-4 \
  --rounds    10 \
  --client-id client_A
```

---

## FedAvg — LoRA (adapter-only exchange)

Same shape as regular FedAvg, but only the LoRA `{A,B}` adapters are
trained and exchanged — the backbone loaded from `--model` is frozen and
identical across all clients and the server. This makes both the
per-round wire payload and the server's `round_accum` much smaller than
full-weight FedAvg. **`--lora-rank`/`--lora-alpha` must match exactly
across the server and every client**, or the server rejects the
connection during the handshake (see below).

```bash
# Terminal 1 — server, LoRA mode
./bin/server \
  --model      SLM/gpt2-small-f32.gguf \
  --port       8080 \
  --clients    2 \
  --rounds     10 \
  --lora \
  --lora-rank  8 \
  --lora-alpha 16.0 \
  --prompt     "the data type is int"
```

```bash
# Terminal 2 — client A, LoRA mode, matching rank/alpha
./bin/client \
  --server     127.0.0.1 \
  --port       8080 \
  --model      SLM/gpt2-small-f32.gguf \
  --dataset    Dataset/client_A \
  --iters      50 --batch 4 --block 512 --lr 1e-4 \
  --rounds     10 \
  --client-id  client_A \
  --lora \
  --lora-rank  8 \
  --lora-alpha 16.0
```

```bash
# Terminal 3 — client B, LoRA mode, matching rank/alpha
./bin/client \
  --server     127.0.0.1 \
  --port       8080 \
  --model      SLM/gpt2-small-f32.gguf \
  --dataset    Dataset/client_B \
  --iters      50 --batch 4 --block 512 --lr 1e-4 \
  --rounds     10 \
  --client-id  client_B \
  --lora \
  --lora-rank  8 \
  --lora-alpha 16.0
```

```bash
# What a MISMATCH looks like — client omits --lora, or uses a different
# rank/alpha than the server. The server logs the mismatch and drops the
# connection; the client logs the server's rejection reason and falls back
# to local-only training instead of sending deltas that won't fit.
./bin/client \
  --server     127.0.0.1 \
  --port       8080 \
  --model      SLM/gpt2-small-f32.gguf \
  --dataset    Dataset/client_C \
  --iters      50 --batch 4 --block 512 --lr 1e-4 \
  --client-id  client_C \
  --lora \
  --lora-rank  16          # ← server is rank 8, this client is rank 16
```

---

## Resuming a FedAvg session

```bash
# Server — resume the global model from a previous run, keep going for 10 more rounds
./bin/server \
  --model     SLM/gpt2-small-f32.gguf \
  --port      8080 \
  --clients   2 \
  --rounds    10 \
  --load-path checkpoints/global/model.tnsf
```

```bash
# Client — resume its own student too (mainly matters if it was also
# trained locally between federated sessions)
./bin/client \
  --server     127.0.0.1 \
  --port       8080 \
  --model      SLM/gpt2-small-f32.gguf \
  --dataset    Dataset/client_A \
  --iters      50 --batch 4 --block 512 --lr 1e-4 \
  --rounds     10 \
  --client-id  client_A \
  --load-path  checkpoints/client_client_A/student.tnsf
```

```bash
# Resuming a LoRA session — --lora/--lora-rank/--lora-alpha must still match
# on every resume, same as a fresh session; the checkpoint only carries the
# adapter weights, not which rank/alpha it was trained with.
./bin/server \
  --model      SLM/gpt2-small-f32.gguf \
  --port       8080 \
  --clients    2 \
  --rounds     10 \
  --lora \
  --lora-rank  8 \
  --lora-alpha 16.0 \
  --load-path  checkpoints/global/model.tnsf
```

---

## FedDistill — logit-exchange rounds

```bash
# Terminal 1 — server in FedDistill mode.
# Server averages client logits → broadcasts consensus back.
# Global model weights are NOT updated server-side (pure distill exchange),
# so the server has nothing of its own worth checkpointing in this mode —
# see run_feddistill_round()'s comment for why.
./bin/server \
  --model      SLM/gpt2-small-f32.gguf \
  --port       8080 \
  --clients    2 \
  --feddistill \
  --prompt     "the data type is int" \
  --tokens     30
```

```bash
# Terminal 2 — client A in FedDistill mode.
# Computes proxy-batch logits → sends → receives consensus → distill_logits().
# Each client's student is personalized, so each saves its own checkpoint —
# --client-id keeps client A's and client B's files apart.
./bin/client \
  --server     127.0.0.1 \
  --port       8080 \
  --model      SLM/gpt2-small-f32.gguf \
  --dataset    Dataset/client_A \
  --iters      50 \
  --batch      4 \
  --block      512 \
  --lr         1e-4 \
  --client-id  client_A \
  --feddistill
```

```bash
# Terminal 3 — client B in FedDistill mode
./bin/client \
  --server     127.0.0.1 \
  --port       8080 \
  --model      SLM/gpt2-small-f32.gguf \
  --dataset    Dataset/client_B \
  --iters      50 \
  --batch      4 \
  --block      512 \
  --lr         1e-4 \
  --client-id  client_B \
  --feddistill
```

---

## FedDistill — heterogeneous clients (key advantage of logit exchange)

Different model files, same vocab/proxy-batch shape. FedAvg can't do this —
FedDistill can, because only logits are exchanged, not weights.

```bash
./bin/server \
  --model      SLM/gpt2-small-f32.gguf \
  --port       8080 \
  --clients    2 \
  --rounds     20 \
  --feddistill
```

```bash
# Small model client — its own checkpoint, its own architecture
./bin/client \
  --server     127.0.0.1 \
  --port       8080 \
  --model      SLM/gpt2-small-f32.gguf \
  --dataset    Dataset/client_A \
  --batch      4 --block 512 --lr 1e-4 \
  --rounds     20 \
  --client-id  small_client \
  --feddistill
```

```bash
# Larger model client — different architecture, same vocab (50257)
./bin/client \
  --server     127.0.0.1 \
  --port       8080 \
  --model      SLM/gpt2-medium-f32.gguf \
  --dataset    Dataset/client_B \
  --batch      2 --block 512 --lr 5e-5 \
  --rounds     20 \
  --client-id  medium_client \
  --feddistill
```

Note: `--lora` is orthogonal to FedDistill's heterogeneity story — the LoRA
handshake still requires an exact rank/alpha match, so it doesn't buy you
anything extra here beyond a smaller local optimizer footprint per client.
FedDistill's own logit-exchange path already tolerates different backbones
without any LoRA involvement.

---

## Compressed checkpoints (the "smaller distilled model" path)

```bash
# Client — save an fp4 checkpoint (≈8x smaller, nibble-packed) every 5 rounds,
# alongside the resumable float one
./bin/client \
  --server     127.0.0.1 \
  --port       8080 \
  --model      SLM/gpt2-small-f32.gguf \
  --dataset    Dataset/client_A \
  --batch      4 --block 512 --lr 1e-4 \
  --rounds     50 \
  --client-id  client_A \
  --save-every 5 \
  --quantize   fp4
# → checkpoints/client_client_A/student.tnsf      (float, resumable)
# → checkpoints/client_client_A/student.tnsf.fp4   (compressed, deployment-sized)
```

```bash
# Server — fp8 checkpoint of the FedAvg global model every round
./bin/server \
  --model      SLM/gpt2-small-f32.gguf \
  --port       8080 \
  --clients    2 \
  --rounds     10 \
  --quantize   fp8
```

---

## Quick smoke test (fast iteration)

```bash
# --save-every 0 on both sides keeps this from writing checkpoints at all
./bin/server \
  --model   SLM/gpt2-small-f32.gguf \
  --port    8080 \
  --clients 1 \
  --rounds  3 \
  --no-profile \
  --quiet \
  --save-every 0
```

```bash
./bin/client \
  --server  127.0.0.1 \
  --port    8080 \
  --model   SLM/gpt2-small-f32.gguf \
  --dataset Dataset \
  --iters   5 \
  --eval    1 \
  --batch   1 \
  --block   64 \
  --rounds  3 \
  --no-profile \
  --save-every 0
```

```bash
# Same smoke test, LoRA — quick sanity check that the handshake and
# adapter-only exchange both work before running a real session
./bin/server \
  --model   SLM/gpt2-small-f32.gguf \
  --port    8080 \
  --clients 1 \
  --rounds  3 \
  --no-profile \
  --quiet \
  --save-every 0 \
  --lora \
  --lora-rank 4 \
  --lora-alpha 8.0
```

```bash
./bin/client \
  --server  127.0.0.1 \
  --port    8080 \
  --model   SLM/gpt2-small-f32.gguf \
  --dataset Dataset \
  --iters   5 \
  --eval    1 \
  --batch   1 \
  --block   64 \
  --rounds  3 \
  --no-profile \
  --save-every 0 \
  --lora \
  --lora-rank 4 \
  --lora-alpha 8.0
```

---

## Profiling + JSON logging

```bash
./bin/server \
  --model    SLM/gpt2-small-f32.gguf \
  --port     8080 \
  --clients  2 \
  --rounds   5 \
  --json-log          # writes server_profile.json
```

```bash
./bin/client \
  --server   127.0.0.1 \
  --port     8080 \
  --model    SLM/gpt2-small-f32.gguf \
  --dataset  Dataset \
  --iters    50 \
  --batch    4 --block 512 --lr 1e-4 \
  --rounds   5 \
  --json-log          # writes client_profile.json
```

---

## Mode summary

| Scenario                     | Server flag                                    | Client flag                                    | Wire payload        | Checkpoint owner                  |
|-------------------------------|-------------------------------------------------|-------------------------------------------------|----------------------|------------------------------------|
| Local finetune                | —                                                | `--no-federated`                                | nothing              | client (`--save-path`)            |
| Local LoRA finetune           | —                                                | `--no-federated --lora`                         | nothing              | client (`--save-path`, adapters only) |
| FedAvg                        | (default)                                        | (default)                                        | full weight deltas   | **server** — global model         |
| FedAvg, N rounds              | `--rounds N`                                     | `--rounds N` (for a clean exit)                 | full weight deltas   | **server** — global model         |
| FedAvg, LoRA                  | `--lora --lora-rank R --lora-alpha A`            | `--lora --lora-rank R --lora-alpha A` (must match) | adapter deltas only | **server** — global adapters      |
| FedDistill                    | `--feddistill`                                   | `--feddistill`                                   | logits only           | **each client** — its own student |
| FedDistill, heterogeneous     | `--feddistill`                                   | `--feddistill` + different `--model`             | logits only           | **each client** — its own student |
| Compressed checkpoint         | `--quantize fp8\|fp4`                            | `--quantize fp8\|fp4`                            | unchanged              | same as above + a `.fp8`/`.fp4` file |
| Resume                        | `--load-path PATH`                               | `--load-path PATH`                               | unchanged              | same as above                     |

**FedDistill's hard constraint is unchanged:** every client and the server
must agree on vocab size (50257 for the GPT-2 family) and use the same
`--batch`×`--block` proxy-batch shape, since that determines the logit
tensor dimensions on the wire. Weights never leave the client in FedDistill
mode.

**LoRA's hard constraint:** every client and the server must agree on
`--lora` (on/off), `--lora-rank`, and `--lora-alpha` exactly. This is
enforced automatically — on connect, the client sends its config and the
server accepts or rejects before the client is allowed to send any round
data. A rejected client prints the server's mismatch reason and continues
in local-only mode rather than exiting, so a typo'd flag doesn't kill an
otherwise-fine local training run.