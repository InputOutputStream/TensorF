# TensorF: language models

Run GPT-2 and Llama-family (SmolLM2) models from GGUF files, train a small model from scratch, and fine-tune across several machines with federated learning. Back to the framework: [README.md](README.md).

## What is supported

| | |
|---|---|
| Models | GPT-2 (`GPT<T, LinearT>`), Llama-style (`Llama<T, LinearT>`) |
| Weights | GGUF with F32, F16, Q8_0 or Q4_0 tensors, loaded into `float` |
| Tokenizers | GPT-2 byte-level BPE (`vocab.json` + `merges.txt`), Llama / SmolLM (from the GGUF) |
| Sampling | temperature + top-k |
| Adapters | `LoRALinear` for adapter-only training and exchange |
| Training | local loop (AdamW), FedAvg, FedDistill |

Everything runs on CPU in `float`. Quantized GGUF tensors are dequantized on load, so a Q4 file saves disk and download, not RAM.

## Setup

```bash
sudo apt-get install -y build-essential cmake pkg-config libopenblas-dev nlohmann-json3-dev   # json is required by the tokenizer
git clone https://github.com/InputOutputStream/TensorF.git && cd TensorF
mkdir -p SLM && ./scripts/download_llms.sh          # GPT-2 small: F32 654 MB, Q8_0 178 MB, Q4_K_S 135 MB
```

Run the script from the repo root (it does `cd SLM`). It also expects the tokenizer files in `SLM/gpt2-tokenizer/{vocab.json,merges.txt}`. For SmolLM2, download the F16 GGUF from Hugging Face and save it as `SLM/lSmolLM2-135M-Instruct-f16.gguf` (the name `examples/SmollLLM.cpp` loads). GPT-2 Medium/Large/XL lines in the script are commented out. Use the F32 or Q8_0 file: the script's Q4_K_S file uses a k-quant format that the GGUF reader does not load (see Current state).

## Inference

```bash
make gpt2   && ./bin/gpt2        # "The data type is int", 30 tokens, T=0.8, top-k 30
make smollm && ./bin/smollm      # SmolLM2-135M with the chat template, 30 tokens, T=0.6, top-k 40
```

In code (this is `examples/GPT2.cpp`):

```cpp
GPT2HyperParams hp{ .vocab_size = 50257, .d_model = 768, .block_size = 1024, .n_layer = 12, .n_head = 12 };
GPTGGUFLoader<float, Linear> loader;
auto model = loader.load_model("SLM/gpt2-small-f32.gguf", hp);
auto tok   = loader.load_tokenizer();

auto ids = tok.encode("The data type is int");
Tensor_t<float> ctx = make_tensor<float>(
    Matrix<float>(std::vector<float>(ids.begin(), ids.end()), {1, ids.size()}));

auto out = model.generate(ctx, /*max_new_tokens=*/30, /*temperature=*/0.8f, /*top_k=*/30);
```

Llama: `LlamaHyperParams` + `LlamaGGUFLoader<float, Linear>`, same pattern; the tokenizer comes from `loader.load_tokenizer()`.

Sizes to plan for: weights are held as `float`, so GPT-2 small needs about 0.5 GB for parameters alone and SmolLM2-135M about the same; training adds gradients and Adam state (about 3 more copies).

## Train a small model from scratch

```bash
make transformer && ./bin/transformer
```

Trains a character-level Llama on `Datasets/`, saves `Models/Llama.hge`, then generates 50 characters from "the data type is int". Edit the constants at the top of `examples/transformer.cpp` (heads, layers, width, block size, batch size). `TextDataset::load()` writes `train_split.txt`, `val_split.txt`, `vocab.txt` into the dataset folder.

For your own loop:

```cpp
Llama<float, Linear> model(vocab, dim, block, n_heads, n_layers, ffn_hidden);
model.train(get_batch, /*iters=*/1000, /*eval_interval=*/100);   // AdamW, lr 1e-4
model.save("Models/Llama.hge");
```

`train()` fixes AdamW and lr 1e-4. For anything else write the loop yourself: `zero_grad` → `forward(x, y)` (returns the loss when targets are given) → `backward` → `step` → `loss->reset_graph()`.

## Fine-tuning with LoRA

Construct the model with `LoRALinear` instead of `Linear` (`GPT<float, LoRALinear>(…, rank, alpha)`), load the dense backbone from GGUF, and train only `lora_parameters()`. Federated runs do this with `--lora`.

## Federated training

Clients train locally and only weights (FedAvg) or logits (FedDistill) cross the network; raw data stays on each machine.

| | FedAvg | FedDistill |
|---|---|---|
| Sent each round | weights | logits on a shared probe batch |
| Use when | clients share one architecture | clients differ (e.g. a LoRA student per client) |

Weights stream in chunks (`--chunk-mb`), so a memory-limited client never holds a second full copy of the model.

```bash
# terminal 1
./bin/server --port 8080 --clients 2 --rounds 5 --model SLM/gpt2-small-f32.gguf

# terminals 2 and 3 (one --client-id each when on the same machine)
./bin/client --server 127.0.0.1 --port 8080 --client-id a --model SLM/gpt2-small-f32.gguf \
             --dataset Datasets/client1 --iters 100 --batch 4 --block 512 --lr 1e-4
./bin/client --server 127.0.0.1 --port 8080 --client-id b --model SLM/gpt2-small-f32.gguf \
             --dataset Datasets/client2 --iters 100 --batch 4 --block 512 --lr 1e-4
```

Switches (the same on server and clients unless noted):

| Flag | Effect |
|---|---|
| `--lora --lora-rank 8 --lora-alpha 16` | exchange only adapter weights; **must match** on every process or the server rejects the client (it then trains locally) |
| `--feddistill` | logit exchange instead of weights |
| `--save-path P --load-path P --save-every N` | checkpoints; default `checkpoints/global/model.tnsf` (server) and `checkpoints/client_<id>/student.tnsf` (client), every round; `--save-every 0` disables |
| `--quantize fp8` / `fp4` | also writes `<path>.fp8` / `.fp4`, about 4× smaller; resume from the float file |
| `--rounds N` | stop after N rounds |
| `--no-federated` (client) | local fine-tune only |
| `--no-profile`, `--json-log`, `--quiet` | skip hardware profiling, write `client_profile.json`, silence per-iteration output |

Every combination (resume, local-only, FedDistill with mixed clients, smoke tests) is written out in `tests/GPT/Modes.md`.

## Current state

- **`server` and `client` do not compile right now.** `tests/GPT/Server.cpp` and `Client.cpp` still include `Profiler/Profiler.hpp` and `Profiler/HyperparamAdvisor.hpp` from the old profiler, and `DataLoader/DataLoading.hpp`; `Trainner.hpp` includes `DataLoader/Quantizer.hpp` (those files are now directly under `include/data/`). They must be moved to the new profiler API (`collect()`, `run_device_benchmark`, `plan_budget`, `HyperparamAdvisor`) before `make server client` works. The networking classes in `include/net/Network/` are fine. The profiler wire protocol in `include/net/Protocol/` has the same problem.
- `Llama` attention does not implement grouped-query attention with a separate KV head count; check a checkpoint's K/V shapes before loading it.
- `Llama::generate` stops on EOS (`id 2`) only for the first batch element.
- `GPT::train` and `Llama::train` hard-code AdamW at lr 1e-4, and `Optimizer::step` clips gradients to norm 1.0.
- GGUF reading supports F32, F16, Q8_0 and Q4_0; other quantization types (e.g. Q4_K) are not loaded as such.

## On a phone

Federated clients are the intended Android use: the profiler sizes the batch to the device's memory budget, chunked transfer bounds peak RAM, and LoRA keeps the exchanged state small. Build and device setup: [README_ANDROID.md](README_ANDROID.md).
