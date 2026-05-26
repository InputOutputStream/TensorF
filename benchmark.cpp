/*
 * main.cpp — TensorF Client Profiler
 * ====================================
 * Orchestrates the full profiling pipeline:
 *   1. Hardware detection + benchmarks   (Profiler::run)
 *   2. Optional workload memory profiling (profile_load / profile_train_step)
 *   3. JSON summary to stdout            (Profiler::to_json)
 *   4. Binary wire serialization         (ProfilerProtocol)
 *   5. Transport dispatch                (send_profile)
 *
 * Build (single-translation-unit, all headers are #pragma once):
 *   g++ -O2 -std=c++20 -o tensorf_profiler benchmark.cpp -lpthread -lblas
 *
 * Run:
 *   ./tensorf_profiler                          # local Unix socket (default)
 *   ./tensorf_profiler --host 192.168.1.10      # TCP to remote server
 *   ./tensorf_profiler --udp                    # UDP broadcast
 *   ./tensorf_profiler --json-only              # no network, just print JSON
 *   ./tensorf_profiler --quiet                  # suppress benchmark progress
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <functional>
#include <chrono>
#include <unistd.h>
#include <sys/types.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// ── TensorF profiler headers (all header-only) ────────────────────────────────
#include "Profiler/Profiler.hpp"
#include "Profiler/HyperparamAdvisor.hpp"   // for QuantPolicy in workload lambdas
#include "Profiler/protocol/ProfilerProtocol.hpp"
#include "Profiler/protocol/ProfilerTransport.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

// Derive a stable 64-bit client ID from the machine's hostname + primary MAC.
// Not cryptographically secure; just needs to be stable across reboots and
// unique within a federation of hundreds of nodes.
static uint64_t make_client_id() {
    // Mix hostname bytes
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);

    // Try to read the MAC of the first non-loopback interface via ioctl
    uint8_t mac[6] = {};
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock >= 0) {
        struct ifreq ifr;
        // Common interface names in priority order
        const char* ifaces[] = {"eth0","enp0s3","ens3","wlan0","en0", nullptr};
        for (int i = 0; ifaces[i]; i++) {
            memset(&ifr, 0, sizeof(ifr));
            strncpy(ifr.ifr_name, ifaces[i], IFNAMSIZ - 1);
            if (ioctl(sock, SIOCGIFHWADDR, &ifr) == 0) {
                memcpy(mac, ifr.ifr_hwaddr.sa_data, 6);
                break;
            }
        }
        close(sock);
    }

    // FNV-1a 64-bit over (hostname + mac)
    uint64_t h = 14695981039346656037ULL;
    auto fnv = [&](const uint8_t* p, size_t n) {
        for (size_t i = 0; i < n; i++)
            h = (h ^ p[i]) * 1099511628211ULL;
    };
    fnv(reinterpret_cast<const uint8_t*>(host), strlen(host));
    fnv(mac, 6);
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ─────────────────────────────────────────────────────────────────────────────

struct CLIOptions {
    std::string server_host  = "127.0.0.1";
    uint16_t    tcp_port     = DEFAULT_TCP_PORT;
    uint16_t    udp_port     = DEFAULT_UDP_PORT;
    std::string unix_path    = DEFAULT_UNIX_PATH;

    bool        json_only    = false;   // skip network, only print JSON
    bool        quiet        = false;   // suppress benchmark progress output
    bool        use_udp      = false;   // force UDP transport only
    bool        use_tcp      = false;   // force TCP transport only
    bool        internet     = false;   // use xxHash (better for WAN)
    bool        dry_run      = false;   // run benchmarks but don't send
    size_t      vocab_size   = 50257;   // GPT-2 default
    bool        help         = false;
};

static void print_usage(const char* prog) {
    printf(
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --host <ip>          Server IP for TCP/UDP (default: 127.0.0.1)\n"
        "  --tcp-port <port>    TCP port (default: %u)\n"
        "  --udp-port <port>    UDP port (default: %u)\n"
        "  --unix <path>        Unix socket path (default: %s)\n"
        "  --tcp                Force TCP transport only\n"
        "  --udp                Force UDP transport only\n"
        "  --internet           Use xxHash checksum (better for WAN)\n"
        "  --json-only          Print JSON to stdout, skip network\n"
        "  --dry-run            Run benchmarks but do not send\n"
        "  --quiet              Suppress benchmark progress output\n"
        "  --vocab <n>          Vocabulary size for RAM estimates (default: 50257)\n"
        "  --help               Show this help\n"
        "\n"
        "Transport fallback order (default): Unix → TCP → UDP\n",
        prog,
        DEFAULT_TCP_PORT, DEFAULT_UDP_PORT, DEFAULT_UNIX_PATH
    );
}

static CLIOptions parse_args(int argc, char* argv[]) {
    CLIOptions opts;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--help")         opts.help       = true;
        else if (a == "--json-only")    opts.json_only  = true;
        else if (a == "--quiet")        opts.quiet      = true;
        else if (a == "--tcp")          opts.use_tcp    = true;
        else if (a == "--udp")          opts.use_udp    = true;
        else if (a == "--internet")     opts.internet   = true;
        else if (a == "--dry-run")      opts.dry_run    = true;
        else if (a == "--host"   && i+1 < argc) opts.server_host = argv[++i];
        else if (a == "--unix"   && i+1 < argc) opts.unix_path   = argv[++i];
        else if (a == "--tcp-port" && i+1 < argc) opts.tcp_port  = (uint16_t)atoi(argv[++i]);
        else if (a == "--udp-port" && i+1 < argc) opts.udp_port  = (uint16_t)atoi(argv[++i]);
        else if (a == "--vocab"  && i+1 < argc) opts.vocab_size  = (size_t)atoi(argv[++i]);
        else {
            fprintf(stderr, "[main] Unknown option: %s\n", a.c_str());
            opts.help = true;
        }
    }
    return opts;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    CLIOptions opts = parse_args(argc, argv);
    if (opts.help) { print_usage(argv[0]); return 0; }

    auto wall_start = std::chrono::steady_clock::now();

    // ── 1. Run the full profiling pipeline ───────────────────────────────────
    Profiler profiler;
    profiler.run(opts.vocab_size, /*verbose=*/!opts.quiet);

    // ── 2. Profile actual workloads using synthetic data sized to the advised config ──
    //
    // These lambdas simulate the three lifecycle stages of a transformer model
    // (load, train step, inference) using the HyperparamConfig the advisor produced.
    // No real model is needed: we allocate and operate on tensors of the exact
    // same shape a real model would use, so the RSS deltas are accurate.

    const HyperparamConfig& cfg = profiler.config();
    const size_t B  = cfg.batch_size;
    const size_t T  = cfg.block_size;
    const size_t C  = cfg.n_embed;
    const size_t L  = cfg.n_layers;
    const size_t V  = opts.vocab_size;

    // Byte-width per element (respects advised quantization)
    const size_t bpp = [&]() -> size_t {
        switch (cfg.algo.quant) {
            case QuantPolicy::FP16:
            case QuantPolicy::FP8_E4M3:
            case QuantPolicy::FP8_E5M2: return 2;
            case QuantPolicy::INT8:     return 1;
            case QuantPolicy::INT4:     return 1;  // packed; real size is 0.5B/elem
            default:                    return 4;  // float32
        }
    }();

    if (!opts.quiet) {
        printf("\n[Workload] Simulating model lifecycle (B=%zu T=%zu C=%zu L=%zu V=%zu bpp=%zu)\n",
               B, T, C, L, V, bpp);
    }

    // ── STAGE: LOADED — allocate weight tensors (parameters only, no activations) ──
    //
    // Standard transformer parameter layout:
    //   token embedding:  V × C
    //   position embed:   T × C
    //   per layer: Q/K/V/O projections (4 × C×C), FFN up/down/gate (3 × C×4C),
    //              2 LayerNorm (γ+β, each C)
    //   final LN + lm_head: C + C + V×C

    profiler.profile_load([&]() {
        size_t tok_embed  = V * C * bpp;
        size_t pos_embed  = T * C * bpp;
        size_t per_layer  = (4 * C * C          // QKV + O projections
                           + 3 * C * (4 * C)    // FFN: gate, up, down
                           + 4 * C) * bpp;      // 2 LayerNorm × (γ+β)
        size_t lm_head    = V * C * bpp + 2 * C * bpp;  // weight + final LN

        size_t total_bytes = tok_embed + pos_embed + L * per_layer + lm_head;

        if (!opts.quiet)
            printf("[Workload] Allocating %.1f MB of parameter tensors...\n",
                   total_bytes / 1024.0 / 1024.0);

        // Allocate and touch every page so the OS actually commits the memory.
        // Using volatile to prevent the compiler from eliding the writes.
        auto params = std::make_unique<uint8_t[]>(total_bytes);
        volatile uint8_t* p = params.get();
        // Touch in 4KB strides (one write per page is enough to commit)
        for (size_t i = 0; i < total_bytes; i += 4096)
            p[i] = static_cast<uint8_t>(i & 0xFF);
        // Force the allocation to live until the snapshot is taken
        p[total_bytes - 1] = 0xFF;

        // Keep params alive for the snapshot — release after the lambda returns.
        // The snapshot is taken by profile_load() after workload() returns.
        // We intentionally do NOT release here so the RSS reflects loaded state.
        params.release();  // intentional: simulates weights staying in RAM
    });

    // ── STAGE: TRAIN — forward + backward + gradient/optimizer state ──────────
    //
    // Training RSS = params + activations (forward) + gradients + Adam m1+m2
    //   Activations per layer:  B × T × C × 4 (residuals, attn scores, FFN intermediates)
    //   Gradients:              same size as params
    //   Adam m1 + m2:           2 × params size

    profiler.profile_train_step([&]() {
        size_t per_layer  = (4 * C * C + 3 * C * 4 * C + 4 * C) * bpp;
        size_t param_bytes = (V * C + T * C + L * per_layer + V * C) * bpp;

        // Activations: B × T × C × factor (4 covers: input, after attn, after FFN, residual)
        size_t act_bytes   = B * T * C * 4 * sizeof(float);  // always float32 for gradients
        // Gradients: same shape as params
        size_t grad_bytes  = param_bytes;
        // Adam moments: 2 × params
        size_t adam_bytes  = 2 * param_bytes;

        size_t total_bytes = act_bytes + grad_bytes + adam_bytes;

        if (!opts.quiet)
            printf("[Workload] Allocating %.1f MB for activations + gradients + Adam state...\n",
                   total_bytes / 1024.0 / 1024.0);

        auto train_buf = std::make_unique<uint8_t[]>(total_bytes);
        volatile uint8_t* p = train_buf.get();

        // Simulate a forward pass: sequential writes (activations building up)
        for (size_t i = 0; i < act_bytes; i += 4096)
            p[i] = static_cast<uint8_t>(i);

        // Simulate backward pass: gradient accumulation
        size_t grad_start = act_bytes;
        for (size_t i = 0; i < grad_bytes; i += 4096)
            p[grad_start + i] = static_cast<uint8_t>(i ^ 0xAA);

        // Simulate Adam update: read+write moment buffers
        size_t adam_start = act_bytes + grad_bytes;
        for (size_t i = 0; i < adam_bytes; i += 4096) {
            p[adam_start + i] = static_cast<uint8_t>(p[grad_start + (i % grad_bytes)] + 1);
        }

        p[total_bytes - 1] = 0x01;
        train_buf.release();  // keep RSS elevated for snapshot
    });

    // ── STAGE: INFER — forward pass only (KV cache + activations, no grads) ───
    //
    // Inference RSS = params + KV cache (per-layer K and V for all T positions)
    //                + activations (one layer at a time → smaller than training)
    //   KV cache: 2 × L × T × C × bpp  (K and V for every layer)
    //   Activations: B × T × C × 2 (input + one layer's output, reused)

    profiler.profile_infer_step([&]() {
        // KV cache: K and V for each layer, full context length
        size_t kv_cache_bytes = 2 * L * T * C * bpp;

        // Activations during inference: much smaller than training (no grad storage)
        // Just input embeddings + one layer's intermediate buffers (reused per layer)
        size_t act_bytes = B * T * C * 2 * sizeof(float);

        size_t total_bytes = kv_cache_bytes + act_bytes;

        if (!opts.quiet)
            printf("[Workload] Allocating %.1f MB for KV cache + inference activations...\n",
                   total_bytes / 1024.0 / 1024.0);

        auto infer_buf = std::make_unique<uint8_t[]>(total_bytes);
        volatile uint8_t* p = infer_buf.get();

        // Simulate autoregressive decode: fill KV cache token by token
        for (size_t tok = 0; tok < T; tok++) {
            size_t offset = tok * C * bpp;
            if (offset + C * bpp <= kv_cache_bytes) {
                for (size_t c = 0; c < C * bpp; c += 64)   // 64B = cache line
                    p[offset + c] = static_cast<uint8_t>(tok ^ c);
            }
        }

        // Simulate activation reuse per layer
        for (size_t layer = 0; layer < L; layer++) {
            size_t act_offset = kv_cache_bytes;
            for (size_t i = 0; i < act_bytes; i += 4096)
                p[act_offset + i] = static_cast<uint8_t>(layer);
        }

        p[total_bytes - 1] = 0x02;
        infer_buf.release();  // keep RSS elevated for snapshot
    });

    // ── 3. Human-readable summary ────────────────────────────────────────────
    if (!opts.quiet)
        profiler.print_summary();

    // ── 4. JSON output ───────────────────────────────────────────────────────
    std::string json = profiler.to_json();

    if (opts.json_only || opts.dry_run) {
        printf("%s\n", json.c_str());
        if (opts.json_only) return 0;
    }

    // ── 5. Binary serialization ──────────────────────────────────────────────
    uint64_t client_id = make_client_id();

    ChecksumType chk = opts.internet
        ? ChecksumType::XXHASH
        : ChecksumType::CRC32;

    ProtoFlags flags = ProtoFlags::NONE;
    // Signal that memory phase snapshots are live (baseline at minimum)
    flags = flags | ProtoFlags::HAS_MEMORY_PHASES;

    WireBuffer wb = serialize(
        profiler.fingerprint(),
        profiler.benchmarks(),
        profiler.config(),
        profiler.memory_report(),
        client_id,
        chk,
        flags
    );

    if (!opts.quiet) {
        printf("\n[Serializer] Wire buffer: %zu bytes (JSON equivalent: %zu bytes, %.1fx smaller)\n",
               wb.size, json.size(), (double)json.size() / (double)wb.size);
        print_wire(wb);   // debug dump from ProfilerProtocol.hpp
    }

    if (opts.dry_run) {
        printf("[main] Dry-run: serialization OK, skipping network send.\n");
        return 0;
    }

    // ── 6. Transport dispatch ─────────────────────────────────────────────────
    TransportConfig tcfg;
    tcfg.server_host   = opts.server_host;
    tcfg.tcp_port      = opts.tcp_port;
    tcfg.udp_port      = opts.udp_port;
    tcfg.unix_path     = opts.unix_path;
    tcfg.internet_mode = opts.internet;

    // If user forced a specific transport, disable the others
    if (opts.use_udp) {
        tcfg.try_unix = false;
        tcfg.try_tcp  = false;
        tcfg.try_udp  = true;
    } else if (opts.use_tcp) {
        tcfg.try_unix = false;
        tcfg.try_tcp  = true;
        tcfg.try_udp  = false;
    }
    // Default: try_unix=true, try_tcp=true, try_udp=true (fallback chain)

    bool sent = send_profile(wb, tcfg);

    auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - wall_start).count();

    if (!opts.quiet) {
        printf("\n[main] Total profiling time: %lld ms\n", (long long)wall_ms);
        printf("[main] Profile send: %s\n", sent ? "OK" : "FAILED");
    }

    return sent ? 0 : 1;
}