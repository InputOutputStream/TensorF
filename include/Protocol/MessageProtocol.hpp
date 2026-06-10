#pragma once

/*
 * ProfilerProtocol.hpp
 * ====================
 * Binary wire protocol for sending profiler results from client to server.
 *
 * Design goals:
 *   - Zero allocation on read path (direct cast to struct)
 *   - Zero parsing (no string scanning, no branches per field)
 *   - Transport-agnostic: same bytes over TCP / Unix socket / UDP
 *   - Integrity: optional CRC32 (LAN) or xxHash32 (WAN/internet)
 *   - Extensible: VERSION field + FLAGS bitmask + RESERVED bytes
 *
 * Wire layout (little-endian throughout):
 *
 *   ┌──────────────────────────────────────────┐
 *   │  ProfilerHeader   (64 bytes, fixed)       │
 *   ├──────────────────────────────────────────┤
 *   │  ProfilerBody     (~224 bytes, fixed)     │
 *   ├──────────────────────────────────────────┤
 *   │  StorageEntry[]   (32 bytes × N, variable)│
 *   └──────────────────────────────────────────┘
 *
 * Total minimum size: 64 + 224 = 288 bytes
 * With 4 storage devices: 288 + 128 = 416 bytes
 *
 * Compare to JSON equivalent: ~1800–2400 bytes
 */

#include <cstdint>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <string>

// ─── Compile-time layout guarantees ─────────────────────────────────────────

// All structs use __attribute__((packed)) to prevent padding surprises.
// On MSVC use #pragma pack(1) — we target GCC/Clang here.

#define TF_PACKED __attribute__((packed))

// ─── Constants ───────────────────────────────────────────────────────────────

static constexpr uint32_t PROTO_MAGIC   = 0x54465052; // "TFPR" (TensorF PRofiler)
static constexpr uint8_t  PROTO_VERSION = 0x01;

// Message types
enum class MsgType : uint8_t {
    PROFILE_REPORT   = 0x01,  // client → server: full profiler result
    PROFILE_ACK      = 0x02,  // server → client: acknowledged + override config
    PING             = 0x03,
    PONG             = 0x04,
};

// Checksum types
enum class ChecksumType : uint8_t {
    NONE    = 0x00,  // no checksum (trusted LAN, Unix socket)
    CRC32   = 0x01,  // CRC32 — fast, good enough for LAN
    XXHASH  = 0x02,  // xxHash32 — faster than CRC32, better for WAN/internet
};

// Flags bitmask (FLAGS field in header)
enum class ProtoFlags : uint8_t {
    NONE              = 0x00,
    HAS_MEMORY_PHASES = 0x01,  // loaded/train/infer snapshots are valid
    WANTS_OVERRIDE    = 0x02,  // client asks server to override its config
    IS_RETRY          = 0x04,  // this is a retransmission
    INTERNET_MODE     = 0x08,  // checksum is xxHash instead of CRC32
};

inline ProtoFlags operator|(ProtoFlags a, ProtoFlags b) {
    return static_cast<ProtoFlags>(
        static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}
inline bool has_flag(ProtoFlags flags, ProtoFlags bit) {
    return (static_cast<uint8_t>(flags) & static_cast<uint8_t>(bit)) != 0;
}

// ISA encoding (fits in 4 bits)
enum class ISACode : uint8_t {
    BASELINE    = 0,
    AVX         = 1,
    AVX2        = 2,
    AVX512      = 3,
    AVX512_VNNI = 4,
    AMX         = 5,
};

// CPU generation encoding
enum class CPUGenCode : uint8_t {
    UNKNOWN             = 0,
    INTEL_PRE_HASWELL   = 1,
    INTEL_HASWELL       = 2,
    INTEL_SKYLAKE       = 3,
    INTEL_ICE_LAKE      = 4,
    INTEL_ALDER_LAKE    = 5,
    INTEL_SAPPHIRE_RAPIDS = 6,
    AMD_ZEN1            = 10,
    AMD_ZEN2            = 11,
    AMD_ZEN3            = 12,
    AMD_ZEN4            = 13,
    AMD_ZEN5            = 14,
};

// RAM type encoding
enum class RAMTypeCode : uint8_t {
    UNKNOWN = 0,
    DDR3    = 1,
    DDR4    = 2,
    DDR5    = 3,
    LPDDR4  = 4,
    LPDDR5  = 5,
};

// Storage type encoding
enum class StorageCode : uint8_t {
    UNKNOWN   = 0,
    HDD       = 1,
    SATA_SSD  = 2,
    NVME_GEN3 = 3,
    NVME_GEN4 = 4,
    NVME_GEN5 = 5,
};

// Quantization policy encoding
enum class QuantCode : uint8_t {
    NONE    = 0,
    FP16    = 1,
    INT8    = 2,
    FP8_E4M3 = 3,
    FP8_E5M2 = 4,
    INT4    = 5,
};

// ─── Header (64 bytes, fixed) ─────────────────────────────────────────────────

struct TF_PACKED ProfilerHeader {
    uint32_t magic;           //  4B  — PROTO_MAGIC = 0x54465052
    uint8_t  version;         //  1B  — PROTO_VERSION
    uint8_t  flags;           //  1B  — ProtoFlags bitmask
    uint8_t  msg_type;        //  1B  — MsgType
    uint8_t  num_storage;     //  1B  — number of StorageEntry records
    uint64_t client_id;       //  8B  — unique client identifier (hash of MAC/hostname)
    uint64_t timestamp_us;    //  8B  — unix timestamp in microseconds
    uint32_t body_len;        //  4B  — bytes of ProfilerBody + StorageEntry[]
    uint8_t  checksum_type;   //  1B  — ChecksumType
    uint8_t  _pad[3];         //  3B  — padding to align checksum to 4B
    uint32_t checksum;        //  4B  — CRC32 or xxHash32 of body bytes, 0 if NONE
    uint8_t  reserved[28];    // 28B  — future use, must be zero
                              // ─────
                              // 64B total
};
static_assert(sizeof(ProfilerHeader) == 64, "ProfilerHeader must be 64 bytes");

// ─── Body fixed section (~224 bytes) ─────────────────────────────────────────

struct TF_PACKED ProfilerBody {

    // ── CPU (16 bytes) ────────────────────────────────────────────────────────
    uint8_t  cpu_gen;           // CPUGenCode
    uint8_t  cpu_isa;           // ISACode (best available)
    uint8_t  cpu_vendor;        // 0=Intel 1=AMD 2=other
    uint8_t  cpu_flags;         // bit0=avx bit1=avx2 bit2=fma bit3=avx512f
                                // bit4=vnni bit5=amx bit6=hyper_threading
    uint8_t  cpu_phys_cores;    // physical core count (capped at 255)
    uint8_t  cpu_logic_cores;   // logical core count
    uint16_t cpu_max_freq_100mhz; // max freq in units of 100MHz (4200=4.2GHz)
    uint8_t  embed_alignment;   // floats per vector register (4/8/16)
    uint8_t  capability_score;  // 0–100 overall hardware score
    uint8_t  _pad_cpu[6];

    // ── Cache (12 bytes) ──────────────────────────────────────────────────────
    uint16_t l1d_kb;            // L1 data cache in KB
    uint16_t l2_kb;             // L2 cache in KB
    uint32_t l3_kb;             // L3 cache in KB (up to 4GB)
    uint8_t  _pad_cache[4];

    // ── RAM (16 bytes) ────────────────────────────────────────────────────────
    uint8_t  ram_type;          // RAMTypeCode
    uint8_t  ram_channels;      // 1/2/4/8
    uint16_t ram_speed_100mts;  // speed in units of 100 MT/s (36=DDR4-3600)
    uint32_t ram_total_mb;      // total RAM in MB
    uint32_t ram_available_mb;  // available RAM in MB
    uint16_t ram_bandwidth_100mbs; // theoretical BW in units of 100 MB/s
                                   // (576 = 57.6 GB/s)
    uint8_t  _pad_ram[2];

    // ── Benchmark results (48 bytes) ─────────────────────────────────────────
    // All float values encoded as uint16_t fixed-point to save space
    // Matmul (GFLOP/s × 100, so 1234 = 12.34 GFLOP/s)
    uint16_t bench_matmul_peak_cgf;   // peak GFLOP/s × 100
    uint16_t bench_matmul_l3_cgf;     // L3-fitting GFLOP/s × 100
    uint16_t bench_matmul_ram_cgf;    // RAM-spilling GFLOP/s × 100
    uint16_t bench_matmul_opt_n;      // optimal matrix N for L3

    // Bandwidth (GB/s × 100)
    uint16_t bench_bw_read_cgbs;      // read GB/s × 100
    uint16_t bench_bw_write_cgbs;     // write GB/s × 100
    uint16_t bench_bw_copy_cgbs;      // copy GB/s × 100

    // Latency (ns, rounded to uint16, max 65535 ns)
    uint16_t bench_lat_l1_ns;
    uint16_t bench_lat_l2_ns;
    uint16_t bench_lat_l3_ns;
    uint16_t bench_lat_ram_ns;

    // Storage (MB/s, fits in uint16 up to 65535 MB/s)
    uint16_t bench_storage_read_mbs;
    uint16_t bench_storage_write_mbs;

    uint8_t  _pad_bench[18];

    // ── Memory profiling (32 bytes) ───────────────────────────────────────────
    uint32_t mem_baseline_mb;
    uint32_t mem_loaded_mb;
    uint32_t mem_train_peak_mb;
    uint32_t mem_infer_peak_mb;
    int32_t  mem_param_cost_mb;      // signed: delta loaded - baseline
    int32_t  mem_train_overhead_mb;  // signed: delta train - loaded
    uint32_t mem_total_peak_mb;
    uint32_t mem_recommended_free_mb;

    // ── Recommended config (32 bytes) ─────────────────────────────────────────
    uint16_t cfg_batch_size;
    uint16_t cfg_block_size;
    uint16_t cfg_n_embed;
    uint8_t  cfg_n_layers;
    uint8_t  cfg_n_heads;
    uint8_t  cfg_grad_accum;
    uint8_t  cfg_quant;              // QuantCode
    uint8_t  cfg_num_threads;
    uint8_t  cfg_matmul_tile;        // tile_size / 4 (so 8=32, 16=64, 32=128)
    uint8_t  cfg_algo_flags;         // bit0=use_blas bit1=mmap bit2=flash_attn
                                     // bit3=prefetch bit4=numa_bind
    uint8_t  cfg_fit_score;          // 0–100
    uint32_t cfg_estimated_param_mb;
    uint32_t cfg_estimated_train_mb;
    uint32_t cfg_estimated_infer_mb;
    uint16_t cfg_lr_1e6;             // learning_rate × 1e6 (300=3e-4)
    uint8_t  _pad_cfg[6];

    // ── Totals ────────────────────────────────────────────────────────────────
    // 16 + 12 + 16 + 48 + 32 + 32 = 156 bytes
    // With padding fields: need to verify at compile time
};

// We'll verify the size below after seeing the actual layout.
// 224 bytes is our target — use static_assert.

// ─── Per-device storage entry (32 bytes, variable section) ───────────────────

struct TF_PACKED StorageEntry {
    char     dev_name[12];     // e.g. "nvme0n1\0"
    uint8_t  storage_type;     // StorageCode
    uint8_t  _pad[3];
    uint64_t size_bytes;       // device size
    uint16_t seq_read_mbs;     // measured sequential read in MB/s (0 if not measured)
    uint16_t seq_write_mbs;
    uint32_t _pad2;
                               // ─────
                               // 32B total
};
static_assert(sizeof(StorageEntry) == 32, "StorageEntry must be 32 bytes");

// ─── Complete message ─────────────────────────────────────────────────────────

struct ProfilerMessage {
    ProfilerHeader header;
    ProfilerBody   body;
    // StorageEntry[] follows in the wire buffer — not embedded here
    // to keep the struct sendable as a flat array + separate entries
};

// ─── Checksum implementations ─────────────────────────────────────────────────

// CRC32 — table-based, no external dep
namespace crc32_impl {
    static uint32_t table[256];
    static bool table_ready = false;

    static void build_table() {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int j = 0; j < 8; j++)
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        table_ready = true;
    }

    static uint32_t compute(const void* data, size_t len) {
        if (!table_ready) build_table();
        const uint8_t* buf = static_cast<const uint8_t*>(data);
        uint32_t crc = 0xFFFFFFFFu;
        for (size_t i = 0; i < len; i++)
            crc = table[(crc ^ buf[i]) & 0xFF] ^ (crc >> 8);
        return crc ^ 0xFFFFFFFFu;
    }
}

// xxHash32 — public domain, very fast, good avalanche
namespace xxhash32_impl {
    static constexpr uint32_t PRIME1 = 0x9E3779B1u;
    static constexpr uint32_t PRIME2 = 0x85EBCA77u;
    static constexpr uint32_t PRIME3 = 0xC2B2AE3Du;
    static constexpr uint32_t PRIME4 = 0x27D4EB2Fu;
    static constexpr uint32_t PRIME5 = 0x165667B1u;

    static inline uint32_t rotl32(uint32_t x, int r) {
        return (x << r) | (x >> (32 - r));
    }
    static inline uint32_t round(uint32_t acc, uint32_t input) {
        return rotl32(acc + input * PRIME2, 13) * PRIME1;
    }

    static uint32_t compute(const void* data, size_t len, uint32_t seed = 0) {
        const uint8_t* p   = static_cast<const uint8_t*>(data);
        const uint8_t* end = p + len;
        uint32_t h32;

        if (len >= 16) {
            uint32_t v1 = seed + PRIME1 + PRIME2;
            uint32_t v2 = seed + PRIME2;
            uint32_t v3 = seed;
            uint32_t v4 = seed - PRIME1;

            while (p <= end - 16) {
                uint32_t tmp;
                memcpy(&tmp, p,      4); v1 = round(v1, tmp);
                memcpy(&tmp, p + 4,  4); v2 = round(v2, tmp);
                memcpy(&tmp, p + 8,  4); v3 = round(v3, tmp);
                memcpy(&tmp, p + 12, 4); v4 = round(v4, tmp);
                p += 16;
            }
            h32 = rotl32(v1, 1) + rotl32(v2, 7) +
                  rotl32(v3, 12) + rotl32(v4, 18);
        } else {
            h32 = seed + PRIME5;
        }

        h32 += (uint32_t)len;

        while (p <= end - 4) {
            uint32_t tmp; memcpy(&tmp, p, 4);
            h32 = rotl32(h32 + tmp * PRIME3, 17) * PRIME4;
            p += 4;
        }
        while (p < end) {
            h32 = rotl32(h32 + (*p++) * PRIME5, 11) * PRIME1;
        }

        h32 ^= h32 >> 15; h32 *= PRIME2;
        h32 ^= h32 >> 13; h32 *= PRIME3;
        h32 ^= h32 >> 16;
        return h32;
    }
}

// Unified checksum dispatch
static inline uint32_t compute_checksum(ChecksumType type,
                                         const void* data, size_t len) {
    switch (type) {
        case ChecksumType::CRC32:   return crc32_impl::compute(data, len);
        case ChecksumType::XXHASH:  return xxhash32_impl::compute(data, len);
        default:                    return 0;
    }
}

// ─── Wire buffer ─────────────────────────────────────────────────────────────
// Flat byte buffer holding the full serialized message.
// Avoids heap allocation: max_storage_devices = 16 → 512 extra bytes.

static constexpr size_t MAX_STORAGE_DEVICES = 16;
static constexpr size_t MAX_MSG_SIZE =
    sizeof(ProfilerHeader) +
    sizeof(ProfilerBody)   +
    MAX_STORAGE_DEVICES * sizeof(StorageEntry);

struct WireBuffer {
    uint8_t  data[MAX_MSG_SIZE];
    size_t   size = 0;  // actual bytes written

    const ProfilerHeader& header() const {
        return *reinterpret_cast<const ProfilerHeader*>(data);
    }
    const ProfilerBody& body() const {
        return *reinterpret_cast<const ProfilerBody*>(data + sizeof(ProfilerHeader));
    }
    const StorageEntry* storage_entries() const {
        return reinterpret_cast<const StorageEntry*>(
            data + sizeof(ProfilerHeader) + sizeof(ProfilerBody));
    }
    uint8_t num_storage() const { return header().num_storage; }
};

// ─── Serializer ──────────────────────────────────────────────────────────────

#include "../HardwareFingerprint.hpp"
#include "../BenchmarkRunner.hpp"
#include "../HyperparamAdvisor.hpp"
#include "../MemoryProfiler.hpp"

// Encode float→uint16 with a given scale factor
static inline uint16_t f2u16(double v, double scale) {
    int64_t x = (int64_t)(v * scale + 0.5);
    if (x < 0)      x = 0;
    if (x > 65535)  x = 65535;
    return (uint16_t)x;
}

static WireBuffer serialize(
    const HardwareFingerprint& fp,
    const BenchmarkResult&     bench,
    const HyperparamConfig&    cfg,
    const MemoryReport&        mem,
    uint64_t                   client_id,
    ChecksumType               chk_type = ChecksumType::CRC32,
    ProtoFlags                 flags     = ProtoFlags::NONE)
{
    WireBuffer wb;
    memset(wb.data, 0, sizeof(wb.data));

    // ── Body ──────────────────────────────────────────────────────────────────

    ProfilerBody& body = *reinterpret_cast<ProfilerBody*>(
        wb.data + sizeof(ProfilerHeader));

    // CPU
    body.cpu_gen      = (uint8_t)static_cast<CPUGenCode>([&]{
        switch (fp.cpu.gen) {
            case CPUGen::INTEL_PRE_HASWELL:     return CPUGenCode::INTEL_PRE_HASWELL;
            case CPUGen::INTEL_HASWELL:         return CPUGenCode::INTEL_HASWELL;
            case CPUGen::INTEL_SKYLAKE:         return CPUGenCode::INTEL_SKYLAKE;
            case CPUGen::INTEL_ICE_LAKE:        return CPUGenCode::INTEL_ICE_LAKE;
            case CPUGen::INTEL_ALDER_LAKE:      return CPUGenCode::INTEL_ALDER_LAKE;
            case CPUGen::INTEL_SAPPHIRE_RAPIDS: return CPUGenCode::INTEL_SAPPHIRE_RAPIDS;
            case CPUGen::AMD_ZEN1:              return CPUGenCode::AMD_ZEN1;
            case CPUGen::AMD_ZEN2:              return CPUGenCode::AMD_ZEN2;
            case CPUGen::AMD_ZEN3:              return CPUGenCode::AMD_ZEN3;
            case CPUGen::AMD_ZEN4:              return CPUGenCode::AMD_ZEN4;
            case CPUGen::AMD_ZEN5:              return CPUGenCode::AMD_ZEN5;
            default:                            return CPUGenCode::UNKNOWN;
        }
    }());

    body.cpu_isa = (uint8_t)[&]{
        switch (fp.cpu.best_isa()) {
            case ISA::AMX:          return ISACode::AMX;
            case ISA::AVX512_VNNI:  return ISACode::AVX512_VNNI;
            case ISA::AVX512:       return ISACode::AVX512;
            case ISA::AVX2:         return ISACode::AVX2;
            case ISA::AVX:          return ISACode::AVX;
            default:                return ISACode::BASELINE;
        }
    }();

    body.cpu_vendor      = (fp.cpu.vendor == CPUVendor::INTEL) ? 0 :
                           (fp.cpu.vendor == CPUVendor::AMD)   ? 1 : 2;
    body.cpu_flags       = (fp.cpu.has_avx     ? 0x01 : 0)
                         | (fp.cpu.has_avx2    ? 0x02 : 0)
                         | (fp.cpu.has_fma     ? 0x04 : 0)
                         | (fp.cpu.has_avx512f ? 0x08 : 0)
                         | (fp.cpu.has_avx512_vnni ? 0x10 : 0)
                         | (fp.cpu.has_amx     ? 0x20 : 0)
                         | (fp.cpu.logical_cores > fp.cpu.physical_cores ? 0x40 : 0);

    body.cpu_phys_cores  = (uint8_t)std::min(fp.cpu.physical_cores, 255u);
    body.cpu_logic_cores = (uint8_t)std::min(fp.cpu.logical_cores,  255u);
    body.cpu_max_freq_100mhz = (uint16_t)(fp.cpu.max_freq_mhz / 100.0 + 0.5);
    body.embed_alignment = (uint8_t)fp.cpu.embed_alignment();
    body.capability_score = (uint8_t)fp.capability_score;

    // Cache
    body.l1d_kb = (uint16_t)std::min(fp.cache.l1d_bytes / 1024, (size_t)65535);
    body.l2_kb  = (uint16_t)std::min(fp.cache.l2_bytes  / 1024, (size_t)65535);
    body.l3_kb  = (uint32_t)std::min(fp.cache.l3_bytes  / 1024, (size_t)0xFFFFFFFFu);

    // RAM
    body.ram_type = (uint8_t)[&]{
        switch (fp.ram.type) {
            case RAMType::DDR3:   return RAMTypeCode::DDR3;
            case RAMType::DDR4:   return RAMTypeCode::DDR4;
            case RAMType::DDR5:   return RAMTypeCode::DDR5;
            case RAMType::LPDDR4: return RAMTypeCode::LPDDR4;
            case RAMType::LPDDR5: return RAMTypeCode::LPDDR5;
            default:              return RAMTypeCode::UNKNOWN;
        }
    }();
    body.ram_channels        = fp.ram.channels;
    body.ram_speed_100mts    = (uint16_t)(fp.ram.speed_mts / 100);
    body.ram_total_mb        = fp.ram.total_mb;
    body.ram_available_mb    = fp.ram.available_mb;
    body.ram_bandwidth_100mbs = (uint16_t)(fp.ram.bandwidth_gbs * 10.0 + 0.5);

    // Benchmarks
    body.bench_matmul_peak_cgf  = f2u16(bench.matmul.peak_gflops, 100.0);
    body.bench_matmul_l3_cgf    = f2u16(bench.matmul.l3_gflops,   100.0);
    body.bench_matmul_ram_cgf   = f2u16(bench.matmul.ram_gflops,  100.0);
    body.bench_matmul_opt_n     = (uint16_t)std::min(bench.matmul.l3_optimal_n, (size_t)65535);
    body.bench_bw_read_cgbs     = f2u16(bench.bandwidth.read_gbs,  100.0);
    body.bench_bw_write_cgbs    = f2u16(bench.bandwidth.write_gbs, 100.0);
    body.bench_bw_copy_cgbs     = f2u16(bench.bandwidth.copy_gbs,  100.0);
    body.bench_lat_l1_ns        = (uint16_t)std::min((int)bench.latency.l1_latency_ns,  65535);
    body.bench_lat_l2_ns        = (uint16_t)std::min((int)bench.latency.l2_latency_ns,  65535);
    body.bench_lat_l3_ns        = (uint16_t)std::min((int)bench.latency.l3_latency_ns,  65535);
    body.bench_lat_ram_ns       = (uint16_t)std::min((int)bench.latency.ram_latency_ns, 65535);
    body.bench_storage_read_mbs = (uint16_t)std::min((int)bench.storage.seq_read_mbs,   65535);
    body.bench_storage_write_mbs= (uint16_t)std::min((int)bench.storage.seq_write_mbs,  65535);

    // Memory profiling
    body.mem_baseline_mb        = (uint32_t)mem.baseline_rss_mb;
    body.mem_loaded_mb          = (uint32_t)mem.loaded_rss_mb;
    body.mem_train_peak_mb      = (uint32_t)mem.train_rss_mb;
    body.mem_infer_peak_mb      = (uint32_t)mem.infer_rss_mb;
    body.mem_param_cost_mb      = (int32_t)mem.param_ram_mb;
    body.mem_train_overhead_mb  = (int32_t)mem.train_overhead_mb;
    body.mem_total_peak_mb      = (uint32_t)mem.total_peak_mb;
    body.mem_recommended_free_mb= (uint32_t)mem.recommended_free_mb;

    // Config
    body.cfg_batch_size         = (uint16_t)cfg.batch_size;
    body.cfg_block_size         = (uint16_t)cfg.block_size;
    body.cfg_n_embed            = (uint16_t)cfg.n_embed;
    body.cfg_n_layers           = (uint8_t)cfg.n_layers;
    body.cfg_n_heads            = (uint8_t)cfg.n_heads;
    body.cfg_grad_accum         = (uint8_t)std::min(cfg.grad_accum_steps, (size_t)255);
    body.cfg_quant              = (uint8_t)[&]{
        switch (cfg.algo.quant) {
            case QuantPolicy::FP16:     return QuantCode::FP16;
            case QuantPolicy::INT8:     return QuantCode::INT8;
            case QuantPolicy::FP8_E4M3: return QuantCode::FP8_E4M3;
            case QuantPolicy::FP8_E5M2: return QuantCode::FP8_E5M2;
            case QuantPolicy::INT4:     return QuantCode::INT4;
            default:                    return QuantCode::NONE;
        }
    }();
    body.cfg_num_threads        = (uint8_t)std::min(cfg.algo.num_threads, 255);
    body.cfg_matmul_tile        = (uint8_t)(cfg.algo.matmul_tile_size / 4);
    body.cfg_algo_flags         = (cfg.algo.use_blas               ? 0x01 : 0)
                                | (cfg.algo.use_mmap_weights        ? 0x02 : 0)
                                | (cfg.algo.use_flash_attn_style    ? 0x04 : 0)
                                | (cfg.algo.prefetch_next_batch     ? 0x08 : 0)
                                | (cfg.algo.use_numa_bind           ? 0x10 : 0);
    body.cfg_fit_score          = (uint8_t)cfg.fit_score;
    body.cfg_estimated_param_mb = (uint32_t)cfg.estimated_param_mb;
    body.cfg_estimated_train_mb = (uint32_t)cfg.estimated_train_mb;
    body.cfg_estimated_infer_mb = (uint32_t)cfg.estimated_infer_mb;
    body.cfg_lr_1e6             = (uint16_t)(cfg.learning_rate * 1e6f + 0.5f);

    // ── Variable section: storage entries ────────────────────────────────────

    uint8_t num_devs = (uint8_t)std::min(fp.storage.size(), (size_t)MAX_STORAGE_DEVICES);
    uint8_t* storage_ptr = wb.data + sizeof(ProfilerHeader) + sizeof(ProfilerBody);

    for (uint8_t i = 0; i < num_devs; i++) {
        StorageEntry entry{};
        const auto& s = fp.storage[i];
        strncpy(entry.dev_name, s.name.c_str(), sizeof(entry.dev_name) - 1);
        entry.storage_type = (uint8_t)[&]{
            switch (s.type) {
                case StorageType::HDD:       return StorageCode::HDD;
                case StorageType::SATA_SSD:  return StorageCode::SATA_SSD;
                case StorageType::NVME_GEN3: return StorageCode::NVME_GEN3;
                case StorageType::NVME_GEN4: return StorageCode::NVME_GEN4;
                case StorageType::NVME_GEN5: return StorageCode::NVME_GEN5;
                default:                     return StorageCode::UNKNOWN;
            }
        }();
        entry.size_bytes      = s.size_bytes;
        entry.seq_read_mbs    = (uint16_t)std::min((int)s.seq_read_mbs,  65535);
        entry.seq_write_mbs   = (uint16_t)std::min((int)s.seq_write_mbs, 65535);
        memcpy(storage_ptr + i * sizeof(StorageEntry), &entry, sizeof(StorageEntry));
    }

    // ── Header (written last so we can compute checksum) ─────────────────────

    size_t body_len = sizeof(ProfilerBody) + num_devs * sizeof(StorageEntry);
    wb.size = sizeof(ProfilerHeader) + body_len;

    // Checksum covers body + storage entries (not header itself)
    uint32_t chk = compute_checksum(chk_type,
        wb.data + sizeof(ProfilerHeader), body_len);

    ProfilerHeader& hdr = *reinterpret_cast<ProfilerHeader*>(wb.data);
    hdr.magic          = PROTO_MAGIC;
    hdr.version        = PROTO_VERSION;
    hdr.flags          = (uint8_t)flags;
    hdr.msg_type       = (uint8_t)MsgType::PROFILE_REPORT;
    hdr.num_storage    = num_devs;
    hdr.client_id      = client_id;
    hdr.body_len       = (uint32_t)body_len;
    hdr.checksum_type  = (uint8_t)chk_type;
    hdr.checksum       = chk;

    // Timestamp
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    hdr.timestamp_us = (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000ULL;

    return wb;
}

// ─── Deserializer ────────────────────────────────────────────────────────────

struct ParseResult {
    bool       ok           = false;
    bool       checksum_ok  = true;
    const char* error       = nullptr;

    // Parsed fields (server side — read directly from body)
    const ProfilerBody*  body    = nullptr;
    const StorageEntry*  storage = nullptr;
    uint8_t              num_storage = 0;
    uint64_t             client_id   = 0;
    uint64_t             timestamp_us = 0;
};

static ParseResult deserialize(const uint8_t* data, size_t len) {
    ParseResult r;

    if (len < sizeof(ProfilerHeader)) {
        r.error = "too short for header"; return r;
    }

    const ProfilerHeader& hdr = *reinterpret_cast<const ProfilerHeader*>(data);

    if (hdr.magic != PROTO_MAGIC) {
        r.error = "bad magic"; return r;
    }
    if (hdr.version != PROTO_VERSION) {
        r.error = "version mismatch"; return r;
    }
    if (hdr.msg_type != (uint8_t)MsgType::PROFILE_REPORT) {
        r.error = "unexpected msg_type"; return r;
    }

    size_t expected = sizeof(ProfilerHeader) + hdr.body_len;
    if (len < expected) {
        r.error = "truncated body"; return r;
    }

    // Verify checksum
    auto chk_type = static_cast<ChecksumType>(hdr.checksum_type);
    if (chk_type != ChecksumType::NONE) {
        uint32_t expected_chk = compute_checksum(chk_type,
            data + sizeof(ProfilerHeader), hdr.body_len);
        if (expected_chk != hdr.checksum) {
            r.checksum_ok = false;
            r.error = "checksum mismatch";
            return r;
        }
    }

    r.body        = reinterpret_cast<const ProfilerBody*>(data + sizeof(ProfilerHeader));
    r.storage     = reinterpret_cast<const StorageEntry*>(
                        data + sizeof(ProfilerHeader) + sizeof(ProfilerBody));
    r.num_storage = hdr.num_storage;
    r.client_id   = hdr.client_id;
    r.timestamp_us = hdr.timestamp_us;
    r.ok          = true;
    return r;
}

// ─── Debug printer ───────────────────────────────────────────────────────────

static void print_wire(const WireBuffer& wb) {
    auto r = deserialize(wb.data, wb.size);
    if (!r.ok) { printf("[Proto] Parse error: %s\n", r.error); return; }

    const ProfilerBody& b = *r.body;
    printf("\n[Protocol Wire — %zu bytes]\n", wb.size);
    printf("  client_id   : 0x%016llx\n", (unsigned long long)r.client_id);
    printf("  cpu_gen     : %u\n", b.cpu_gen);
    printf("  cpu_isa     : %u\n", b.cpu_isa);
    printf("  cpu_cores   : %u phys / %u logic\n", b.cpu_phys_cores, b.cpu_logic_cores);
    printf("  cpu_freq    : %u00 MHz\n", b.cpu_max_freq_100mhz);
    printf("  L3          : %u KB\n", b.l3_kb);
    printf("  RAM         : %u MB total, %u MB avail, type=%u @ %u00 MT/s\n",
           b.ram_total_mb, b.ram_available_mb, b.ram_type, b.ram_speed_100mts);
    printf("  matmul_peak : %.2f GFLOP/s\n", b.bench_matmul_peak_cgf / 100.0);
    printf("  matmul_l3   : %.2f GFLOP/s\n", b.bench_matmul_l3_cgf   / 100.0);
    printf("  bw_read     : %.2f GB/s\n",    b.bench_bw_read_cgbs    / 100.0);
    printf("  lat_l3      : %u ns\n", b.bench_lat_l3_ns);
    printf("  lat_ram     : %u ns\n", b.bench_lat_ram_ns);
    printf("  cfg batch   : %u\n", b.cfg_batch_size);
    printf("  cfg block   : %u\n", b.cfg_block_size);
    printf("  cfg n_embed : %u\n", b.cfg_n_embed);
    printf("  cfg n_layers: %u\n", b.cfg_n_layers);
    printf("  cfg n_heads : %u\n", b.cfg_n_heads);
    printf("  cfg quant   : %u\n", b.cfg_quant);
    printf("  cfg threads : %u\n", b.cfg_num_threads);
    printf("  fit_score   : %u\n", b.cfg_fit_score);
    printf("  param est   : %u MB\n", b.cfg_estimated_param_mb);
    printf("  train est   : %u MB\n", b.cfg_estimated_train_mb);
    printf("  Storage devices: %u\n", r.num_storage);
    for (uint8_t i = 0; i < r.num_storage; i++) {
        const StorageEntry& s = r.storage[i];
        printf("    [%u] %-12s type=%u  %llu GB  %u MB/s read\n",
               i, s.dev_name, s.storage_type,
               (unsigned long long)(s.size_bytes / (1024*1024*1024)),
               s.seq_read_mbs);
    }
    printf("  [Checksum OK]\n\n");
}
