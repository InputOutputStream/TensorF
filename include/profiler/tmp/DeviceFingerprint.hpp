#pragma once

/*
 * DeviceFingerprint.hpp
 * =====================
 * Layer 1 of the rewritten profiler: FACTS about the device, no benchmarking.
 *
 * Replaces HardwareFingerprint.hpp. What changed and why:
 *
 *   - No popen / lscpu / dmidecode / lsblk. Everything comes from /proc and /sys,
 *     so it works on Android where those tools do not exist and on Linux without root.
 *   - ARM is detected (CPU implementer/part, Features), including armv7 phones.
 *   - Core CLASSES (prime / performance / efficiency) by capacity and frequency,
 *     not "N cores". A big.LITTLE phone is described as clusters, each with its
 *     own frequency and caches. Intel hybrid (P/E) is recognised too.
 *   - Microarchitecture comes from numeric family/model, not substring guesses
 *     ("Ryzen 9 7950X" contains a 9 but is Zen4, not Zen5).
 *   - All numbers parse through 64-bit-safe helpers (no sscanf %lu on armv7).
 *   - Memory limits, not memory size: MemAvailable, cgroup limit, kernel reserve,
 *     combined into one safe budget.
 *   - Thermal zones and battery are read as facts; the same probe is used by the
 *     capability suite to stop a run before the device gets hot.
 *   - No single "capability_score". The capability VECTOR is produced by
 *     CapabilitySuite.hpp from measurements.
 *
 * Testability: every adapter takes a filesystem root prefix, so tests can point
 * it at a fixture tree (see tests/profiler_capability_test.cpp) instead of the
 * real machine. The parsers are pure functions over text.
 *
 * Only the Linux/Android adapter exists. Apple and Windows adapters are step 5
 * of the design (not testable without devices); detect_device() falls back to a
 * minimal generic fingerprint there and says so in `notes`.
 */

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "ProfilerCommon.hpp"

#if defined(__linux__)
#include <dirent.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace tensorf::prof {

// ─── Enums ───────────────────────────────────────────────────────────────────

// NB: enumerators are mixed-case on purpose. The NDK CMake toolchain defines ANDROID as a macro
// (-DANDROID=1) and GNU modes predefine `linux`, so an enumerator named after a platform macro breaks the build.
enum class OS     { Linux, Android, Apple, Windows, Other };
enum class Arch   { X86_64, X86, ARM64, ARM32, OTHER };
enum class Vendor { INTEL, AMD, ARM_LICENSED, APPLE, UNKNOWN };   // ARM_LICENSED: ARM, Qualcomm, Samsung, ... cores
enum class CoreClass { UNIFORM, PRIME, PERFORMANCE, EFFICIENCY };
enum class StorageKind { HDD, SATA_OR_UFS_FLASH, EMMC, NVME, UNKNOWN };

inline const char* to_string(OS o) {
    switch (o) { case OS::Linux: return "linux"; case OS::Android: return "android";
                 case OS::Apple: return "apple"; case OS::Windows: return "windows"; default: return "other"; }
}
inline const char* to_string(Arch a) {
    switch (a) { case Arch::X86_64: return "x86_64"; case Arch::X86: return "x86";
                 case Arch::ARM64: return "arm64"; case Arch::ARM32: return "arm32"; default: return "other"; }
}
inline const char* to_string(Vendor v) {
    switch (v) { case Vendor::INTEL: return "intel"; case Vendor::AMD: return "amd";
                 case Vendor::ARM_LICENSED: return "arm"; case Vendor::APPLE: return "apple"; default: return "unknown"; }
}
inline const char* to_string(CoreClass c) {
    switch (c) { case CoreClass::PRIME: return "prime"; case CoreClass::PERFORMANCE: return "performance";
                 case CoreClass::EFFICIENCY: return "efficiency"; default: return "uniform"; }
}
inline const char* to_string(StorageKind k) {
    switch (k) { case StorageKind::HDD: return "hdd"; case StorageKind::SATA_OR_UFS_FLASH: return "flash";
                 case StorageKind::EMMC: return "emmc"; case StorageKind::NVME: return "nvme"; default: return "unknown"; }
}

// ─── Data structures ─────────────────────────────────────────────────────────

struct SimdFeatures {
    // x86
    bool sse2 = false, avx = false, avx2 = false, fma = false;
    bool avx512f = false, avx512_vnni = false, avx512_bf16 = false, amx = false;
    // Arm
    bool neon = false, fp16 = false, dotprod = false, i8mm = false, bf16 = false;
    bool sve = false, sve2 = false, sme = false, sme2 = false;

    // Float32 lanes per vector register on the best available unit.
    int f32_lanes() const {
        if (avx512f) return 16;
        if (avx)     return 8;
        if (sse2 || neon) return 4;
        return 1;
    }
    std::string summary() const {
        std::string s;
        auto add = [&](bool b, const char* n) { if (b) { if (!s.empty()) s += ' '; s += n; } };
        add(sse2, "sse2"); add(avx, "avx"); add(avx2, "avx2"); add(fma, "fma");
        add(avx512f, "avx512f"); add(avx512_vnni, "avx512_vnni"); add(avx512_bf16, "avx512_bf16"); add(amx, "amx");
        add(neon, "neon"); add(fp16, "fp16"); add(dotprod, "dotprod"); add(i8mm, "i8mm"); add(bf16, "bf16");
        add(sve, "sve"); add(sve2, "sve2"); add(sme, "sme"); add(sme2, "sme2");
        return s;
    }
};

struct CacheLevel {
    int         level = 0;
    std::string type;                 // "Data" | "Instruction" | "Unified"
    uint64_t    size_bytes = 0;
    uint32_t    line_bytes = 0;
    uint32_t    ways = 0;
    std::vector<int> shared_cpus;     // logical CPUs sharing this cache instance
};

// One group of identical cores (a "cluster" in phone terms).
struct CoreCluster {
    CoreClass        cls = CoreClass::UNIFORM;
    std::vector<int> cpus;            // logical CPU ids (online ones)
    int              physical_cores = 0;
    double           max_freq_mhz = 0.0;
    int              capacity = 0;    // Linux cpu_capacity (0..1024), 0 = unknown
    std::string      part_name;       // "cortex-a78", "zen4", ...
    uint64_t         l1d_bytes = 0;   // per core
    uint64_t         l2_bytes = 0;    // per cache instance
    int              l2_sharers = 1;  // logical CPUs sharing one L2 instance
    uint64_t         l3_bytes = 0;    // per cache instance (shared)
};

struct MemoryLimits {
    uint64_t total_bytes = 0;
    uint64_t available_bytes = 0;      // MemAvailable: what we could get without swapping others out
    uint64_t swap_total_bytes = 0;
    uint64_t cgroup_limit_bytes = 0;   // 0 = unlimited / not found
    uint64_t cgroup_usage_bytes = 0;
    uint64_t kernel_reserve_bytes = 0; // vm.min_free_kbytes: the floor the kernel/LMK defends
    bool     low_ram_device = false;   // heuristic: < 3 GiB total
    // The number every memory-fit decision should use. Never total RAM.
    uint64_t safe_budget_bytes = 0;
};

struct StorageDevice {
    std::string name;
    StorageKind kind = StorageKind::UNKNOWN;
    uint64_t    size_bytes = 0;
    bool        rotational = false;
    std::string link_speed;           // NVMe PCIe link ("16.0 GT/s PCIe"), if exposed
};

struct ThermalZone {
    std::string name;                 // sysfs zone "type" ("cpu-1-0", "battery", "soc_thermal")
    double      temp_c = 0.0;
};

struct BatteryInfo {
    bool        present = false;
    int         level_pct = -1;
    double      temp_c = std::numeric_limits<double>::quiet_NaN();   // tenths of °C in sysfs
    std::string status;                                            // "Charging" | "Discharging" | ...
    bool        charging() const { return status == "Charging" || status == "Full"; }
};

struct DeviceFingerprint {
    OS     os   = OS::Other;
    Arch   arch = Arch::OTHER;
    Vendor vendor = Vendor::UNKNOWN;

    std::string model_name;           // marketing / SoC string
    std::string microarch;            // "zen4", "intel_alder_lake", "cortex-a78+cortex-a55", ...
    std::string kernel, os_release, compiler, compile_target;

    int logical_cpus  = 0;            // online
    int physical_cores = 0;
    std::vector<CoreCluster> clusters;   // sorted: strongest class first
    bool hybrid() const { return clusters.size() > 1; }

    SimdFeatures simd;
    MemoryLimits memory;
    std::vector<StorageDevice> storage;
    std::vector<ThermalZone>   thermal_zones;
    BatteryInfo battery;

    std::vector<std::string> notes;   // things we could not determine, for the server to see

    // Convenience: largest per-instance cache the strongest cluster sees.
    uint64_t l1d_bytes() const { return clusters.empty() ? 0 : clusters.front().l1d_bytes; }
    uint64_t l2_bytes()  const { return clusters.empty() ? 0 : clusters.front().l2_bytes; }
    uint64_t l3_bytes()  const {
        uint64_t m = 0; for (const auto& c : clusters) m = std::max(m, c.l3_bytes); return m;
    }
    const CoreCluster* strongest() const { return clusters.empty() ? nullptr : &clusters.front(); }
};

// ─── CPU identification (pure functions) ─────────────────────────────────────

// Arm Ltd part numbers (CPU implementer 0x41). Part ids are from the Arm TRMs.
inline std::string arm_part_name(uint32_t implementer, uint32_t part, bool is_arm32) {
    if (implementer == 0x41) {
        switch (part) {
            case 0xc05: return "cortex-a5";   case 0xc07: return "cortex-a7";   case 0xc08: return "cortex-a8";
            case 0xc09: return "cortex-a9";   case 0xc0d: return "cortex-a12";  case 0xc0e: return "cortex-a17";
            case 0xc0f: return "cortex-a15";
            case 0xd03: return "cortex-a53";  case 0xd04: return "cortex-a35";  case 0xd05: return "cortex-a55";
            case 0xd07: return "cortex-a57";  case 0xd08: return "cortex-a72";  case 0xd09: return "cortex-a73";
            case 0xd0a: return "cortex-a75";  case 0xd0b: return "cortex-a76";  case 0xd0c: return "neoverse-n1";
            case 0xd0d: return "cortex-a77";  case 0xd40: return "neoverse-v1"; case 0xd41: return "cortex-a78";
            case 0xd44: return "cortex-x1";   case 0xd46: return "cortex-a510"; case 0xd47: return "cortex-a710";
            case 0xd48: return "cortex-x2";   case 0xd4d: return "cortex-a715"; case 0xd4e: return "cortex-x3";
            case 0xd4f: return "neoverse-v2"; case 0xd80: return "cortex-a520"; case 0xd81: return "cortex-a720";
            case 0xd82: return "cortex-x4";   case 0xd85: return "cortex-x925"; case 0xd87: return "cortex-a725";
            default: break;
        }
        (void)is_arm32;
        char b[32]; std::snprintf(b, sizeof b, "arm-part-0x%x", part); return b;
    }
    const char* who = implementer == 0x51 ? "qualcomm" : implementer == 0x53 ? "samsung" :
                      implementer == 0x61 ? "apple"    : implementer == 0x48 ? "hisilicon" :
                      implementer == 0x41 ? "arm"      : "arm-licensee";
    char b[48]; std::snprintf(b, sizeof b, "%s-0x%x", who, part); return b;
}

// x86 microarchitecture from numeric family/model. These are the CPUID display
// family/model that Linux prints in /proc/cpuinfo ("cpu family", "model").
inline std::string x86_microarch(Vendor v, uint32_t family, uint32_t model, const SimdFeatures& f) {
    auto in = [&](uint32_t m, std::initializer_list<uint32_t> xs) { for (uint32_t x : xs) if (x == m) return true; return false; };
    if (v == Vendor::AMD) {
        if (family == 0x1a) return "zen5";
        if (family == 0x19) {
            if ((model >= 0x10 && model <= 0x1f) || (model >= 0x60 && model <= 0x7f) || (model >= 0xa0 && model <= 0xaf))
                return "zen4";
            return "zen3";
        }
        if (family == 0x17) {
            if ((model >= 0x30 && model <= 0x3f) || (model >= 0x60 && model <= 0x7f) || (model >= 0x90 && model <= 0xaf))
                return "zen2";
            return "zen1";   // includes Zen+ (0x08, 0x18, 0x50)
        }
        if (family == 0x16) return "jaguar";
        return f.avx2 ? "amd_avx2_unknown" : "amd_unknown";
    }
    if (v == Vendor::INTEL) {
        if (family == 6) {
            if (in(model, {0x3c, 0x3f, 0x45, 0x46})) return "intel_haswell";
            if (in(model, {0x3d, 0x47, 0x4f, 0x56})) return "intel_broadwell";
            if (in(model, {0x4e, 0x5e, 0x55}))       return f.avx512f ? "intel_skylake_x" : "intel_skylake";
            if (in(model, {0x8e, 0x9e, 0xa5, 0xa6})) return "intel_skylake_derivative";   // Kaby/Coffee/Comet Lake
            if (in(model, {0x7e, 0x7d, 0x6a, 0x6c})) return "intel_ice_lake";
            if (in(model, {0x8c, 0x8d, 0xa7}))       return "intel_tiger_rocket_lake";
            if (in(model, {0x97, 0x9a, 0xbe}))       return "intel_alder_lake";
            if (in(model, {0xb7, 0xba, 0xbf}))       return "intel_raptor_lake";
            if (in(model, {0x8f, 0xcf}))             return "intel_sapphire_rapids";
            if (in(model, {0xaa, 0xac, 0xbd}))       return "intel_meteor_lunar_lake";
            if (in(model, {0xc5, 0xc6}))             return "intel_arrow_lake";
            if (in(model, {0xad, 0xae}))             return "intel_granite_rapids";
            if (in(model, {0x1a, 0x1e, 0x25, 0x2c, 0x2a, 0x2d, 0x3a, 0x3e})) return "intel_pre_haswell";
        }
        return f.avx2 ? "intel_avx2_unknown" : "intel_pre_haswell";
    }
    return "unknown";
}

// ─── /proc/cpuinfo parsing ───────────────────────────────────────────────────

struct CpuInfoBlock { int processor = -1; std::map<std::string, std::string> kv; };

struct ParsedCpuInfo {
    std::vector<CpuInfoBlock> blocks;
    std::string common_flags;     // "flags" (x86) or "Features" (Arm), intersected over all blocks
    std::string model_name;
    Vendor      vendor = Vendor::UNKNOWN;
    uint32_t    x86_family = 0, x86_model = 0;
    bool        arm_arch64 = false;
    std::vector<std::string> block_part_names;   // per core block (processor >= 0), in file order, Arm only
};

inline ParsedCpuInfo parse_cpuinfo(const std::string& text) {
    ParsedCpuInfo out;
    CpuInfoBlock cur;
    auto flush = [&] { if (!cur.kv.empty()) out.blocks.push_back(cur); cur = CpuInfoBlock{}; };
    for (const std::string& line : split_lines(text)) {
        if (trim(line).empty()) { flush(); continue; }
        size_t c = line.find(':');
        if (c == std::string::npos) continue;
        std::string k = trim(std::string_view(line).substr(0, c)), v = trim(std::string_view(line).substr(c + 1));
        if (k == "processor") { uint64_t p; if (parse_u64(v, p)) cur.processor = (int)p; }
        cur.kv[k] = v;
    }
    flush();
    if (out.blocks.empty()) return out;

    // Flags: intersection across blocks (hybrid CPUs and mixed clusters).
    auto flag_key = [](const CpuInfoBlock& b) -> const std::string* {
        auto i = b.kv.find("flags");     if (i != b.kv.end()) return &i->second;
        auto j = b.kv.find("Features");  if (j != b.kv.end()) return &j->second;
        return nullptr;
    };
    std::set<std::string> common;
    bool first = true;
    for (const auto& b : out.blocks) {
        const std::string* f = flag_key(b);
        if (!f) continue;
        std::set<std::string> s;
        for (const std::string& t : split(*f, ' ')) if (!t.empty()) s.insert(t);
        if (first) { common = s; first = false; }
        else { std::set<std::string> keep; for (const auto& t : common) if (s.count(t)) keep.insert(t); common = keep; }
    }
    for (const auto& t : common) { out.common_flags += t; out.common_flags += ' '; }
    if (!out.common_flags.empty()) out.common_flags.pop_back();

    // A key may live in any block: on Android the SoC "Hardware" line comes after the
    // last core, and a leading "Processor" line can form a block of its own.
    auto get = [&](const char* k) {
        for (const auto& b : out.blocks) { auto i = b.kv.find(k); if (i != b.kv.end()) return i->second; }
        return std::string();
    };

    // x86 identification
    const std::string vid = get("vendor_id");
    if (vid.find("GenuineIntel") != std::string::npos) out.vendor = Vendor::INTEL;
    else if (vid.find("AuthenticAMD") != std::string::npos) out.vendor = Vendor::AMD;
    uint64_t fam = 0, mod = 0;
    if (parse_u64(get("cpu family"), fam)) out.x86_family = (uint32_t)fam;
    if (parse_u64(get("model"), mod))      out.x86_model  = (uint32_t)mod;

    // Arm identification: ARM cpuinfo has no vendor_id, only "CPU implementer".
    const bool is_arm = !get("CPU implementer").empty() || !get("CPU part").empty();
    if (is_arm) {
        out.vendor = Vendor::ARM_LICENSED;
        const std::string arch = get("CPU architecture");
        uint64_t a = 0; parse_u64(arch, a);
        out.arm_arch64 = (a >= 8) || arch == "AArch64" || has_token(out.common_flags, "asimd");
        for (const auto& b : out.blocks) {
            if (b.processor < 0) continue;                     // not a per-core block
            uint64_t imp = 0, part = 0;
            auto gi = b.kv.find("CPU implementer"); auto gp = b.kv.find("CPU part");
            if (gi != b.kv.end()) parse_u64(gi->second, imp, 16);
            if (gp != b.kv.end()) parse_u64(gp->second, part, 16);
            if (imp == 0x61) out.vendor = Vendor::APPLE;
            out.block_part_names.push_back(arm_part_name((uint32_t)imp, (uint32_t)part, !out.arm_arch64));
        }
    }

    // Model name: Arm prefers "Hardware" (the SoC); x86 uses "model name".
    if (is_arm) {
        for (const char* k : {"Hardware", "model name", "Processor"}) { std::string v = get(k); if (!v.empty()) { out.model_name = v; break; } }
    } else {
        out.model_name = get("model name");
    }
    return out;
}

inline SimdFeatures simd_from_flags(const std::string& flags, Vendor v) {
    SimdFeatures s;
    auto h = [&](const char* t) { return has_token(flags, t); };
    if (v == Vendor::ARM_LICENSED || v == Vendor::APPLE) {
        s.neon    = h("asimd") || h("neon");
        // NB: the armv7 hwcap token "half" means half-WORD loads/stores, not fp16 arithmetic.
        s.fp16    = h("asimdhp") || h("fphp") || h("fp16");
        s.dotprod = h("asimddp") || h("dotprod");
        s.i8mm    = h("i8mm");
        s.bf16    = h("bf16");
        s.sve     = h("sve");   s.sve2 = h("sve2");
        s.sme     = h("sme");   s.sme2 = h("sme2");
    } else {
        s.sse2 = h("sse2"); s.avx = h("avx"); s.avx2 = h("avx2"); s.fma = h("fma");
        s.avx512f = h("avx512f"); s.avx512_vnni = h("avx512_vnni"); s.avx512_bf16 = h("avx512_bf16");
        s.amx = h("amx_bf16") || h("amx_int8") || h("amx_tile");
    }
    return s;
}

// ─── Core classification (pure) ──────────────────────────────────────────────

struct RawCore {
    int      cpu = 0;
    int      capacity = 0;        // cpu_capacity, 0 = unknown
    uint64_t max_freq_khz = 0;    // cpufreq max, 0 = unknown
    std::string part;             // part name, "" = unknown
    bool     hybrid_atom = false; // Intel E-core, from /sys/devices/cpu_atom
    bool     hybrid_core = false; // Intel P-core
};

// Group cores into tiers and label them. Grouping key preference:
//   Intel hybrid flag  >  cpu_capacity  >  cpufreq max  >  part name  >  one group
// Within numeric keys, values within 8% belong to the same tier.
inline std::vector<std::vector<RawCore>> group_cores(std::vector<RawCore> cores) {
    std::vector<std::vector<RawCore>> groups;
    if (cores.empty()) return groups;

    const bool any_hybrid = std::any_of(cores.begin(), cores.end(), [](const RawCore& c) { return c.hybrid_atom || c.hybrid_core; });
    if (any_hybrid) {
        std::vector<RawCore> p, e;
        for (auto& c : cores) (c.hybrid_atom ? e : p).push_back(c);
        if (!p.empty()) groups.push_back(p);
        if (!e.empty()) groups.push_back(e);
        return groups;
    }
    const bool have_cap  = std::all_of(cores.begin(), cores.end(), [](const RawCore& c) { return c.capacity > 0; });
    const bool have_freq = std::all_of(cores.begin(), cores.end(), [](const RawCore& c) { return c.max_freq_khz > 0; });
    if (have_cap || have_freq) {
        auto key = [&](const RawCore& c) { return have_cap ? (double)c.capacity : (double)c.max_freq_khz; };
        std::sort(cores.begin(), cores.end(), [&](const RawCore& a, const RawCore& b) {
            return key(a) != key(b) ? key(a) > key(b) : a.cpu < b.cpu; });
        double ref = 0;
        for (auto& c : cores) {
            if (groups.empty() || key(c) < ref * 0.92) { groups.emplace_back(); ref = key(c); }
            groups.back().push_back(c);
        }
        return groups;
    }
    // Only part names: group by name, order unknown (cannot rank) -> keep first-seen order.
    std::map<std::string, size_t> idx;
    for (auto& c : cores) {
        auto it = idx.find(c.part);
        if (it == idx.end()) { idx[c.part] = groups.size(); groups.emplace_back(); it = idx.find(c.part); }
        groups[it->second].push_back(c);
    }
    return groups;
}

inline CoreClass class_for_tier(size_t tier, size_t n_tiers) {
    if (n_tiers <= 1) return CoreClass::UNIFORM;
    if (tier == n_tiers - 1) return CoreClass::EFFICIENCY;
    if (n_tiers >= 3 && tier == 0) return CoreClass::PRIME;
    return CoreClass::PERFORMANCE;
}

// ─── Memory (pure part) ──────────────────────────────────────────────────────

// cgroup limit files contain "max" (v2) or a huge number (v1) when unlimited.
inline uint64_t parse_cgroup_limit(const std::string& text) {
    std::string t = trim(text);
    if (t.empty() || t == "max") return 0;
    uint64_t v = 0;
    if (!parse_u64(t, v)) return 0;
    if (v >= (1ull << 60)) return 0;      // v1 "unlimited" sentinel is ~2^63
    return v;
}

// One number for "how much may this process plan to use".
//   headroom = min(MemAvailable, cgroup_limit - cgroup_usage)
//   budget   = 80% of headroom, minus the kernel reserve (never below zero)
// The 80% is a margin for allocator overhead and other processes; the reserve is
// what the kernel/LMK defends before killing things.
inline uint64_t compute_safe_budget(const MemoryLimits& m) {
    uint64_t headroom = m.available_bytes;
    if (m.cgroup_limit_bytes > 0) {
        const uint64_t cg = m.cgroup_limit_bytes > m.cgroup_usage_bytes ? m.cgroup_limit_bytes - m.cgroup_usage_bytes : 0;
        headroom = headroom == 0 ? cg : std::min(headroom, cg);
    }
    const uint64_t scaled = (uint64_t)((double)headroom * 0.8);
    return scaled > m.kernel_reserve_bytes ? scaled - m.kernel_reserve_bytes : 0;
}

// ─── Linux / Android adapter ─────────────────────────────────────────────────

namespace linux_adapter {

inline std::string cpu_dir(const std::string& root, int cpu) {
    return root + "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
}

inline std::vector<CacheLevel> read_caches(const std::string& root, int cpu) {
    std::vector<CacheLevel> out;
    for (int i = 0; i < 10; ++i) {
        const std::string d = cpu_dir(root, cpu) + "/cache/index" + std::to_string(i);
        const std::string level = read_trim(d + "/level");
        if (level.empty()) { if (i == 0) break; else continue; }
        CacheLevel c;
        uint64_t v = 0;
        if (parse_u64(level, v)) c.level = (int)v;
        c.type = read_trim(d + "/type");
        parse_size_bytes(read_trim(d + "/size"), c.size_bytes);
        if (parse_u64(read_trim(d + "/coherency_line_size"), v)) c.line_bytes = (uint32_t)v;
        if (parse_u64(read_trim(d + "/ways_of_associativity"), v)) c.ways = (uint32_t)v;
        c.shared_cpus = parse_cpu_list(read_trim(d + "/shared_cpu_list"));
        out.push_back(std::move(c));
    }
    return out;
}

inline std::vector<int> online_cpus(const std::string& root, int fallback_count) {
    std::vector<int> cpus = parse_cpu_list(read_trim(root + "/sys/devices/system/cpu/online"));
    if (!cpus.empty()) return cpus;
    cpus = parse_cpu_list(read_trim(root + "/sys/devices/system/cpu/present"));
    if (!cpus.empty()) return cpus;
    for (int i = 0; i < std::max(1, fallback_count); ++i) cpus.push_back(i);
    return cpus;
}

inline MemoryLimits read_memory(const std::string& root) {
    MemoryLimits m;
    const std::string mi = read_file(root + "/proc/meminfo");
    uint64_t kb = 0;
    if (meminfo_kb(mi, "MemTotal", kb))     m.total_bytes     = kb * 1024;
    if (meminfo_kb(mi, "MemAvailable", kb)) m.available_bytes = kb * 1024;
    else if (meminfo_kb(mi, "MemFree", kb)) {           // very old kernels lack MemAvailable
        uint64_t cached = 0; meminfo_kb(mi, "Cached", cached);
        m.available_bytes = (kb + cached) * 1024;
    }
    if (meminfo_kb(mi, "SwapTotal", kb)) m.swap_total_bytes = kb * 1024;

    // cgroup v2 first, then v1.
    uint64_t lim = parse_cgroup_limit(read_file(root + "/sys/fs/cgroup/memory.max"));
    uint64_t use = 0;
    if (lim > 0 || !read_trim(root + "/sys/fs/cgroup/memory.max").empty()) {
        parse_u64(read_trim(root + "/sys/fs/cgroup/memory.current"), use);
    } else {
        lim = parse_cgroup_limit(read_file(root + "/sys/fs/cgroup/memory/memory.limit_in_bytes"));
        parse_u64(read_trim(root + "/sys/fs/cgroup/memory/memory.usage_in_bytes"), use);
    }
    m.cgroup_limit_bytes = lim;
    m.cgroup_usage_bytes = lim ? use : 0;

    if (parse_u64(read_trim(root + "/proc/sys/vm/min_free_kbytes"), kb)) m.kernel_reserve_bytes = kb * 1024;
    m.low_ram_device = m.total_bytes > 0 && m.total_bytes < (3ull << 30);
    m.safe_budget_bytes = compute_safe_budget(m);
    return m;
}

inline std::vector<StorageDevice> read_storage(const std::string& root) {
    std::vector<StorageDevice> out;
    std::vector<std::string> names;
#if defined(__linux__)
    if (DIR* dir = opendir((root + "/sys/block").c_str())) {
        while (dirent* e = readdir(dir)) { std::string n = e->d_name; if (n != "." && n != "..") names.push_back(n); }
        closedir(dir);
    }
    std::sort(names.begin(), names.end());
#endif
    for (const std::string& n : names) {
        if (starts_with(n, "loop") || starts_with(n, "ram") || starts_with(n, "zram") || starts_with(n, "dm-") ||
            starts_with(n, "sr") || starts_with(n, "md") || starts_with(n, "nbd"))
            continue;
        const std::string b = root + "/sys/block/" + n;
        StorageDevice d;
        d.name = n;
        uint64_t sectors = 0;
        if (parse_u64(read_trim(b + "/size"), sectors)) d.size_bytes = sectors * 512;
        const bool virtual_disk = starts_with(n, "vd") || starts_with(n, "xvd");   // virtio/xen report rotational=1 regardless
        d.rotational = !virtual_disk && read_trim(b + "/queue/rotational") == "1";
        if (d.rotational)                      d.kind = StorageKind::HDD;
        else if (starts_with(n, "nvme"))       d.kind = StorageKind::NVME;
        else if (starts_with(n, "mmcblk"))     d.kind = StorageKind::EMMC;
        else if (starts_with(n, "sd")) d.kind = StorageKind::SATA_OR_UFS_FLASH;
        if (d.kind == StorageKind::NVME) {
            // The link speed lives on the PCI device, one level above the nvme controller.
            for (const char* rel : {"/device/device/current_link_speed", "/device/current_link_speed"}) {
                d.link_speed = read_trim(b + rel);
                if (!d.link_speed.empty()) break;
            }
        }
        if (d.size_bytes == 0 && d.kind == StorageKind::UNKNOWN) continue;
        out.push_back(std::move(d));
    }
    return out;
}

inline std::vector<ThermalZone> read_thermal(const std::string& root) {
    std::vector<ThermalZone> out;
    for (int i = 0; i < 64; ++i) {
        const std::string d = root + "/sys/class/thermal/thermal_zone" + std::to_string(i);
        const std::string type = read_trim(d + "/type");
        if (type.empty()) { if (i > 24) break; else continue; }
        double raw = 0;
        if (!parse_double(read_trim(d + "/temp"), raw)) continue;
        // sysfs reports millidegrees; some vendor nodes report whole degrees.
        const double c = std::abs(raw) > 1000.0 ? raw / 1000.0 : raw;
        if (c < -40.0 || c > 150.0) continue;    // broken sensor
        out.push_back({type, c});
    }
    return out;
}

inline BatteryInfo read_battery(const std::string& root) {
    BatteryInfo b;
    const std::string base = root + "/sys/class/power_supply/";
    for (const char* name : {"battery", "BAT0", "BAT1"}) {
        const std::string d = base + name;
        if (read_trim(d + "/type") != "Battery") continue;
        b.present = true;
        uint64_t v = 0;
        if (parse_u64(read_trim(d + "/capacity"), v)) b.level_pct = (int)v;
        double t = 0;
        if (parse_double(read_trim(d + "/temp"), t)) b.temp_c = t / 10.0;     // tenths of a degree
        b.status = read_trim(d + "/status");
        break;
    }
    return b;
}

inline bool is_android(const std::string& root) {
#if defined(__ANDROID__)
    (void)root; return true;
#else
    if (!read_file(root + "/system/build.prop").empty()) return true;
    return read_file(root + "/proc/sys/kernel/osrelease").find("android") != std::string::npos;
#endif
}

inline std::string os_pretty_name(const std::string& root) {
    for (const std::string& l : split_lines(read_file(root + "/etc/os-release")))
        if (starts_with(l, "PRETTY_NAME=")) {
            std::string v = trim(std::string_view(l).substr(12));
            if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
            return v;
        }
    return {};
}

}  // namespace linux_adapter

// ─── Thermal probe (used by the suite to stop before the device gets hot) ────

class ThermalProbe {
public:
    explicit ThermalProbe(std::string root = "") : root_(std::move(root)) {}
    struct Reading { double max_cpu_like_c = std::numeric_limits<double>::quiet_NaN();
                     double battery_c      = std::numeric_limits<double>::quiet_NaN(); bool ok = false; };
    Reading read() const {
        Reading r;
        for (const auto& z : linux_adapter::read_thermal(root_)) {
            const std::string n = to_lower(z.name);
            const bool battery_like = n.find("batt") != std::string::npos;
            if (battery_like) continue;
            if (!r.ok || z.temp_c > r.max_cpu_like_c) r.max_cpu_like_c = z.temp_c;
            r.ok = true;
        }
        BatteryInfo b = linux_adapter::read_battery(root_);
        if (b.present && std::isfinite(b.temp_c)) { r.battery_c = b.temp_c; r.ok = true; }
        return r;
    }
    // Hottest sensor we can see, NaN if none. On Android 7/8 only battery temp exists.
    double hottest_c() const {
        Reading r = read();
        double h = std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(r.max_cpu_like_c)) h = r.max_cpu_like_c;
        if (std::isfinite(r.battery_c))      h = std::isfinite(h) ? std::max(h, r.battery_c) : r.battery_c;
        return h;
    }
private:
    std::string root_;
};

// ─── Assembly ────────────────────────────────────────────────────────────────

inline Arch compiled_arch() {
#if defined(__x86_64__) || defined(_M_X64)
    return Arch::X86_64;
#elif defined(__i386__) || defined(_M_IX86)
    return Arch::X86;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return Arch::ARM64;
#elif defined(__arm__) || defined(_M_ARM)
    return Arch::ARM32;
#else
    return Arch::OTHER;
#endif
}

inline std::string compile_target_string() {
    std::string s = to_string(compiled_arch());
#if defined(__AVX512F__)
    s += "+avx512f";
#elif defined(__AVX2__)
    s += "+avx2";
#elif defined(__AVX__)
    s += "+avx";
#endif
#if defined(__ARM_NEON)
    s += "+neon";
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    s += "+dotprod";
#endif
#if defined(NDEBUG)
    s += " release";
#else
    s += " DEBUG";
#endif
    return s;
}

// Detect from a (possibly fake) filesystem root. root == "" means the real machine.
inline DeviceFingerprint detect_device_linux(const std::string& root = "") {
    DeviceFingerprint fp;
    using namespace linux_adapter;

    fp.os   = is_android(root) ? OS::Android : OS::Linux;
    fp.arch = compiled_arch();
    fp.compiler = 
#if defined(__VERSION__)
        __VERSION__;
#else
        "unknown";
#endif
    fp.compile_target = compile_target_string();
#if defined(__linux__)
    if (root.empty()) {
        utsname u; if (uname(&u) == 0) fp.kernel = u.release;
    }
#endif
    if (fp.kernel.empty()) fp.kernel = read_trim(root + "/proc/sys/kernel/osrelease");
    fp.os_release = os_pretty_name(root);

    // ---- cpuinfo ----
    const ParsedCpuInfo ci = parse_cpuinfo(read_file(root + "/proc/cpuinfo"));
    fp.vendor     = ci.vendor;
    fp.model_name = ci.model_name;
    fp.simd       = simd_from_flags(ci.common_flags, ci.vendor);
    if (fp.model_name.empty()) {
        std::string dt = read_file(root + "/proc/device-tree/model");
        dt = dt.substr(0, dt.find('\0'));
        fp.model_name = trim(dt);
    }
    if (ci.blocks.empty()) fp.notes.push_back("/proc/cpuinfo unreadable: CPU identity and SIMD flags unknown");
    if (ci.vendor == Vendor::ARM_LICENSED || ci.vendor == Vendor::APPLE)
        fp.arch = ci.arm_arch64 ? Arch::ARM64 : Arch::ARM32;   // runtime truth beats the compile target

    // ---- topology ----
    unsigned hw = std::thread::hardware_concurrency();
    std::vector<int> cpus = online_cpus(root, (int)hw ? (int)hw : (int)ci.blocks.size());
    fp.logical_cpus = (int)cpus.size();

    const std::string hybrid_core = read_trim(root + "/sys/devices/cpu_core/cpus");
    const std::string hybrid_atom = read_trim(root + "/sys/devices/cpu_atom/cpus");
    std::set<int> p_set, e_set;
    for (int c : parse_cpu_list(hybrid_core)) p_set.insert(c);
    for (int c : parse_cpu_list(hybrid_atom)) e_set.insert(c);

    std::vector<RawCore> raw;
    for (int c : cpus) {
        RawCore r;
        r.cpu = c;
        uint64_t v = 0;
        if (parse_u64(read_trim(cpu_dir(root, c) + "/cpu_capacity"), v)) r.capacity = (int)v;
        if (parse_u64(read_trim(cpu_dir(root, c) + "/cpufreq/cpuinfo_max_freq"), v)) r.max_freq_khz = v;
        // blocks are indexed by processor id; Arm part names parallel the blocks.
        {
            size_t k = 0;
            for (const auto& b : ci.blocks) {
                if (b.processor < 0) continue;
                if (b.processor == c && k < ci.block_part_names.size()) r.part = ci.block_part_names[k];
                ++k;
            }
        }
        r.hybrid_core = p_set.count(c) > 0 && !e_set.empty();
        r.hybrid_atom = e_set.count(c) > 0;
        if (r.max_freq_khz == 0) {      // VMs / no cpufreq: fall back to the "cpu MHz" line
            for (const auto& b : ci.blocks) if (b.processor == c) {
                auto it = b.kv.find("cpu MHz"); double mhz = 0;
                if (it != b.kv.end() && parse_double(it->second, mhz)) r.max_freq_khz = (uint64_t)(mhz * 1000.0);
            }
        }
        raw.push_back(r);
    }

    // Physical cores = distinct sibling sets.
    std::set<std::vector<int>> sibling_sets;
    for (int c : cpus) {
        std::string s = read_trim(cpu_dir(root, c) + "/topology/thread_siblings_list");
        if (s.empty()) s = read_trim(cpu_dir(root, c) + "/topology/core_cpus_list");
        std::vector<int> v = parse_cpu_list(s);
        if (v.empty()) v = {c};
        std::sort(v.begin(), v.end());
        sibling_sets.insert(v);
    }
    fp.physical_cores = (int)sibling_sets.size();

    // ---- clusters ----
    auto groups = group_cores(raw);
    std::string micro;
    for (size_t gi = 0; gi < groups.size(); ++gi) {
        const auto& g = groups[gi];
        CoreCluster cl;
        cl.cls = class_for_tier(gi, groups.size());
        uint64_t fmax = 0; int cap = 0;
        std::set<std::vector<int>> sib;
        for (const RawCore& r : g) {
            cl.cpus.push_back(r.cpu);
            fmax = std::max(fmax, r.max_freq_khz);
            cap  = std::max(cap, r.capacity);
            std::string s = read_trim(cpu_dir(root, r.cpu) + "/topology/thread_siblings_list");
            if (s.empty()) s = read_trim(cpu_dir(root, r.cpu) + "/topology/core_cpus_list");
            std::vector<int> v = parse_cpu_list(s); if (v.empty()) v = {r.cpu};
            std::sort(v.begin(), v.end()); sib.insert(v);
        }
        std::sort(cl.cpus.begin(), cl.cpus.end());
        cl.physical_cores = (int)sib.size();
        cl.max_freq_mhz   = (double)fmax / 1000.0;
        cl.capacity       = cap;
        cl.part_name      = g.front().part;

        // Caches from the first CPU of the cluster.
        for (const CacheLevel& c : read_caches(root, cl.cpus.front())) {
            if (c.type == "Instruction") continue;
            if (c.level == 1 && cl.l1d_bytes == 0) cl.l1d_bytes = c.size_bytes;
            else if (c.level == 2 && cl.l2_bytes == 0) { cl.l2_bytes = c.size_bytes; cl.l2_sharers = std::max<int>(1, (int)c.shared_cpus.size()); }
            else if (c.level == 3 && cl.l3_bytes == 0) cl.l3_bytes = c.size_bytes;
        }
#if defined(_SC_LEVEL1_DCACHE_SIZE) && defined(__linux__)
        if (root.empty() && cl.l1d_bytes == 0) { long v = sysconf(_SC_LEVEL1_DCACHE_SIZE); if (v > 0) cl.l1d_bytes = (uint64_t)v; }
#endif
        fp.clusters.push_back(std::move(cl));
    }
    if (fp.clusters.empty() && !cpus.empty()) {    // no usable info at all: one uniform cluster
        CoreCluster cl; cl.cpus = cpus; cl.physical_cores = fp.physical_cores;
        fp.clusters.push_back(cl);
        fp.notes.push_back("no cpufreq/cpu_capacity/part info: treating all cores as one uniform cluster");
    }

    // ---- microarchitecture string ----
    if (fp.vendor == Vendor::INTEL || fp.vendor == Vendor::AMD) {
        fp.microarch = x86_microarch(fp.vendor, ci.x86_family, ci.x86_model, fp.simd);
    } else {
        std::set<std::string> seen;
        for (const auto& c : fp.clusters)
            if (!c.part_name.empty() && seen.insert(c.part_name).second) { if (!micro.empty()) micro += '+'; micro += c.part_name; }
        fp.microarch = micro.empty() ? "unknown" : micro;
    }
    for (auto& c : fp.clusters) if (c.part_name.empty()) c.part_name = fp.microarch;

    // ---- memory, storage, thermal, battery ----
    fp.memory        = read_memory(root);
    fp.storage       = read_storage(root);
    fp.thermal_zones = read_thermal(root);
    fp.battery       = read_battery(root);
    if (fp.memory.total_bytes == 0)   fp.notes.push_back("/proc/meminfo unreadable: memory budget unknown");
    if (fp.thermal_zones.empty() && !fp.battery.present)
        fp.notes.push_back("no thermal zones or battery readable: the suite cannot watch temperature");
    return fp;
}

inline DeviceFingerprint detect_device(const std::string& root = "") {
#if defined(__linux__)
    return detect_device_linux(root);
#else
    DeviceFingerprint fp;
    fp.arch = compiled_arch();
    fp.logical_cpus = fp.physical_cores = (int)std::thread::hardware_concurrency();
    CoreCluster c; for (int i = 0; i < fp.logical_cpus; ++i) c.cpus.push_back(i);
    c.physical_cores = fp.logical_cpus; fp.clusters.push_back(c);
    fp.compile_target = compile_target_string();
    fp.notes.push_back("platform adapter not implemented (Apple/Windows are step 5 of the design): generic fingerprint only");
    (void)root;
    return fp;
#endif
}

// ─── JSON ────────────────────────────────────────────────────────────────────

inline void to_json(Json& j, const DeviceFingerprint& fp, const char* key = "device") {
    j.begin_object(key);
    j.str("os", to_string(fp.os)).str("arch", to_string(fp.arch)).str("vendor", to_string(fp.vendor));
    j.str("model", fp.model_name).str("microarch", fp.microarch).str("kernel", fp.kernel);
    j.str("os_release", fp.os_release).str("compile_target", fp.compile_target);
    j.integer("logical_cpus", fp.logical_cpus).integer("physical_cores", fp.physical_cores);
    j.str("simd", fp.simd.summary());
    j.begin_array("clusters");
    for (const auto& c : fp.clusters) {
        j.begin_object();
        j.str("class", to_string(c.cls)).str("part", c.part_name);
        j.integer("cpus", (int64_t)c.cpus.size()).integer("physical_cores", c.physical_cores);
        j.num("max_freq_mhz", c.max_freq_mhz).integer("capacity", c.capacity);
        j.uinteger("l1d", c.l1d_bytes).uinteger("l2", c.l2_bytes).integer("l2_sharers", c.l2_sharers).uinteger("l3", c.l3_bytes);
        j.end_object();
    }
    j.end_array();
    j.begin_object("memory");
    j.uinteger("total", fp.memory.total_bytes).uinteger("available", fp.memory.available_bytes);
    j.uinteger("cgroup_limit", fp.memory.cgroup_limit_bytes).uinteger("kernel_reserve", fp.memory.kernel_reserve_bytes);
    j.boolean("low_ram_device", fp.memory.low_ram_device).uinteger("safe_budget", fp.memory.safe_budget_bytes);
    j.end_object();
    j.begin_array("storage");
    for (const auto& s : fp.storage) {
        j.begin_object().str("name", s.name).str("kind", to_string(s.kind)).uinteger("size", s.size_bytes);
        if (!s.link_speed.empty()) j.str("link", s.link_speed);
        j.end_object();
    }
    j.end_array();
    j.begin_object("battery").boolean("present", fp.battery.present).integer("level_pct", fp.battery.level_pct)
        .num("temp_c", fp.battery.temp_c).str("status", fp.battery.status).end_object();
    j.integer("thermal_zones", (int64_t)fp.thermal_zones.size());
    j.begin_array("notes");
    for (const auto& n : fp.notes) { j.str(nullptr, n); }
    j.end_array();
    j.end_object();
}

}  // namespace tensorf::prof
