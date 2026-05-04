#pragma once

/*
 * HardwareFingerprint.hpp
 * =======================
 * Captures the complete hardware profile of the client machine:
 *   - CPU model, generation, micro-architecture, ISA extensions
 *   - Cache hierarchy (L1/L2/L3 sizes)
 *   - RAM: type (DDR4/DDR5), frequency, channels, total, bandwidth
 *   - Storage: type (HDD/SATA SSD/NVMe Gen3/Gen4/Gen5), sequential R/W
 *   - Software versions: compiler, BLAS, OS
 *
 * All detection is done without external dependencies beyond Linux sysfs/procfs.
 * This fingerprint drives every downstream decision in HyperparamAdvisor.
 */

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <optional>
#include <unistd.h>
#include <sys/utsname.h>

// ─── Data structures ─────────────────────────────────────────────────────────

enum class ISA {
    BASELINE,   // SSE2 only
    AVX,        // 256-bit float
    AVX2,       // 256-bit int + FMA
    AVX512,     // 512-bit, doubles throughput vs AVX2
    AVX512_VNNI,// int8 dot product native (Ice Lake+)
    AMX         // Tile matmul (Sapphire Rapids+), ~10x int8
};

enum class CPUVendor { INTEL, AMD, ARM, UNKNOWN };

enum class CPUGen {
    UNKNOWN,
    // Intel
    INTEL_PRE_HASWELL,   // < 2013 — no AVX2
    INTEL_HASWELL,       // 2013 — AVX2 + FMA
    INTEL_SKYLAKE,       // 2015 — AVX-512 on server (Skylake-X)
    INTEL_ICE_LAKE,      // 2019 — AVX-512 + VNNI
    INTEL_ALDER_LAKE,    // 2021 — hybrid, AVX-512 disabled
    INTEL_SAPPHIRE_RAPIDS,// 2023 — AMX + full AVX-512
    // AMD
    AMD_ZEN1,            // 2017 — AVX2
    AMD_ZEN2,            // 2019 — AVX2, improved throughput
    AMD_ZEN3,            // 2020 — +19% IPC, better L3
    AMD_ZEN4,            // 2022 — AVX-512 native, DDR5
    AMD_ZEN5,            // 2024 — doubled AVX-512 width
};

enum class RAMType { UNKNOWN, DDR3, DDR4, DDR5, LPDDR4, LPDDR5 };

enum class StorageType {
    HDD,        // rotational — ~150 MB/s
    SATA_SSD,   // ~550 MB/s
    NVME_GEN3,  // ~3.5 GB/s
    NVME_GEN4,  // ~7 GB/s
    NVME_GEN5,  // ~14 GB/s
    UNKNOWN
};

struct CacheInfo {
    size_t l1d_bytes  = 0;
    size_t l2_bytes   = 0;
    size_t l3_bytes   = 0;
};

struct RAMInfo {
    RAMType   type          = RAMType::UNKNOWN;
    uint32_t  speed_mts     = 0;     // MT/s (DDR4-3600 → 3600)
    uint32_t  total_mb      = 0;
    uint32_t  available_mb  = 0;
    uint8_t   channels      = 1;
    double    bandwidth_gbs = 0.0;   // theoretical: speed × width × channels / 8
};

struct StorageDevice {
    std::string  name;
    StorageType  type       = StorageType::UNKNOWN;
    uint64_t     size_bytes = 0;
    double       seq_read_mbs  = 0.0;   // measured
    double       seq_write_mbs = 0.0;
};

struct SoftwareInfo {
    std::string compiler_version;
    std::string os_release;
    std::string blas_backend;    // "OpenBLAS" | "MKL" | "unknown"
    std::string kernel_version;
};

struct CPUInfo {
    CPUVendor   vendor    = CPUVendor::UNKNOWN;
    CPUGen      gen       = CPUGen::UNKNOWN;
    std::string model_name;
    uint32_t    physical_cores = 0;
    uint32_t    logical_cores  = 0;
    double      base_freq_mhz  = 0.0;
    double      max_freq_mhz   = 0.0;

    // ISA flags
    bool has_sse2     = false;
    bool has_avx      = false;
    bool has_avx2     = false;
    bool has_fma      = false;
    bool has_avx512f  = false;
    bool has_avx512_vnni = false;
    bool has_amx      = false;

    ISA best_isa() const {
        if (has_amx)          return ISA::AMX;
        if (has_avx512_vnni)  return ISA::AVX512_VNNI;
        if (has_avx512f)      return ISA::AVX512;
        if (has_avx2)         return ISA::AVX2;
        if (has_avx)          return ISA::AVX;
        return ISA::BASELINE;
    }

    // Natural vector width in bytes for float32
    size_t vector_width_bytes() const {
        if (has_avx512f) return 64;
        if (has_avx)     return 32;
        return 16; // SSE2
    }

    // Optimal n_embed alignment: must be multiple of floats per vector register
    size_t embed_alignment() const {
        return vector_width_bytes() / sizeof(float);
    }
};

struct HardwareFingerprint {
    CPUInfo                  cpu;
    CacheInfo                cache;
    RAMInfo                  ram;
    std::vector<StorageDevice> storage;
    SoftwareInfo             software;

    // Derived score 0–100 (higher = more capable client)
    int capability_score = 0;
};

// ─── Helper: read a whole file into string ───────────────────────────────────

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string exec_cmd(const std::string& cmd) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {};
    char buf[256];
    std::string result;
    while (fgets(buf, sizeof(buf), pipe))
        result += buf;
    pclose(pipe);
    // trim trailing newline
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}

// ─── CPU Detection ───────────────────────────────────────────────────────────

static CPUInfo detect_cpu() {
    CPUInfo cpu;
    std::string cpuinfo = read_file("/proc/cpuinfo");
    if (cpuinfo.empty()) return cpu;

    // Model name
    auto find_field = [&](const std::string& key) -> std::string {
        size_t pos = cpuinfo.find(key);
        if (pos == std::string::npos) return {};
        size_t colon = cpuinfo.find(':', pos);
        if (colon == std::string::npos) return {};
        size_t end = cpuinfo.find('\n', colon);
        std::string val = cpuinfo.substr(colon + 1, end - colon - 1);
        // ltrim
        size_t start = val.find_first_not_of(" \t");
        return (start == std::string::npos) ? "" : val.substr(start);
    };

    cpu.model_name = find_field("model name");

    // Vendor
    std::string vendor_id = find_field("vendor_id");
    if (vendor_id.find("GenuineIntel") != std::string::npos)
        cpu.vendor = CPUVendor::INTEL;
    else if (vendor_id.find("AuthenticAMD") != std::string::npos)
        cpu.vendor = CPUVendor::AMD;

    // ISA flags (first processor block is enough)
    std::string flags_line = find_field("flags");
    auto has_flag = [&](const std::string& f) {
        return flags_line.find(" " + f + " ") != std::string::npos
            || flags_line.find(" " + f + "\n") != std::string::npos
            || flags_line.find(f + " ") == 0;
    };
    cpu.has_sse2        = has_flag("sse2");
    cpu.has_avx         = has_flag("avx");
    cpu.has_avx2        = has_flag("avx2");
    cpu.has_fma         = has_flag("fma");
    cpu.has_avx512f     = has_flag("avx512f");
    cpu.has_avx512_vnni = has_flag("avx512_vnni");
    cpu.has_amx         = has_flag("amx_bf16") || has_flag("amx_int8");

    // Logical cores
    cpu.logical_cores = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);

    // Physical cores
    std::string phys = exec_cmd(
        "lscpu 2>/dev/null | grep '^Core(s) per socket' | awk '{print $NF}'");
    std::string sockets = exec_cmd(
        "lscpu 2>/dev/null | grep '^Socket(s)' | awk '{print $NF}'");
    if (!phys.empty() && !sockets.empty())
        cpu.physical_cores = (uint32_t)(std::stoi(phys) * std::stoi(sockets));
    else
        cpu.physical_cores = cpu.logical_cores;

    // Frequencies
    std::string max_f = exec_cmd(
        "lscpu 2>/dev/null | grep 'CPU max MHz' | awk '{print $NF}'");
    std::string base_f = exec_cmd(
        "lscpu 2>/dev/null | grep 'CPU MHz' | awk '{print $NF}'");
    if (!max_f.empty())  cpu.max_freq_mhz  = std::stod(max_f);
    if (!base_f.empty()) cpu.base_freq_mhz = std::stod(base_f);

    // CPU Generation heuristic
    if (cpu.vendor == CPUVendor::INTEL) {
        if      (cpu.has_amx)          cpu.gen = CPUGen::INTEL_SAPPHIRE_RAPIDS;
        else if (cpu.has_avx512_vnni)  cpu.gen = CPUGen::INTEL_ICE_LAKE;
        else if (cpu.has_avx512f)      cpu.gen = CPUGen::INTEL_SKYLAKE;
        else if (cpu.has_avx2 && cpu.model_name.find("12th") != std::string::npos)
                                        cpu.gen = CPUGen::INTEL_ALDER_LAKE;
        else if (cpu.has_avx2)         cpu.gen = CPUGen::INTEL_HASWELL;
        else                           cpu.gen = CPUGen::INTEL_PRE_HASWELL;
    } else if (cpu.vendor == CPUVendor::AMD) {
        // Detect Zen generation from model name keywords
        const std::string& m = cpu.model_name;
        if      (m.find("9") != std::string::npos && cpu.has_avx512f)
                                        cpu.gen = CPUGen::AMD_ZEN5;
        else if (cpu.has_avx512f)       cpu.gen = CPUGen::AMD_ZEN4;
        else if (m.find("5000") != std::string::npos ||
                 m.find("5600") != std::string::npos ||
                 m.find("5800") != std::string::npos ||
                 m.find("5900") != std::string::npos ||
                 m.find("5950") != std::string::npos)
                                        cpu.gen = CPUGen::AMD_ZEN3;
        else if (m.find("3000") != std::string::npos ||
                 m.find("3600") != std::string::npos ||
                 m.find("3700") != std::string::npos ||
                 m.find("3900") != std::string::npos)
                                        cpu.gen = CPUGen::AMD_ZEN2;
        else if (cpu.has_avx2)          cpu.gen = CPUGen::AMD_ZEN1;
    }

    return cpu;
}

// ─── Cache Detection ─────────────────────────────────────────────────────────

static CacheInfo detect_cache() {
    CacheInfo c;

    // /sys/devices/system/cpu/cpu0/cache/index*/
    auto read_cache_level = [](int idx) -> std::pair<std::string, size_t> {
        std::string base = "/sys/devices/system/cpu/cpu0/cache/index"
                         + std::to_string(idx);
        std::string level = read_file(base + "/level");
        std::string type  = read_file(base + "/type");
        std::string size  = read_file(base + "/size");
        while (!level.empty() && level.back() == '\n') level.pop_back();
        while (!type.empty()  && type.back()  == '\n') type.pop_back();
        while (!size.empty()  && size.back()  == '\n') size.pop_back();

        if (type == "Instruction") return {"", 0};  // skip I-cache

        size_t bytes = 0;
        if (!size.empty()) {
            size_t val = std::stoull(size);
            if (size.back() == 'K') val *= 1024;
            else if (size.back() == 'M') val *= 1024 * 1024;
            bytes = val;
        }
        return {level, bytes};
    };

    // Scan up to 8 cache levels
    for (int i = 0; i < 8; i++) {
        auto [lvl, bytes] = read_cache_level(i);
        if (lvl.empty() && bytes == 0) break;
        if      (lvl == "1" && c.l1d_bytes == 0) c.l1d_bytes = bytes;
        else if (lvl == "2" && c.l2_bytes  == 0) c.l2_bytes  = bytes;
        else if (lvl == "3" && c.l3_bytes  == 0) c.l3_bytes  = bytes;
    }

    // Fallback: getconf
    if (c.l1d_bytes == 0) {
        std::string v = exec_cmd("getconf LEVEL1_DCACHE_SIZE 2>/dev/null");
        if (!v.empty() && v != "undefined") c.l1d_bytes = std::stoull(v);
    }
    if (c.l3_bytes == 0) {
        std::string v = exec_cmd("getconf LEVEL3_CACHE_SIZE 2>/dev/null");
        if (!v.empty() && v != "undefined") c.l3_bytes = std::stoull(v);
    }

    return c;
}

// ─── RAM Detection ───────────────────────────────────────────────────────────

static RAMInfo detect_ram() {
    RAMInfo ram;

    // Total + available from /proc/meminfo
    std::string meminfo = read_file("/proc/meminfo");
    auto parse_kb = [&](const std::string& key) -> uint64_t {
        size_t pos = meminfo.find(key);
        if (pos == std::string::npos) return 0;
        size_t colon = meminfo.find(':', pos);
        std::string rest = meminfo.substr(colon + 1);
        uint64_t kb = std::stoull(rest);
        return kb;
    };
    ram.total_mb     = (uint32_t)(parse_kb("MemTotal") / 1024);
    ram.available_mb = (uint32_t)(parse_kb("MemAvailable") / 1024);

    // Type and speed via dmidecode (needs root — gracefully degrade)
    std::string dmi = exec_cmd("dmidecode -t memory 2>/dev/null");
    if (!dmi.empty()) {
        // Type
        if      (dmi.find("DDR5")   != std::string::npos) ram.type = RAMType::DDR5;
        else if (dmi.find("LPDDR5") != std::string::npos) ram.type = RAMType::LPDDR5;
        else if (dmi.find("LPDDR4") != std::string::npos) ram.type = RAMType::LPDDR4;
        else if (dmi.find("DDR4")   != std::string::npos) ram.type = RAMType::DDR4;
        else if (dmi.find("DDR3")   != std::string::npos) ram.type = RAMType::DDR3;

        // Speed — take the max of all populated slots
        uint32_t max_speed = 0;
        uint32_t slots_populated = 0;
        size_t pos = 0;
        while ((pos = dmi.find("Speed:", pos)) != std::string::npos) {
            std::string line = dmi.substr(pos, 60);
            uint32_t sp = 0;
            if (sscanf(line.c_str(), "Speed: %u MT/s", &sp) == 1 && sp > 0) {
                max_speed = std::max(max_speed, sp);
                slots_populated++;
            }
            pos += 6;
        }
        ram.speed_mts = max_speed;
        ram.channels  = (uint8_t)(slots_populated >= 4 ? 4 :
                                  slots_populated >= 2 ? 2 : 1);
    }

    // Theoretical bandwidth: speed_MT/s × bus_width(8B) × channels / 1000
    if (ram.speed_mts > 0) {
        ram.bandwidth_gbs = (double)ram.speed_mts * 8.0 * ram.channels / 1000.0;
    }

    return ram;
}

// ─── Storage Detection ───────────────────────────────────────────────────────

static std::vector<StorageDevice> detect_storage() {
    std::vector<StorageDevice> devs;

    // lsblk: NAME ROTA SIZE TYPE TRAN
    std::string lsblk = exec_cmd(
        "lsblk -d -b -o NAME,ROTA,SIZE,TYPE,TRAN 2>/dev/null");
    if (lsblk.empty()) return devs;

    std::istringstream ss(lsblk);
    std::string line;
    std::getline(ss, line); // skip header
    while (std::getline(ss, line)) {
        std::istringstream ls(line);
        std::string name, rota, size_str, type, tran;
        ls >> name >> rota >> size_str >> type >> tran;

        if (type != "disk") continue;

        StorageDevice d;
        d.name = name;
        try { d.size_bytes = std::stoull(size_str); } catch (...) {}

        bool rotational = (rota == "1");
        if (rotational) {
            d.type = StorageType::HDD;
        } else {
            // Distinguish SATA SSD vs NVMe by transport field or name
            if (tran == "nvme" || name.find("nvme") != std::string::npos) {
                // Try to detect PCIe gen from sysfs
                std::string link_speed = read_file(
                    "/sys/block/" + name + "/device/current_link_speed");
                while (!link_speed.empty() && link_speed.back() == '\n')
                    link_speed.pop_back();

                // PCIe gen from link speed string: "2.5 GT/s", "5.0 GT/s",
                // "8.0 GT/s", "16.0 GT/s", "32.0 GT/s"
                if (link_speed.find("32.0") != std::string::npos)
                    d.type = StorageType::NVME_GEN5;
                else if (link_speed.find("16.0") != std::string::npos)
                    d.type = StorageType::NVME_GEN4;
                else if (link_speed.find("8.0") != std::string::npos)
                    d.type = StorageType::NVME_GEN3;
                else
                    d.type = StorageType::NVME_GEN3; // conservative
            } else {
                d.type = StorageType::SATA_SSD;
            }
        }

        devs.push_back(d);
    }

    return devs;
}

// ─── Software Detection ──────────────────────────────────────────────────────

static SoftwareInfo detect_software() {
    SoftwareInfo sw;

    sw.compiler_version = exec_cmd("g++ --version 2>/dev/null | head -1");

    struct utsname uts;
    if (uname(&uts) == 0)
        sw.kernel_version = std::string(uts.release);

    sw.os_release = exec_cmd("lsb_release -ds 2>/dev/null");
    if (sw.os_release.empty())
        sw.os_release = read_file("/etc/os-release").substr(0, 80);

    // Detect BLAS: check if libopenblas or libmkl is loaded or available
    std::string blas = exec_cmd(
        "ldconfig -p 2>/dev/null | grep -oE '(openblas|mkl)' | head -1");
    if      (blas.find("mkl")      != std::string::npos) sw.blas_backend = "MKL";
    else if (blas.find("openblas") != std::string::npos) sw.blas_backend = "OpenBLAS";
    else                                                  sw.blas_backend = "unknown";

    return sw;
}

// ─── Capability Score ────────────────────────────────────────────────────────

static int compute_capability_score(const HardwareFingerprint& fp) {
    int score = 0;

    // ISA (0–30)
    switch (fp.cpu.best_isa()) {
        case ISA::AMX:          score += 30; break;
        case ISA::AVX512_VNNI:  score += 25; break;
        case ISA::AVX512:       score += 22; break;
        case ISA::AVX2:         score += 15; break;
        case ISA::AVX:          score += 8;  break;
        default:                score += 0;  break;
    }

    // RAM bandwidth (0–20)
    double bw = fp.ram.bandwidth_gbs;
    if      (bw >= 100) score += 20;
    else if (bw >= 60)  score += 15;
    else if (bw >= 40)  score += 10;
    else if (bw >= 20)  score += 5;

    // RAM total (0–15)
    uint32_t ram_mb = fp.ram.total_mb;
    if      (ram_mb >= 65536) score += 15;
    else if (ram_mb >= 32768) score += 12;
    else if (ram_mb >= 16384) score += 8;
    else if (ram_mb >= 8192)  score += 4;

    // L3 cache (0–15)
    size_t l3 = fp.cache.l3_bytes;
    if      (l3 >= 64 * 1024 * 1024) score += 15;
    else if (l3 >= 32 * 1024 * 1024) score += 12;
    else if (l3 >= 16 * 1024 * 1024) score += 8;
    else if (l3 >= 8  * 1024 * 1024) score += 4;

    // Storage (0–10)
    for (const auto& s : fp.storage) {
        switch (s.type) {
            case StorageType::NVME_GEN5: score += 10; break;
            case StorageType::NVME_GEN4: score += 8;  break;
            case StorageType::NVME_GEN3: score += 6;  break;
            case StorageType::SATA_SSD:  score += 3;  break;
            default: break;
        }
        break; // only count primary drive
    }

    // CPU cores (0–10)
    if      (fp.cpu.physical_cores >= 32) score += 10;
    else if (fp.cpu.physical_cores >= 16) score += 8;
    else if (fp.cpu.physical_cores >= 8)  score += 5;
    else if (fp.cpu.physical_cores >= 4)  score += 2;

    return std::min(score, 100);
}

// ─── Main entry point ────────────────────────────────────────────────────────

inline HardwareFingerprint detect_hardware() {
    HardwareFingerprint fp;
    fp.cpu      = detect_cpu();
    fp.cache    = detect_cache();
    fp.ram      = detect_ram();
    fp.storage  = detect_storage();
    fp.software = detect_software();
    fp.capability_score = compute_capability_score(fp);
    return fp;
}
