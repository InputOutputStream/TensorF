// HardwareFingerprint.hpp
//
// Facts about the device the library is running on. Nothing here measures
// performance (that is DeviceBenchmark's job); it records what the device IS and
// what it will currently ALLOW us to use.
//
// Design rules
//   * No shell-outs (no popen/lscpu/dmidecode/lsblk). Everything comes from
//     /proc, /sys, system properties or the platform C API, so it works inside
//     an Android app sandbox and does not need root.
//   * Every field that can be unreadable is optional or has an explicit
//     "unknown" value. A phone that hides /sys/.../cpufreq still produces a
//     usable fingerprint; the missing pieces are listed in `notes`.
//   * Parsing is separated from reading. All parsers take a string, and the
//     Linux/Android collector reads through the FsSource interface, so the
//     phone code paths are unit-tested on a desktop with a fake /sys tree
//     (see tests/hw_fingerprint_test.cpp).
//   * big.LITTLE is first-class: cores are grouped into CoreClass entries
//     (little / mid / big / prime) with their own caches and frequencies,
//     because the thread pools and the governor must treat them differently.
//   * Features are reported as an intersection across cores (safe to dispatch
//     on when threads may migrate) and a union (what some core can do).
//
// Platform status
//   Linux (desktop/server) and Android : implemented, tested here (Android
//                                        through a fake sysfs, not a device).
//   macOS / iOS                        : implemented from the sysctl/mach API,
//                                        NOT compiled or run here.
//   Windows                            : implemented from the Win32 API,
//                                        NOT compiled or run here.
//   Thermal and battery on iOS, and battery on Android 8+ apps, are normally
//   only available through the OS framework (BatteryManager, thermal status);
//   the app shell can fill HardwareFingerprint::power / thermal itself and the
//   rest of the library treats them the same way.
#pragma once

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

#if defined(__APPLE__)
  #include <TargetConditionals.h>
  #include <mach/mach.h>
  #include <sys/statvfs.h>
  #include <sys/sysctl.h>
  #if TARGET_OS_IPHONE
    #include <os/proc.h>
  #endif
#elif defined(_WIN32)
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  #pragma comment(lib, "advapi32.lib")
#else
  #include <dirent.h>
  #include <sched.h>
  #include <sys/statvfs.h>
  #include <sys/utsname.h>
  #include <unistd.h>
  #if defined(__ANDROID__)
    #include <sys/system_properties.h>
  #endif
#endif

namespace tensorf::profile {

inline constexpr int kFingerprintSchema = 2;

// ───────────────────────────── data model ─────────────────────────────

struct CacheInfo {
    int level = 0;                  // 1, 2, 3
    std::string type;               // "Data" | "Instruction" | "Unified"
    uint64_t size_bytes = 0;
    int line_bytes = 0;
    int ways = 0;
    std::vector<int> shared_cpus;   // cpus sharing this cache (empty = unknown)
};

struct CoreInfo {
    int id = -1;
    bool online = true;
    uint32_t implementer = 0;       // ARM MIDR implementer (0 on x86)
    uint32_t part = 0;              // ARM MIDR part number
    std::string uarch;              // "Cortex-A55", "Cortex-X1", ... or ""
    uint64_t max_khz = 0;           // 0 = unreadable
    uint64_t min_khz = 0;
    uint64_t observed_khz = 0;      // x86 "cpu MHz": a current value, only a hint
    int capacity = 0;               // Linux cpu_capacity, 0..1024, 0 = unknown
    int tier = 1;                   // coarse uarch tier 0 efficiency / 1 mid / 2 big, used to order classes when capacity and frequency are hidden
};

struct CoreClass {
    int index = 0;                  // 0 = slowest class
    std::string name;               // "little" | "mid" | "big" | "prime" | "uniform"
    std::vector<int> cpus;
    uint64_t max_khz = 0;
    int capacity = 0;
    std::string uarch;
    std::vector<CacheInfo> caches;  // caches as seen from the first cpu of the class
    // Relative speed of this class, 0..1, 1 = fastest class. Taken from
    // cpu_capacity when present, otherwise from max frequency, otherwise 1.
    double rel_perf = 1.0;
    int tier = 1;                   // see CoreInfo::tier
};

struct SimdFeatures {
    // x86
    bool sse42 = false, avx = false, avx2 = false, fma = false, f16c = false;
    bool avx512f = false, avx512_vnni = false, avx512_bf16 = false, avx512_fp16 = false;
    bool avx_vnni = false, amx = false;
    // ARM
    bool neon = false, fp16 = false, dotprod = false, i8mm = false, bf16 = false;
    bool sve = false, sve2 = false, sme = false;
};

struct CpuInfo {
    std::string arch;               // "x86_64" "x86" "aarch64" "armv7" "riscv64" "unknown"
    int pointer_bits = int(sizeof(void*) * 8);
    std::string vendor;             // "GenuineIntel", "AuthenticAMD", "ARM", "Qualcomm", ...
    std::string model_name;
    std::string soc_hint;           // "Hardware" line on ARM, board/soc property on Android
    int logical_cpus = 0;           // present cpus (offline ones included)
    int online_cpus = 0;
    std::vector<CoreInfo> cores;
    std::vector<CoreClass> classes; // ascending speed
    std::set<std::string> features_all;  // present on every core with a feature list
    std::set<std::string> features_any;  // present on at least one core
    SimdFeatures simd;              // derived from features_all
};

struct MemoryInfo {
    uint64_t total = 0;
    uint64_t available = 0;         // kernel estimate of memory allocatable without swapping
    uint64_t swap_total = 0;
    uint64_t swap_free = 0;
    bool swap_is_zram = false;      // compressed RAM swap: counts against RAM, not "extra"
    std::optional<uint64_t> cgroup_limit;   // container / app limit, if one applies
    std::optional<uint64_t> cgroup_usage;
    std::optional<double> psi_some_avg10;   // memory pressure stall %, if readable
    std::optional<uint64_t> lmk_min_free_bytes;  // Android lowmemorykiller minfree, last tier
    uint64_t page_bytes = 4096;
    // What we may plan to use right now: available, further capped by the
    // cgroup headroom (limit - usage). Conservative: cgroup usage includes
    // reclaimable page cache.
    uint64_t effective_available = 0;
};

struct StorageInfo {
    std::string path;
    uint64_t total = 0;
    uint64_t free = 0;
    bool known = false;
};

struct ThermalZone {
    std::string name;
    double temp_c = 0;
};

struct PowerState {
    bool available = false;         // false = unreadable (not "no battery")
    bool battery_present = false;
    bool on_external_power = false;
    std::optional<int> level_pct;
    std::string status;             // "Charging" "Discharging" "Full" "Not charging"
    std::optional<double> battery_temp_c;
    std::optional<double> voltage_mv;
    std::optional<double> current_ma;       // sign convention is vendor specific
};

struct OsInfo {
    std::string name;               // "Linux" "Android" "macOS" "iOS" "Windows"
    std::string version;
    std::string kernel;
    int android_api = 0;            // 0 when not Android
    std::string manufacturer;
    std::string model;
    // "phone" "tablet" "laptop" "desktop" "server" "unknown": a heuristic, used
    // only to pick default budgets, never to gate behaviour.
    std::string form_factor = "unknown";
    bool in_container = false;
};

struct HardwareFingerprint {
    int schema = kFingerprintSchema;
    OsInfo os;
    CpuInfo cpu;
    MemoryInfo memory;
    StorageInfo storage;
    PowerState power;
    std::vector<ThermalZone> thermal;
    int usable_threads = 1;         // min(online, affinity, cgroup cpu quota), >= 1
    std::vector<std::string> notes; // what could not be read and why it matters

    // Stable key for caching benchmark results: same device model + same core
    // layout + same RAM size bucket gives the same id; free memory, temperature
    // and battery do not enter it.
    std::string id() const;
    // CPUs of the fastest class (and, if `include_mid`, the class below it).
    std::vector<int> performance_cpus(bool include_mid = false) const;
    uint64_t usable_memory_bytes(double fraction) const {
        return static_cast<uint64_t>(double(memory.effective_available) * fraction);
    }
    std::string to_json() const;
    std::string summary() const;
};

// ───────────────────────────── file access ─────────────────────────────

struct FsSource {
    virtual ~FsSource() = default;
    virtual std::optional<std::string> read(const std::string& path) = 0;
    virtual std::vector<std::string> list(const std::string& dir) = 0;  // child names
};

#if !defined(_WIN32)
struct RealFs final : FsSource {
    std::optional<std::string> read(const std::string& path) override {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return std::nullopt;
        std::string out;
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) {
            out.append(buf, n);
            if (out.size() > (4u << 20)) break;     // /proc files are small
        }
        std::fclose(f);
        return out;
    }
    std::vector<std::string> list(const std::string& dir) override {
        std::vector<std::string> out;
        if (DIR* d = ::opendir(dir.c_str())) {
            while (dirent* e = ::readdir(d)) {
                std::string n = e->d_name;
                if (n != "." && n != "..") out.push_back(n);
            }
            ::closedir(d);
        }
        std::sort(out.begin(), out.end());
        return out;
    }
};
#endif

// In-memory file tree, for tests and for replaying a dump taken from a device.
struct MemFs final : FsSource {
    std::map<std::string, std::string> files;
    void put(const std::string& path, const std::string& content) { files[path] = content; }
    std::optional<std::string> read(const std::string& path) override {
        auto it = files.find(path);
        if (it == files.end()) return std::nullopt;
        return it->second;
    }
    std::vector<std::string> list(const std::string& dir) override {
        std::string pre = dir;
        if (pre.empty() || pre.back() != '/') pre += '/';
        std::set<std::string> names;
        for (auto it = files.lower_bound(pre); it != files.end() && it->first.compare(0, pre.size(), pre) == 0; ++it) {
            std::string rest = it->first.substr(pre.size());
            names.insert(rest.substr(0, rest.find('/')));
        }
        return {names.begin(), names.end()};
    }
};

struct DiskSpace { uint64_t total = 0, free = 0; };

// Everything the Linux/Android collector needs from the outside world.
struct Probe {
    FsSource* fs = nullptr;
    std::function<std::string(const std::string&)> prop;              // Android system property, "" if none
    std::function<std::optional<DiskSpace>(const std::string&)> disk;  // free space of a path
    std::function<int()> affinity_cpus;                               // cpus this process may run on, <=0 unknown
    std::string arch;                                                  // compiled architecture
    bool is_android = false;
    std::string kernel;                                                // uname release, optional
};

// ───────────────────────────── parsing helpers ─────────────────────────────
namespace detail {

inline std::string trim(std::string s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

inline std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == sep) { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    out.push_back(cur);
    return out;
}

inline std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream is(s);
    std::string w;
    while (is >> w) out.push_back(w);
    return out;
}

inline bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

// Accepts decimal or 0x-prefixed hex. Rejects garbage and trailing junk.
inline std::optional<uint64_t> parse_u64(const std::string& raw) {
    std::string s = trim(raw);
    if (s.empty() || s[0] == '-') return std::nullopt;
    char* end = nullptr;
    errno = 0;
    unsigned long long v = std::strtoull(s.c_str(), &end, 0);
    if (errno != 0 || end == s.c_str() || *end != '\0') return std::nullopt;
    return static_cast<uint64_t>(v);
}

inline std::optional<int64_t> parse_i64(const std::string& raw) {
    std::string s = trim(raw);
    if (s.empty()) return std::nullopt;
    char* end = nullptr;
    errno = 0;
    long long v = std::strtoll(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0') return std::nullopt;
    return static_cast<int64_t>(v);
}

inline std::optional<double> parse_double(const std::string& raw) {
    std::string s = trim(raw);
    if (s.empty()) return std::nullopt;
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0' || !std::isfinite(v)) return std::nullopt;
    return v;
}

// "0-3,6,8-9" -> {0,1,2,3,6,8,9}
inline std::vector<int> parse_cpulist(const std::string& raw) {
    std::vector<int> out;
    for (const auto& part : split(trim(raw), ',')) {
        if (part.empty()) continue;
        auto dash = part.find('-');
        if (dash == std::string::npos) {
            if (auto v = parse_u64(part)) out.push_back(int(*v));
        } else {
            auto a = parse_u64(part.substr(0, dash)), b = parse_u64(part.substr(dash + 1));
            if (a && b && *b >= *a && *b - *a < 4096)
                for (uint64_t i = *a; i <= *b; ++i) out.push_back(int(i));
        }
    }
    return out;
}

// "32K" "1M" "512" "2048K" -> bytes (sysfs cache sizes)
inline uint64_t parse_size(const std::string& raw) {
    std::string s = trim(raw);
    if (s.empty()) return 0;
    uint64_t mult = 1;
    char last = s.back();
    if (last == 'K' || last == 'k') mult = 1024, s.pop_back();
    else if (last == 'M' || last == 'm') mult = 1024 * 1024, s.pop_back();
    else if (last == 'G' || last == 'g') mult = 1024ull * 1024 * 1024, s.pop_back();
    auto v = parse_u64(s);
    return v ? *v * mult : 0;
}

// "MemTotal:  8000000 kB" lines -> bytes by key (kB converted)
inline std::map<std::string, uint64_t> parse_meminfo(const std::string& text) {
    std::map<std::string, uint64_t> out;
    for (const auto& line : split(text, '\n')) {
        auto c = line.find(':');
        if (c == std::string::npos) continue;
        auto toks = split_ws(line.substr(c + 1));
        if (toks.empty()) continue;
        auto v = parse_u64(toks[0]);
        if (!v) continue;
        uint64_t mul = (toks.size() > 1 && (toks[1] == "kB" || toks[1] == "KB")) ? 1024 : 1;
        out[line.substr(0, c)] = *v * mul;
    }
    return out;
}

// /proc/cpuinfo -> one key/value map per "processor" block.
using KV = std::map<std::string, std::string>;
inline std::vector<KV> parse_cpuinfo(const std::string& text) {
    std::vector<KV> blocks;
    KV cur;
    auto flush = [&] { if (!cur.empty()) { blocks.push_back(std::move(cur)); cur.clear(); } };
    for (const auto& line : split(text, '\n')) {
        if (trim(line).empty()) { flush(); continue; }
        auto c = line.find(':');
        if (c == std::string::npos) continue;
        std::string k = trim(line.substr(0, c)), v = trim(line.substr(c + 1));
        // Older 32-bit ARM kernels print "Hardware" / "Processor" once, after
        // the per-cpu blocks, with no blank-line separation needed here.
        cur[k] = v;
    }
    flush();
    return blocks;
}

struct ArmPart { uint32_t implementer, part; const char* name; int tier; };  // tier: 0 efficiency, 1 mid, 2 big/prime
inline const ArmPart* arm_lookup(uint32_t implementer, uint32_t part);
inline const char* arm_uarch_name(uint32_t implementer, uint32_t part) {
    const ArmPart* e = arm_lookup(implementer, part);
    return e ? e->name : "";
}
inline const ArmPart* arm_lookup(uint32_t implementer, uint32_t part) {
    static const ArmPart table[] = {
        {0x41, 0xc05, "Cortex-A5", 0},  {0x41, 0xc07, "Cortex-A7", 0},  {0x41, 0xc08, "Cortex-A8", 1},
        {0x41, 0xc09, "Cortex-A9", 1},  {0x41, 0xc0e, "Cortex-A17", 1}, {0x41, 0xc0f, "Cortex-A15", 1},
        {0x41, 0xd01, "Cortex-A32", 0}, {0x41, 0xd02, "Cortex-A34", 0}, {0x41, 0xd03, "Cortex-A53", 0},
        {0x41, 0xd04, "Cortex-A35", 0}, {0x41, 0xd05, "Cortex-A55", 0}, {0x41, 0xd06, "Cortex-A65", 0},
        {0x41, 0xd07, "Cortex-A57", 1}, {0x41, 0xd08, "Cortex-A72", 1}, {0x41, 0xd09, "Cortex-A73", 1},
        {0x41, 0xd0a, "Cortex-A75", 1}, {0x41, 0xd0b, "Cortex-A76", 1}, {0x41, 0xd0c, "Neoverse-N1", 1},
        {0x41, 0xd0d, "Cortex-A77", 1}, {0x41, 0xd41, "Cortex-A78", 1}, {0x41, 0xd44, "Cortex-X1", 2},
        {0x41, 0xd46, "Cortex-A510", 0},{0x41, 0xd47, "Cortex-A710", 1},{0x41, 0xd48, "Cortex-X2", 2},
        {0x41, 0xd4d, "Cortex-A715", 1},{0x41, 0xd4e, "Cortex-X3", 2},  {0x41, 0xd80, "Cortex-A520", 0},
        {0x41, 0xd81, "Cortex-A720", 1},{0x41, 0xd82, "Cortex-X4", 2},  {0x41, 0xd85, "Cortex-X925", 2},
        {0x41, 0xd87, "Cortex-A725", 1},
    };
    for (const auto& e : table)
        if (e.implementer == implementer && e.part == part) return &e;
    return nullptr;
}

inline const char* arm_implementer_name(uint32_t impl) {
    switch (impl) {
        case 0x41: return "ARM";
        case 0x42: return "Broadcom";
        case 0x43: return "Cavium";
        case 0x48: return "HiSilicon";
        case 0x4e: return "NVIDIA";
        case 0x51: return "Qualcomm";
        case 0x53: return "Samsung";
        case 0x61: return "Apple";
        default:   return "";
    }
}

inline std::set<std::string> feature_set(const KV& block) {
    std::set<std::string> out;
    for (const char* key : {"flags", "Features", "features"}) {
        auto it = block.find(key);
        if (it == block.end()) continue;
        for (const auto& w : split_ws(it->second)) out.insert(w);
        break;
    }
    return out;
}

inline SimdFeatures derive_simd(const std::set<std::string>& f) {
    auto has = [&](const char* k) { return f.count(k) > 0; };
    SimdFeatures s;
    s.sse42 = has("sse4_2");
    s.avx = has("avx");
    s.avx2 = has("avx2");
    s.fma = has("fma");
    s.f16c = has("f16c");
    s.avx512f = has("avx512f");
    s.avx512_vnni = has("avx512_vnni");
    s.avx512_bf16 = has("avx512_bf16");
    s.avx512_fp16 = has("avx512_fp16");
    s.avx_vnni = has("avx_vnni");
    s.amx = has("amx_tile");
    s.neon = has("neon") || has("asimd");
    s.fp16 = has("asimdhp") || has("fphp") || has("fp16");
    s.dotprod = has("asimddp") || has("dotprod");
    s.i8mm = has("i8mm");
    s.bf16 = has("bf16");
    s.sve = has("sve");
    s.sve2 = has("sve2");
    s.sme = has("sme");
    return s;
}

// Group cores into performance classes.
//   key = cpu_capacity when every core has one (the scheduler's own view),
//         otherwise (uarch part, max frequency in 200 MHz buckets).
inline std::vector<CoreClass> build_classes(const std::vector<CoreInfo>& cores) {
    bool all_cap = !cores.empty();
    for (const auto& c : cores) all_cap = all_cap && c.capacity > 0;
    using Key = std::tuple<uint64_t, uint64_t, uint64_t>;
    std::map<Key, CoreClass> groups;
    for (const auto& c : cores) {
        Key k = all_cap ? Key{uint64_t(c.capacity), 0, 0}
                        : Key{0, uint64_t(c.part), c.max_khz / 200000};
        auto& g = groups[k];
        g.cpus.push_back(c.id);
        g.max_khz = std::max(g.max_khz, c.max_khz);
        g.capacity = std::max(g.capacity, c.capacity);
        if (g.uarch.empty()) g.uarch = c.uarch;
        g.tier = c.tier;
    }
    std::vector<CoreClass> out;
    for (auto& kv : groups) out.push_back(std::move(kv.second));
    // Slowest first: by capacity, then by frequency.
    std::stable_sort(out.begin(), out.end(), [](const CoreClass& a, const CoreClass& b) {
        if (a.capacity != b.capacity) return a.capacity < b.capacity;
        if (a.tier != b.tier) return a.tier < b.tier;
        return a.max_khz < b.max_khz;
    });
    const size_t n = out.size();
    int top_cap = 0;
    uint64_t top_khz = 0;
    for (const auto& c : out) { top_cap = std::max(top_cap, c.capacity); top_khz = std::max(top_khz, c.max_khz); }
    for (size_t i = 0; i < n; ++i) {
        auto& c = out[i];
        c.index = int(i);
        if (n == 1) c.name = "uniform";
        else if (i == 0) c.name = "little";
        else if (i + 1 == n) c.name = (n >= 3 && c.cpus.size() == 1) ? "prime" : "big";
        else c.name = "mid";
        if (top_cap > 0 && c.capacity > 0) c.rel_perf = double(c.capacity) / top_cap;
        else if (top_khz > 0 && c.max_khz > 0) c.rel_perf = double(c.max_khz) / double(top_khz);
        else c.rel_perf = 1.0;
        std::sort(c.cpus.begin(), c.cpus.end());
    }
    return out;
}

// cgroup limits. "max" and absurdly large values mean "no limit".
inline std::optional<uint64_t> parse_cgroup_limit(const std::string& raw) {
    std::string s = trim(raw);
    if (s.empty() || s == "max") return std::nullopt;
    auto v = parse_u64(s);
    if (!v || *v >= (1ull << 60)) return std::nullopt;
    return v;
}

// cpu.max "200000 100000" or "max 100000" -> number of cpus, nullopt if unlimited
inline std::optional<double> parse_cpu_max(const std::string& raw) {
    auto t = split_ws(raw);
    if (t.size() < 2 || t[0] == "max") return std::nullopt;
    auto q = parse_u64(t[0]), p = parse_u64(t[1]);
    if (!q || !p || *p == 0) return std::nullopt;
    return double(*q) / double(*p);
}

inline std::string json_escape(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += char(c);
        }
    }
    return o;
}

// Minimal streaming JSON writer (no dependency).
class JsonOut {
public:
    void begin_obj() { pre(); s_ += '{'; first_.push_back(true); }
    void end_obj()   { s_ += '}'; first_.pop_back(); }
    void begin_arr() { pre(); s_ += '['; first_.push_back(true); }
    void end_arr()   { s_ += ']'; first_.pop_back(); }
    void key(const std::string& k) { sep(); s_ += '"' + json_escape(k) + "\":"; after_key_ = true; }
    void str(const std::string& v) { pre(); s_ += '"' + json_escape(v) + '"'; }
    void num(double v) {
        pre();
        if (!std::isfinite(v)) { s_ += "null"; return; }
        char b[48];
        std::snprintf(b, sizeof b, "%.9g", v);
        s_ += b;
    }
    void u64(uint64_t v) { pre(); s_ += std::to_string(v); }
    void i64(int64_t v) { pre(); s_ += std::to_string(v); }
    void boolean(bool v) { pre(); s_ += v ? "true" : "false"; }
    void null() { pre(); s_ += "null"; }
    // key + value shorthands
    void kv(const std::string& k, const std::string& v) { key(k); str(v); }
    void kv(const std::string& k, const char* v) { key(k); str(v); }
    void kv(const std::string& k, uint64_t v) { key(k); u64(v); }
    void kv(const std::string& k, int v) { key(k); i64(v); }
    void kv(const std::string& k, double v) { key(k); num(v); }
    void kv(const std::string& k, bool v) { key(k); boolean(v); }
    template <class T> void kv(const std::string& k, const std::optional<T>& v) {
        key(k);
        if (!v) null();
        else if constexpr (std::is_floating_point_v<T>) num(*v);
        else u64(uint64_t(*v));
    }
    void ints(const std::string& k, const std::vector<int>& v) {
        key(k); begin_arr(); for (int x : v) i64(x); end_arr();
    }
    const std::string& str() const { return s_; }
private:
    void sep() { if (!first_.empty()) { if (!first_.back()) s_ += ','; first_.back() = false; } }
    void pre() { if (after_key_) after_key_ = false; else sep(); }
    std::string s_;
    std::vector<bool> first_;
    bool after_key_ = false;
};

}  // namespace detail

// ───────────────────────────── Linux / Android collector ─────────────────────────────

// Memory facts only (cheap: a few small file reads). Used by the full collector and
// by the runtime memory guard, which polls it.
inline MemoryInfo read_memory_linux(FsSource& fs, std::vector<std::string>* notes = nullptr) {
    using namespace detail;
    auto rd = [&](const std::string& path) { return fs.read(path); };
    auto rd_trim = [&](const std::string& path) -> std::optional<std::string> {
        auto r = fs.read(path);
        if (!r) return std::nullopt;
        return trim(*r);
    };
    MemoryInfo m;
    if (auto t = rd("/proc/meminfo")) {
        auto mi = parse_meminfo(*t);
        m.total = mi.count("MemTotal") ? mi["MemTotal"] : 0;
        if (mi.count("MemAvailable")) m.available = mi["MemAvailable"];
        else m.available = (mi.count("MemFree") ? mi["MemFree"] : 0) + (mi.count("Cached") ? mi["Cached"] : 0) +
                           (mi.count("Buffers") ? mi["Buffers"] : 0);   // pre-3.14 kernels
        m.swap_total = mi.count("SwapTotal") ? mi["SwapTotal"] : 0;
        m.swap_free = mi.count("SwapFree") ? mi["SwapFree"] : 0;
    } else if (notes) notes->push_back("/proc/meminfo unreadable: memory budget unknown");
    if (auto sw = rd("/proc/swaps")) m.swap_is_zram = sw->find("zram") != std::string::npos;
    // cgroup v2 then v1
    if (auto v = rd("/sys/fs/cgroup/memory.max")) {
        m.cgroup_limit = parse_cgroup_limit(*v);
        if (auto u = rd("/sys/fs/cgroup/memory.current")) m.cgroup_usage = parse_u64(*u);
    } else if (auto v1 = rd("/sys/fs/cgroup/memory/memory.limit_in_bytes")) {
        m.cgroup_limit = parse_cgroup_limit(*v1);
        if (auto u = rd("/sys/fs/cgroup/memory/memory.usage_in_bytes")) m.cgroup_usage = parse_u64(*u);
    }
    if (m.cgroup_limit && m.total && *m.cgroup_limit >= m.total) m.cgroup_limit.reset();  // not a real cap
    if (auto psi = rd("/proc/pressure/memory")) {
        // "some avg10=0.00 avg60=0.00 avg300=0.00 total=0"
        for (const auto& line : split(*psi, '\n')) {
            if (!starts_with(line, "some")) continue;
            for (const auto& tok : split_ws(line))
                if (starts_with(tok, "avg10=")) m.psi_some_avg10 = parse_double(tok.substr(6));
        }
    }
    if (auto mf = rd_trim("/sys/module/lowmemorykiller/parameters/minfree")) {
        uint64_t last = 0;
        for (const auto& t : split(*mf, ',')) if (auto v = parse_u64(t)) last = std::max(last, *v);
        if (last) m.lmk_min_free_bytes = last * m.page_bytes;
    }
    m.effective_available = m.available;
    if (m.cgroup_limit && m.cgroup_usage) {
        uint64_t head = *m.cgroup_limit > *m.cgroup_usage ? *m.cgroup_limit - *m.cgroup_usage : 0;
        m.effective_available = std::min(m.effective_available, head);
    } else if (m.cgroup_limit) {
        m.effective_available = std::min(m.effective_available, *m.cgroup_limit);
    }

    return m;
}

inline HardwareFingerprint collect_linux(const Probe& p) {
    using namespace detail;
    HardwareFingerprint fp;
    FsSource& fs = *p.fs;
    auto rd = [&](const std::string& path) { return fs.read(path); };
    auto rd_trim = [&](const std::string& path) -> std::optional<std::string> {
        auto r = fs.read(path);
        if (!r) return std::nullopt;
        return trim(*r);
    };
    auto prop = [&](const char* k) { return p.prop ? p.prop(k) : std::string(); };
    auto note = [&](const std::string& s) { fp.notes.push_back(s); };

    // ---- OS ----
    fp.os.name = p.is_android ? "Android" : "Linux";
    fp.os.kernel = p.kernel;
    if (p.is_android) {
        fp.os.android_api = int(parse_i64(prop("ro.build.version.sdk")).value_or(0));
        fp.os.version = prop("ro.build.version.release");
        fp.os.manufacturer = prop("ro.product.manufacturer");
        fp.os.model = prop("ro.product.model");
        const std::string chars = prop("ro.build.characteristics");
        fp.os.form_factor = chars.find("tablet") != std::string::npos ? "tablet" : "phone";
    }
    if (!p.is_android)
        if (auto t = rd("/etc/os-release"))
            for (const auto& line : split(*t, '\n'))
                if (starts_with(line, "PRETTY_NAME=")) {
                    std::string v = trim(line.substr(12));
                    if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
                    fp.os.version = v;
                }
    if (auto c = rd("/proc/1/cgroup"); c && (c->find("docker") != std::string::npos || c->find("kubepods") != std::string::npos))
        fp.os.in_container = true;
    if (rd("/.dockerenv")) fp.os.in_container = true;

    // ---- CPU: cpuinfo ----
    CpuInfo& cpu = fp.cpu;
    cpu.arch = p.arch;
    std::vector<KV> blocks;
    if (auto t = rd("/proc/cpuinfo")) blocks = parse_cpuinfo(*t);
    else note("/proc/cpuinfo unreadable: cpu model, features and core part ids unknown");

    // Blocks that carry a "processor" key are per-cpu; a trailing block with
    // "Hardware"/"Processor" only is the 32-bit ARM footer.
    std::map<int, KV> per_cpu;
    KV footer;
    for (auto& b : blocks) {
        auto it = b.find("processor");
        if (it != b.end()) {
            if (auto id = parse_u64(it->second)) per_cpu[int(*id)] = b;
        } else {
            for (auto& kv : b) footer[kv.first] = kv.second;
        }
    }
    // Kernels that put Hardware inside the last cpu block.
    auto first_of = [&](const char* k) -> std::string {
        for (auto& kv : per_cpu) { auto it = kv.second.find(k); if (it != kv.second.end() && !it->second.empty()) return it->second; }
        auto it = footer.find(k);
        return it != footer.end() ? it->second : std::string();
    };
    cpu.model_name = first_of("model name");
    if (cpu.model_name.empty()) cpu.model_name = first_of("Processor");
    cpu.vendor = first_of("vendor_id");
    cpu.soc_hint = first_of("Hardware");
    if (cpu.soc_hint.empty()) cpu.soc_hint = prop("ro.soc.model");
    if (cpu.soc_hint.empty()) cpu.soc_hint = prop("ro.board.platform");

    // ---- CPU: present / online lists ----
    std::vector<int> present, online;
    if (auto t = rd_trim("/sys/devices/system/cpu/present")) present = parse_cpulist(*t);
    if (auto t = rd_trim("/sys/devices/system/cpu/online")) online = parse_cpulist(*t);
    // /proc/cpuinfo only lists ONLINE cpus, and on phones the offline ones are
    // exactly the cores the governor may want later, so prefer sysfs.
    std::set<int> ids(present.begin(), present.end());
    if (ids.empty()) for (auto& kv : per_cpu) ids.insert(kv.first);
    if (ids.empty()) {
        note("no cpu list readable: assuming std::thread::hardware_concurrency");
        int n = std::max(1, int(std::thread::hardware_concurrency()));
        for (int i = 0; i < n; ++i) ids.insert(i);
    }
    std::set<int> online_set(online.begin(), online.end());
    if (online_set.empty()) {
        if (!per_cpu.empty()) for (auto& kv : per_cpu) online_set.insert(kv.first);
        else online_set = ids;
    }

    // ---- CPU: per-core facts ----
    bool saw_freq = false, saw_cap = false;
    for (int id : ids) {
        CoreInfo c;
        c.id = id;
        c.online = online_set.count(id) > 0;
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(id);
        auto it = per_cpu.find(id);
        if (it != per_cpu.end()) {
            const KV& b = it->second;
            auto g = [&](const char* k) -> std::optional<uint64_t> {
                auto f = b.find(k);
                return f == b.end() ? std::nullopt : parse_u64(f->second);
            };
            c.implementer = uint32_t(g("CPU implementer").value_or(0));
            c.part = uint32_t(g("CPU part").value_or(0));
            if (auto f = b.find("cpu MHz"); f != b.end())
                if (auto mhz = parse_double(f->second)) c.observed_khz = uint64_t(*mhz * 1000.0);
        }
        c.uarch = arm_uarch_name(c.implementer, c.part);
        if (const ArmPart* ap = arm_lookup(c.implementer, c.part)) c.tier = ap->tier;
        auto freq = rd_trim(base + "/cpufreq/cpuinfo_max_freq");
        if (!freq) freq = rd_trim(base + "/cpufreq/scaling_max_freq");
        if (freq) if (auto v = parse_u64(*freq)) { c.max_khz = *v; saw_freq = true; }
        if (auto m = rd_trim(base + "/cpufreq/cpuinfo_min_freq")) if (auto v = parse_u64(*m)) c.min_khz = *v;
        if (auto cap = rd_trim(base + "/cpu_capacity")) if (auto v = parse_u64(*cap)) { c.capacity = int(*v); saw_cap = true; }
        cpu.cores.push_back(c);
    }
    if (!saw_freq) note("cpufreq not readable: core classes fall back to uarch part ids; frequencies unknown");
    if (!saw_cap) note("cpu_capacity not readable: relative core speed estimated from frequency, or assumed equal");
    cpu.logical_cpus = int(cpu.cores.size());
    for (auto& c : cpu.cores) cpu.online_cpus += c.online ? 1 : 0;

    // implementer -> vendor on ARM
    if (cpu.vendor.empty())
        for (auto& c : cpu.cores) if (c.implementer) { cpu.vendor = arm_implementer_name(c.implementer); break; }
    if (cpu.vendor.empty() && (cpu.arch == "aarch64" || cpu.arch.rfind("arm", 0) == 0)) cpu.vendor = "ARM";
    if (cpu.model_name.empty()) {
        // derive a readable name from the class layout later; keep the soc hint now
        cpu.model_name = cpu.soc_hint;
    }

    // ---- CPU: features (intersection / union over per-cpu blocks) ----
    bool first = true;
    for (auto& kv : per_cpu) {
        auto f = feature_set(kv.second);
        if (f.empty()) continue;
        cpu.features_any.insert(f.begin(), f.end());
        if (first) { cpu.features_all = f; first = false; }
        else {
            std::set<std::string> inter;
            std::set_intersection(cpu.features_all.begin(), cpu.features_all.end(), f.begin(), f.end(),
                                  std::inserter(inter, inter.begin()));
            cpu.features_all = std::move(inter);
        }
    }
    cpu.simd = derive_simd(cpu.features_all);

    // ---- CPU: classes and caches ----
    cpu.classes = build_classes(cpu.cores);
    for (auto& cls : cpu.classes) {
        if (cls.cpus.empty()) continue;
        const std::string cdir = "/sys/devices/system/cpu/cpu" + std::to_string(cls.cpus.front()) + "/cache";
        for (const auto& name : fs.list(cdir)) {
            if (!starts_with(name, "index")) continue;
            const std::string d = cdir + "/" + name + "/";
            CacheInfo ci;
            if (auto v = rd_trim(d + "level")) ci.level = int(parse_u64(*v).value_or(0));
            if (auto v = rd_trim(d + "type")) ci.type = *v;
            if (auto v = rd_trim(d + "size")) ci.size_bytes = parse_size(*v);
            if (auto v = rd_trim(d + "coherency_line_size")) ci.line_bytes = int(parse_u64(*v).value_or(0));
            if (auto v = rd_trim(d + "ways_of_associativity")) ci.ways = int(parse_u64(*v).value_or(0));
            if (auto v = rd_trim(d + "shared_cpu_list")) ci.shared_cpus = parse_cpulist(*v);
            if (ci.level > 0 && ci.size_bytes > 0) cls.caches.push_back(ci);
        }
        std::sort(cls.caches.begin(), cls.caches.end(), [](const CacheInfo& a, const CacheInfo& b) {
            return a.level != b.level ? a.level < b.level : a.type < b.type;
        });
        if (cls.caches.empty()) note("cache topology unreadable for class " + cls.name);
    }

    // ---- Memory ----
    fp.memory = read_memory_linux(fs, &fp.notes);

    // ---- Storage ----
    fp.storage.path = ".";
    if (p.disk) if (auto d = p.disk(".")) { fp.storage.total = d->total; fp.storage.free = d->free; fp.storage.known = true; }

    // ---- Thermal ----
    for (const auto& z : fs.list("/sys/class/thermal")) {
        if (!starts_with(z, "thermal_zone")) continue;
        const std::string d = "/sys/class/thermal/" + z + "/";
        auto t = rd_trim(d + "temp");
        if (!t) continue;
        auto raw = parse_i64(*t);
        if (!raw) continue;
        double c = std::abs(*raw) >= 1000 ? double(*raw) / 1000.0 : double(*raw);
        if (c < -40.0 || c > 150.0) continue;          // sensor garbage
        ThermalZone tz;
        tz.name = rd_trim(d + "type").value_or(z);
        tz.temp_c = c;
        fp.thermal.push_back(tz);
    }
    if (fp.thermal.empty()) note("no thermal zones readable: the governor must use the OS thermal API or a heuristic");

    // ---- Power ----
    {
        bool any = false;
        for (const auto& s : fs.list("/sys/class/power_supply")) {
            const std::string d = "/sys/class/power_supply/" + s + "/";
            auto type = rd_trim(d + "type");
            if (!type) continue;
            any = true;
            if (*type == "Battery") {
                fp.power.battery_present = true;
                if (auto v = rd_trim(d + "capacity")) if (auto n = parse_i64(*v)) fp.power.level_pct = int(*n);
                if (auto v = rd_trim(d + "status")) fp.power.status = *v;
                if (auto v = rd_trim(d + "temp")) if (auto n = parse_i64(*v)) fp.power.battery_temp_c = double(*n) / 10.0;
                if (auto v = rd_trim(d + "voltage_now")) if (auto n = parse_i64(*v)) fp.power.voltage_mv = double(*n) / 1000.0;
                if (auto v = rd_trim(d + "current_now")) if (auto n = parse_i64(*v)) fp.power.current_ma = double(*n) / 1000.0;
            } else if (auto on = rd_trim(d + "online")) {
                if (*on == "1") fp.power.on_external_power = true;
            }
        }
        fp.power.available = any;
        if (!any) note("power_supply not readable: battery and charger state unknown (use BatteryManager in the app shell)");
    }

    // ---- Threads ----
    int usable = cpu.online_cpus;
    if (p.affinity_cpus) { int a = p.affinity_cpus(); if (a > 0) usable = std::min(usable, a); }
    double quota = 0;
    bool have_quota = false;
    if (auto v = rd("/sys/fs/cgroup/cpu.max")) {
        if (auto q = parse_cpu_max(*v)) { quota = *q; have_quota = true; }
    } else if (auto q1 = rd_trim("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")) {
        auto per = rd_trim("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
        auto qv = parse_i64(*q1);
        auto pv = per ? parse_i64(*per) : std::nullopt;
        if (qv && pv && *qv > 0 && *pv > 0) { quota = double(*qv) / double(*pv); have_quota = true; }
    }
    if (have_quota) usable = std::min(usable, std::max(1, int(std::ceil(quota))));
    fp.usable_threads = std::max(1, usable);

    // ---- form factor for non-Android ----
    if (fp.os.form_factor == "unknown") {
        if (fp.power.battery_present) fp.os.form_factor = "laptop";
        else if (cpu.logical_cpus >= 64) fp.os.form_factor = "server";
        else fp.os.form_factor = "desktop";
    }
    return fp;
}

// ───────────────────────────── Apple (macOS / iOS) ─────────────────────────────
// NOT compiled or run in the environment this was written in.
#if defined(__APPLE__)
namespace detail {
inline std::optional<std::string> sysctl_str(const char* name) {
    size_t n = 0;
    if (sysctlbyname(name, nullptr, &n, nullptr, 0) != 0 || n == 0) return std::nullopt;
    std::string s(n, '\0');
    if (sysctlbyname(name, s.data(), &n, nullptr, 0) != 0) return std::nullopt;
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}
inline std::optional<uint64_t> sysctl_u64(const char* name) {
    uint64_t v = 0;                         // zero-init: a 4-byte write lands in the low word (little endian)
    size_t n = sizeof v;
    if (sysctlbyname(name, &v, &n, nullptr, 0) != 0) return std::nullopt;
    return v;
}
}  // namespace detail

inline HardwareFingerprint collect_apple() {
    using namespace detail;
    HardwareFingerprint fp;
    CpuInfo& cpu = fp.cpu;
#if TARGET_OS_IPHONE
    fp.os.name = "iOS";
#else
    fp.os.name = "macOS";
#endif
    fp.os.version = sysctl_str("kern.osproductversion").value_or("");
    fp.os.kernel = sysctl_str("kern.osrelease").value_or("");
    fp.os.model = sysctl_str("hw.machine").value_or("");      // "iPhone15,2", "arm64"
    if (auto m = sysctl_str("hw.model")) if (fp.os.model == "arm64" || fp.os.model.empty()) fp.os.model = *m;
    fp.os.manufacturer = "Apple";
    if (fp.os.model.rfind("iPad", 0) == 0) fp.os.form_factor = "tablet";
    else if (fp.os.model.rfind("iPhone", 0) == 0) fp.os.form_factor = "phone";
    else fp.os.form_factor = "laptop";   // Macs: laptop vs desktop needs the model table; not guessed here

#if defined(__aarch64__)
    cpu.arch = "aarch64"; cpu.vendor = "Apple";
#else
    cpu.arch = "x86_64";
#endif
    cpu.model_name = sysctl_str("machdep.cpu.brand_string").value_or(fp.os.model);
    cpu.logical_cpus = int(sysctl_u64("hw.ncpu").value_or(std::thread::hardware_concurrency()));
    cpu.online_cpus = int(sysctl_u64("hw.activecpu").value_or(cpu.logical_cpus));

    const int levels = int(sysctl_u64("hw.nperflevels").value_or(1));
    int next_cpu = 0;
    for (int lv = levels - 1; lv >= 0; --lv) {            // perflevel0 = performance, last = efficiency; we want slowest first
        const std::string pfx = "hw.perflevel" + std::to_string(lv) + ".";
        CoreClass cc;
        const int n = int(sysctl_u64((pfx + "logicalcpu").c_str()).value_or(levels == 1 ? cpu.logical_cpus : 0));
        if (n <= 0) continue;
        cc.name = sysctl_str((pfx + "name").c_str()).value_or("");
        for (int i = 0; i < n; ++i) cc.cpus.push_back(next_cpu++);
        auto cache = [&](const char* key, int level, const char* type) {
            if (auto v = sysctl_u64((pfx + key).c_str()))  { CacheInfo ci; ci.level = level; ci.type = type; ci.size_bytes = *v;
                ci.line_bytes = int(sysctl_u64("hw.cachelinesize").value_or(0)); cc.caches.push_back(ci); }
        };
        cache("l1dcachesize", 1, "Data"); cache("l1icachesize", 1, "Instruction"); cache("l2cachesize", 2, "Unified");
        cpu.classes.push_back(cc);
    }
    if (cpu.classes.empty()) {                              // Intel Mac / simulator
        CoreClass cc; cc.name = "uniform";
        for (int i = 0; i < cpu.logical_cpus; ++i) cc.cpus.push_back(i);
        if (auto v = sysctl_u64("hw.l1dcachesize")) { CacheInfo ci; ci.level = 1; ci.type = "Data"; ci.size_bytes = *v; cc.caches.push_back(ci); }
        if (auto v = sysctl_u64("hw.l2cachesize")) { CacheInfo ci; ci.level = 2; ci.type = "Unified"; ci.size_bytes = *v; cc.caches.push_back(ci); }
        cpu.classes.push_back(cc);
    }
    // Names: first class is the efficiency class when there is more than one.
    if (cpu.classes.size() > 1) { cpu.classes.front().name = "little"; cpu.classes.back().name = "big"; }
    for (size_t i = 0; i < cpu.classes.size(); ++i) {
        cpu.classes[i].index = int(i);
        cpu.classes[i].rel_perf = cpu.classes.size() == 1 ? 1.0 : (i + 1 == cpu.classes.size() ? 1.0 : 0.5);  // no capacity API
        for (int id : cpu.classes[i].cpus) { CoreInfo c; c.id = id; c.online = true; cpu.cores.push_back(c); }
    }
    auto feat = [&](const char* name) { return sysctl_u64((std::string("hw.optional.") + name).c_str()).value_or(0) != 0; };
    cpu.simd.neon = feat("neon") || cpu.arch == "aarch64";
    cpu.simd.fp16 = feat("arm.FEAT_FP16");
    cpu.simd.dotprod = feat("arm.FEAT_DotProd");
    cpu.simd.i8mm = feat("arm.FEAT_I8MM");
    cpu.simd.bf16 = feat("arm.FEAT_BF16");
    cpu.simd.sme = feat("arm.FEAT_SME");
    cpu.simd.avx2 = feat("avx2_0"); cpu.simd.avx = feat("avx1_0"); cpu.simd.avx512f = feat("avx512f");

    MemoryInfo& m = fp.memory;
    m.total = sysctl_u64("hw.memsize").value_or(0);
    m.page_bytes = sysctl_u64("hw.pagesize").value_or(16384);
#if TARGET_OS_IPHONE
    m.available = os_proc_available_memory();                // bytes before jetsam, the number that matters on iOS
#else
    {
        vm_statistics64_data_t vm; mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &cnt) == KERN_SUCCESS)
            m.available = uint64_t(vm.free_count + vm.inactive_count + vm.purgeable_count) * m.page_bytes;
    }
#endif
    m.effective_available = m.available;
    struct statvfs sv;
    if (statvfs(".", &sv) == 0) { fp.storage.path = "."; fp.storage.total = uint64_t(sv.f_blocks) * sv.f_frsize;
        fp.storage.free = uint64_t(sv.f_bavail) * sv.f_frsize; fp.storage.known = true; }
    fp.usable_threads = std::max(1, cpu.online_cpus);
    fp.notes.push_back("thermal state and battery need the app shell (ProcessInfo.thermalState, UIDevice); not readable from C++");
    return fp;
}
#endif  // __APPLE__

// ───────────────────────────── Windows ─────────────────────────────
// NOT compiled or run in the environment this was written in.
#if defined(_WIN32)
inline HardwareFingerprint collect_windows() {
    using namespace detail;
    HardwareFingerprint fp;
    fp.os.name = "Windows";
    CpuInfo& cpu = fp.cpu;
    SYSTEM_INFO si; GetNativeSystemInfo(&si);
    switch (si.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_AMD64: cpu.arch = "x86_64"; break;
        case PROCESSOR_ARCHITECTURE_ARM64: cpu.arch = "aarch64"; break;
        case PROCESSOR_ARCHITECTURE_INTEL: cpu.arch = "x86"; break;
        default: cpu.arch = "unknown";
    }
    {   char buf[256]; DWORD n = sizeof buf;
        if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString",
                         RRF_RT_REG_SZ, nullptr, buf, &n) == ERROR_SUCCESS) cpu.model_name = trim(buf);
        n = sizeof buf;
        if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "VendorIdentifier",
                         RRF_RT_REG_SZ, nullptr, buf, &n) == ERROR_SUCCESS) cpu.vendor = trim(buf); }
    // Cores with EfficiencyClass (hybrid Intel / ARM64 Windows).
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    std::map<int, CoreClass> by_eff;
    int next = 0;
    if (len && GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) {
        for (DWORD off = 0; off < len;) {
            auto* r = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
            if (r->Relationship == RelationProcessorCore) {
                int threads = 0;
                for (WORD g = 0; g < r->Processor.GroupCount; ++g) threads += int(__popcnt64(r->Processor.GroupMask[g].Mask));
                auto& cc = by_eff[r->Processor.EfficiencyClass];
                for (int i = 0; i < std::max(1, threads); ++i) cc.cpus.push_back(next++);
            }
            off += r->Size;
        }
    }
    for (auto& kv : by_eff) cpu.classes.push_back(kv.second);   // ascending EfficiencyClass = slowest first
    if (cpu.classes.empty()) { CoreClass cc; for (int i = 0; i < int(si.dwNumberOfProcessors); ++i) cc.cpus.push_back(i); cpu.classes.push_back(cc); }
    for (size_t i = 0; i < cpu.classes.size(); ++i) {
        auto& c = cpu.classes[i]; c.index = int(i);
        c.name = cpu.classes.size() == 1 ? "uniform" : (i == 0 ? "little" : (i + 1 == cpu.classes.size() ? "big" : "mid"));
        c.rel_perf = cpu.classes.size() == 1 ? 1.0 : (i + 1 == cpu.classes.size() ? 1.0 : 0.5);
        for (int id : c.cpus) { CoreInfo ci; ci.id = id; cpu.cores.push_back(ci); }
    }
    cpu.logical_cpus = cpu.online_cpus = int(cpu.cores.size());
    MEMORYSTATUSEX ms{}; ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) { fp.memory.total = ms.ullTotalPhys; fp.memory.available = ms.ullAvailPhys; }
    fp.memory.effective_available = fp.memory.available;
    ULARGE_INTEGER fb, tb, tf;
    if (GetDiskFreeSpaceExA(".", &fb, &tb, &tf)) { fp.storage.path = "."; fp.storage.total = tb.QuadPart; fp.storage.free = fb.QuadPart; fp.storage.known = true; }
    SYSTEM_POWER_STATUS sp;
    if (GetSystemPowerStatus(&sp)) {
        fp.power.available = true;
        fp.power.battery_present = sp.BatteryFlag != 128 && sp.BatteryFlag != 255;
        fp.power.on_external_power = sp.ACLineStatus == 1;
        if (sp.BatteryLifePercent != 255) fp.power.level_pct = sp.BatteryLifePercent;
    }
    fp.os.form_factor = fp.power.battery_present ? "laptop" : "desktop";
    fp.usable_threads = std::max(1, cpu.online_cpus);
    fp.notes.push_back("windows: caches, features and thermals are not collected yet");
    return fp;
}
#endif  // _WIN32

// ───────────────────────────── entry point ─────────────────────────────

inline std::string compiled_arch() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
    return "armv7";
#elif defined(__riscv) && __riscv_xlen == 64
    return "riscv64";
#else
    return "unknown";
#endif
}

#if !defined(__APPLE__) && !defined(_WIN32)
inline Probe system_probe(FsSource& fs) {
    Probe p;
    p.fs = &fs;
    p.arch = compiled_arch();
#if defined(__ANDROID__)
    p.is_android = true;
    p.prop = [](const std::string& k) {
        char v[PROP_VALUE_MAX] = {0};
        __system_property_get(k.c_str(), v);
        return std::string(v);
    };
#endif
    p.disk = [](const std::string& path) -> std::optional<DiskSpace> {
        struct statvfs sv;
        if (::statvfs(path.c_str(), &sv) != 0) return std::nullopt;
        return DiskSpace{uint64_t(sv.f_blocks) * sv.f_frsize, uint64_t(sv.f_bavail) * sv.f_frsize};
    };
    p.affinity_cpus = []() -> int {
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof set, &set) != 0) return -1;
        return CPU_COUNT(&set);
    };
    struct utsname u;
    if (::uname(&u) == 0) p.kernel = u.release;
    return p;
}
#endif

// Collect for the machine we are running on.
inline HardwareFingerprint collect() {
#if defined(__APPLE__)
    return collect_apple();
#elif defined(_WIN32)
    return collect_windows();
#else
    static RealFs fs;       // stateless; static so the Probe pointer stays valid
    return collect_linux(system_probe(fs));
#endif
}

// ───────────────────────────── members ─────────────────────────────

inline std::string HardwareFingerprint::id() const {
    std::ostringstream s;
    s << os.name << '|' << os.manufacturer << '|' << os.model << '|' << cpu.arch << '|' << cpu.model_name << '|';
    for (const auto& c : cpu.classes) s << c.cpus.size() << ':' << c.capacity << ':' << c.max_khz / 100000 << ':' << c.uarch << ';';
    s << '|' << (memory.total >> 29);          // total RAM in 512 MB buckets
    // FNV-1a 64
    uint64_t h = 1469598103934665603ull;
    for (unsigned char ch : s.str()) { h ^= ch; h *= 1099511628211ull; }
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(h));
    return b;
}

inline std::vector<int> HardwareFingerprint::performance_cpus(bool include_mid) const {
    std::vector<int> out;
    const size_t n = cpu.classes.size();
    if (n == 0) return out;
    out = cpu.classes.back().cpus;
    if (include_mid && n >= 3) out.insert(out.end(), cpu.classes[n - 2].cpus.begin(), cpu.classes[n - 2].cpus.end());
    std::sort(out.begin(), out.end());
    return out;
}

inline std::string HardwareFingerprint::to_json() const {
    detail::JsonOut j;
    j.begin_obj();
    j.kv("schema", schema);
    j.kv("id", id());
    j.key("os"); j.begin_obj();
    j.kv("name", os.name); j.kv("version", os.version); j.kv("kernel", os.kernel);
    j.kv("android_api", os.android_api); j.kv("manufacturer", os.manufacturer); j.kv("model", os.model);
    j.kv("form_factor", os.form_factor); j.kv("in_container", os.in_container);
    j.end_obj();
    j.key("cpu"); j.begin_obj();
    j.kv("arch", cpu.arch); j.kv("pointer_bits", cpu.pointer_bits); j.kv("vendor", cpu.vendor);
    j.kv("model_name", cpu.model_name); j.kv("soc_hint", cpu.soc_hint);
    j.kv("logical_cpus", cpu.logical_cpus); j.kv("online_cpus", cpu.online_cpus);
    j.key("classes"); j.begin_arr();
    for (const auto& c : cpu.classes) {
        j.begin_obj();
        j.kv("index", c.index); j.kv("name", c.name); j.ints("cpus", c.cpus);
        j.kv("max_khz", uint64_t(c.max_khz)); j.kv("capacity", c.capacity); j.kv("uarch", c.uarch);
        j.kv("rel_perf", c.rel_perf);
        j.key("caches"); j.begin_arr();
        for (const auto& k : c.caches) {
            j.begin_obj();
            j.kv("level", k.level); j.kv("type", k.type); j.kv("size_bytes", uint64_t(k.size_bytes));
            j.kv("line_bytes", k.line_bytes); j.kv("ways", k.ways); j.ints("shared_cpus", k.shared_cpus);
            j.end_obj();
        }
        j.end_arr();
        j.end_obj();
    }
    j.end_arr();
    j.key("features_all"); j.begin_arr(); for (const auto& f : cpu.features_all) j.str(f); j.end_arr();
    j.key("features_any_only"); j.begin_arr();
    for (const auto& f : cpu.features_any) if (!cpu.features_all.count(f)) j.str(f);
    j.end_arr();
    j.key("simd"); j.begin_obj();
    const auto& s = cpu.simd;
    j.kv("sse42", s.sse42); j.kv("avx", s.avx); j.kv("avx2", s.avx2); j.kv("fma", s.fma); j.kv("f16c", s.f16c);
    j.kv("avx512f", s.avx512f); j.kv("avx512_vnni", s.avx512_vnni); j.kv("avx512_bf16", s.avx512_bf16);
    j.kv("avx512_fp16", s.avx512_fp16); j.kv("avx_vnni", s.avx_vnni); j.kv("amx", s.amx);
    j.kv("neon", s.neon); j.kv("fp16", s.fp16); j.kv("dotprod", s.dotprod); j.kv("i8mm", s.i8mm);
    j.kv("bf16", s.bf16); j.kv("sve", s.sve); j.kv("sve2", s.sve2); j.kv("sme", s.sme);
    j.end_obj();
    j.end_obj();
    j.key("memory"); j.begin_obj();
    j.kv("total", uint64_t(memory.total)); j.kv("available", uint64_t(memory.available));
    j.kv("effective_available", uint64_t(memory.effective_available));
    j.kv("swap_total", uint64_t(memory.swap_total)); j.kv("swap_free", uint64_t(memory.swap_free));
    j.kv("swap_is_zram", memory.swap_is_zram);
    j.kv("cgroup_limit", memory.cgroup_limit); j.kv("cgroup_usage", memory.cgroup_usage);
    j.kv("psi_some_avg10", memory.psi_some_avg10); j.kv("lmk_min_free_bytes", memory.lmk_min_free_bytes);
    j.kv("page_bytes", uint64_t(memory.page_bytes));
    j.end_obj();
    j.key("storage"); j.begin_obj();
    j.kv("known", storage.known); j.kv("path", storage.path);
    j.kv("total", uint64_t(storage.total)); j.kv("free", uint64_t(storage.free));
    j.end_obj();
    j.key("power"); j.begin_obj();
    j.kv("available", power.available); j.kv("battery_present", power.battery_present);
    j.kv("on_external_power", power.on_external_power);
    j.key("level_pct"); if (power.level_pct) j.i64(*power.level_pct); else j.null();
    j.kv("status", power.status);
    j.kv("battery_temp_c", power.battery_temp_c); j.kv("voltage_mv", power.voltage_mv); j.kv("current_ma", power.current_ma);
    j.end_obj();
    j.key("thermal"); j.begin_arr();
    for (const auto& z : thermal) { j.begin_obj(); j.kv("name", z.name); j.kv("temp_c", z.temp_c); j.end_obj(); }
    j.end_arr();
    j.kv("usable_threads", usable_threads);
    j.key("notes"); j.begin_arr(); for (const auto& n : notes) j.str(n); j.end_arr();
    j.end_obj();
    return j.str();
}

inline std::string HardwareFingerprint::summary() const {
    std::ostringstream o;
    o << os.name << ' ' << os.version << " (" << os.form_factor << ") " << os.manufacturer << ' ' << os.model << '\n';
    o << cpu.arch << ' ' << cpu.model_name << ", " << cpu.online_cpus << '/' << cpu.logical_cpus << " cpus online, "
      << usable_threads << " usable\n";
    for (auto it = cpu.classes.rbegin(); it != cpu.classes.rend(); ++it) {
        o << "  class " << it->name << ": " << it->cpus.size() << " cpu(s)";
        if (!it->uarch.empty()) o << ' ' << it->uarch;
        if (it->max_khz) o << ' ' << it->max_khz / 1000 << " MHz";
        if (it->capacity) o << " cap " << it->capacity;
        for (const auto& c : it->caches) if (c.type != "Instruction") o << " L" << c.level << '=' << c.size_bytes / 1024 << 'K';
        o << '\n';
    }
    o << "memory " << memory.total / (1 << 20) << " MB total, " << memory.effective_available / (1 << 20) << " MB usable now";
    if (memory.swap_is_zram) o << " (zram swap)";
    o << '\n';
    for (const auto& n : notes) o << "  note: " << n << '\n';
    return o.str();
}

}  // namespace tensorf::profile
