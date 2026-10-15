/*
 * profiler_capability_test.cpp
 * ============================
 * Tests for the rewritten profiler (ProfilerCommon, DeviceFingerprint,
 * CapabilityKernels, CapabilitySuite).
 *
 *   g++ -std=c++20 -O2 -Wall -Wextra -Iinclude tests/profiler_capability_test.cpp -o profiler_test -pthread
 *   ./profiler_test            # ~15 s: includes two real suite runs
 *   ./profiler_test --quick    # skips the real suite runs
 *
 * What is covered, grouped by the bug it guards against (see the design document):
 *   - ARM never detected              -> phone fixtures, vendor/part/class checks
 *   - "Ryzen 9" matched as Zen5       -> numeric family/model fixtures
 *   - sscanf("%lu") on 32-bit ARM     -> 64-bit parse helpers + armv7 fixture
 *   - L1/L2 latency returned 0        -> plan_levels + chase on small buffers
 *   - pointer chase not one cycle     -> make_chase_next cycle length
 *   - lscpu/dmidecode/lsblk           -> everything runs from a fake /proc + /sys tree
 *   - thermal safety                  -> guard verdicts and an immediate stop on a hot fixture
 *   - budget                          -> time budget respected
 *
 * Fixtures are written to a temp dir and removed afterwards, so the tests do not
 * depend on the machine they run on (except the two "real machine" sections).
 */

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <future>
#include <string>
#include <unistd.h>

#include "profiler/tmp/CapabilitySuite.hpp"

namespace fs = std::filesystem;
using namespace tensorf::prof;
namespace K = tensorf::prof::kernels;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failed; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while (0)
#define CHECK_NEAR(a, b, tol) do { ++g_checks; double _a=(double)(a), _b=(double)(b); if (!(std::fabs(_a-_b) <= (tol))) { ++g_failed; \
    std::printf("  FAIL %s:%d  %s ~ %s  (%g vs %g)\n", __FILE__, __LINE__, #a, #b, _a, _b); } } while (0)
#define SECTION(name) std::printf("[%s]\n", name)

// ─── Fixture tree ────────────────────────────────────────────────────────────

struct Fx {
    fs::path root;
    Fx() {
        static int n = 0;
        root = fs::temp_directory_path() / ("tf_prof_fx_" + std::to_string(getpid()) + "_" + std::to_string(n++));
        fs::create_directories(root);
    }
    ~Fx() { std::error_code ec; fs::remove_all(root, ec); }
    std::string str() const { return root.string(); }
    void put(const std::string& rel, const std::string& text) {
        fs::path p = root / rel.substr(rel[0] == '/' ? 1 : 0);
        fs::create_directories(p.parent_path());
        FILE* f = std::fopen(p.c_str(), "wb");
        std::fwrite(text.data(), 1, text.size(), f);
        std::fclose(f);
    }
    std::string cpu(int c) const { return "/sys/devices/system/cpu/cpu" + std::to_string(c); }
    void cache(int c, int idx, int level, const char* type, const char* size, const char* shared) {
        const std::string d = cpu(c) + "/cache/index" + std::to_string(idx);
        put(d + "/level", std::to_string(level) + "\n"); put(d + "/type", std::string(type) + "\n");
        put(d + "/size", std::string(size) + "\n");      put(d + "/shared_cpu_list", std::string(shared) + "\n");
        put(d + "/coherency_line_size", "64\n");
    }
    void core(int c, int capacity, long max_khz) {
        if (capacity) put(cpu(c) + "/cpu_capacity", std::to_string(capacity) + "\n");
        if (max_khz)  put(cpu(c) + "/cpufreq/cpuinfo_max_freq", std::to_string(max_khz) + "\n");
        put(cpu(c) + "/topology/thread_siblings_list", std::to_string(c) + "\n");
    }
};

static std::string arm64_block(int cpu, const char* part) {
    return "processor\t: " + std::to_string(cpu) + "\nBogoMIPS\t: 38.40\n"
           "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp\n"
           "CPU implementer\t: 0x41\nCPU architecture: 8\nCPU variant\t: 0x1\nCPU part\t: " + part + "\nCPU revision\t: 0\n\n";
}

// Snapdragon-888-like: 4x A55 (efficiency), 3x A78 (performance), 1x X1 (prime).
static void build_phone_arm64(Fx& fx) {
    std::string ci;
    const char* parts[8] = {"0xd05", "0xd05", "0xd05", "0xd05", "0xd41", "0xd41", "0xd41", "0xd44"};
    for (int i = 0; i < 8; ++i) ci += arm64_block(i, parts[i]);
    ci += "Hardware\t: Qualcomm Technologies, Inc SM8350\n";
    fx.put("/proc/cpuinfo", ci);
    fx.put("/sys/devices/system/cpu/online", "0-7\n");
    for (int i = 0; i < 8; ++i) {
        if (i < 4)      { fx.core(i, 325, 1804800);  fx.cache(i, 0, 1, "Data", "32K", std::to_string(i).c_str());
                          fx.cache(i, 1, 1, "Instruction", "32K", std::to_string(i).c_str());
                          fx.cache(i, 2, 2, "Unified", "128K", std::to_string(i).c_str()); }
        else if (i < 7) { fx.core(i, 871, 2419200);  fx.cache(i, 0, 1, "Data", "64K", std::to_string(i).c_str());
                          fx.cache(i, 2, 2, "Unified", "512K", std::to_string(i).c_str()); }
        else            { fx.core(i, 1024, 2841600); fx.cache(i, 0, 1, "Data", "64K", "7");
                          fx.cache(i, 2, 2, "Unified", "1024K", "7"); }
        fx.cache(i, 3, 3, "Unified", "4096K", "0-7");
    }
    fx.put("/proc/meminfo", "MemTotal:        7864320 kB\nMemFree:          300000 kB\nMemAvailable:    2900000 kB\nSwapTotal:       2097152 kB\n");
    fx.put("/proc/sys/vm/min_free_kbytes", "27386\n");
    fx.put("/system/build.prop", "ro.build.version.sdk=33\n");
    fx.put("/sys/class/power_supply/battery/type", "Battery\n");
    fx.put("/sys/class/power_supply/battery/capacity", "87\n");
    fx.put("/sys/class/power_supply/battery/temp", "312\n");
    fx.put("/sys/class/power_supply/battery/status", "Discharging\n");
    fx.put("/sys/class/thermal/thermal_zone0/type", "cpu-1-0\n");
    fx.put("/sys/class/thermal/thermal_zone0/temp", "38000\n");
    fx.put("/sys/block/sda/size", "250000000\n");
    fx.put("/sys/block/sda/queue/rotational", "0\n");
    fx.put("/sys/block/loop0/size", "100\n");
}

// Android 7, armv7, 4x A53 at one speed, no cpufreq, no cpu_capacity, battery temp only.
static void build_phone_armv7(Fx& fx) {
    std::string ci = "Processor\t: ARMv7 Processor rev 4 (v7l)\n";
    for (int i = 0; i < 4; ++i)
        ci += "processor\t: " + std::to_string(i) + "\nBogoMIPS\t: 38.40\n"
              "Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm\n"
              "CPU implementer\t: 0x41\nCPU architecture: 7\nCPU variant\t: 0x0\nCPU part\t: 0xd03\nCPU revision\t: 4\n\n";
    ci += "Hardware\t: Qualcomm Technologies, Inc MSM8916\nRevision\t: 0000\n";
    fx.put("/proc/cpuinfo", ci);
    fx.put("/sys/devices/system/cpu/online", "0-3\n");
    fx.put("/proc/meminfo", "MemTotal:        1835008 kB\nMemFree:          100000 kB\nCached:           600000 kB\n");   // no MemAvailable
    fx.put("/system/build.prop", "ro.build.version.sdk=24\n");
    fx.put("/sys/class/power_supply/battery/type", "Battery\n");
    fx.put("/sys/class/power_supply/battery/temp", "358\n");
    fx.put("/sys/class/power_supply/battery/status", "Charging\n");
    fx.put("/sys/block/mmcblk0/size", "62537728\n");
    fx.put("/sys/block/mmcblk0/queue/rotational", "0\n");
}

static void build_x86(Fx& fx, const char* vendor, int family, int model, const char* name, const char* flags, int ncpu) {
    std::string ci;
    for (int i = 0; i < ncpu; ++i)
        ci += "processor\t: " + std::to_string(i) + "\nvendor_id\t: " + vendor + "\ncpu family\t: " + std::to_string(family) +
              "\nmodel\t\t: " + std::to_string(model) + "\nmodel name\t: " + name + "\ncpu MHz\t\t: 3000.000\nflags\t\t: " + flags + "\n\n";
    fx.put("/proc/cpuinfo", ci);
    fx.put("/sys/devices/system/cpu/online", "0-" + std::to_string(ncpu - 1) + "\n");
}

// ─── Tests ───────────────────────────────────────────────────────────────────

static void test_common() {
    SECTION("ProfilerCommon: parsing helpers");
    uint64_t v = 0;
    CHECK(parse_size_bytes("32K", v) && v == 32768);
    CHECK(parse_size_bytes("8 M", v) && v == (8u << 20));
    CHECK(parse_size_bytes("1024K\n", v) && v == (1u << 20));
    CHECK(parse_u64("17179869184", v) && v == 17179869184ull);     // > 32 bit: what %lu broke on armv7
    CHECK(!parse_u64("-5", v));
    CHECK(!parse_u64("abc", v));
    auto l = parse_cpu_list("0-3,6,8-9\n");
    CHECK(l.size() == 7 && l[3] == 3 && l[4] == 6 && l[6] == 9);
    CHECK(parse_cpu_list("").empty());
    CHECK(has_token("fp asimd aes", "asimd"));
    CHECK(!has_token("avx512f", "avx"));            // no substring matches
    CHECK(has_token("a b avx2", "avx2"));           // last token, no trailing space
    CHECK(median({3, 1, 2}) == 2.0 && median({1, 2, 3, 4}) == 2.5);
    CHECK_NEAR(percentile({1, 2, 3, 4, 5}, 0.5), 3.0, 1e-12);
    CHECK_NEAR(percentile({1, 2, 3, 4, 5}, 1.0), 5.0, 1e-12);

    SECTION("ProfilerCommon: JSON writer");
    Json j;
    j.begin_object().num("a", 1.5).num("nan", std::nan("")).str("s", "x\"y\n").begin_array("v").num(1).num(2).end_array().boolean("b", true).end_object();
    CHECK(j.finish() == "{\"a\":1.5,\"nan\":null,\"s\":\"x\\\"y\\n\",\"v\":[1,2],\"b\":true}");
}

static void test_arm_phone() {
    SECTION("Fingerprint: big.LITTLE arm64 phone (Android)");
    Fx fx; build_phone_arm64(fx);
    DeviceFingerprint fp = detect_device(fx.str());
    CHECK(fp.os == OS::Android);
    CHECK(fp.vendor == Vendor::ARM_LICENSED);                 // ARM used to fall through to INTEL_PRE_HASWELL
    CHECK(fp.arch == Arch::ARM64);
    CHECK(fp.model_name.find("SM8350") != std::string::npos);
    CHECK(fp.simd.neon && fp.simd.fp16 && fp.simd.dotprod && !fp.simd.i8mm && !fp.simd.sve);
    CHECK(fp.logical_cpus == 8 && fp.physical_cores == 8);
    CHECK(fp.clusters.size() == 3);
    if (fp.clusters.size() == 3) {
        CHECK(fp.clusters[0].cls == CoreClass::PRIME       && fp.clusters[0].cpus == std::vector<int>{7});
        CHECK(fp.clusters[1].cls == CoreClass::PERFORMANCE && fp.clusters[1].cpus == (std::vector<int>{4, 5, 6}));
        CHECK(fp.clusters[2].cls == CoreClass::EFFICIENCY  && fp.clusters[2].cpus == (std::vector<int>{0, 1, 2, 3}));
        CHECK(fp.clusters[0].part_name == "cortex-x1" && fp.clusters[1].part_name == "cortex-a78" && fp.clusters[2].part_name == "cortex-a55");
        CHECK_NEAR(fp.clusters[0].max_freq_mhz, 2841.6, 0.01);
        CHECK(fp.clusters[0].l1d_bytes == 65536 && fp.clusters[0].l2_bytes == (1u << 20) && fp.clusters[0].l3_bytes == (4u << 20));
        CHECK(fp.clusters[2].l1d_bytes == 32768 && fp.clusters[2].l2_bytes == 131072);   // I-cache is not counted as L1d
    }
    CHECK(fp.microarch == "cortex-x1+cortex-a78+cortex-a55");
    CHECK(fp.hybrid());
    CHECK(fp.memory.total_bytes == 7864320ull * 1024 && fp.memory.available_bytes == 2900000ull * 1024);
    CHECK(fp.memory.safe_budget_bytes < fp.memory.available_bytes && fp.memory.safe_budget_bytes > 0);
    CHECK(fp.battery.present && fp.battery.level_pct == 87 && !fp.battery.charging());
    CHECK_NEAR(fp.battery.temp_c, 31.2, 1e-9);
    CHECK(fp.thermal_zones.size() == 1 && std::fabs(fp.thermal_zones[0].temp_c - 38.0) < 1e-9);
    CHECK(fp.storage.size() == 1 && fp.storage[0].name == "sda" && fp.storage[0].kind == StorageKind::SATA_OR_UFS_FLASH);   // loop0 skipped
    Json j; to_json(j, fp);                                   // serialises without throwing
    CHECK(j.finish().find("\"class\":\"prime\"") != std::string::npos);
}

static void test_armv7_phone() {
    SECTION("Fingerprint: armv7 Android 7 phone");
    Fx fx; build_phone_armv7(fx);
    DeviceFingerprint fp = detect_device(fx.str());
    CHECK(fp.vendor == Vendor::ARM_LICENSED);
    CHECK(fp.arch == Arch::ARM32);                            // runtime truth, not the host compile target
    CHECK(fp.model_name.find("MSM8916") != std::string::npos);
    CHECK(fp.simd.neon && !fp.simd.dotprod && !fp.simd.fp16);
    CHECK(fp.logical_cpus == 4);
    CHECK(fp.clusters.size() == 1 && fp.clusters[0].cls == CoreClass::UNIFORM && fp.clusters[0].cpus.size() == 4);
    CHECK(fp.microarch == "cortex-a53");
    CHECK(fp.memory.available_bytes == (100000ull + 600000ull) * 1024);     // fallback when MemAvailable is missing
    CHECK(fp.memory.low_ram_device);
    CHECK(fp.battery.present && fp.battery.charging() && std::fabs(fp.battery.temp_c - 35.8) < 1e-9);
    CHECK(fp.thermal_zones.empty());                          // Android 7/8: battery temperature is the only signal
    CHECK(fp.storage.size() == 1 && fp.storage[0].kind == StorageKind::EMMC);
}

static void test_x86_generations() {
    SECTION("Fingerprint: x86 microarchitecture from family/model");
    {   // Ryzen 9 7950X: the name contains a '9' but it is Zen4
        Fx fx; build_x86(fx, "AuthenticAMD", 25, 97, "AMD Ryzen 9 7950X 16-Core Processor", "fpu sse2 avx fma avx2 avx512f avx512_vnni avx512_bf16", 2);
        DeviceFingerprint fp = detect_device(fx.str());
        CHECK(fp.vendor == Vendor::AMD && fp.microarch == "zen4");
        CHECK(fp.simd.avx512f && fp.simd.avx512_vnni && fp.simd.f32_lanes() == 16);
    }
    {   // Ryzen 9 9950X: family 0x1a
        Fx fx; build_x86(fx, "AuthenticAMD", 26, 68, "AMD Ryzen 9 9950X 16-Core Processor", "sse2 avx avx2 avx512f", 1);
        CHECK(detect_device(fx.str()).microarch == "zen5");
    }
    CHECK(x86_microarch(Vendor::AMD, 23, 1, {}) == "zen1");
    CHECK(x86_microarch(Vendor::AMD, 23, 113, {}) == "zen2");
    CHECK(x86_microarch(Vendor::AMD, 25, 33, {}) == "zen3");
    CHECK(x86_microarch(Vendor::INTEL, 6, 60, {}) == "intel_haswell");
    CHECK(x86_microarch(Vendor::INTEL, 6, 151, {}) == "intel_alder_lake");
    CHECK(x86_microarch(Vendor::INTEL, 6, 143, {}) == "intel_sapphire_rapids");

    SECTION("Fingerprint: Intel hybrid (P/E cores) and SMT");
    Fx fx; build_x86(fx, "GenuineIntel", 6, 151, "12th Gen Intel(R) Core(TM) i9-12900K", "sse2 avx avx2 fma", 2);
    fx.put("/sys/devices/system/cpu/online", "0-23\n");
    fx.put("/sys/devices/cpu_core/cpus", "0-15\n");
    fx.put("/sys/devices/cpu_atom/cpus", "16-23\n");
    for (int c = 0; c < 24; ++c) {
        if (c < 16) { fx.put(fx.cpu(c) + "/topology/thread_siblings_list", std::to_string(c & ~1) + "-" + std::to_string(c | 1) + "\n"); fx.put(fx.cpu(c) + "/cpufreq/cpuinfo_max_freq", "5200000\n"); }
        else        { fx.put(fx.cpu(c) + "/topology/thread_siblings_list", std::to_string(c) + "\n");                                          fx.put(fx.cpu(c) + "/cpufreq/cpuinfo_max_freq", "3900000\n"); }
    }
    fx.cache(0, 0, 1, "Data", "48K", "0-1");  fx.cache(0, 1, 2, "Unified", "1280K", "0-1"); fx.cache(0, 2, 3, "Unified", "30720K", "0-23");
    fx.cache(16, 0, 1, "Data", "32K", "16"); fx.cache(16, 1, 2, "Unified", "2048K", "16-19");
    DeviceFingerprint fp = detect_device(fx.str());
    CHECK(fp.logical_cpus == 24 && fp.physical_cores == 16);
    CHECK(fp.clusters.size() == 2);
    if (fp.clusters.size() == 2) {
        CHECK(fp.clusters[0].cls == CoreClass::PERFORMANCE && fp.clusters[0].cpus.size() == 16 && fp.clusters[0].physical_cores == 8);
        CHECK(fp.clusters[1].cls == CoreClass::EFFICIENCY  && fp.clusters[1].cpus.size() == 8  && fp.clusters[1].physical_cores == 8);
        CHECK(fp.clusters[0].l1d_bytes == 49152 && fp.clusters[1].l2_sharers == 4);
    }
}

static void test_core_grouping() {
    SECTION("Core grouping and classes (pure)");
    CHECK(class_for_tier(0, 1) == CoreClass::UNIFORM);
    CHECK(class_for_tier(0, 2) == CoreClass::PERFORMANCE && class_for_tier(1, 2) == CoreClass::EFFICIENCY);
    CHECK(class_for_tier(0, 3) == CoreClass::PRIME && class_for_tier(1, 3) == CoreClass::PERFORMANCE && class_for_tier(2, 3) == CoreClass::EFFICIENCY);

    std::vector<RawCore> by_freq;      // no cpu_capacity: group by frequency, 8% tolerance
    for (int i = 0; i < 4; ++i) { RawCore r; r.cpu = i;     r.max_freq_khz = 2900000 - (uint64_t)i * 30000; by_freq.push_back(r); }
    for (int i = 4; i < 8; ++i) { RawCore r; r.cpu = i;     r.max_freq_khz = 1800000; by_freq.push_back(r); }
    auto g = group_cores(by_freq);
    CHECK(g.size() == 2 && g[0].size() == 4 && g[1].size() == 4 && g[0][0].cpu == 0);

    std::vector<RawCore> none(3); for (int i = 0; i < 3; ++i) none[(size_t)i].cpu = i;
    CHECK(group_cores(none).size() == 1);
    CHECK(group_cores({}).empty());
}

static void test_memory() {
    SECTION("Memory budget: cgroup + kernel reserve");
    CHECK(parse_cgroup_limit("max\n") == 0);
    CHECK(parse_cgroup_limit("9223372036854771712") == 0);    // cgroup v1 "unlimited"
    CHECK(parse_cgroup_limit("1073741824") == 1073741824ull);
    CHECK(parse_cgroup_limit("") == 0);

    Fx fx;
    fx.put("/proc/meminfo", "MemTotal:  8000000 kB\nMemAvailable:  3000000 kB\n");
    fx.put("/proc/sys/vm/min_free_kbytes", "16384\n");
    fx.put("/sys/fs/cgroup/memory.max", "1073741824\n");
    fx.put("/sys/fs/cgroup/memory.current", "419430400\n");
    MemoryLimits m = linux_adapter::read_memory(fx.str());
    CHECK(m.cgroup_limit_bytes == 1073741824ull && m.cgroup_usage_bytes == 419430400ull);
    // headroom = min(3,072,000,000, 654,311,424); 80% of it, minus 16 MiB
    CHECK(m.safe_budget_bytes == (uint64_t)(654311424 * 0.8) - 16777216ull);

    Fx fx2;                                                   // unlimited cgroup: MemAvailable governs
    fx2.put("/proc/meminfo", "MemTotal: 8000000 kB\nMemAvailable: 4000000 kB\n");
    fx2.put("/sys/fs/cgroup/memory.max", "max\n");
    MemoryLimits m2 = linux_adapter::read_memory(fx2.str());
    CHECK(m2.cgroup_limit_bytes == 0 && m2.safe_budget_bytes == (uint64_t)(4000000ull * 1024 * 0.8));

    MemoryLimits z; z.available_bytes = 10; z.kernel_reserve_bytes = 1000;
    CHECK(compute_safe_budget(z) == 0);                       // never underflows

    DeviceFingerprint fp; fp.memory.safe_budget_bytes = 4ull << 30;
    SuiteConfig c;
    CHECK(memory_cap_bytes(fp, c) == c.memory_hard_cap);      // 25% of 4 GiB = 1 GiB, capped at 512 MiB
    fp.memory.safe_budget_bytes = 200ull << 20;
    CHECK(memory_cap_bytes(fp, c) == (size_t)((200ull << 20) * 0.25));
    fp.memory.safe_budget_bytes = 0;
    CHECK(memory_cap_bytes(fp, c) == c.memory_floor);
}

static void test_level_plan_and_latency() {
    SECTION("Level plan and latency on small caches");
    LevelPlan p = plan_levels(32u << 10, 256u << 10, 8u << 20, 1ull << 30);
    CHECK(p.l1_ws == (16u << 10) && p.l2_ws == (128u << 10) && p.l3_ws == (4u << 20) && p.dram_ws == (64u << 20));
    CHECK(!p.cache_sizes_assumed && !p.dram_may_be_cached);
    LevelPlan u = plan_levels(0, 0, 0, 1ull << 30);
    CHECK(u.cache_sizes_assumed && u.l3_ws == 0 && u.l1_ws > 0 && u.l2_ws > u.l1_ws);
    LevelPlan big = plan_levels(48u << 10, 2u << 20, 272629760ull, 1ull << 30);       // a VM reporting the host's L3
    CHECK(big.l3_ws == (32u << 20) && big.dram_ws == (256u << 20) && big.dram_may_be_cached);
    CHECK(plan_levels(32u << 10, 256u << 10, 8u << 20, 16u << 20).dram_ws == (16u << 20));

    // The old code returned 0 for L1 and for any L2 <= 512 KB (it only tried sizes >= 1 MB).
    Deadline dl(5);
    K::LatencyPoint l1 = K::chase_latency(p.l1_ws, dl, 1ull << 30);
    K::LatencyPoint l2 = K::chase_latency(p.l2_ws, dl, 1ull << 30);
    CHECK(l1.ok && l1.ns > 0 && l2.ok && l2.ns > 0);
    CHECK(!K::chase_latency(64ull << 20, dl, 1ull << 20).ok);                         // over the memory cap: refuses, does not throw

    SECTION("Pointer chase is ONE cycle");
    for (size_t n : {8u, 64u, 1000u, 4097u}) {
        auto next = K::make_chase_next(n, 12345);
        size_t steps = 0; uint32_t i = 0;
        do { i = next[i]; ++steps; } while (i != 0 && steps <= n);
        CHECK(steps == n);                                    // returns to 0 only after visiting every slot
    }
    CHECK(K::make_chase_next(100, 1) == K::make_chase_next(100, 1));
    CHECK(K::make_chase_next(100, 1) != K::make_chase_next(100, 2));
}

static void test_thermal_guard() {
    SECTION("Thermal guard verdicts (pure)");
    SuiteConfig c;       // stop_cpu 45, battery 40 (38 while charging), rise 8
    ThermalProbe::Reading cool; cool.ok = true; cool.max_cpu_like_c = 36; cool.battery_c = 33;
    CHECK(thermal_verdict(cool, 34, false, c).empty());
    ThermalProbe::Reading hot = cool; hot.max_cpu_like_c = 46;
    CHECK(thermal_verdict(hot, 34, false, c).find("zone") != std::string::npos);
    ThermalProbe::Reading warm_bat = cool; warm_bat.battery_c = 39;
    CHECK(thermal_verdict(warm_bat, 34, false, c).empty());                              // 39 < 40 discharging
    CHECK(thermal_verdict(warm_bat, 34, true, c).find("battery") != std::string::npos);   // 39 >= 38 charging
    ThermalProbe::Reading rising = cool; rising.max_cpu_like_c = 43;
    CHECK(thermal_verdict(rising, 34, false, c).find("rose") != std::string::npos);      // 9 C above the start
    CHECK(thermal_verdict(rising, 41, false, c).empty());                                // only 2 C above this start
    ThermalProbe::Reading none;                                                         // no sensors: cannot judge
    CHECK(thermal_verdict(none, 30, false, c).empty());

    SECTION("ThermalProbe on fixtures");
    Fx fx; fx.put("/sys/class/thermal/thermal_zone0/type", "battery\n"); fx.put("/sys/class/thermal/thermal_zone0/temp", "30000\n");
    fx.put("/sys/class/thermal/thermal_zone1/type", "cpu-0-0\n");        fx.put("/sys/class/thermal/thermal_zone1/temp", "47500\n");
    fx.put("/sys/class/thermal/thermal_zone2/type", "broken\n");         fx.put("/sys/class/thermal/thermal_zone2/temp", "-274000\n");
    auto r = ThermalProbe(fx.str()).read();
    CHECK(r.ok && std::fabs(r.max_cpu_like_c - 47.5) < 1e-9);            // battery zone is not CPU-like; broken sensor ignored
}

static void test_kernels() {
    SECTION("Kernels: GEMM correctness (shapes crossing the JB/KB block edges)");
    {
        const size_t M = 37, N = 300, Kd = 200;
        K::Rng r;
        std::vector<float> A(M * Kd), B(Kd * N), C1(M * N, 0.f), C2(M * N, 0.f);
        for (auto& x : A) x = r.uniformf(-1, 1);
        for (auto& x : B) x = r.uniformf(-1, 1);
        K::gemm_blocked<float, float>(A.data(), B.data(), C1.data(), M, N, Kd);
        K::gemm_naive<float, float>(A.data(), B.data(), C2.data(), M, N, Kd);
        double e = 0; for (size_t i = 0; i < C1.size(); ++i) e = std::max(e, (double)std::fabs(C1[i] - C2[i]));
        CHECK(e < 1e-4);
        std::vector<int8_t> a8(M * Kd), b8(Kd * N); std::vector<int32_t> c1(M * N, 0), c2(M * N, 0);
        for (auto& x : a8) x = (int8_t)((int)(r.next() % 7) - 3);
        for (auto& x : b8) x = (int8_t)((int)(r.next() % 7) - 3);
        K::gemm_blocked<int8_t, int32_t>(a8.data(), b8.data(), c1.data(), M, N, Kd);
        K::gemm_naive<int8_t, int32_t>(a8.data(), b8.data(), c2.data(), M, N, Kd);
        CHECK(c1 == c2);
        std::vector<double> ad(M * Kd), bd(Kd * N), d1(M * N, 0.0), d2(M * N, 0.0);
        for (auto& x : ad) x = r.uniform();
        for (auto& x : bd) x = r.uniform();
        K::gemm_blocked<double, double>(ad.data(), bd.data(), d1.data(), M, N, Kd);
        K::gemm_naive<double, double>(ad.data(), bd.data(), d2.data(), M, N, Kd);
        double ed = 0; for (size_t i = 0; i < d1.size(); ++i) ed = std::max(ed, std::fabs(d1[i] - d2[i]));
        CHECK(ed < 1e-9);
    }

    SECTION("Kernels: RNG, graph, barrier, window");
    { K::Rng a(5), b(5), c(6); CHECK(a.next() == b.next() && a.next() != c.next()); }
    {
        K::CsrGraph g = K::make_power_law_graph(4096, 8.0);
        CHECK(g.rowptr.size() == 4097 && g.rowptr.back() == g.col.size());
        bool mono = true, inrange = true; uint32_t maxdeg = 0;
        for (size_t i = 0; i < 4096; ++i) { mono &= g.rowptr[i] <= g.rowptr[i + 1]; maxdeg = std::max(maxdeg, g.rowptr[i + 1] - g.rowptr[i]); }
        for (uint32_t c : g.col) inrange &= c < 4096;
        CHECK(mono && inrange);
        const double avg = (double)g.col.size() / 4096.0;
        CHECK(avg > 6.0 && avg < 14.0);
        CHECK((double)maxdeg > 20.0 * avg);                    // heavy tail: hubs exist
        CHECK(K::make_power_law_graph(4096, 8.0).col == g.col); // deterministic
    }
    {
        constexpr int T = 4, R = 500;
        K::SpinBarrier bar(T);
        std::atomic<int> counter{0}; std::atomic<bool> bad{false};
        std::vector<std::thread> ts;
        for (int t = 0; t < T; ++t) ts.emplace_back([&] {
            for (int r = 1; r <= R; ++r) { counter.fetch_add(1); bar.wait(); if (counter.load() < r * T) bad = true; bar.wait(); }
        });
        for (auto& t : ts) t.join();
        CHECK(!bad.load() && counter.load() == T * R);
    }
    {
        auto w = K::run_window(3, {}, 0.05, [](int) { return 0; }, [](int&) { return 1.0; });
        CHECK(w.ok && w.seconds >= 0.05 && w.units > 3);
        auto bad = K::run_window(2, {}, 0.05, [](int) -> int { throw std::bad_alloc(); }, [](int&) { return 1.0; });
        CHECK(!bad.ok);                                       // a failing allocation is reported, not thrown across threads
    }

    SECTION("Kernels: measurements produce sane numbers");
    {
        Deadline dl(20); const size_t cap = 256ull << 20;
        auto g32 = K::gemm(K::Precision::F32, {128, 128, 128, "t"}, 1, {}, 0.03, 3, dl, cap);
        auto g8  = K::gemm(K::Precision::I8,  {128, 128, 128, "t"}, 1, {}, 0.03, 3, dl, cap);
        CHECK(g32.ok && g32.gops > 0 && g8.ok && g8.gops > 0);
        CHECK(!K::gemm(K::Precision::F32, {4096, 4096, 4096, "huge"}, 1, {}, 0.03, 3, dl, 1u << 20).ok);   // over the cap: refused
        auto bw = K::read_bandwidth("L2", 128u << 10, 1, {}, 0.03, 3, dl, cap);
        CHECK(bw.ok && bw.read_gbs > 0);
        auto tr = K::triad_bandwidth(4u << 20, 1, {}, 0.03, 3, dl, cap);
        CHECK(tr.ok && tr.value > 0);
        auto sp = K::sparse_gather_scatter(8192, 32, 8.0, dl, cap);
        CHECK(sp.ok && sp.gather_gbs > 0 && sp.scatter_gbs > 0 && sp.edges > 8192);
        auto cv = K::conv3x3(8, 32, 1, {}, 0.03, 3, dl, cap);
        CHECK(cv.ok && cv.gflops > 0);
        auto dc = K::dependent_chains(64, dl);
        CHECK(dc.ok && dc.fma_latency_ns > 0.1 && dc.rnn_step_us > 0);
        auto nb = K::nbody(256, 1, {}, 0.03, 3, dl, cap);
        CHECK(nb.ok && nb.interactions_per_s > 0);
        auto ov = K::overheads(1, dl);
        CHECK(ov.ok && ov.thread_spawn_us > 0 && ov.barrier_roundtrip_us == 0.0);   // 1 thread: no pool barrier measured
        auto sc = K::thread_scaling({}, 2, 128, 0.03, 3, dl, cap);
        CHECK(sc.ok && sc.points.size() == 2 && sc.points[0].efficiency == 1.0 && sc.best_threads >= 1 && sc.best_threads <= 2);
        auto sg = K::sync_gemm({}, 2, 128, 0.03, 3, dl, cap);
        CHECK(sg.ok && sg.value > 0);
    }

    SECTION("Kernels: sustained run honours the stop callback");
    {
        int calls = 0;
        auto sr = K::sustained_gemm(1, {}, 128, 5.0, 0.1, 256ull << 20, [&](double) { return ++calls > 3 ? std::string("test-stop") : std::string(); });
        CHECK(sr.stopped_reason == "test-stop" && sr.window_gflops.size() == 3);
        CHECK(!sr.ok);                                        // 3 windows: refuses to claim a throttling fraction
        CHECK(sr.seconds_run < 2.0);
    }
}

// Regression: sync_gemm (barrier inside the timed loop) used to deadlock when one worker
// left its loop on its own clock while a peer was already waiting at the barrier. The
// stop decision is now collective. A watchdog turns a hang into a failure.
static void test_barrier_no_deadlock() {
    SECTION("Kernels: barrier-paced loop never deadlocks");
    Deadline dl(60);
    auto fut = std::async(std::launch::async, [&] {
        int ok = 0;
        for (int rep = 0; rep < 40; ++rep) {
            const int threads = 2 + (rep % 3);                 // 2, 3, 4 threads (oversubscribed on small machines)
            auto m = K::sync_gemm({}, threads, 32, 0.002, 2, dl, 64ull << 20);
            if (m.ok && m.value > 0) ++ok;
        }
        return ok;
    });
    if (fut.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
        ++g_checks; ++g_failed;
        std::printf("  FAIL sync_gemm hung (barrier deadlock)\n");
        std::fflush(stdout);
        std::_Exit(2);                                         // the hung threads can never be joined
    }
    CHECK(fut.get() == 40);
}

static bool json_balanced_and_clean(const std::string& s) {
    int brace = 0, brack = 0; bool in_str = false, esc = false;
    for (char c : s) {
        if (in_str) { if (esc) esc = false; else if (c == '\\') esc = true; else if (c == '"') in_str = false; continue; }
        if (c == '"') in_str = true;
        else if (c == '{') ++brace; else if (c == '}') --brace;
        else if (c == '[') ++brack; else if (c == ']') --brack;
        if (brace < 0 || brack < 0) return false;
    }
    std::string low = to_lower(s);
    return !in_str && brace == 0 && brack == 0 && low.find("nan") == std::string::npos && low.find("inf") == std::string::npos;
}

static void test_real_machine(bool quick) {
    SECTION("Real machine: fingerprint sanity");
    DeviceFingerprint fp = detect_device();
    CHECK(fp.os == OS::Linux || fp.os == OS::Android);
    CHECK(fp.logical_cpus >= 1 && !fp.clusters.empty());
    int sum = 0; for (const auto& c : fp.clusters) sum += (int)c.cpus.size();
    CHECK(sum == fp.logical_cpus);
    CHECK(fp.physical_cores >= 1 && fp.physical_cores <= fp.logical_cpus);
    CHECK(fp.memory.total_bytes > 0 && fp.memory.safe_budget_bytes < fp.memory.total_bytes);
    CHECK(!fp.simd.summary().empty());
    if (quick) return;

    SECTION("Suite: hot device stops immediately");
    {
        Fx fx; fx.put("/sys/class/thermal/thermal_zone0/type", "soc\n"); fx.put("/sys/class/thermal/thermal_zone0/temp", "52000\n");
        SuiteConfig c; c.time_budget_s = 10; c.sysroot = fx.str();
        CapabilityVector cv = CapabilitySuite(c).run(fp);
        CHECK(cv.stopped_reason.rfind("thermal:", 0) == 0);
        CHECK(cv.time_used_s < 1.0);
        CHECK(cv.f32_gflops_best == 0.0 && cv.bw_dram_gbs == 0.0);
        bool any_ran = false; for (const auto& g : cv.groups) any_ran |= g.ran;
        CHECK(!any_ran && cv.groups.size() == 11);
    }

    SECTION("Suite: full run under a 9 s budget");
    {
        SuiteConfig c; c.time_budget_s = 9.0; c.sustained_s = 2.0;
        Fx fx; c.sysroot = fx.str();                           // no sensors in the fixture: nothing to trip
        CapabilityVector cv = CapabilitySuite(c).run(fp);
        CHECK(cv.time_used_s <= c.time_budget_s + 1.0);       // budget respected (one in-flight window of slack)
        CHECK(cv.groups.size() == 11);
        CHECK(cv.group_ran("gemm") && cv.group_ran("latency") && cv.group_ran("bandwidth") && cv.group_ran("sparse"));
        CHECK(cv.f32_gflops_best > 0 && cv.f64_gflops > 0 && cv.i8_gops > 0 && cv.conv_gflops > 0 && cv.nbody_gflops > 0);
        CHECK(cv.f32_gflops_skinny > 0);
        CHECK(cv.bw_dram_gbs > 0 && cv.triad_max_gbs >= cv.triad_1t_gbs && !cv.triad_sweep.empty());
        CHECK(cv.dram_saturation_threads >= 1 && cv.dram_saturation_threads <= (int)cv.triad_sweep.size());
        CHECK(cv.gather_gbs > 0 && cv.gather_efficiency > 0 && cv.gather_efficiency < 50);
        CHECK(cv.fma_latency_ns > 0 && cv.rnn_step_us > 0 && cv.thread_spawn_us > 0 && cv.tiny_op_ns > 0);
        CHECK(!cv.scaling.empty() && cv.recommended_threads >= 1);
        CHECK(cv.ridge_flop_per_byte > 0);
        if (cv.build_is_optimised) {                           // timing relationships only mean something when optimised
            CHECK(cv.lat_l1_ns > 0 && cv.lat_dram_ns > cv.lat_l1_ns);
            CHECK(cv.bw_l1_gbs > cv.bw_dram_gbs);
            CHECK(cv.f32_gflops_best >= cv.f64_gflops * 0.8);  // f32 is never much slower than f64
        }
        std::string js = suite_to_json(fp, cv);
        CHECK(json_balanced_and_clean(js));
        CHECK(js.find("\"profiler_version\":\"2.0\"") != std::string::npos);
        std::printf("  (info) f32 %.1f GFLOP/s, DRAM read %.1f GB/s, noise %.0f%%, used %.1f s\n", cv.f32_gflops_best, cv.bw_dram_gbs, cv.noise * 100, cv.time_used_s);
    }
}

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    test_common();
    test_arm_phone();
    test_armv7_phone();
    test_x86_generations();
    test_core_grouping();
    test_memory();
    test_level_plan_and_latency();
    test_thermal_guard();
    test_kernels();
    test_barrier_no_deadlock();
    test_real_machine(quick);
    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
