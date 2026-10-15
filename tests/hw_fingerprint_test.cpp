#include <cassert>
#include <cstdio>
#include <fstream>
#include "profiler/HardwareFingerprint.hpp"
using namespace tensorf::profile;
using namespace tensorf::profile::detail;

static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++fails; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

static void t_parsers() {
    CHECK(parse_cpulist("0-3,6,8-9") == (std::vector<int>{0,1,2,3,6,8,9}));
    CHECK(parse_cpulist("0\n") == (std::vector<int>{0}));
    CHECK(parse_cpulist("").empty());
    CHECK(parse_cpulist("5-3").empty());          // reversed range rejected
    CHECK(parse_cpulist("0-99999999").empty());   // absurd range rejected
    CHECK(parse_size("32K") == 32768 && parse_size("2048K") == 2097152 && parse_size("1M") == 1048576 && parse_size("512") == 512);
    CHECK(parse_size("junk") == 0 && parse_size("") == 0);
    CHECK(parse_u64("0xd05") == 0xd05 && parse_u64(" 42\n") == 42 && !parse_u64("12abc") && !parse_u64("-1") && !parse_u64(""));
    CHECK(!parse_cgroup_limit("max") && !parse_cgroup_limit("9223372036854771712") && parse_cgroup_limit("1073741824") == 1073741824ull);
    CHECK(parse_cpu_max("200000 100000") == 2.0 && !parse_cpu_max("max 100000") && parse_cpu_max("50000 100000") == 0.5);
    auto mi = parse_meminfo("MemTotal:        7933580 kB\nMemAvailable:    3000000 kB\nHugePages_Total:       0\n");
    CHECK(mi["MemTotal"] == 7933580ull * 1024 && mi["MemAvailable"] == 3000000ull * 1024 && mi["HugePages_Total"] == 0);
    CHECK(std::string(arm_uarch_name(0x41, 0xd05)) == "Cortex-A55" && std::string(arm_uarch_name(0x41, 0xd44)) == "Cortex-X1");
    CHECK(std::string(arm_uarch_name(0x41, 0xffff)).empty());
    JsonOut j; j.begin_obj(); j.kv("a", "q\"\n\\"); j.key("n"); j.num(NAN); j.kv("o", std::optional<uint64_t>{}); j.ints("v", {1,2}); j.end_obj();
    CHECK(j.str() == "{\"a\":\"q\\\"\\n\\\\\",\"n\":null,\"o\":null,\"v\":[1,2]}");
}

// ---- A fake 8-core big.LITTLE phone: 4x A55, 3x A78, 1x X1 (Android 12, API 31) ----
static void add_phone_cpu(MemFs& fs, bool with_freq, bool with_cap) {
    std::string info;
    int part[8] = {0xd05,0xd05,0xd05,0xd05,0xd41,0xd41,0xd41,0xd44};
    for (int i = 0; i < 8; ++i) {
        info += "processor\t: " + std::to_string(i) + "\nBogoMIPS\t: 38.40\nFeatures\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp\n";
        char b[16]; std::snprintf(b, sizeof b, "0x%x", part[i]);
        info += "CPU implementer\t: 0x41\nCPU architecture: 8\nCPU variant\t: 0x1\nCPU part\t: " + std::string(b) + "\nCPU revision\t: 0\n\n";
    }
    fs.put("/proc/cpuinfo", info);
    fs.put("/sys/devices/system/cpu/present", "0-7\n");
    fs.put("/sys/devices/system/cpu/online", "0-7\n");
    for (int i = 0; i < 8; ++i) {
        std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(i);
        int cap = i < 4 ? 462 : (i < 7 ? 871 : 1024);
        int khz = i < 4 ? 1804800 : (i < 7 ? 2419200 : 2841600);
        if (with_freq) { fs.put(base + "/cpufreq/cpuinfo_max_freq", std::to_string(khz) + "\n"); fs.put(base + "/cpufreq/cpuinfo_min_freq", "300000\n"); }
        if (with_cap) fs.put(base + "/cpu_capacity", std::to_string(cap) + "\n");
    }
    auto cache = [&](int cpu, int idx, int lvl, const char* type, const char* size, const char* shared) {
        std::string d = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache/index" + std::to_string(idx) + "/";
        fs.put(d + "level", std::to_string(lvl) + "\n"); fs.put(d + "type", std::string(type) + "\n");
        fs.put(d + "size", std::string(size) + "\n"); fs.put(d + "coherency_line_size", "64\n");
        fs.put(d + "ways_of_associativity", "4\n"); fs.put(d + "shared_cpu_list", std::string(shared) + "\n");
    };
    cache(0,0,1,"Data","32K","0"); cache(0,1,1,"Instruction","32K","0"); cache(0,2,2,"Unified","128K","0-3"); cache(0,3,3,"Unified","2048K","0-7");
    cache(4,0,1,"Data","64K","4"); cache(4,1,1,"Instruction","64K","4"); cache(4,2,2,"Unified","512K","4-6"); cache(4,3,3,"Unified","2048K","0-7");
    cache(7,0,1,"Data","64K","7"); cache(7,1,1,"Instruction","64K","7"); cache(7,2,2,"Unified","1024K","7"); cache(7,3,3,"Unified","2048K","0-7");
}

static Probe phone_probe(MemFs& fs) {
    Probe p; p.fs = &fs; p.arch = "aarch64"; p.is_android = true;
    p.prop = [](const std::string& k) -> std::string {
        if (k == "ro.build.version.sdk") return "31";
        if (k == "ro.build.version.release") return "12";
        if (k == "ro.product.manufacturer") return "Acme";
        if (k == "ro.product.model") return "Phone 9";
        if (k == "ro.soc.model") return "SM8350";
        if (k == "ro.build.characteristics") return "nosdcard";
        return "";
    };
    p.disk = [](const std::string&) { return std::optional<DiskSpace>(DiskSpace{128ull << 30, 40ull << 30}); };
    p.affinity_cpus = [] { return 8; };
    return p;
}

static void t_phone() {
    MemFs fs;
    add_phone_cpu(fs, true, true);
    fs.put("/proc/meminfo", "MemTotal:        7800000 kB\nMemFree:          300000 kB\nMemAvailable:    2900000 kB\nSwapTotal:       4194300 kB\nSwapFree:        4000000 kB\n");
    fs.put("/proc/swaps", "Filename Type Size Used Priority\n/dev/block/zram0 partition 4194300 194300 -2\n");
    fs.put("/sys/module/lowmemorykiller/parameters/minfree", "18432,23040,27648,32256,55296,80640\n");
    fs.put("/proc/pressure/memory", "some avg10=1.50 avg60=0.40 avg300=0.10 total=12345\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n");
    fs.put("/sys/class/thermal/thermal_zone0/type", "cpu-1-0\n"); fs.put("/sys/class/thermal/thermal_zone0/temp", "41500\n");
    fs.put("/sys/class/thermal/thermal_zone1/type", "battery\n");  fs.put("/sys/class/thermal/thermal_zone1/temp", "33000\n");
    fs.put("/sys/class/thermal/thermal_zone2/type", "broken\n");   fs.put("/sys/class/thermal/thermal_zone2/temp", "-274000\n");
    fs.put("/sys/class/thermal/cooling_device0/temp", "1\n");
    fs.put("/sys/class/power_supply/battery/type", "Battery\n"); fs.put("/sys/class/power_supply/battery/capacity", "63\n");
    fs.put("/sys/class/power_supply/battery/status", "Discharging\n"); fs.put("/sys/class/power_supply/battery/temp", "312\n");
    fs.put("/sys/class/power_supply/battery/voltage_now", "3950000\n"); fs.put("/sys/class/power_supply/battery/current_now", "-450000\n");
    fs.put("/sys/class/power_supply/usb/type", "USB\n"); fs.put("/sys/class/power_supply/usb/online", "0\n");

    auto fp = collect_linux(phone_probe(fs));
    CHECK(fp.os.name == "Android" && fp.os.android_api == 31 && fp.os.model == "Phone 9" && fp.os.form_factor == "phone");
    CHECK(fp.cpu.vendor == "ARM" && fp.cpu.arch == "aarch64" && fp.cpu.soc_hint == "SM8350");
    CHECK(fp.cpu.logical_cpus == 8 && fp.cpu.online_cpus == 8);
    CHECK(fp.cpu.classes.size() == 3);
    if (fp.cpu.classes.size() == 3) {
        auto& L = fp.cpu.classes[0]; auto& M = fp.cpu.classes[1]; auto& B = fp.cpu.classes[2];
        CHECK(L.name == "little" && L.cpus == (std::vector<int>{0,1,2,3}) && L.uarch == "Cortex-A55");
        CHECK(M.name == "mid" && M.cpus == (std::vector<int>{4,5,6}) && M.uarch == "Cortex-A78");
        CHECK(B.name == "prime" && B.cpus == (std::vector<int>{7}) && B.uarch == "Cortex-X1" && B.max_khz == 2841600);
        CHECK(std::abs(L.rel_perf - 462.0/1024) < 1e-9 && B.rel_perf == 1.0);
        // caches are per class, from that class's first cpu
        CHECK(L.caches.size() == 4 && L.caches[0].level == 1 && L.caches[0].size_bytes == 32768 && L.caches[0].line_bytes == 64);
        CHECK(B.caches.size() == 4 && B.caches[2].level == 2 && B.caches[2].size_bytes == 1024 * 1024);
        CHECK(M.caches[2].shared_cpus == (std::vector<int>{4,5,6}));
    }
    CHECK(fp.cpu.simd.neon && fp.cpu.simd.fp16 && fp.cpu.simd.dotprod && !fp.cpu.simd.i8mm && !fp.cpu.simd.sve);
    CHECK(fp.performance_cpus() == (std::vector<int>{7}) && fp.performance_cpus(true) == (std::vector<int>{4,5,6,7}));
    CHECK(fp.memory.total == 7800000ull * 1024 && fp.memory.available == 2900000ull * 1024 && fp.memory.swap_is_zram);
    CHECK(fp.memory.lmk_min_free_bytes == 80640ull * 4096);
    CHECK(fp.memory.psi_some_avg10 && std::abs(*fp.memory.psi_some_avg10 - 1.5) < 1e-9);
    CHECK(fp.thermal.size() == 2 && fp.thermal[0].name == "cpu-1-0" && std::abs(fp.thermal[0].temp_c - 41.5) < 1e-9);   // garbage zone dropped
    CHECK(fp.power.available && fp.power.battery_present && !fp.power.on_external_power && fp.power.level_pct == 63);
    CHECK(fp.power.battery_temp_c && std::abs(*fp.power.battery_temp_c - 31.2) < 1e-9 && fp.power.current_ma && *fp.power.current_ma < 0);
    CHECK(fp.storage.known && fp.storage.free == (40ull << 30));
    CHECK(fp.usable_threads == 8);
    CHECK(fp.notes.empty());
    // id is stable and independent of volatile state
    auto id1 = fp.id();
    fs.put("/proc/meminfo", "MemTotal:        7800000 kB\nMemAvailable:    100000 kB\n");
    fs.put("/sys/class/power_supply/battery/capacity", "5\n");
    CHECK(collect_linux(phone_probe(fs)).id() == id1);
    // ...but changes with the hardware
    fs.put("/proc/meminfo", "MemTotal:        3800000 kB\nMemAvailable:    100000 kB\n");
    CHECK(collect_linux(phone_probe(fs)).id() != id1);
    // JSON contains the essentials and parses (checked externally in main via python)
    std::string js = fp.to_json();
    CHECK(js.find("\"name\":\"prime\"") != std::string::npos && js.find("\"android_api\":31") != std::string::npos);
    std::ofstream("/tmp/fp_phone.json") << js;
}

static void t_phone_hidden_sysfs() {
    // Android 8+ style: cpufreq and cpu_capacity are not readable. Cores must still be classed by uarch part.
    MemFs fs;
    add_phone_cpu(fs, false, false);
    fs.put("/proc/meminfo", "MemTotal: 4000000 kB\nMemAvailable: 1000000 kB\n");
    auto fp = collect_linux(phone_probe(fs));
    CHECK(fp.cpu.classes.size() == 3);
    CHECK(!fp.notes.empty());
    bool freq_note = false, cap_note = false;
    for (auto& n : fp.notes) { freq_note |= n.find("cpufreq") != std::string::npos; cap_note |= n.find("cpu_capacity") != std::string::npos; }
    CHECK(freq_note && cap_note);
    // without capacity or frequency we cannot rank classes by speed; order is part-id order (little->big happens to match here)
    CHECK(fp.cpu.classes[0].cpus.size() == 4);
    CHECK(fp.thermal.empty() && !fp.power.available);
}

static void t_tier_order() {
    // Capacity and frequency hidden; part ids are NOT in speed order (A520=0xd80 is slower than A715=0xd4d, X4=0xd82).
    MemFs fs; std::string info;
    int part[8] = {0xd80,0xd80,0xd80,0xd4d,0xd4d,0xd81,0xd81,0xd82};
    for (int i = 0; i < 8; ++i) {
        char b[16]; std::snprintf(b, sizeof b, "0x%x", part[i]);
        info += "processor\t: " + std::to_string(i) + "\nFeatures\t: fp asimd\nCPU implementer\t: 0x41\nCPU part\t: " + b + "\n\n";
    }
    fs.put("/proc/cpuinfo", info); fs.put("/sys/devices/system/cpu/present", "0-7\n");
    Probe p; p.fs = &fs; p.arch = "aarch64";
    auto fp = collect_linux(p);
    CHECK(fp.cpu.classes.size() == 4);
    if (fp.cpu.classes.size() == 4) {
        CHECK(fp.cpu.classes[0].uarch == "Cortex-A520" && fp.cpu.classes[3].uarch == "Cortex-X4");
        CHECK(fp.cpu.classes[3].name == "prime" && fp.cpu.classes[0].name == "little");
        CHECK(fp.cpu.classes[1].tier == 1 && fp.cpu.classes[2].tier == 1);
    }
}

static void t_arm32_and_offline() {
    // 32-bit ARMv7 phone, old kernel: Processor/Hardware in a footer, only 2 of 4 cores online, no MemAvailable.
    MemFs fs;
    fs.put("/proc/cpuinfo",
        "Processor\t: ARMv7 Processor rev 4 (v7l)\nprocessor\t: 0\nBogoMIPS\t: 38.40\nFeatures\t: swp half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt\n"
        "CPU implementer\t: 0x41\nCPU architecture: 7\nCPU part\t: 0xc07\n\n"
        "processor\t: 1\nFeatures\t: swp half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt\nCPU implementer\t: 0x41\nCPU part\t: 0xc07\n\n"
        "Hardware\t: Qualcomm Technologies, Inc MSM8226\nRevision\t: 0000\n");
    fs.put("/sys/devices/system/cpu/present", "0-3\n");
    fs.put("/sys/devices/system/cpu/online", "0-1\n");
    fs.put("/proc/meminfo", "MemTotal: 1000000 kB\nMemFree: 100000 kB\nBuffers: 50000 kB\nCached: 150000 kB\n");
    Probe p; p.fs = &fs; p.arch = "armv7"; p.is_android = true;
    p.prop = [](const std::string& k) { return k == "ro.build.version.sdk" ? std::string("24") : std::string(); };
    auto fp = collect_linux(p);
    CHECK(fp.cpu.vendor == "ARM");                        // old code fell through to INTEL_PRE_HASWELL
    CHECK(fp.cpu.logical_cpus == 4 && fp.cpu.online_cpus == 2);
    CHECK(!fp.cpu.cores[2].online && fp.cpu.cores[0].online);
    CHECK(fp.cpu.soc_hint == "Qualcomm Technologies, Inc MSM8226");
    CHECK(fp.cpu.simd.neon && !fp.cpu.simd.fp16 && !fp.cpu.simd.dotprod);
    CHECK(fp.cpu.features_all.count("vfpv4") == 1);
    CHECK(fp.memory.available == (100000ull + 50000 + 150000) * 1024);   // fallback estimate
    CHECK(fp.os.android_api == 24);
    CHECK(fp.usable_threads == 2);                         // only online cores count
    // large values do not truncate on 32-bit: parse_u64 returns uint64_t
    CHECK(parse_u64("5000000000") == 5000000000ull);
}

static void t_cgroups() {
    MemFs fs;
    fs.put("/proc/cpuinfo", "processor\t: 0\nmodel name\t: Test CPU\nflags\t: fpu sse4_2 avx avx2 fma\nvendor_id\t: GenuineIntel\n\nprocessor\t: 1\nmodel name\t: Test CPU\nflags\t: fpu sse4_2 avx avx2 avx512f\nvendor_id\t: GenuineIntel\n");
    fs.put("/proc/meminfo", "MemTotal: 16000000 kB\nMemAvailable: 12000000 kB\n");
    // cgroup v2 limit 2 GiB, usage 0.5 GiB, 1.5 cpus
    fs.put("/sys/fs/cgroup/memory.max", "2147483648\n"); fs.put("/sys/fs/cgroup/memory.current", "536870912\n");
    fs.put("/sys/fs/cgroup/cpu.max", "150000 100000\n");
    Probe p; p.fs = &fs; p.arch = "x86_64";
    auto fp = collect_linux(p);
    CHECK(fp.memory.cgroup_limit == 2147483648ull);
    CHECK(fp.memory.effective_available == 2147483648ull - 536870912ull);
    CHECK(fp.usable_threads == 2);                         // ceil(1.5)
    CHECK(fp.cpu.features_all.count("avx512f") == 0 && fp.cpu.features_any.count("avx512f") == 1);   // intersection vs union
    CHECK(fp.cpu.simd.avx2 && !fp.cpu.simd.avx512f);
    CHECK(fp.os.form_factor == "desktop");
    // v1 + "no limit" sentinel
    MemFs fs2; fs2.files = fs.files;
    fs2.files.erase("/sys/fs/cgroup/memory.max"); fs2.files.erase("/sys/fs/cgroup/memory.current"); fs2.files.erase("/sys/fs/cgroup/cpu.max");
    fs2.put("/sys/fs/cgroup/memory/memory.limit_in_bytes", "9223372036854771712\n");
    fs2.put("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "-1\n"); fs2.put("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "100000\n");
    p.fs = &fs2;
    auto fp2 = collect_linux(p);
    CHECK(!fp2.memory.cgroup_limit && fp2.memory.effective_available == 12000000ull * 1024 && fp2.usable_threads == 2);
    // a cgroup limit that is not below total RAM is not a limit
    fs2.put("/sys/fs/cgroup/memory/memory.limit_in_bytes", "32000000000\n");
    CHECK(!collect_linux(p).memory.cgroup_limit);
    // v1 quota
    fs2.put("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "100000\n");
    CHECK(collect_linux(p).usable_threads == 1);
    // laptop heuristic
    fs2.put("/sys/class/power_supply/BAT0/type", "Battery\n"); fs2.put("/sys/class/power_supply/AC/type", "Mains\n"); fs2.put("/sys/class/power_supply/AC/online", "1\n");
    auto fp3 = collect_linux(p);
    CHECK(fp3.os.form_factor == "laptop" && fp3.power.on_external_power);
}

static void t_empty() {
    // Nothing readable at all must still produce a sane, non-crashing fingerprint with notes.
    MemFs fs; Probe p; p.fs = &fs; p.arch = "unknown";
    auto fp = collect_linux(p);
    CHECK(fp.cpu.logical_cpus >= 1 && fp.usable_threads >= 1 && !fp.notes.empty());
    CHECK(fp.cpu.classes.size() == 1 && fp.cpu.classes[0].name == "uniform");
    CHECK(fp.memory.total == 0 && fp.memory.effective_available == 0);
    (void)fp.to_json(); (void)fp.summary(); (void)fp.id();
}

static void t_real() {
    auto fp = collect();
    std::printf("---- real machine ----\n%s", fp.summary().c_str());
    CHECK(fp.cpu.logical_cpus >= 1 && fp.cpu.online_cpus >= 1 && fp.usable_threads >= 1);
    CHECK(!fp.cpu.arch.empty() && fp.cpu.arch != "unknown");
    CHECK(!fp.cpu.classes.empty() && !fp.cpu.classes[0].cpus.empty());
    CHECK(fp.memory.total > (64u << 20) && fp.memory.available > 0 && fp.memory.available <= fp.memory.total);
    CHECK(fp.memory.effective_available <= fp.memory.available);
    CHECK(fp.storage.known && fp.storage.total >= fp.storage.free && fp.storage.free > 0);
    CHECK(!fp.os.kernel.empty() && !fp.os.version.empty());
    CHECK(!fp.cpu.model_name.empty());
    CHECK(fp.id() == collect().id());
    std::ofstream("/tmp/fp_real.json") << fp.to_json();
}

int main() {
    t_parsers(); t_phone(); t_phone_hidden_sysfs(); t_tier_order(); t_arm32_and_offline(); t_cgroups(); t_empty(); t_real();
    std::printf("%d/%d checks passed\n", checks - fails, checks);
    return fails ? 1 : 0;
}
