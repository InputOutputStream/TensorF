/*
 * profiler_android_test.cpp - AndroidProbe.hpp on fixture trees (no device needed).
 *   g++ -std=c++20 -O2 -Wall -Wextra -Iinclude -Iinclude/tools tests/profiler_android_test.cpp -o bin/profiler_android_test -pthread
 */
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "Profiler/device/AndroidProbe.hpp"

using namespace tensorf::prof;
namespace an = tensorf::prof::android;
namespace fs = std::filesystem;

static int g_checks = 0, g_failed = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_failed; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define SECTION(n) std::printf("[%s]\n", n)

struct Fx {
    std::string root;
    Fx() { root = (fs::temp_directory_path() / ("tensorf_android_" + std::to_string((long long)(now_s() * 1e6)))).string(); fs::create_directories(root); }
    ~Fx() { std::error_code ec; fs::remove_all(root, ec); }
    void put(const std::string& rel, const std::string& text) const {
        const fs::path p = fs::path(root) / rel.substr(1); fs::create_directories(p.parent_path()); std::ofstream(p) << text;
    }
    void dev(const std::string& rel) const { put(rel, ""); }
};

static bool balanced(const std::string& s) {
    int b = 0, k = 0; bool in = false, esc = false;
    for (char c : s) { if (in) { if (esc) esc = false; else if (c == '\\') esc = true; else if (c == '"') in = false; continue; }
        if (c == '"') in = true; else if (c == '{') ++b; else if (c == '}') --b; else if (c == '[') ++k; else if (c == ']') --k; if (b < 0 || k < 0) return false; }
    return !in && b == 0 && k == 0;
}
static bool has_note(const an::AndroidSystem& s, const char* n) { for (const auto& x : s.notes) if (x.find(n) != std::string::npos) return true; return false; }

static void test_units() {
    SECTION("frequency units are normalised and the guess is recorded");
    using U = an::Unit;
    an::Freq a = an::parse_freq("587000000\n", "n", U::HZ);   CHECK(a.read && a.mhz == 587.0 && a.unit_guess == "Hz");
    an::Freq b = an::parse_freq("1804800", "n", U::KHZ);       CHECK(b.read && b.mhz == 1804.8 && b.unit_guess == "kHz");   // cpufreq is kHz: 1804800 is 1.8 GHz, not 1.8 MHz
    an::Freq c = an::parse_freq("600000", "n");                CHECK(c.read && c.mhz == 600.0 && c.unit_guess == "kHz (guessed)");
    an::Freq d = an::parse_freq("650", "n");                   CHECK(d.read && d.mhz == 650.0 && d.unit_guess == "MHz (assumed)");
    an::Freq e = an::parse_freq("680", "n", U::HZ);            CHECK(e.read && e.mhz == 680.0 && e.unit_guess == "MHz (assumed)");   // implausible as Hz: falls back, says so
    CHECK(!an::parse_freq("", "n").read && !an::parse_freq("0", "n").read && !an::parse_freq("abc", "n").read);
    auto m = an::parse_build_prop("# c\nro.hardware=qcom\n\nro.product.model = Pixel X \nbad line\n=novalue\n");
    CHECK(m.size() == 2 && m["ro.hardware"] == "qcom" && m["ro.product.model"] == "Pixel X");
}

static void test_phone() {
    SECTION("a Snapdragon-like phone tree");
    Fx fx;
    fx.put("/system/build.prop", "ro.hardware=qcom\nro.product.board=kalama\nro.board.platform=kalama\nro.product.model=SM-X\nro.product.manufacturer=acme\n"
                                 "ro.build.version.release=14\nro.build.version.sdk=34\nro.build.version.security_patch=2025-01-05\nro.hardware.egl=adreno\n");
    fx.put("/vendor/build.prop", "ro.soc.model=SM8550\nro.soc.manufacturer=QTI\nro.hardware=qcom_vendor\n");
    fx.put("/proc/meminfo", "MemTotal: 8000000 kB\nBuffers: 1000 kB\nCached: 2000 kB\nSwapTotal: 4000000 kB\nSwapFree: 3999000 kB\nSlab: 300 kB\n");
    fx.put("/sys/block/zram0/disksize", "4294967296\n");
    fx.put("/sys/class/devfreq/soc:qcom,ddr-minmax/cur_freq", "2736000000\n");   // matched by the "ddr" glob, as the sample's `cat *ddr-minmax`
    fx.put("/sys/class/kgsl/kgsl-3d0/gpu_model", "Adreno740\n");
    fx.put("/sys/class/kgsl/kgsl-3d0/gpuclk", "680000000\n");
    fx.dev("/dev/kgsl-3d0"); fx.dev("/dev/adsprpc-smd"); fx.dev("/dev/ion");
    fx.put("/sys/class/scsi_host/host0/proc_name", "ufshcd\n");
    fx.put("/sys/block/sda/size", "1000\n");
    fx.put("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "1804800\n");
    fx.put("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "2016000\n");
    fx.put("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "walt\n");
    fx.put("/sys/devices/system/cpu/cpu1/online", "1\n");
    fx.put("/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq", "1000000\n");
    fx.put("/sys/devices/system/cpu/cpu2/online", "0\n");                          // genuinely offline
    fx.put("/sys/devices/system/cpu/cpu3/online", "1\n");                          // online but cpufreq unreadable
    fx.put("/sys/devices/system/cpu/cpufreq/policy0/x", "1\n"); fx.put("/sys/devices/system/cpu/cpuidle/x", "1\n");   // not cores
    fx.put("/proc/sys/kernel/osrelease", "5.15.0-android13\n");
    fx.put("/sys/fs/selinux/enforce", "1\n");
    fx.put("/sys/class/thermal/thermal_zone0/type", "cpu-1-0\n"); fx.put("/sys/class/thermal/thermal_zone0/temp", "41500\n");
    fx.put("/data/.keep", "");

    const an::AndroidSystem s = an::collect_android(fx.root);
    CHECK(s.android);
    CHECK(s.memory.meminfo_read && s.memory.swap_total_kb == 4000000 && s.memory.swap_free_kb == 3999000 && s.memory.cached_kb == 2000 && s.memory.slab_kb == 300);
    CHECK(s.memory.zram_disk_bytes == 4294967296ull);
    CHECK(s.memory.ddr.read && s.memory.ddr.mhz == 2736.0);
    CHECK(s.data_fs.read && s.data_fs.total_bytes > 0 && s.data_fs.avail_bytes <= s.data_fs.total_bytes);
    CHECK(s.storage_link.interface_kind == "UFS");                                  // by host driver, not by the "sda" name
    CHECK(s.gpu.model == "Adreno740" && s.gpu.cur.read && s.gpu.cur.mhz == 680.0 && s.gpu.kernel_nodes.size() == 1);
    CHECK(s.dsp.present.size() == 1 && s.dsp.present[0] == "/dev/adsprpc-smd");     // /dev/ion is not evidence of an NPU
    CHECK(s.configured_cpus == 4 && s.online_cpus == 3);
    CHECK(s.cores[0].cur.mhz == 1804.8 && s.cores[0].max.mhz == 2016.0 && s.cores[0].governor == "walt" && !s.cores[0].online_known && s.cores[0].online);
    CHECK(s.cores[2].online_known && !s.cores[2].online);
    CHECK(s.cores[3].online && !s.cores[3].cur.read);                               // unreadable is NOT reported as offline
    CHECK(has_note(s, "no readable scaling_cur_freq"));
    CHECK(s.soc.hardware == "qcom_vendor");                                         // vendor build.prop overrides system
    CHECK(s.soc.platform == "kalama" && s.soc.soc_model == "SM8550" && s.soc.manufacturer == "acme");
    CHECK(s.os.sdk == "34" && s.os.android_release == "14" && s.os.security_patch == "2025-01-05");
    CHECK(s.os.kernel == "5.15.0-android13" && s.os.selinux == "enforcing");
    CHECK(s.thermal.size() == 1 && s.thermal[0].temp_c == 41.5);

    const std::string j = an::to_json(s);
    CHECK(balanced(j) && j.find("\"interface\":\"UFS\"") != std::string::npos && j.find("nan") == std::string::npos);
    std::ostringstream os; an::print_android(s, os);
    CHECK(os.str().find("=== GPU METRICS & FREQUENCIES ===") != std::string::npos && os.str().find("Adreno740") != std::string::npos);
    CHECK(os.str().find("Offline / Deep Sleep") == std::string::npos);
}

static void test_variants() {
    SECTION("other vendors' nodes, and a custom property reader");
    {
        Fx fx;
        fx.put("/sys/class/devfreq/gpufreq/cur_freq", "700000000\n");                // explicit node, no kgsl
        fx.put("/sys/class/devfreq/ddrfreq/cur_freq", "3200000000\n");
        fx.put("/sys/block/mmcblk0/size", "100\n"); fx.dev("/dev/mali0");
        const an::AndroidSystem s = an::collect_android(fx.root, [](const std::string& k) { return k == "ro.hardware.egl" ? std::string("mali") : std::string(); });
        CHECK(s.gpu.cur.mhz == 700.0 && s.gpu.model == "mali" && s.gpu.model_source.find("ro.hardware.egl") != std::string::npos);
        CHECK(s.memory.ddr.mhz == 3200.0);
        CHECK(s.storage_link.interface_kind == "eMMC");
        CHECK(s.dsp.present.empty() && s.gpu.kernel_nodes.size() == 1 && s.gpu.kernel_nodes[0] == "/dev/mali0");   // mali0 is a GPU, never an NPU
    }
    {
        Fx fx; fx.put("/sys/block/sda/size", "10\n");
        CHECK(an::read_storage_link(fx.root).interface_kind == "SCSI/USB (not distinguishable)");
        fx.put("/sys/block/mmcblk0boot0/size", "1\n");
        CHECK(an::read_storage_link(fx.root).interface_kind != "eMMC");              // boot partitions do not define the interface
    }
}

static void test_empty() {
    SECTION("an empty tree says what it could not read and invents nothing");
    Fx fx;
    const an::AndroidSystem s = an::collect_android(fx.root);
    CHECK(!s.android && s.cores.empty() && s.gpu.model.empty() && !s.gpu.cur.read && !s.memory.ddr.read && s.dsp.present.empty());
    CHECK(s.storage_link.interface_kind == "unknown" && s.os.selinux == "unknown");
    for (const char* n : {"/proc/meminfo unreadable", "no readable DDR clock", "/data not readable", "no readable GPU clock", "per-core frequencies unknown",
                          "no Android system properties", "SELinux state unknown", "no readable thermal zones", "does not look like Android"})
        CHECK(has_note(s, n));
    CHECK(balanced(an::to_json(s)));
    std::ostringstream os; an::print_android(s, os); CHECK(os.str().find("not read") != std::string::npos);
}

static void test_real() {
    SECTION("this machine: must not crash, must stay consistent");
    const an::AndroidSystem s = an::collect_android("");
    CHECK(s.configured_cpus >= 1 && s.online_cpus >= 1 && s.online_cpus <= s.configured_cpus);
    CHECK(balanced(an::to_json(s)));
    std::printf("  (info) %d cpus, android=%d, notes=%zu\n", s.configured_cpus, (int)s.android, s.notes.size());
}

int main() {
    test_units(); test_phone(); test_variants(); test_empty(); test_real();
    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
