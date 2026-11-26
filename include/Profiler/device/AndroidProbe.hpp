#pragma once
/*
 * AndroidProbe.hpp
 * ================
 * Android/ARM system facts beyond DeviceFingerprint: DDR and GPU clocks, per-core frequencies, swap/zram, the
 * /data filesystem, the storage interface, DSP/NPU device nodes, SoC/board and OS properties, SELinux state.
 * Layout follows get_system_info.cpp (the same sysfs/proc paths and the same nine sections) with these rules of the suite:
 *
 *   - no popen / dumpsys / getprop / cmd / getenforce: every value is a direct file read, a statvfs, or an Android
 *     system property (__system_property_get on a real device, build.prop files under a fake root);
 *   - every function takes a root prefix ("" = the real machine) so tests can build a fake /sys and /proc;
 *   - a value that could not be read is "not read" with a note, never invented: an unreadable cpufreq node does not
 *     mean the core is asleep, a /dev/mali0 node is a GPU (not an NPU), and /dev/ion is a generic buffer allocator;
 *   - frequency units are normalised and the guess is recorded (devfreq/kgsl nodes are Hz by kernel ABI, some vendor nodes are not).
 *
 * Status: tested on a Linux host against fixture trees only (tests/profiler_android_test.cpp). Not run on an Android
 * device: node names vary by SoC vendor and SELinux restricts app reads on Android 8+, so expect "not read" entries.
 */
#include "Profiler/core/ProfilerCommon.hpp"
#include "Profiler/device/DeviceFingerprint.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif
#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

namespace tensorf::prof::android {

// ─── Small pure helpers ──────────────────────────────────────────────────────

inline std::vector<std::string> list_dir(const std::string& dir) {
    std::vector<std::string> out;
#if defined(__linux__)
    if (DIR* d = opendir(dir.c_str())) {
        while (dirent* e = readdir(d)) { std::string n = e->d_name; if (n != "." && n != "..") out.push_back(n); }
        closedir(d);
    }
#else
    (void)dir;
#endif
    std::sort(out.begin(), out.end());
    return out;
}

inline bool path_exists(const std::string& path) {
#if defined(__linux__)
    return access(path.c_str(), F_OK) == 0;
#else
    (void)path; return false;
#endif
}

// A clock read from a node whose unit is not guaranteed. Kernel devfreq and kgsl nodes report Hz; cpufreq reports kHz.
struct Freq {
    bool        read = false;
    double      mhz = 0.0;
    std::string node;            // where it came from
    std::string unit_guess;      // declared unit ("Hz"/"kHz") or a guess ("... (guessed)", "MHz (assumed)")
};

// `declared` is the unit the kernel ABI promises for that node (cpufreq: kHz, devfreq and kgsl: Hz). A value that
// makes no sense in the declared unit falls back to magnitude detection, and unit_guess says which was used.
enum class Unit { AUTO, HZ, KHZ };

inline Freq parse_freq(const std::string& raw_text, const std::string& node, Unit declared = Unit::AUTO) {
    Freq f; double v = 0;
    if (!parse_double(trim(raw_text), v) || v <= 0) return f;
    f.read = true; f.node = node;
    if (declared == Unit::HZ  && v >= 1e6) { f.mhz = v / 1e6; f.unit_guess = "Hz";  return f; }
    if (declared == Unit::KHZ && v >= 1e3) { f.mhz = v / 1e3; f.unit_guess = "kHz"; return f; }
    // undeclared, or implausible in the declared unit: guess from magnitude (ambiguous between 1e3 and 1e6)
    if (v >= 1e6)      { f.mhz = v / 1e6; f.unit_guess = "Hz (guessed)"; }
    else if (v >= 1e3) { f.mhz = v / 1e3; f.unit_guess = "kHz (guessed)"; }
    else               { f.mhz = v;       f.unit_guess = "MHz (assumed)"; }
    return f;
}

// "key=value" lines of build.prop; comments and blank lines skipped.
inline std::map<std::string, std::string> parse_build_prop(const std::string& text) {
    std::map<std::string, std::string> m;
    for (const std::string& l0 : split_lines(text)) {
        const std::string l = trim(l0);
        if (l.empty() || l[0] == '#') continue;
        const size_t eq = l.find('=');
        if (eq == std::string::npos || eq == 0) continue;
        m[trim(std::string_view(l).substr(0, eq))] = trim(std::string_view(l).substr(eq + 1));
    }
    return m;
}

using PropFn = std::function<std::string(const std::string&)>;   // "" = not set

// A real device (root == "") reads the live property service. A fake or non-Android root reads the build.prop files
// found under it, vendor and odm last so they override like the real property merge does.
inline PropFn make_prop_reader(const std::string& root) {
#if defined(__ANDROID__)
    if (root.empty()) {
        return [](const std::string& key) {
            char v[PROP_VALUE_MAX] = {0};
            __system_property_get(key.c_str(), v);
            return std::string(v);
        };
    }
#endif
    auto props = std::make_shared<std::map<std::string, std::string>>();
    for (const char* p : {"/system/build.prop", "/product/build.prop", "/odm/etc/build.prop", "/vendor/build.prop"})
        for (const auto& kv : parse_build_prop(read_file(root + p))) (*props)[kv.first] = kv.second;
    return [props](const std::string& key) {
        auto it = props->find(key);
        return it == props->end() ? std::string() : it->second;
    };
}

// ─── Result model ────────────────────────────────────────────────────────────

struct CoreFreq {
    int         id = 0;
    bool        online = true;
    bool        online_known = false;     // cpu0 has no "online" node: assumed online
    Freq        cur;                      // scaling_cur_freq
    Freq        max;                      // cpuinfo_max_freq
    std::string governor;
};

struct MemoryExtra {
    uint64_t swap_total_kb = 0, swap_free_kb = 0, buffers_kb = 0, cached_kb = 0, slab_kb = 0;
    uint64_t zram_disk_bytes = 0;         // /sys/block/zram0/disksize
    Freq     ddr;                         // memory bus clock, if a devfreq node exposes it
    bool     meminfo_read = false;
};

struct DataFs {
    bool     read = false;
    uint64_t total_bytes = 0, avail_bytes = 0;
};

struct StorageLink {
    std::string interface_kind = "unknown";   // "UFS" | "eMMC" | "NVMe" | "SCSI/USB (not distinguishable)" | "unknown"
    std::string evidence;                     // the node that said so
};

struct GpuInfo {
    std::string model;                        // kgsl gpu_model ("Adreno660"), else empty
    std::string model_source;
    Freq        cur, max;
    int         busy_pct = -1;                // kgsl gpubusy is not parsed: reserved for a device-verified reader
    std::vector<std::string> kernel_nodes;    // /dev/kgsl-3d0, /dev/mali0, ... : "a GPU driver node exists"
};

struct DspNodes {
    std::vector<std::string> present;         // /dev/adsprpc-smd ... : a DSP RPC channel exists. Not a measured NPU.
};

struct SocProps {
    std::string hardware, board, platform, model, manufacturer, soc_model, soc_manufacturer;
};

struct OsProps {
    std::string android_release, sdk, security_patch, kernel;
    std::string selinux = "unknown";          // "enforcing" | "permissive" | "unknown"
};

struct AndroidSystem {
    bool                      android = false;   // is_android(root)
    MemoryExtra               memory;
    DataFs                    data_fs;
    StorageLink               storage_link;
    GpuInfo                   gpu;
    DspNodes                  dsp;
    int                       configured_cpus = 0;
    int                       online_cpus = 0;
    std::vector<CoreFreq>     cores;
    SocProps                  soc;
    OsProps                   os;
    std::vector<ThermalZone>  thermal;            // reuses DeviceFingerprint's reader (sysfs zones, no `cmd thermalservice`)
    BatteryInfo               battery;
    std::vector<std::string>  notes;
};

// ─── Readers (each takes the root prefix) ────────────────────────────────────

inline bool contains_any(const std::string& lower, std::initializer_list<const char*> keys) {
    for (const char* k : keys) if (lower.find(k) != std::string::npos) return true;
    return false;
}

// First readable clock among explicit nodes, then devfreq entries whose directory name matches `keys`.
inline Freq read_devfreq(const std::string& root, const std::vector<std::string>& explicit_nodes,
                         std::initializer_list<const char*> keys) {
    for (const std::string& rel : explicit_nodes) {
        Freq f = parse_freq(read_trim(root + rel), rel, Unit::HZ);
        if (f.read) return f;
    }
    const std::string base = "/sys/class/devfreq/";
    for (const std::string& n : list_dir(root + base)) {
        if (!contains_any(to_lower(n), keys)) continue;
        Freq f = parse_freq(read_trim(root + base + n + "/cur_freq"), base + n + "/cur_freq", Unit::HZ);
        if (f.read) return f;
    }
    return {};
}

inline void read_memory_extra(const std::string& root, AndroidSystem& s) {
    const std::string mi = read_file(root + "/proc/meminfo");
    s.memory.meminfo_read = !mi.empty();
    meminfo_kb(mi, "SwapTotal", s.memory.swap_total_kb);
    meminfo_kb(mi, "SwapFree",  s.memory.swap_free_kb);
    meminfo_kb(mi, "Buffers",   s.memory.buffers_kb);
    meminfo_kb(mi, "Cached",    s.memory.cached_kb);
    meminfo_kb(mi, "Slab",      s.memory.slab_kb);
    uint64_t z = 0;
    if (parse_u64(read_trim(root + "/sys/block/zram0/disksize"), z)) s.memory.zram_disk_bytes = z;
    s.memory.ddr = read_devfreq(root, {"/sys/class/devfreq/ddrfreq/cur_freq"}, {"ddr", "dram", "dmc", "mif"});
    if (!s.memory.meminfo_read) s.notes.push_back("/proc/meminfo unreadable: swap and cache figures unknown");
    if (!s.memory.ddr.read)     s.notes.push_back("no readable DDR clock node under /sys/class/devfreq (restricted or vendor-specific)");
}

inline void read_data_fs(const std::string& root, AndroidSystem& s) {
#if defined(__linux__)
    struct statvfs st;
    if (statvfs((root + "/data").c_str(), &st) == 0) {
        s.data_fs.read = true;
        s.data_fs.total_bytes = (uint64_t)st.f_blocks * (uint64_t)st.f_frsize;
        s.data_fs.avail_bytes = (uint64_t)st.f_bavail * (uint64_t)st.f_frsize;   // 64-bit before the multiply: no overflow on 32-bit ABIs
        return;
    }
#endif
    s.notes.push_back("/data not readable (statvfs failed): storage capacity unknown");
}

// UFS is recognised by its SCSI host driver, not by the block name: "sda" is also USB sticks and SD readers.
inline StorageLink read_storage_link(const std::string& root) {
    StorageLink l;
    for (const std::string& h : list_dir(root + "/sys/class/scsi_host")) {
        const std::string p = "/sys/class/scsi_host/" + h + "/proc_name";
        if (to_lower(read_trim(root + p)).find("ufs") != std::string::npos) { l.interface_kind = "UFS"; l.evidence = p; return l; }
    }
    for (const std::string& n : list_dir(root + "/sys/block")) {
        if (starts_with(n, "mmcblk") && !contains_any(n, {"boot", "rpmb"})) { l.interface_kind = "eMMC"; l.evidence = "/sys/block/" + n; return l; }
        if (starts_with(n, "nvme"))  { l.interface_kind = "NVMe"; l.evidence = "/sys/block/" + n; return l; }
    }
    for (const std::string& n : list_dir(root + "/sys/block"))
        if (starts_with(n, "sd")) { l.interface_kind = "SCSI/USB (not distinguishable)"; l.evidence = "/sys/block/" + n; return l; }
    return l;
}

inline void read_gpu(const std::string& root, const PropFn& prop, AndroidSystem& s) {
    GpuInfo& g = s.gpu;
    for (const char* rel : {"/sys/class/kgsl/kgsl-3d0/gpu_model"}) {
        const std::string m = read_trim(root + rel);
        if (!m.empty()) { g.model = m; g.model_source = rel; break; }
    }
    if (g.model.empty()) {
        const std::string egl = prop("ro.hardware.egl");
        if (!egl.empty()) { g.model = egl; g.model_source = "property ro.hardware.egl"; }   // a driver name ("adreno", "mali"), not a model
    }
    g.cur = read_devfreq(root, {"/sys/class/kgsl/kgsl-3d0/gpuclk", "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq",
                                "/sys/class/devfreq/gpufreq/cur_freq"}, {"gpu", "kgsl", "mali", "g3d"});
    g.max = parse_freq(read_trim(root + "/sys/class/kgsl/kgsl-3d0/max_gpuclk"), "/sys/class/kgsl/kgsl-3d0/max_gpuclk", Unit::HZ);
    for (const char* dev : {"/dev/kgsl-3d0", "/dev/mali0", "/dev/mali"}) if (path_exists(root + dev)) g.kernel_nodes.push_back(dev);
    if (!g.cur.read) s.notes.push_back("no readable GPU clock node (SELinux or vendor-specific path)");
}

inline void read_dsp(const std::string& root, AndroidSystem& s) {
    for (const char* dev : {"/dev/adsprpc-smd", "/dev/adsprpc-smd-secure", "/dev/cdsprpc-smd", "/dev/cdsprpc-smd-secure", "/dev/sdsprpc-smd"})
        if (path_exists(root + dev)) s.dsp.present.push_back(dev);
}

inline void read_cores(const std::string& root, AndroidSystem& s) {
    const std::string base = root + "/sys/devices/system/cpu/";
    std::vector<int> ids;
    for (const std::string& n : list_dir(base)) {
        if (n.size() > 3 && n.compare(0, 3, "cpu") == 0) {
            uint64_t v = 0;
            if (parse_u64(n.substr(3), v)) ids.push_back((int)v);        // "cpufreq", "cpuidle" do not parse and are skipped
        }
    }
    std::sort(ids.begin(), ids.end());
    s.configured_cpus = (int)ids.size();
    for (int id : ids) {
        CoreFreq c; c.id = id;
        const std::string d = base + "cpu" + std::to_string(id);
        const std::string on = read_trim(d + "/online");
        if (!on.empty()) { c.online_known = true; c.online = on != "0"; }
        c.cur = parse_freq(read_trim(d + "/cpufreq/scaling_cur_freq"), "cpu" + std::to_string(id) + "/cpufreq/scaling_cur_freq", Unit::KHZ);
        c.max = parse_freq(read_trim(d + "/cpufreq/cpuinfo_max_freq"), "cpu" + std::to_string(id) + "/cpufreq/cpuinfo_max_freq", Unit::KHZ);
        c.governor = read_trim(d + "/cpufreq/scaling_governor");
        if (c.online) ++s.online_cpus;
        s.cores.push_back(std::move(c));
    }
    if (s.cores.empty()) s.notes.push_back("no cpuN directories under /sys/devices/system/cpu: per-core frequencies unknown");
    else {
        int unread = 0; for (const auto& c : s.cores) if (c.online && !c.cur.read) ++unread;
        if (unread) s.notes.push_back(std::to_string(unread) + " online core(s) have no readable scaling_cur_freq (not the same as offline)");
    }
}

inline void read_props(const PropFn& prop, AndroidSystem& s) {
    s.soc.hardware = prop("ro.hardware");           s.soc.board = prop("ro.product.board");
    s.soc.platform = prop("ro.board.platform");     s.soc.model = prop("ro.product.model");
    s.soc.manufacturer = prop("ro.product.manufacturer");
    s.soc.soc_model = prop("ro.soc.model");         s.soc.soc_manufacturer = prop("ro.soc.manufacturer");   // Android 12+
    s.os.android_release = prop("ro.build.version.release");
    s.os.sdk = prop("ro.build.version.sdk");
    s.os.security_patch = prop("ro.build.version.security_patch");
    if (s.soc.hardware.empty() && s.soc.platform.empty() && s.os.sdk.empty())
        s.notes.push_back("no Android system properties readable: SoC and OS identity unknown");
}

inline void read_os(const std::string& root, AndroidSystem& s) {
    s.os.kernel = read_trim(root + "/proc/sys/kernel/osrelease");        // uname -r
    const std::string se = read_trim(root + "/sys/fs/selinux/enforce");  // getenforce
    if (se == "1") s.os.selinux = "enforcing"; else if (se == "0") s.os.selinux = "permissive";
    else s.notes.push_back("/sys/fs/selinux/enforce not readable: SELinux state unknown");
}

// Everything above in one call. `props` defaults to make_prop_reader(root).
inline AndroidSystem collect_android(const std::string& root = "", PropFn props = nullptr) {
    AndroidSystem s;
    if (!props) props = make_prop_reader(root);
    s.android = linux_adapter::is_android(root);
    read_memory_extra(root, s);
    read_data_fs(root, s);
    s.storage_link = read_storage_link(root);
    read_gpu(root, props, s);
    read_dsp(root, s);
    read_cores(root, s);
    read_props(props, s);
    read_os(root, s);
    s.thermal = linux_adapter::read_thermal(root);
    s.battery = linux_adapter::read_battery(root);
    if (s.thermal.empty()) s.notes.push_back("no readable thermal zones (restricted by the OEM security profile)");
    if (!s.android) s.notes.push_back("root does not look like Android: values come from whatever Linux nodes exist");
    return s;
}

// ─── Output ──────────────────────────────────────────────────────────────────

inline void write_freq(Json& j, const char* key, const Freq& f) {
    j.begin_object(key);
    j.boolean("read", f.read);
    if (f.read) { j.num("mhz", f.mhz); j.str("node", f.node); j.str("unit_guess", f.unit_guess); }
    j.end_object();
}

// Writes the members only; the caller opens and closes the object (same convention as TensorAccounting::to_json).
inline void to_json(Json& j, const AndroidSystem& s) {
    j.boolean("android", s.android);
    j.begin_object("memory");
    j.uinteger("swap_total_kb", s.memory.swap_total_kb); j.uinteger("swap_free_kb", s.memory.swap_free_kb);
    j.uinteger("buffers_kb", s.memory.buffers_kb); j.uinteger("cached_kb", s.memory.cached_kb); j.uinteger("slab_kb", s.memory.slab_kb);
    j.uinteger("zram_disk_bytes", s.memory.zram_disk_bytes);
    write_freq(j, "ddr_clock", s.memory.ddr);
    j.end_object();
    j.begin_object("data_fs"); j.boolean("read", s.data_fs.read);
    if (s.data_fs.read) { j.uinteger("total_bytes", s.data_fs.total_bytes); j.uinteger("avail_bytes", s.data_fs.avail_bytes); }
    j.end_object();
    j.begin_object("storage_link"); j.str("interface", s.storage_link.interface_kind); j.str("evidence", s.storage_link.evidence); j.end_object();
    j.begin_object("gpu"); j.str("model", s.gpu.model); j.str("model_source", s.gpu.model_source);
    write_freq(j, "clock", s.gpu.cur); write_freq(j, "max_clock", s.gpu.max);
    j.begin_array("kernel_nodes"); for (const auto& n : s.gpu.kernel_nodes) j.str(nullptr, n); j.end_array(); j.end_object();
    j.begin_array("dsp_rpc_nodes"); for (const auto& n : s.dsp.present) j.str(nullptr, n); j.end_array();
    j.integer("configured_cpus", s.configured_cpus); j.integer("online_cpus", s.online_cpus);
    j.begin_array("cores");
    for (const auto& c : s.cores) {
        j.begin_object(); j.integer("id", c.id); j.boolean("online", c.online); j.boolean("online_known", c.online_known);
        write_freq(j, "cur", c.cur); write_freq(j, "max", c.max); j.str("governor", c.governor); j.end_object();
    }
    j.end_array();
    j.begin_object("soc"); j.str("hardware", s.soc.hardware); j.str("board", s.soc.board); j.str("platform", s.soc.platform);
    j.str("model", s.soc.model); j.str("manufacturer", s.soc.manufacturer);
    j.str("soc_model", s.soc.soc_model); j.str("soc_manufacturer", s.soc.soc_manufacturer); j.end_object();
    j.begin_object("os"); j.str("android_release", s.os.android_release); j.str("sdk", s.os.sdk);
    j.str("security_patch", s.os.security_patch); j.str("kernel", s.os.kernel); j.str("selinux", s.os.selinux); j.end_object();
    j.begin_array("thermal");
    for (const auto& z : s.thermal) { j.begin_object(); j.str("zone", z.name); j.num("temp_c", z.temp_c); j.end_object(); }
    j.end_array();
    j.begin_array("notes"); for (const auto& n : s.notes) j.str(nullptr, n); j.end_array();
}

inline std::string to_json(const AndroidSystem& s) {
    Json j; j.begin_object(); to_json(j, s); j.end_object(); return j.finish();
}

// The nine sections of get_system_info.cpp, in the same order, with "not read" instead of guesses.
inline void print_android(const AndroidSystem& s, std::ostream& os) {
    auto mhz = [](const Freq& f) { char b[64]; if (!f.read) return std::string("not read"); std::snprintf(b, sizeof b, "%.0f MHz", f.mhz); return std::string(b); };
    auto gb  = [](double bytes) { char b[32]; std::snprintf(b, sizeof b, "%.2f GB", bytes / 1073741824.0); return std::string(b); };
    os << "=== RAM & MEMORY METRICS ===\n";
    os << "LPDDR clock:            " << mhz(s.memory.ddr) << "\n";
    os << "\n=== EXTENDED MEMORY & ZRAM METRICS ===\n";
    os << "  SwapTotal: " << s.memory.swap_total_kb << " kB   SwapFree: " << s.memory.swap_free_kb << " kB\n"
       << "  Buffers: " << s.memory.buffers_kb << " kB   Cached: " << s.memory.cached_kb << " kB   Slab: " << s.memory.slab_kb << " kB\n"
       << "  zram disk: " << gb((double)s.memory.zram_disk_bytes) << "\n";
    os << "\n=== STORAGE & DATA BUS METRICS ===\n";
    if (s.data_fs.read) os << "Total (/data):          " << gb((double)s.data_fs.total_bytes) << "\nAvailable:              " << gb((double)s.data_fs.avail_bytes) << "\n";
    else os << "/data:                  not read\n";
    os << "Storage interface:      " << s.storage_link.interface_kind << (s.storage_link.evidence.empty() ? "" : "  (" + s.storage_link.evidence + ")") << "\n";
    os << "\n=== GPU METRICS & FREQUENCIES ===\n";
    os << "GPU:                    " << (s.gpu.model.empty() ? "not read" : s.gpu.model) << "\n";
    os << "Current clock:          " << mhz(s.gpu.cur) << "\n";
    os << "\n=== DSP / AI ACCELERATOR NODES ===\n";
    if (s.dsp.present.empty()) os << "No DSP RPC device node visible.\n";
    for (const auto& n : s.dsp.present) os << "  " << n << "  (a DSP channel exists; no NPU throughput is measured)\n";
    os << "\n=== CPU CORE METRICS & CLOCK FREQUENCIES ===\n";
    os << "Configured / online:    " << s.configured_cpus << " / " << s.online_cpus << "\n";
    for (const auto& c : s.cores) {
        os << "  Core [" << c.id << "] " << (c.online ? "online " : "offline") << "  cur " << mhz(c.cur) << "  max " << mhz(c.max)
           << (c.governor.empty() ? "" : "  governor " + c.governor) << "\n";
    }
    os << "\n=== SOC & BOARD HARDWARE INFO ===\n";
    os << "Hardware: " << s.soc.hardware << "\nBoard: " << s.soc.board << "\nPlatform: " << s.soc.platform
       << "\nModel: " << s.soc.model << "\nManufacturer: " << s.soc.manufacturer << "\n";
    if (!s.soc.soc_model.empty()) os << "SoC: " << s.soc.soc_manufacturer << " " << s.soc.soc_model << "\n";
    os << "\n=== OS & SECURITY RUNTIME METRICS ===\n";
    os << "Android: " << s.os.android_release << "  SDK " << s.os.sdk << "  patch " << s.os.security_patch
       << "\nSELinux: " << s.os.selinux << "\nKernel: " << s.os.kernel << "\n";
    os << "\n=== THERMAL SENSOR METRICS ===\n";
    if (s.thermal.empty()) os << "No thermal zone readable.\n";
    for (const auto& z : s.thermal) { char b[64]; std::snprintf(b, sizeof b, "  %-24s %.1f C\n", z.name.c_str(), z.temp_c); os << b; }
    if (!s.notes.empty()) { os << "\nNot determined:\n"; for (const auto& n : s.notes) os << "  - " << n << "\n"; }
}

}  // namespace tensorf::prof::android
