#pragma once

/*
 * ProfilerCommon.hpp
 * ==================
 * Shared plumbing for the rewritten profiler suite (namespace tensorf::prof).
 *
 *   - clocks (wall + per-thread CPU time), deadlines / time budgets
 *   - robust statistics (median, percentile, trimmed mean)
 *   - optimiser barriers so benchmark kernels are not deleted
 *   - safe file + text parsing helpers (NO popen, NO shell-outs, no sscanf %lu)
 *   - a tiny JSON writer (NaN/inf become null, so the payload is always valid)
 *
 * Everything here is portable C++20 plus POSIX clock_gettime. Nothing reads
 * /proc or /sys; that belongs to the platform adapters in DeviceFingerprint.hpp.
 */

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <time.h>
#endif

namespace tensorf::prof {

// ─── Clocks ──────────────────────────────────────────────────────────────────

using SteadyClock = std::chrono::steady_clock;

inline double now_s() {
    return std::chrono::duration<double>(SteadyClock::now().time_since_epoch()).count();
}

// CPU time consumed by the calling thread, in seconds. Used to tell "we were
// slow" from "the kernel preempted us" (wall >> cpu means contention).
// Returns a negative value when the platform has no per-thread clock.
inline double thread_cpu_s() {
#if defined(CLOCK_THREAD_CPUTIME_ID)
    timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0)
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
    return -1.0;
}

class Stopwatch {
public:
    Stopwatch() : t0_(now_s()) {}
    void   reset()         { t0_ = now_s(); }
    double elapsed() const { return now_s() - t0_; }
private:
    double t0_;
};

// A wall-clock budget. Every long-running piece of the suite checks it, which is
// what makes the suite safe on a phone: it never runs past what it was granted.
class Deadline {
public:
    explicit Deadline(double seconds = std::numeric_limits<double>::infinity())
        : start_(now_s()), limit_(seconds) {}
    bool   expired()   const { return elapsed() >= limit_; }
    double elapsed()   const { return now_s() - start_; }
    double remaining() const { return std::max(0.0, limit_ - elapsed()); }
    double limit()     const { return limit_; }
    // A child budget: at most `fraction` of what is left, never more than `cap`.
    Deadline slice(double fraction, double cap = std::numeric_limits<double>::infinity()) const {
        return Deadline(std::min(cap, remaining() * fraction));
    }
private:
    double start_, limit_;
};

// ─── Optimiser barriers ──────────────────────────────────────────────────────

inline void clobber_memory() {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : : "memory");
#endif
}

template <class T>
inline void do_not_optimize(const T& v) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r,m"(v) : "memory");
#else
    volatile const T* sink = &v; (void)sink;
#endif
}

// ─── Statistics ──────────────────────────────────────────────────────────────

inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// p in [0,1], linear interpolation.
inline double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    p = std::clamp(p, 0.0, 1.0);
    const double pos = p * (double)(v.size() - 1);
    const size_t lo = (size_t)pos, hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - (double)lo);
}

inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0; for (double x : v) s += x;
    return s / (double)v.size();
}

inline double min_of(const std::vector<double>& v) {
    return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
}
inline double max_of(const std::vector<double>& v) {
    return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end());
}

// Relative spread (p90 - p10) / median: a cheap "how noisy was this timing".
inline double rel_spread(const std::vector<double>& v) {
    const double m = median(v);
    if (m <= 0.0) return 0.0;
    return (percentile(v, 0.9) - percentile(v, 0.1)) / m;
}

// Run `fn` (returning seconds) until `reps` samples are collected, the deadline
// expires, or at least `min_reps` exist. Returns the samples.
template <class F>
inline std::vector<double> sample_seconds(F&& fn, int reps, const Deadline& dl, int min_reps = 3) {
    std::vector<double> out;
    out.reserve((size_t)reps);
    for (int i = 0; i < reps; ++i) {
        out.push_back(fn());
        if ((int)out.size() >= min_reps && dl.expired()) break;
    }
    return out;
}

// ─── Text / file helpers (no shell-outs) ─────────────────────────────────────

inline std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline std::string trim(std::string_view s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' || s[b - 1] == '\r')) --b;
    return std::string(s.substr(a, b - a));
}

inline std::string read_trim(const std::string& path) { return trim(read_file(path)); }

inline std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find('\n', i);
        if (j == std::string::npos) j = s.size();
        out.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return out;
}

inline std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(sep, i);
        if (j == std::string::npos) j = s.size();
        out.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return out;
}

inline bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

inline std::string to_lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// Parse an unsigned integer from the start of `s` (after blanks). Always 64-bit,
// independent of what `long` is, which is what broke sscanf("%lu") on armv7.
inline bool parse_u64(std::string_view s, uint64_t& out, int base = 10) {
    std::string t = trim(s);
    if (t.empty() || t[0] == '-') return false;
    char* end = nullptr;
    errno = 0;
    unsigned long long v = std::strtoull(t.c_str(), &end, base);
    if (end == t.c_str() || errno == ERANGE) return false;
    out = (uint64_t)v;
    return true;
}

inline bool parse_double(std::string_view s, double& out) {
    std::string t = trim(s);
    if (t.empty()) return false;
    char* end = nullptr;
    double v = std::strtod(t.c_str(), &end);
    if (end == t.c_str()) return false;
    out = v;
    return true;
}

// "32K", "32 KB", "1M", "8192K", "512", "2G" -> bytes. Sysfs cache sizes use K/M.
inline bool parse_size_bytes(std::string_view s, uint64_t& out) {
    std::string t = trim(s);
    if (t.empty()) return false;
    char* end = nullptr;
    unsigned long long v = std::strtoull(t.c_str(), &end, 10);
    if (end == t.c_str()) return false;
    uint64_t mult = 1;
    while (*end == ' ') ++end;
    switch (std::toupper((unsigned char)*end)) {
        case 'K': mult = 1ull << 10; break;
        case 'M': mult = 1ull << 20; break;
        case 'G': mult = 1ull << 30; break;
        default: break;
    }
    out = (uint64_t)v * mult;
    return true;
}

// Linux cpu-list syntax: "0-3,6,8-9" -> {0,1,2,3,6,8,9}. Used by sysfs
// (shared_cpu_list, online, possible) so we never need lscpu.
inline std::vector<int> parse_cpu_list(const std::string& s) {
    std::vector<int> out;
    for (const std::string& part : split(trim(s), ',')) {
        if (part.empty()) continue;
        size_t dash = part.find('-');
        uint64_t a = 0, b = 0;
        if (dash == std::string::npos) {
            if (parse_u64(part, a)) out.push_back((int)a);
        } else if (parse_u64(part.substr(0, dash), a) && parse_u64(part.substr(dash + 1), b) && b >= a && b - a < 65536) {
            for (uint64_t i = a; i <= b; ++i) out.push_back((int)i);
        }
    }
    return out;
}

// "key : value" lookup in /proc/cpuinfo-style text. `nth` selects the nth match
// (0-based); returns empty when absent.
inline std::string cpuinfo_field(const std::string& text, std::string_view key, int nth = 0) {
    int seen = 0;
    for (const std::string& line : split_lines(text)) {
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        if (trim(std::string_view(line).substr(0, colon)) != key) continue;
        if (seen++ == nth) return trim(std::string_view(line).substr(colon + 1));
    }
    return {};
}

// "Key:   123 kB" lookup in /proc/meminfo-style text -> value in kB.
inline bool meminfo_kb(const std::string& text, std::string_view key, uint64_t& out) {
    for (const std::string& line : split_lines(text)) {
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        if (trim(std::string_view(line).substr(0, colon)) != key) continue;
        return parse_u64(std::string_view(line).substr(colon + 1), out);
    }
    return false;
}

// Whole-word membership in a space separated token list (cpuinfo flags/Features).
inline bool has_token(const std::string& list, std::string_view token) {
    size_t pos = 0;
    while ((pos = list.find(token, pos)) != std::string::npos) {
        const bool left  = pos == 0 || list[pos - 1] == ' ';
        const size_t e   = pos + token.size();
        const bool right = e == list.size() || list[e] == ' ' || list[e] == '\n';
        if (left && right) return true;
        pos = e;
    }
    return false;
}

// ─── Minimal JSON writer ─────────────────────────────────────────────────────
// Streaming, comma-managing. Non-finite doubles are written as null.

class Json {
public:
    Json() { stack_.push_back(true); }

    Json& begin_object(const char* key = nullptr) { sep(key); os_ << '{'; stack_.push_back(true); return *this; }
    Json& end_object()                            { stack_.pop_back(); os_ << '}'; return *this; }
    Json& begin_array(const char* key = nullptr)  { sep(key); os_ << '['; stack_.push_back(true); return *this; }
    Json& end_array()                             { stack_.pop_back(); os_ << ']'; return *this; }

    Json& num(const char* key, double v) {
        sep(key);
        if (std::isfinite(v)) { char b[40]; std::snprintf(b, sizeof b, "%.6g", v); os_ << b; }
        else os_ << "null";
        return *this;
    }
    Json& num(double v) { return num(nullptr, v); }
    Json& integer(const char* key, int64_t v)  { sep(key); os_ << v; return *this; }
    Json& uinteger(const char* key, uint64_t v){ sep(key); os_ << v; return *this; }
    Json& boolean(const char* key, bool v)     { sep(key); os_ << (v ? "true" : "false"); return *this; }
    Json& str(const char* key, const std::string& v) { sep(key); quote(v); return *this; }

    std::string finish() const { return os_.str(); }

    static std::string escape(const std::string& s) {
        std::string o;
        for (unsigned char c : s) {
            switch (c) {
                case '"':  o += "\\\""; break;
                case '\\': o += "\\\\"; break;
                case '\n': o += "\\n";  break;
                case '\r': o += "\\r";  break;
                case '\t': o += "\\t";  break;
                default:
                    if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                    else o += (char)c;
            }
        }
        return o;
    }

private:
    std::ostringstream os_;
    std::vector<bool>  stack_;   // true = next element is the first in its container

    void sep(const char* key) {
        if (!stack_.back()) os_ << ',';
        stack_.back() = false;
        if (key) { quote(key); os_ << ':'; }
    }
    void quote(const std::string& s) { os_ << '"' << escape(s) << '"'; }
};

}  // namespace tensorf::prof
