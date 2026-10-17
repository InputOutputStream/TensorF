/*
 * profiler_suite_test.cpp
 * =======================
 * Tests for ProfilerSuite.hpp: the orchestrator. Defines TENSORF_MEMTRACK_IMPLEMENT (the allocation hook is
 * linked), so the memory stage can run. tests/profiler_suite_nohook_test.cpp covers the program without it.
 *
 *   g++ -std=c++20 -O2 -Wall -Wextra -Iinclude -Iinclude/core -Iinclude/tools \
 *       tests/profiler_suite_test.cpp -o bin/profiler_suite_test -pthread
 *   ./bin/profiler_suite_test            # ~35 s: one full live profile
 *   ./bin/profiler_suite_test --quick    # ~3 s: fixtures and budgets only
 */
#define TENSORF_MEMTRACK_IMPLEMENT
#include "profiler/tmp/ProfilerSuite.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

using namespace tensorf::prof;
namespace fs = std::filesystem;
namespace A = tensorf::prof::adv;
namespace wl = tensorf::prof::wl;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failed; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while (0)
#define SECTION(name) std::printf("[%s]\n", name)

class FakeRoot {
public:
    FakeRoot() { path_ = (fs::temp_directory_path() / ("tensorf_suite_" + std::to_string((long long)(now_s() * 1e6)))).string(); fs::create_directories(path_); }
    ~FakeRoot() { std::error_code ec; fs::remove_all(path_, ec); }
    const std::string& root() const { return path_; }
    void thermal(const char* type, long millideg) const {
        const fs::path d = fs::path(path_) / "sys/class/thermal/thermal_zone0"; fs::create_directories(d);
        std::ofstream(d / "type") << type << "\n"; std::ofstream(d / "temp") << millideg << "\n";
    }
private:
    std::string path_;
};

static bool balanced_clean(const std::string& s) {
    int brace = 0, brack = 0; bool in_str = false, esc = false;
    for (char ch : s) {
        if (in_str) { if (esc) esc = false; else if (ch == '\\') esc = true; else if (ch == '"') in_str = false; continue; }
        if (ch == '"') in_str = true;
        else if (ch == '{') ++brace; else if (ch == '}') --brace;
        else if (ch == '[') ++brack; else if (ch == ']') --brack;
        if (brace < 0 || brack < 0) return false;
    }
    const std::string low = to_lower(s);
    for (const char* bad : {":nan", ":-nan", ":inf", ":-inf", ",nan", ",inf", "[nan", "[inf"}) if (low.find(bad) != std::string::npos) return false;
    return !in_str && brace == 0 && brack == 0;
}

static bool has_note(const A::Advice& a, const char* needle) { for (const auto& n : a.notes) if (n.find(needle) != std::string::npos) return true; return false; }

// A small MLP algorithm with tunables, as a user would declare one.
static wl::Algorithm make_alg() {
    wl::Algorithm a; a.name = "mlp";
    a.tunables = {{"micro_batch", {8, 16, 32, 64}, true, "step"}, {"ckpt", {1, 2}, true, "step"}, {"batch", {64, 128}, false, "step"}};
    a.build = [](const wl::Config& c) {
        const double mb = c.at("micro_batch"), batch = c.at("batch");
        wl::RunSpec r; r.mode = wl::Mode::TRAIN; r.opt_states = 2;
        wl::Workload w; w.units_per_micro_step = mb; w.micro_steps = (int)std::max(1.0, std::round(batch / mb));
        w.layers = {wl::dense(mb, 784, 256, r, "fc1"), wl::elementwise(mb * 256, 1, 2, r, "relu"), wl::dense(mb, 256, 10, r, "fc2"),
                    wl::optimizer_update(784.0 * 256 + 256.0 * 10, r)};
        return (int)c.at("ckpt") > 1 ? wl::with_checkpointing(w, (int)c.at("ckpt")) : w;
    };
    return a;
}
static wl::Config base() { return {{"micro_batch", 32}, {"ckpt", 1}, {"batch", 64}}; }

static void test_hot_device() {
    SECTION("hot device: the profile stops cleanly and says why");
    FakeRoot fx; fx.thermal("cpu-0-0", 62000);
    ProfileOptions o; o.sysroot = fx.root(); o.total_budget_s = 30;
    Stopwatch sw;
    Profile p = ProfilerSuite().run(o);
    CHECK(sw.elapsed() < 5.0);                                                      // nothing heavy ran
    CHECK(p.stopped_reason.find("thermal") != std::string::npos);
    CHECK(!p.have_capability && !p.have_probe && !p.have_memory);
    CHECK(p.calibration() == nullptr);
    CHECK(p.stages.size() == 4);
    CHECK(p.stage("device") && p.stage("device")->status == "ok");
    CHECK(p.stage("capability") && p.stage("capability")->status == "stopped");     // a guard stop is not a failure
    CHECK(p.stage("validation") && p.stage("validation")->status == "skipped");
    CHECK(p.stage("memory") && p.stage("memory")->status == "skipped");
    CHECK(p.stage("memory")->note.find("too hot") != std::string::npos);            // even the light stage respects the thermal guard
    for (const auto& s : p.stages) if (s.status != "ok") CHECK(!s.note.empty());    // every non-ok stage explains itself

    A::Advice a = ProfilerSuite().advise(p, make_alg(), base());
    CHECK(!a.ok);
    CHECK(has_note(a, "no capability data"));

    const std::string j = ProfilerSuite::to_json(p, &a);
    CHECK(balanced_clean(j));
    CHECK(j.find("\"schema\":\"tensorf.profile/2\"") != std::string::npos);
    CHECK(j.find("\"stopped_reason\":\"thermal") != std::string::npos);
    CHECK(j.find("\"capability\":{") == std::string::npos);                         // no capability OBJECT is invented (the stage name still appears)
    ProfilerSuite::print_summary(p);                                                // must cope with a mostly empty profile
}

static void test_budgets() {
    SECTION("budgets: stages that cannot start are skipped with a reason");
    {
        ProfileOptions o; o.total_budget_s = 1.0;                                    // below min_stage_s for capability
        Stopwatch sw;
        Profile p = ProfilerSuite().run(o);
        CHECK(sw.elapsed() < 3.0);
        CHECK(p.stage("capability")->status == "skipped");
        CHECK(p.stage("capability")->note.find("time budget") != std::string::npos);
        CHECK(!p.stopped_reason.empty());
        CHECK(p.stage("validation")->status == "skipped" && p.stage("memory")->status == "skipped");
        CHECK(balanced_clean(ProfilerSuite::to_json(p)));
    }
    {
        ProfileOptions o; o.run_probe = false; o.measure_memory = false; o.capability_budget_s = 14; o.total_budget_s = 30;
        Profile p = ProfilerSuite().run(o);
        CHECK(p.stage("validation")->status == "skipped" && p.stage("validation")->note.find("disabled") != std::string::npos);
        CHECK(p.stage("memory")->status == "skipped" && p.stage("memory")->note.find("disabled") != std::string::npos);
        CHECK(p.have_capability && p.calibration() == nullptr);
        CHECK(!p.mem_factor_measured && p.mem_overhead_factor == 1.25);
        // advice without validation or measured memory says so, and still works
        A::Advice a = ProfilerSuite().advise(p, make_alg(), base());
        CHECK(a.ok);
        CHECK(has_note(a, "uncalibrated"));
        CHECK(has_note(a, "not measured on this device"));
        CHECK(a.prediction.memory.overhead_factor == 1.25);
    }
}

static void test_full_live(bool quick) {
    SECTION("a full live profile on this machine");
    if (quick) { std::printf("  (skipped: --quick)\n"); return; }
    ProfileOptions o; o.total_budget_s = 45; o.capability_budget_s = 14; o.probe_budget_s = 14; o.probe_size_scale = 0.5;
    Stopwatch sw;
    Profile p = ProfilerSuite().run(o);
    const double wall = sw.elapsed();
    ProfilerSuite::print_summary(p);

    CHECK(wall < o.total_budget_s + 3.0);                                           // one budget for everything
    CHECK(p.total_seconds <= wall + 0.01);
    CHECK(p.stages.size() == 4);
    const char* order[] = {"device", "memory", "capability", "validation"};   // memory runs before the heavy stages (heap pollution)
    for (size_t i = 0; i < 4 && i < p.stages.size(); ++i) CHECK(p.stages[i].name == order[i]);
    double sum = 0; for (const auto& s : p.stages) sum += s.seconds;
    CHECK(sum <= p.total_seconds + 0.05);                                           // stage times account for the total
    CHECK(p.stage("device")->status == "ok");
    for (const auto& s : p.stages) if (s.status != "ok") CHECK(!s.note.empty());

    if (!p.stopped_reason.empty()) { std::printf("  (profile stopped: %s; skipping value checks)\n", p.stopped_reason.c_str()); return; }
    CHECK(p.stage("capability")->status == "ok" && p.have_capability);
    CHECK(p.stage("validation")->status == "ok" && p.have_probe);
    CHECK(p.calibration() != nullptr);
    CHECK(p.probe.usable >= 12);
    CHECK(p.mem_overhead_factor >= 1.0 && p.mem_overhead_factor <= 4.0);
    // Run first, in a fresh heap, the memory stage can measure the factor (it could not when it ran last: freed pages stay resident).
    if (p.memory.hwm_reset) {
        CHECK(p.stage("memory")->status == "ok");
        CHECK(p.mem_factor_measured && p.have_memory && p.memory.factor_valid);
        CHECK(p.memory.rss_growth() > 0);
    } else std::printf("  (info) this machine cannot reset the kernel peak counter: factor stays at the default\n");

    // advice from the real profile
    A::Advice a = ProfilerSuite().advise(p, make_alg(), base());
    A::print_advice(a);
    CHECK(a.ok);
    CHECK(a.prediction.calibrated);                                                 // the validation stage's calibration was used
    CHECK(a.prediction.memory.overhead_factor == p.mem_overhead_factor);            // so was the measured overhead factor
    CHECK(a.prediction.memory_fits);
    CHECK(a.config.at("batch") == 64);                                              // result-changing tunable left alone
    CHECK(a.rel_error <= wl::Calibration::kUncalibratedError + 1e-12);
    // the range is never tighter than the device's own repeatability, and says so when it had to be widened
    const double floor = ProfilerSuite::noise_floor(p);
    CHECK(floor > 0.0);
    CHECK(a.rel_error >= floor - 1e-12);
    if (a.prediction.rel_error < floor) CHECK(has_note(a, "run-to-run noise measured on this device"));

    const std::string j = ProfilerSuite::to_json(p, &a);
    CHECK(balanced_clean(j));
    for (const char* key : {"\"schema\"", "\"stages\"", "\"device\"", "\"capability\"", "\"validation\"", "\"advice\"", "\"memory_overhead_factor\"", "\"timestamp\""})
        CHECK(j.find(key) != std::string::npos);
    CHECK(j.size() > 3000);
}

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    test_hot_device();
    test_budgets();
    test_full_live(quick);
    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
