/*
 * profiler_suite_nohook_test.cpp
 * ==============================
 * ProfilerSuite in a program where the allocation hook is NOT linked (no TENSORF_MEMTRACK_IMPLEMENT).
 * The memory stage must be skipped WITH the reason, the default overhead factor kept, and the advisor must say
 * that the factor was not measured. Everything else still works.
 *
 *   g++ -std=c++20 -O2 -Wall -Wextra -Iinclude -Iinclude/core -Iinclude/tools \
 *       tests/profiler_suite_nohook_test.cpp -o bin/profiler_suite_nohook_test -pthread
 */
#include <cstdio>
#include "profiler/tmp/ProfilerSuite.hpp"

using namespace tensorf::prof;
namespace A = tensorf::prof::adv;
namespace wl = tensorf::prof::wl;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failed; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while (0)

int main() {
    std::printf("TensorF profiler suite: no allocation hook\n");
    ProfileOptions o; o.run_probe = false; o.capability_budget_s = 14; o.total_budget_s = 25;
    Profile p = ProfilerSuite().run(o);

    CHECK(!mem::hook_active());
    CHECK(p.stage("memory") && p.stage("memory")->status == "skipped");
    CHECK(p.stage("memory")->note.find("hook not linked") != std::string::npos);
    CHECK(!p.have_memory && !p.mem_factor_measured && p.mem_overhead_factor == 1.25);
    CHECK(p.stage("capability") && p.stage("capability")->status == "ok" && p.have_capability);

    wl::Algorithm alg; alg.name = "tiny";
    alg.tunables = {{"micro_batch", {16, 32, 64}, true, "step"}};
    alg.build = [](const wl::Config& c) {
        wl::RunSpec r; r.mode = wl::Mode::TRAIN; r.opt_states = 2;
        wl::Workload w; w.units_per_micro_step = c.at("micro_batch");
        w.layers = {wl::dense(c.at("micro_batch"), 256, 256, r, "fc")};
        return w;
    };
    A::Advice a = ProfilerSuite().advise(p, alg, {{"micro_batch", 32}});
    CHECK(a.ok);
    bool said = false;
    for (const auto& n : a.notes) if (n.find("not measured on this device") != std::string::npos) said = true;
    CHECK(said);
    CHECK(a.prediction.memory.overhead_factor == 1.25);

    const std::string j = ProfilerSuite::to_json(p, &a);
    CHECK(j.find("\"memory_overhead_measured\":false") != std::string::npos);
    CHECK(j.find("\"memory_measurement\":{") == std::string::npos);            // nothing is invented for a stage that did not run

    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
