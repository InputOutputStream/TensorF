/*
 * profiler_memory_nohook_test.cpp
 * ===============================
 * TensorAccounting.hpp in a program where the allocation hook is NOT linked (no TENSORF_MEMTRACK_IMPLEMENT).
 * Everything must still compile and run, and must say what is missing instead of returning silent zeros.
 *
 *   g++ -std=c++20 -O2 -Wall -Wextra -Iinclude -Iinclude/core -Iinclude/tools \
 *       tests/profiler_memory_nohook_test.cpp -o bin/profiler_memory_nohook_test -pthread
 */
#include <cstdio>
#include <vector>
#include "profiler/tmp/TensorAccounting.hpp"

using namespace tensorf::prof;
using namespace tensorf::prof::mem;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failed; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while (0)

int main() {
    std::printf("TensorF profiler step 3: no-hook behaviour\n");
    CHECK(!hook_active());

    // Scopes are harmless and counters stay at zero: nothing is counted, and nothing pretends to be.
    {
        MemScope s(Cat::PARAMS);
        std::vector<float> v(1u << 20);
        do_not_optimize(v[0]);
        const Snapshot sn = snapshot();
        CHECK(sn.allocs == 0 && sn.frees == 0 && sn.live_total == 0 && sn.peak_total == 0);
        CHECK(sn.live[Cat::PARAMS] == 0);
    }

    // measure_memory reports the missing hook instead of a made-up factor.
    MeasureOptions o; o.warmup = 0;
    MemoryMeasurement m = measure_memory([] { std::vector<char> v(32u << 20, 1); do_not_optimize(v[0]); }, o);
    CHECK(!m.hook_active);
    CHECK(!m.factor_valid);
    CHECK(m.tensor_growth() == 0);
    CHECK(m.recommended_factor(1.25) == 1.25);
    bool said = false;
    for (const auto& n : m.notes) if (n.find("hook not linked") != std::string::npos) said = true;
    CHECK(said);
    Json j; j.begin_object(); to_json(j, m); j.end_object();
    CHECK(j.finish().find("\"hook_active\":false") != std::string::npos);

    // The parts that do not need the hook keep working: process memory and the graph census.
    ProcessMem pm = read_process_mem();
    CHECK(pm.ok && pm.rss > 0);
    auto a = make_tensor<float>(Matrix<float>::randu(-1.f, 1.f, {16, 16}));
    auto loss = (a * a)->sum();
    CatBytes c = census<float>(loss);
    CHECK(c[Cat::PARAMS] == 16 * 16 * 4);
    CHECK(c[Cat::ACTIVATIONS] == 16 * 16 * 4 + 4);

    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
