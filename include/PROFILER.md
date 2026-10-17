# TensorF profiler suite (rewrite)

Replaces the old `Profiler.hpp` / `BenchmarkRunner.hpp` / `HyperparamAdvisor.hpp` pipeline, which benchmarked one
operation (dense matmul) and let one function pick a whole transformer. The new suite measures what a device can
do across workload classes, checks its own predictions against real ops, and advises only on the tunables an
algorithm declares. It never chooses a model size.

## Layers

| Layer | Header | What it does |
|---|---|---|
| device | `DeviceFingerprint.hpp` | Facts only. Core classes by capacity/frequency, per-cluster caches, SIMD, safe memory budget (available RAM, cgroup, kernel reserve; never total RAM), thermal and battery sensors. Reads `/proc` and `/sys` directly; no `lscpu`, `dmidecode`, `popen`. |
| capability | `CapabilityKernels.hpp`, `CapabilitySuite.hpp` | Roofline-style kernels: GEMM (square, skinny, f64, int8), conv, STREAM-style bandwidth by thread count, single-cycle pointer chase L1..DRAM, power-law gather, dependent chain, n-body, fork-join and per-op overhead, per-core-class scaling, optional sustained run. Output is a **vector**, with a time budget and a thermal stop. |
| workload model | `WorkloadModel.hpp` | Layer cost library (dense, conv2d, attention, spmm, recurrent, elementwise, norm, embedding, particles, optimizer), `Algorithm` = factory + tunables, `predict()`, activation checkpointing, `Calibration`. |
| validation | `WorkloadProbe.hpp` | Runs 14 real `Matrix`/`Tensor` cases, fits `Calibration`, reports **leave-one-out** error. |
| memory | `TensorAccounting.hpp` | Allocation hook (live/peak by category), autograd graph census, `VmHWM` process peak, measured overhead factor. |
| advice | `Advisor.hpp` | Searches declared tunables; reports config, predicted time/memory with an error range, the binding constraint, runners-up, and what is unvalidated. Optional measured trial. |
| orchestrator | `ProfilerSuite.hpp` | Runs the stages under one time budget and thermal guard; one JSON profile (`tensorf.profile/2`). |

Stage order in `ProfilerSuite::run`: device, **memory**, capability, validation. Memory runs second, before the
heavy stages, because they leave freed pages resident and a later step would then reuse them (measured: factor
0.99 cold, 0.54 warmed, 0.17 after the heavy stages).

## Use

```cpp
#define TENSORF_MEMTRACK_IMPLEMENT          // in exactly ONE .cpp of the program; optional, enables the memory stage
#include "tools/Profiler/ProfilerSuite.hpp"
using namespace tensorf::prof;

ProfilerSuite suite;
Profile p = suite.run();                    // ~20-30 s on a desktop; budgets in ProfileOptions
wl::Algorithm alg = /* factory + tunables; a tunable named "threads" is the thread count */;
adv::Advice a = suite.advise(p, alg, base_config, constraints, {}, optional_trial_fn);
std::string payload = ProfilerSuite::to_json(p, &a);
ProfilerSuite::print_summary(p);
```

Build: `-std=c++20 -Iinclude -Iinclude/core -Iinclude/tools -pthread` (the validation stage runs real framework ops).

CMake, one target per test (the include paths above are the only requirement):

```cmake
foreach(t profiler_capability_test profiler_workload_test profiler_probe_test profiler_memory_test
          profiler_memory_nohook_test profiler_advisor_test profiler_suite_test profiler_suite_nohook_test)
  add_executable(${t} tests/${t}.cpp)
  target_include_directories(${t} PRIVATE include include/core include/tools)
  target_compile_features(${t} PRIVATE cxx_std_20)
  find_package(Threads REQUIRED)
  target_link_libraries(${t} PRIVATE Threads::Threads)
endforeach()
```

## What the numbers mean

- **Calibration** is per build and per device. `ProbeReport::blas` records which BLAS was measured (`shim` is the
  portable loop in `include/tools/cblas.h`, not OpenBLAS).
- **Error range.** Uncalibrated predictions carry 60%. After validation each layer kind carries its measured
  leave-one-out error. The advised range is additionally floored at the device's measured run-to-run noise
  (`ProfilerSuite::noise_floor`: median probe spread / 2.56, one standard deviation of one step).
- **Memory.** Parameters, gradients and activations match real autograd to the byte for dense and elementwise
  layers; matmul backward temporaries are `3MK + 3KN + MN` elements (fitted to <0.1%). The MLP training peak is
  predicted within ~2%. Other layer kinds are analytic only.
- **Thermal.** Stop limits (45 C CPU zone, 40 C battery, 2 C tighter while charging, +8 C rise) are starting
  points from the governor design, **not verified per device**.

## Framework finding

`MatmulOperation::backward` computes both `dA` and `dB` and only afterwards asks `accumulate` to drop the one for
an input without `requires_grad`. A first layer fed by a data batch pays for a gradient it never uses, in time and in
temporaries. Guarding the product with `requires_grad` would save roughly 25-50% of that layer's backward. The
workload model describes TensorF as it is (`need_input_grad = true`).

## Status against the design document

| Item | Status |
|---|---|
| Step 1: capability suite + fingerprint on Linux, with tests | Done, tested |
| Step 2: workload cost library + validation loop vs `Matrix`/`Tensor` | Done, tested. Conv, particle, attention, norm, embedding have **no op to measure**: uncalibrated |
| Step 3: memory accounting hook | Done, tested. Android behaviour untested |
| Step 4: advisor | Done, tested (including mutation checks) |
| Orchestrator + JSON profile | Done, tested |
| Step 5: Android and Apple adapters | **Not done.** Needs devices (see below) |

## Not verified / known limits

- **No Android or iOS device has run any of this.** Android build errors found by a user build (`ANDROID` macro,
  `aligned_alloc` below API 28) were fixed and are guarded by an emulation check, but the NDK itself was not available
  here. `/proc` and `/sys` reads are restricted for apps on Android 8+; the fingerprint degrades (and says what it
  could not read), but real behaviour is untested. `clear_refs` (peak-RSS reset) may be denied.
- **Single thread only.** The framework has no thread pool, so every validation case runs on one thread and
  multi-thread predictions are unvalidated (`threads_validated: 1`). An algorithm without a `threads` tunable is
  predicted single-threaded.
- **No energy model.** `Prediction` carries total flops and bytes; joules need the governor's per-op calibration.
- **Shared-VM noise.** On the 1-vCPU development VM, case noise was 10-66% and the capability vector moved ~40%
  between runs. Accuracy claims should be re-measured on target hardware.
- **Allocation hook** costs 16 bytes per allocation and replaces global `operator new`: profiling builds only.
  Not supported on Windows. A foreign pointer passed to `operator delete` is detected by a magic word, not guaranteed.
- **Step 5 (not written).** The Android adapter's missing parts are the Kotlin sensor shim (battery level/charging,
  power-save, thermal status/headroom) and a foreground service; the Apple adapter needs `sysctl`/`ProcessInfo`
  reads and Apple's documentation did not render when the design was written. Shipping that code without a device
  to compile and run it on would be guesswork, so it is left out rather than faked.
