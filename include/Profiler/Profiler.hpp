#pragma once
/*
 * Profiler.hpp - single entry point of the fused profiler.
 *
 *   core/        ProfilerCommon        timing, stats, parsing, Json, thermal limits       (tensorf::prof)
 *   device/      DeviceFingerprint     facts, safe memory budget, sensors                 (tensorf::prof)
 *                HardwareFingerprint   legacy fingerprint, incl. Apple/Windows collectors (tensorf::profile)
 *   capability/  CapabilityKernels/Suite, FrameworkBench                                  (tensorf::prof, tensorf::fwbench)
 *                DeviceBenchmark       legacy capability profile                          (tensorf::profile)
 *   workload/    WorkloadModel, WorkloadProbe   cost library + validation                 (tensorf::prof::wl)
 *                AlgorithmProfiler     legacy cost model, 7 prebuilt algorithms           (tensorf::profile)
 *   memory/      TensorAccounting      allocation hook, census, overhead factor           (tensorf::prof::mem)
 *                MemoryProfiler        legacy ledger, budget policy, guard, dtype sizes   (tensorf::profile)
 *   advice/      Advisor               tunable search                                      (tensorf::prof::adv)
 *                HyperparamAdvisor     legacy advisor                                      (tensorf::profile)
 *   ProfilerSuite                      orchestrator, JSON profile tensorf.profile/2
 *
 * Both generations live in different namespaces and compile together in one TU. The old flat include
 * names (e.g. "Profiler/CapabilitySuite.hpp") still work through forwarders at this directory's root.
 * TENSORF_PROFILER_NO_LEGACY drops the legacy generation.
 */
#include "Profiler/core/ProfilerCommon.hpp"
#include "Profiler/device/DeviceFingerprint.hpp"
#include "Profiler/capability/FrameworkBench.hpp"
#include "Profiler/capability/CapabilityKernels.hpp"
#include "Profiler/capability/CapabilitySuite.hpp"
#include "Profiler/workload/WorkloadModel.hpp"
#include "Profiler/advice/Advisor.hpp"
#include "Profiler/device/HardwareFingerprint.hpp"
#include "Profiler/capability/DeviceBenchmark.hpp"
#include "Profiler/memory/MemoryProfiler.hpp"
#include "Profiler/workload/AlgorithmProfiler.hpp"
#include "Profiler/advice/HyperparamAdvisor.hpp"
#include "Profiler/memory/TensorAccounting.hpp"
#include "Profiler/workload/WorkloadProbe.hpp"
#include "Profiler/ProfilerSuite.hpp"
