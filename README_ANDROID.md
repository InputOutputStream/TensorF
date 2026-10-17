# TensorF on Android

Build the tests with the Android NDK, push them to a phone over `adb`, run them, and profile the device. Back to the framework: [README.md](README.md).

TensorF runs as a native command-line binary from the `adb` shell (`/data/local/tmp`). It is not an app and does not need root.

## 1. Install the tools

You need a Linux or macOS computer, a phone with a USB cable, and about 2 GB of disk.

```bash
# JDK for sdkmanager
sudo apt-get install -y openjdk-17-jdk cmake unzip        # macOS: brew install openjdk@17 cmake

# Android command-line tools: download "Command line tools only" from
# https://developer.android.com/studio#command-line-tools-only
mkdir -p ~/Android/Sdk/cmdline-tools
unzip commandlinetools-linux-*_latest.zip -d ~/Android/Sdk/cmdline-tools
mv ~/Android/Sdk/cmdline-tools/cmdline-tools ~/Android/Sdk/cmdline-tools/latest

export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH
yes | sdkmanager --licenses
sdkmanager --install "platform-tools" "ndk;29.0.14206865"
```

Already have Android Studio? Install "NDK (Side by side) 29.0.14206865" and "Android SDK Platform-Tools" from *Settings → Android SDK → SDK Tools* instead.

Add to `~/.bashrc`:

```bash
export ANDROID_HOME=$HOME/Android/Sdk
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/29.0.14206865
export PATH=$ANDROID_HOME/platform-tools:$PATH
```

`scripts/run_android.sh` assigns its own `ANDROID_NDK_HOME` near the top, so if you installed a different NDK version, edit that line in the script.

## 2. Prepare the phone

1. Settings → About phone → tap **Build number** seven times.
2. Settings → Developer options → turn on **USB debugging**.
3. Plug in the cable and accept the "Allow USB debugging?" prompt.
4. Check:

```bash
adb devices                                  # must say "device", not "unauthorized"
adb shell getprop ro.product.cpu.abi         # arm64-v8a on almost every current phone
```

Wireless instead of a cable (Android 11+): Developer options → Wireless debugging → *Pair with code*, then `adb pair <ip>:<pair-port>` and `adb connect <ip>:<port>`.

If `adb devices` shows nothing: try another cable or port, `adb kill-server && adb start-server`, and on Linux install `android-sdk-platform-tools-common` for udev rules.

## 3. Build, push, run

```bash
./scripts/download_cora.sh          # optional: real graph data for the GCN run
./scripts/run_android.sh            # arm64-v8a, API 24
```

Options:

```bash
ABI=x86_64 ./scripts/run_android.sh                  # emulator (checks Android only, not ARM/NEON speed)
CORA=0 ./scripts/run_android.sh                      # skip Cora
EPOCHS=50 ./scripts/run_android.sh                   # shorter Cora run
./scripts/run_android.sh -DTENSORF_USE_OPENBLAS=ON   # extra arguments go to cmake
```

The script builds with CMake and the NDK toolchain (the bundled BLAS shim, so no OpenBLAS is needed), pushes the binaries to `/data/local/tmp/tensorf`, and runs `spmmtest`, `gcntest`, `algorithm_profiler_test`, `profiler_workload_test`, `profiler_probe_test`, then `coratrain` on Cora. The fingerprint, device-benchmark, memory and capability tests are built and pushed but their run lines are commented out in the script; run them by hand:

```bash
adb shell /data/local/tmp/tensorf/hw_fingerprint_test
adb shell /data/local/tmp/tensorf/device_benchmark_test
adb shell /data/local/tmp/tensorf/memory_profiler_test
adb shell /data/local/tmp/tensorf/profiler_capability_test
```

Always run from `/data/local/tmp` (`/sdcard` is mounted `noexec`).

## 4. Profile the phone

The profiler measures what the device is and what it can sustain, so training can be sized to it.

```cpp
auto fp  = tensorf::profile::collect();                              // cores, caches, memory limits, thermal
auto cap = tensorf::profile::run_device_benchmark(fp, BenchConfig::quick());
auto bud = tensorf::profile::plan_budget(fp, read_process_memory(), BudgetPolicy{});
```

What it handles on a phone:

- **big.LITTLE:** cores are grouped into classes from `cpu_capacity`, else the ARM part-id table, else max frequency; the benchmark reports per-class and per-thread rates and how threads scale.
- **Memory:** cgroup limits, PSI pressure, Android low-memory-killer thresholds and zram are read; `plan_budget` keeps a reserve and shrinks the budget as pressure rises. `MemoryGuard` returns Ok / Shrink / Stop so a training loop can reduce its batch instead of being killed.
- **Throttling:** a sustained probe shows how far throughput falls after a few seconds. It is seconds long by default; lengthen `BenchConfig::sustained_seconds` for a real answer.
- **Governor hooks:** `BenchConfig::should_stop` and `read_temp_c` let your app stop or pace a run.

`include/profiler/tmp/CapabilitySuite.hpp` additionally stops a run when a sensor passes 45 °C (battery 40 °C); those limits are documented as unverified starting points.

## 5. What to expect

Reference phone (from `REPORT2.TXT`): Xiaomi 23053RN02A, Android 15, 8 cores (2× Cortex-A75 @ 2.0 GHz, 6× Cortex-A55 @ 1.8 GHz), 7.6 GB RAM. `spmmtest`, `gcntest`, `algorithm_profiler_test`, `profiler_workload_test` and `profiler_probe_test` pass there. Four checks fail and are open:

| Test | Failing check | Cause |
|---|---|---|
| `memory_profiler_test` | RSS grows by ≥ 50 MB after a 64 MB allocation | pages already resident; overhead factor reads 1.000 |
| `hw_fingerprint_test` | `fp.storage.known` | cache topology, thermal zones and `power_supply` are not readable from the adb shell |
| `device_benchmark_test` | sustained-probe checks | the time budget skipped the probe |
| `profiler_capability_test` | `bw_l1 > bw_dram` | measured 4.0 GFLOP/s f32, 8.7 GB/s DRAM, 11% noise |

The adb shell does not have an app's permissions: `/sys` and PSI are partly hidden, and `smaps_rollup` needs Android 8+. Battery and thermal data for an installed app should come from the Android framework APIs, not from `/sys`. These code paths are tested against fake filesystem text and on the one phone above, nothing else, and not on 32-bit ARM.

## No computer?

Termux (from F-Droid) can build on the phone itself: `pkg install clang cmake make git`, clone, then `cmake -S . -B build && cmake --build build -j$(nproc)`. This route is untested by the project.

## Related

LLM inference and federated training on a phone: [README_LLM.md](README_LLM.md) (memory-bounded chunked transfer, LoRA, FP8/FP4 checkpoints).
