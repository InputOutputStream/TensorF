#!/usr/bin/env bash
# Build gcntest + spmmtest with the Android NDK and run them on a connected device/emulator.
#
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$HOME/Android/Sdk/ndk/29.0.14206865}"   # override by exporting it first
export PATH=$HOME/Android/Sdk/platform-tools:$PATH
#   ./scripts/run_android.sh                   # arm64-v8a (real phones)
#   ABI=x86_64 ./scripts/run_android.sh        # x86_64 emulator: checks Android only, not ARM/NEON
#   BLAS=SHIM ./scripts/run_android.sh         # portable shim instead of OpenBLAS (slow; for comparison)
#   ./scripts/run_android.sh -DTENSORF_OPENBLAS_TARGET=CORTEXA76          # extra args go to cmake
#   ./scripts/run_android.sh -DTENSORF_OPENBLAS_DYNAMIC_ARCH=ON           # pick OpenBLAS kernels per CPU at run time
#   ./scripts/run_android.sh -DTENSORF_OPENBLAS_ROOT=/path/to/prebuilt    # skip the OpenBLAS source build
#
# By default OpenBLAS (static) is downloaded and cross-compiled for $ABI with the NDK on the first run (a few
# minutes, needs network + host gcc/make); it is cached in build-android-$ABI/ and reused after that.

set -euo pipefail

: "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME to your NDK folder}"
ABI="${ABI:-arm64-v8a}"
API="${API:-24}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build-android-$ABI"
REMOTE=/data/local/tmp/tensorf

[ -f "$ROOT/CMakeLists.txt" ] || { echo "No CMakeLists.txt in $ROOT (must be spelled exactly 'CMakeLists.txt')"; exit 1; }

# Reject OpenBLAS targets OpenBLAS does not know about.
for arg in "$@"; do
  case "$arg" in
    -DTENSORF_OPENBLAS_TARGET=*)
      t="${arg#-DTENSORF_OPENBLAS_TARGET=}"
      case "$t" in
        CORTEXA53|CORTEXA55|CORTEXA57|CORTEXA72|CORTEXA73|CORTEXA76|CORTEXA710|CORTEXA510|\
        CORTEXX1|VULCAN|THUNDERX|THUNDERX2T99|EMAG|ARMV8|ARMV7|NEHALEM) ;;
        *) echo "Unknown TENSORF_OPENBLAS_TARGET='$t'."; \
           echo "This phone is 2x A75 + 6x A55; use -DTENSORF_OPENBLAS_DYNAMIC_ARCH=ON, or TARGET=CORTEXA55."; \
           exit 1 ;;
      esac ;;
  esac
done

cmake -S "$ROOT" -B "$BUILD" -G "Unix Makefiles" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ABI" -DANDROID_PLATFORM="android-$API" \
  -DCMAKE_BUILD_TYPE=Release ${BLAS:+-DTENSORF_BLAS=$BLAS} "$@"
cmake --build "$BUILD" -j"$(nproc)" --target blas_check gcntest spmmtest coratrain algorithm_profiler_test memory_profiler_test device_benchmark_test hw_fingerprint_test profiler_capability_test  profiler_workload_test profiler_probe_test profiler_advisor_test hyperparam_advisor_test profiler_memory_nohook_test profiler_memory_test

adb get-state >/dev/null            # fails early if no device is connected/authorised
echo "device ABI: $(adb shell getprop ro.product.cpu.abi | tr -d '\r')"
adb shell "mkdir -p $REMOTE"
adb push "$BUILD/bin/blas_check" "$BUILD/bin/gcntest" "$BUILD/bin/spmmtest" "$BUILD/bin/coratrain" "$BUILD/bin/algorithm_profiler_test" "$BUILD/bin/memory_profiler_test" "$BUILD/bin/hw_fingerprint_test" "$REMOTE/"
adb push "$BUILD/bin/device_benchmark_test" "$BUILD/bin/profiler_capability_test" "$BUILD/bin/profiler_workload_test" "$BUILD/bin/profiler_probe_test" "$REMOTE/"
adb push "$BUILD/bin/profiler_advisor_test" "$BUILD/bin/hyperparam_advisor_test" "$BUILD/bin/profiler_memory_nohook_test" "$BUILD/bin/profiler_memory_test" "$REMOTE/"

adb shell "chmod +x $REMOTE/blas_check $REMOTE/gcntest $REMOTE/spmmtest $REMOTE/coratrain $REMOTE/algorithm_profiler_test $REMOTE/hw_fingerprint_test $REMOTE/memory_profiler_test"
adb shell "chmod +x $REMOTE/device_benchmark_test $REMOTE/profiler_capability_test $REMOTE/profiler_workload_test $REMOTE/profiler_probe_test"
adb shell "chmod +x $REMOTE/profiler_advisor_test $REMOTE/hyperparam_advisor_test $REMOTE/profiler_memory_nohook_test $REMOTE/profiler_memory_test"

# Proves which BLAS is really in the binary on the device and that it computes correctly; aborts the run otherwise.
EXPECT_BLAS="$(echo "${BLAS:-auto}" | tr '[:upper:]' '[:lower:]')"
echo "=== blas_check (expect: $EXPECT_BLAS) ==="; adb shell "$REMOTE/blas_check --expect=$EXPECT_BLAS" || { echo "BLAS mismatch, aborting"; exit 1; }
echo "=== spmmtest ==="; adb shell "$REMOTE/spmmtest"
echo "=== gcntest ===";  adb shell "$REMOTE/gcntest"
echo "=== algorithm_profiler_test ===";  adb shell "$REMOTE/algorithm_profiler_test"
# echo "=== memory_profiler_test ==="; adb shell "$REMOTE/memory_profiler_test"
# echo "=== hw_fingerprint_test ===";  adb shell "$REMOTE/hw_fingerprint_test"
echo "=== device_benchmark_test ==="; adb shell "$REMOTE/device_benchmark_test"
# echo "=== profiler_capability_test ===";  adb shell "$REMOTE/profiler_capability_test"
echo "=== profiler_workload_test ==="; adb shell "$REMOTE/profiler_workload_test"
echo "=== profiler_probe_test ==="; adb shell "OPENBLAS_NUM_THREADS=1 $REMOTE/profiler_probe_test"
echo "=== profiler_advisor_test ==="; adb shell "$REMOTE/profiler_advisor_test"
echo "=== hyperparam_advisor_test ==="; adb shell "$REMOTE/hyperparam_advisor_test"
# echo "=== profiler_memory_test ==="; adb shell "$REMOTE/profiler_memory_test"
echo "=== profiler_memory_nohook_test ==="; adb shell "$REMOTE/profiler_memory_nohook_test"

# Real data: Cora (run ./scripts/download_cora.sh first). Set CORA=0 to skip, EPOCHS=n to change length.
if [ "${CORA:-1}" = "1" ] && [ -s "$ROOT/Datasets/cora/cora.content" ]; then
  adb shell "mkdir -p $REMOTE/cora"
  adb push "$ROOT/Datasets/cora/cora.content" "$ROOT/Datasets/cora/cora.cites" "$REMOTE/cora/"
  echo "=== coratrain (Cora) ==="
  adb shell "OPENBLAS_NUM_THREADS=1 $REMOTE/coratrain $REMOTE/cora/cora.content $REMOTE/cora/cora.cites ${EPOCHS:-200}"
else
  echo "(skipping Cora: run ./scripts/download_cora.sh to enable)"
fi


