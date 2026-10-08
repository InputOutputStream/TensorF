#!/usr/bin/env bash
# Build gcntest + spmmtest with the Android NDK and run them on a connected device/emulator.
#
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/29.0.14206865
export PATH=$HOME/Android/Sdk/platform-tools:$PATH
#   ./scripts/run_android.sh                   # arm64-v8a (real phones)
#   ABI=x86_64 ./scripts/run_android.sh        # x86_64 emulator: checks Android only, not ARM/NEON
#   ./scripts/run_android.sh -DTENSORF_USE_OPENBLAS=ON   # extra args go to cmake

set -euo pipefail

: "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME to your NDK folder}"
ABI="${ABI:-arm64-v8a}"
API="${API:-24}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build-android-$ABI"
REMOTE=/data/local/tmp/tensorf

[ -f "$ROOT/CMakeLists.txt" ] || { echo "No CMakeLists.txt in $ROOT (must be spelled exactly 'CMakeLists.txt')"; exit 1; }

cmake -S "$ROOT" -B "$BUILD" -G "Unix Makefiles" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ABI" -DANDROID_PLATFORM="android-$API" \
  -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$BUILD" -j"$(nproc)" --target gcntest spmmtest coratrain

adb get-state >/dev/null            # fails early if no device is connected/authorised
echo "device ABI: $(adb shell getprop ro.product.cpu.abi | tr -d '\r')"
adb shell "mkdir -p $REMOTE"
adb push "$BUILD/bin/gcntest" "$BUILD/bin/spmmtest" "$BUILD/bin/coratrain" "$REMOTE/"
adb shell "chmod +x $REMOTE/gcntest $REMOTE/spmmtest $REMOTE/coratrain"

echo "=== spmmtest ==="; adb shell "$REMOTE/spmmtest"
echo "=== gcntest ===";  adb shell "$REMOTE/gcntest"

# Real data: Cora (run ./scripts/download_cora.sh first). Set CORA=0 to skip, EPOCHS=n to change length.
if [ "${CORA:-1}" = "1" ] && [ -s "$ROOT/Datasets/cora/cora.content" ]; then
  adb shell "mkdir -p $REMOTE/cora"
  adb push "$ROOT/Datasets/cora/cora.content" "$ROOT/Datasets/cora/cora.cites" "$REMOTE/cora/"
  echo "=== coratrain (Cora) ==="
  adb shell "$REMOTE/coratrain $REMOTE/cora/cora.content $REMOTE/cora/cora.cites ${EPOCHS:-200}"
else
  echo "(skipping Cora: run ./scripts/download_cora.sh to enable)"
fi