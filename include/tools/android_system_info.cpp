/*
 * android_system_info.cpp - replaces get_system_info.cpp. Same nine sections, but through
 * profiler/device/AndroidProbe.hpp: no popen, no dumpsys/getprop/cmd/getenforce, "not read" instead of guesses.
 *
 *   g++ -std=c++20 -O2 -Iinclude -Iinclude/tools tools/android_system_info.cpp -o bin/android_system_info
 *   Android NDK: build the same file with the NDK toolchain and run it with adb shell.
 *   --json            print the machine-readable form
 *   --root=PATH       read a copy of /sys and /proc under PATH instead of the live system (for tests)
 */
#include <cstring>
#include <iostream>
#include "profiler/device/AndroidProbe.hpp"

int main(int argc, char** argv) {
    bool json = false; std::string root;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0) json = true;
        else if (std::strncmp(argv[i], "--root=", 7) == 0) root = argv[i] + 7;
    }
    const auto s = tensorf::prof::android::collect_android(root);
    if (json) std::cout << tensorf::prof::android::to_json(s) << "\n";
    else tensorf::prof::android::print_android(s, std::cout);
    return 0;
}
