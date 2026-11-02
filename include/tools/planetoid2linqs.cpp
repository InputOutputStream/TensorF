// Command-line driver for include/data/planetoid2linqs.hpp (used by scripts/download_datasets.sh).
//
// Build:  g++ -std=c++17 -O2 -Iinclude/data tools/planetoid2linqs.cpp -o bin/planetoid2linqs
// Usage:  planetoid2linqs <raw_dir> <name> <out_dir>

#include "planetoid2linqs.hpp"

#include <iostream>

int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: " << argv[0] << " <raw_dir> <name> <out_dir>\n"; return 2; }
    try {
        std::cout << planetoid_to_linqs(argv[1], argv[2], argv[3]) << "\n";
    } catch (const std::exception& e) {
        std::cerr << "planetoid2linqs: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
