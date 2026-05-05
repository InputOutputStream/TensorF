#pragma once
#include <vector>
#include <fstream>
#include <string>

class LazyDataLoader {
    std::ifstream file;

public:
    LazyDataLoader(const std::string& path) {
        file.open(path);
    }

    bool getNextBatch(std::vector<double>& batch) {
        batch.clear();
        double val;

        for (int i = 0; i < 10 && file >> val; i++) {
            batch.push_back(val);
        }

        return !batch.empty();
    }
};
