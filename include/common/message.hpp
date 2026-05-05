#pragma once
#include <vector>

struct ModelUpdate {
    std::vector<double> weights;
};

struct Message {
    enum Type { UPDATE, GLOBAL_MODEL } type;
    ModelUpdate data;
};
