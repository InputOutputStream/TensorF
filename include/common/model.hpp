#pragma once
#include <vector>

class Model {
public:
    std::vector<double> weights;

    Model(int size) : weights(size, 0.0) {}

    void update(const std::vector<double>& grad, double lr) {
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] -= lr * grad[i];
        }
    }
};
