#include <vector>

std::vector<double> federatedAverage(
    const std::vector<std::vector<double>>& updates) {

    int n = updates.size();
    int dim = updates[0].size();

    std::vector<double> avg(dim, 0.0);

    for (const auto& u : updates) {
        for (int i = 0; i < dim; i++) {
            avg[i] += u[i];
        }
    }

    for (int i = 0; i < dim; i++) {
        avg[i] /= n;
    }

    return avg;
}