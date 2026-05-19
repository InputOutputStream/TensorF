#include "lazy_loader.hpp"
#include <netinet/in.h>
#include <unistd.h>
#include <vector>
#include <iostream>

int main() {
    LazyDataLoader loader("data.txt");

    std::vector<double> weights(10, 0.5);

    // simulation entraînement
    std::vector<double> batch;
    while (loader.getNextBatch(batch)) {
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] += 0.01; // fake training
        }
    }

    int sock = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in serv{};
    serv.sin_family = AF_INET;
    serv.sin_port = htons(8080);
    serv.sin_addr.s_addr = INADDR_ANY;

    connect(sock, (sockaddr*)&serv, sizeof(serv));

    write(sock, weights.data(), sizeof(double)*weights.size());

    close(sock);

    std::cout << "Update sent\n";
}
