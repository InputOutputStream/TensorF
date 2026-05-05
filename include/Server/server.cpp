#include "thread_pool.hpp"
#include <netinet/in.h>
#include <unistd.h>
#include <vector>
#include <iostream>


ThreadPool pool(4);
std::vector<std::vector<double>> updates;

void handleClient(int clientSock) {
    std::vector<double> weights(10);

    read(clientSock, weights.data(), sizeof(double)*10);

    {
        // critique section
        updates.push_back(weights);
    }

    close(clientSock);
}


int main() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8080);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(server_fd, (sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 5);

    std::cout << "Server started...\n";

    while (true) {
        int clientSock = accept(server_fd, nullptr, nullptr);

        pool.enqueue([clientSock]() {
            handleClient(clientSock);
        });
    }
}




