#pragma once
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

// Fixed-size worker pool.
// Fixes vs. the previous version: `stop` is now only touched under the mutex
// (it was written without it in ~ThreadPool -> data race with the workers'
// wait predicate), tasks are moved instead of copied, and enqueue() after
// shutdown is refused (returns false) instead of silently queuing a task that
// would never run.
class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;

public:
    explicit ThreadPool(size_t n) {
        for (size_t i = 0; i < n; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait(lock, [this]() { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    bool enqueue(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (stop) return false;
            tasks.push(std::move(task));
        }
        cv.notify_one();
        return true;
    }

    /// Tasks waiting for a free worker (e.g. detect "more clients than threads").
    size_t pending() {
        std::lock_guard<std::mutex> lock(mtx);
        return tasks.size();
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        for (auto& t : workers) t.join();
    }
};
