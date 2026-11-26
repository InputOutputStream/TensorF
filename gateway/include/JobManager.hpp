#pragma once
// JobManager — spawns TensorF binaries (server/client/benchmark/examples) as
// child processes, captures their stdout/stderr line-by-line, and keeps an
// in-memory registry of job state + rolling log buffer. Consumed by the
// HTTP/WebSocket gateway so the web dashboard can start/stop/watch jobs
// without any changes to the existing C++ training/federated code.

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <memory>
#include <chrono>
#include <deque>
#include <functional>
#include <algorithm>
#include <csignal>
#include <stdexcept>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

enum class JobStatus { Running, Exited, Failed, Killed };

inline std::string to_string(JobStatus s) {
    switch (s) {
        case JobStatus::Running: return "running";
        case JobStatus::Exited:  return "exited";
        case JobStatus::Failed:  return "failed";
        case JobStatus::Killed:  return "killed";
    }
    return "unknown";
}

struct Job {
    std::string id;
    std::string kind;              // "server" | "client" | "benchmark" | "gpt2" | "smollm" | "tests"
    std::string command;           // full command line, for display
    pid_t pid = -1;
    std::atomic<bool> running{true};
    JobStatus status = JobStatus::Running;
    int exit_code = 0;
    std::chrono::system_clock::time_point started_at;
    std::chrono::system_clock::time_point ended_at;
    std::deque<std::string> log_lines;     // rolling buffer
    size_t log_seq = 0;                    // monotonically increasing, for incremental fetch
    std::mutex log_mutex;
    static constexpr size_t MAX_LOG_LINES = 5000;

    // Called whenever a new log line arrives (used to fan out to WebSocket subscribers)
    std::function<void(const std::string& job_id, const std::string& line)> on_line;

    void push_line(const std::string& line) {
        std::lock_guard<std::mutex> lock(log_mutex);
        log_lines.push_back(line);
        log_seq++;
        if (log_lines.size() > MAX_LOG_LINES) log_lines.pop_front();
        if (on_line) on_line(id, line);
    }
};

class JobManager {
public:
    static JobManager& instance() {
        static JobManager mgr;
        return mgr;
    }

    // Launch a binary with args, tagged with `kind` for the UI. Returns the new job id.
    std::string launch(const std::string& kind, const std::string& binary_path,
                        const std::vector<std::string>& args,
                        std::function<void(const std::string&, const std::string&)> on_line,
                        std::function<void(const std::shared_ptr<Job>&)> on_exit = nullptr) {
        // Build argv BEFORE fork(): allocating in the child of a multithreaded
        // process is not async-signal-safe (the 2nd job is forked while the
        // 1st job's reader thread is alive).
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(binary_path.c_str()));
        for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);

        int out_pipe[2];
        if (pipe(out_pipe) != 0) throw std::runtime_error("pipe() failed");

        pid_t pid = fork();
        if (pid < 0) throw std::runtime_error("fork() failed");

        if (pid == 0) {
            // child
            dup2(out_pipe[1], STDOUT_FILENO);
            dup2(out_pipe[1], STDERR_FILENO);
            close(out_pipe[0]);
            close(out_pipe[1]);
            // Don't leak the gateway's listening/client sockets and other jobs'
            // pipes into the child (a surviving child would keep the port bound).
            for (int fd = 3; fd < 1024; ++fd) close(fd);
            execv(binary_path.c_str(), argv.data());
            _exit(127); // execv failed
        }

        // parent
        close(out_pipe[1]);
        fcntl(out_pipe[0], F_SETFL, O_NONBLOCK);

        auto job = std::make_shared<Job>();
        job->id = generate_id();
        job->kind = kind;
        job->pid = pid;
        job->started_at = std::chrono::system_clock::now();
        job->on_line = on_line;

        std::ostringstream cmd;
        cmd << binary_path;
        for (auto& a : args) cmd << " " << a;
        job->command = cmd.str();

        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            jobs_[job->id] = job;
            prune_locked();
        }

        // reader thread: pull stdout/stderr, split into lines, push into job log
        std::thread([this, job, on_exit, fd = out_pipe[0]]() {
            std::string buf;
            char chunk[4096];
            while (true) {
                ssize_t n = read(fd, chunk, sizeof(chunk));
                if (n > 0) {
                    buf.append(chunk, n);
                    size_t pos;
                    while ((pos = buf.find('\n')) != std::string::npos) {
                        job->push_line(buf.substr(0, pos));
                        buf.erase(0, pos + 1);
                    }
                } else if (n == 0) {
                    break; // EOF: child closed pipe
                } else {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        continue;
                    }
                    break;
                }
            }
            if (!buf.empty()) job->push_line(buf);
            close(fd);

            int status = 0;
            waitpid(job->pid, &status, 0);
            {
                // status/exit_code/ended_at are read by job_to_json() under log_mutex
                std::lock_guard<std::mutex> lk(job->log_mutex);
                job->ended_at = std::chrono::system_clock::now();
                if (WIFEXITED(status)) {
                    job->exit_code = WEXITSTATUS(status);
                    job->status = (job->exit_code == 0) ? JobStatus::Exited : JobStatus::Failed;
                } else if (WIFSIGNALED(status)) {
                    job->exit_code = WTERMSIG(status);
                    job->status = JobStatus::Killed;
                } else {
                    job->status = JobStatus::Failed;
                }
                job->running = false;
            }
            if (on_exit) on_exit(job);   // lets the gateway broadcast "job_exited"
        }).detach();

        return job->id;
    }

    bool kill_job(const std::string& id) {
        std::shared_ptr<Job> job = get(id);
        if (!job || !job->running) return false;
        return ::kill(job->pid, SIGTERM) == 0;
    }

    std::shared_ptr<Job> get(const std::string& id) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto it = jobs_.find(id);
        return it == jobs_.end() ? nullptr : it->second;
    }

    std::vector<std::shared_ptr<Job>> list() {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        std::vector<std::shared_ptr<Job>> out;
        out.reserve(jobs_.size());
        for (auto& [id, job] : jobs_) out.push_back(job);
        return out;
    }

    json job_to_json(const std::shared_ptr<Job>& job, size_t log_tail = 200) {
        std::lock_guard<std::mutex> lock(job->log_mutex);
        json j;
        j["id"] = job->id;
        j["kind"] = job->kind;
        j["command"] = job->command;
        j["pid"] = job->pid;
        j["status"] = to_string(job->status);
        j["running"] = job->running.load();
        j["exit_code"] = job->exit_code;
        j["log_seq"] = job->log_seq;
        json lines = json::array();
        size_t start = job->log_lines.size() > log_tail ? job->log_lines.size() - log_tail : 0;
        for (size_t i = start; i < job->log_lines.size(); i++) lines.push_back(job->log_lines[i]);
        j["log"] = lines;
        return j;
    }

private:
    JobManager() = default;
    // Keep the registry (and its 5000-line log buffers) bounded: drop the oldest
    // FINISHED jobs beyond MAX_FINISHED. Running jobs are never dropped.
    void prune_locked() {
        constexpr size_t MAX_FINISHED = 100;
        std::vector<std::shared_ptr<Job>> done;
        for (auto& [id, j] : jobs_) if (!j->running) done.push_back(j);
        if (done.size() <= MAX_FINISHED) return;
        std::sort(done.begin(), done.end(), [](auto& a, auto& b) { return a->started_at < b->started_at; });
        for (size_t i = 0; i + MAX_FINISHED < done.size(); i++) jobs_.erase(done[i]->id);
    }
    std::string generate_id() {
        static std::atomic<uint64_t> counter{0};
        auto now = std::chrono::system_clock::now().time_since_epoch().count();
        std::ostringstream oss;
        oss << std::hex << now << "-" << counter.fetch_add(1);
        return oss.str();
    }

    std::unordered_map<std::string, std::shared_ptr<Job>> jobs_;
    std::mutex registry_mutex_;
};
