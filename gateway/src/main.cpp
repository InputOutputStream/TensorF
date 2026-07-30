// TensorF Gateway
// A small Boost.Beast HTTP + WebSocket server that sits in front of the
// existing TensorF binaries (server/client/benchmark/examples) and exposes
// a REST + realtime API for the web dashboard to manage training and
// federated learning jobs, without touching TensorF's own C++ code.
//
// REST API:
//   GET  /api/health                       -> {"status":"ok"}
//   GET  /api/jobs                         -> list of jobs (summary)
//   GET  /api/jobs/{id}                    -> single job detail + log tail
//   POST /api/jobs/benchmark               -> launch bin/benchmark
//   POST /api/jobs/tests                   -> launch bin/basic_tests
//   POST /api/jobs/gpt2                    -> launch bin/gpt2
//   POST /api/jobs/smollm                  -> launch bin/smollm
//   POST /api/jobs/server   {"port":N}     -> launch bin/server (federated)
//   POST /api/jobs/client   {"host":..,"port":N} -> launch bin/client
//   POST /api/jobs/{id}/kill               -> SIGTERM the job
//
// WebSocket:
//   GET  /ws  (upgrade) -> streams JSON events:
//        {"type":"job_started","job":{...}}
//        {"type":"log","job_id":"...","line":"..."}
//        {"type":"job_exited","job":{...}}

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <cstdlib>
#include <nlohmann/json.hpp>

#include "JobManager.hpp"
#include "WsHub.hpp"

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

static std::string BIN_DIR = "../bin"; // relative to gateway/build by default, overridable via TENSORF_BIN_DIR

static std::string bin_path(const std::string& name) {
    return BIN_DIR + "/" + name;
}

static void on_job_line(const std::string& job_id, const std::string& line) {
    json msg;
    msg["type"] = "log";
    msg["job_id"] = job_id;
    msg["line"] = line;
    WsHub::instance().broadcast(msg);
}

static json launch_and_announce(const std::string& kind, const std::string& binary,
                                 const std::vector<std::string>& args) {
    auto& mgr = JobManager::instance();
    std::string id = mgr.launch(kind, binary, args, on_job_line);
    auto job = mgr.get(id);
    json msg;
    msg["type"] = "job_started";
    msg["job"] = mgr.job_to_json(job);
    WsHub::instance().broadcast(msg);
    return mgr.job_to_json(job);
}

// Simple CORS + JSON helpers ------------------------------------------------

static http::response<http::string_body> json_response(
    const http::request<http::string_body>& req, int status, const json& body) {
    http::response<http::string_body> res{static_cast<http::status>(status), req.version()};
    res.set(http::field::server, "tensorf-gateway");
    res.set(http::field::access_control_allow_origin, "*");
    res.set(http::field::access_control_allow_headers, "content-type");
    res.set(http::field::access_control_allow_methods, "GET,POST,OPTIONS");
    res.keep_alive(req.keep_alive());

    // RFC 9110: 1xx/204/304 responses MUST NOT have a message body.
    // Beast's serializer enforces this and throws std::invalid_argument
    // ("invalid response body") if handed a non-empty body on a 204 —
    // the old OPTIONS preflight handler dumped "{}" into a 204 body,
    // which aborted the whole process on the first CORS preflight
    // (async_write -> serializer -> uncaught exception -> terminate).
    // Leave body/content-type unset for these statuses.
    const auto st = static_cast<http::status>(status);
    bool no_body = (status >= 100 && status < 200) || st == http::status::no_content ||
                   st == http::status::not_modified;
    if (!no_body) {
        res.set(http::field::content_type, "application/json");
        res.body() = body.dump();
    }
    res.prepare_payload();
    return res;
}

static json parse_body(const http::request<http::string_body>& req) {
    if (req.body().empty()) return json::object();
    try { return json::parse(req.body()); } catch (...) { return json::object(); }
}

// Route table ----------------------------------------------------------------

static http::response<http::string_body> route(const http::request<http::string_body>& req) {
    std::string target = std::string(req.target());
    std::string path = target;
    if (auto q = target.find('?'); q != std::string::npos) path = target.substr(0, q);

    if (req.method() == http::verb::options) {
        return json_response(req, 204, json::object());
    }

    if (path == "/api/health" && req.method() == http::verb::get) {
        return json_response(req, 200, {{"status", "ok"}, {"ws_clients", WsHub::instance().count()}});
    }

    if (path == "/api/jobs" && req.method() == http::verb::get) {
        json arr = json::array();
        for (auto& job : JobManager::instance().list()) {
            arr.push_back(JobManager::instance().job_to_json(job, 0));
        }
        return json_response(req, 200, {{"jobs", arr}});
    }

    if (path.rfind("/api/jobs/", 0) == 0) {
        std::string rest = path.substr(std::string("/api/jobs/").size());
        if (rest.size() >= 5 && rest.substr(rest.size() - 5) == "/kill" && req.method() == http::verb::post) {
            std::string id = rest.substr(0, rest.size() - 5);
            bool ok = JobManager::instance().kill_job(id);
            return json_response(req, ok ? 200 : 404, {{"killed", ok}});
        }
        if (req.method() == http::verb::get) {
            auto job = JobManager::instance().get(rest);
            if (!job) return json_response(req, 404, {{"error", "job not found"}});
            return json_response(req, 200, JobManager::instance().job_to_json(job, 500));
        }
    }

    if (req.method() == http::verb::post) {
        json body = parse_body(req);

        if (path == "/api/jobs/benchmark")
            return json_response(req, 200, launch_and_announce("benchmark", bin_path("benchmark"), {}));

        if (path == "/api/jobs/tests")
            return json_response(req, 200, launch_and_announce("tests", bin_path("basic_tests"), {}));

        if (path == "/api/jobs/gpt2")
            return json_response(req, 200, launch_and_announce("gpt2", bin_path("gpt2"), {}));

        if (path == "/api/jobs/smollm")
            return json_response(req, 200, launch_and_announce("smollm", bin_path("smollm"), {}));

        if (path == "/api/jobs/server") {
            std::vector<std::string> args;
            if (body.contains("port")) args = {"--port", std::to_string(body["port"].get<int>())};
            return json_response(req, 200, launch_and_announce("server", bin_path("server"), args));
        }

        if (path == "/api/jobs/client") {
            std::vector<std::string> args;
            if (body.contains("host")) { args.push_back("--host"); args.push_back(body["host"].get<std::string>()); }
            if (body.contains("port")) { args.push_back("--port"); args.push_back(std::to_string(body["port"].get<int>())); }
            return json_response(req, 200, launch_and_announce("client", bin_path("client"), args));
        }
    }

    return json_response(req, 404, {{"error", "not found"}});
}

// Per-connection session (handles both plain HTTP and WS upgrade) -----------

class HttpSession : public std::enable_shared_from_this<HttpSession> {
public:
    explicit HttpSession(tcp::socket socket) : socket_(std::move(socket)) {}

    void run() { do_read(); }

private:
    void do_read() {
        auto self = shared_from_this();
        req_ = {};
        http::async_read(socket_, buffer_, req_,
            [this, self](beast::error_code ec, size_t) {
                if (ec) return;
                if (websocket::is_upgrade(req_)) {
                    auto ws_session = std::make_shared<WsSession>(std::move(socket_));
                    ws_session->run(std::move(req_));
                    return;
                }
                auto res = std::make_shared<http::response<http::string_body>>(route(req_));
                http::async_write(socket_, *res, [this, self, res](beast::error_code ec2, size_t) {
                    socket_.shutdown(tcp::socket::shutdown_send, ec2);
                });
            });
    }

    tcp::socket socket_;
    beast::flat_buffer buffer_;
    http::request<http::string_body> req_;
};

class Listener : public std::enable_shared_from_this<Listener> {
public:
    Listener(net::io_context& ioc, tcp::endpoint endpoint) : ioc_(ioc), acceptor_(ioc) {
        beast::error_code ec;
        acceptor_.open(endpoint.protocol(), ec);
        acceptor_.set_option(net::socket_base::reuse_address(true), ec);
        acceptor_.bind(endpoint, ec);
        if (ec) { std::cerr << "bind failed: " << ec.message() << std::endl; std::exit(1); }
        acceptor_.listen(net::socket_base::max_listen_connections, ec);
        if (ec) { std::cerr << "listen failed: " << ec.message() << std::endl; std::exit(1); }
    }

    void run() { do_accept(); }

private:
    void do_accept() {
        acceptor_.async_accept(net::make_strand(ioc_),
            [self = shared_from_this()](beast::error_code ec, tcp::socket socket) {
                if (!ec) std::make_shared<HttpSession>(std::move(socket))->run();
                self->do_accept();
            });
    }

    net::io_context& ioc_;
    tcp::acceptor acceptor_;
};

int main(int argc, char** argv) {
    int port = 8080;
    if (const char* p = std::getenv("TENSORF_GATEWAY_PORT")) port = std::atoi(p);
    if (const char* b = std::getenv("TENSORF_BIN_DIR")) BIN_DIR = b;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        if (a == "--bin-dir" && i + 1 < argc) BIN_DIR = argv[++i];
    }

    std::cout << "TensorF Gateway starting on port " << port
              << " (bin dir: " << BIN_DIR << ")" << std::endl;

    // Single-threaded io_context: this process calls fork()+execv() to launch
    // TensorF jobs, which is unsafe to mix with a multithreaded Boost.Asio
    // event loop (only the calling thread survives fork(), and other threads
    // can leave internal locks held, corrupting the parent). Job stdout is
    // still read on dedicated reader threads spawned per-job in JobManager,
    // which is safe since those threads are created *after* the fork, not
    // running concurrently with it.
    net::io_context ioc{1};
    std::make_shared<Listener>(ioc, tcp::endpoint{tcp::v4(), static_cast<unsigned short>(port)})->run();

    // Defense in depth: ioc.run() rethrows any exception that escapes a
    // completion handler (this is how the 204-body bug above took the
    // whole process down instead of just failing one request). Loop and
    // swallow+log rather than let a single malformed request or future
    // bug of the same shape kill every in-flight job's connection.
    for (;;) {
        try {
            ioc.run();
            break; // ran to completion (no more work), normal shutdown
        } catch (const std::exception& e) {
            std::cerr << "[gateway] unhandled exception in io_context: " << e.what()
                      << " — continuing" << std::endl;
        }
    }
    return 0;
}
