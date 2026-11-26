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
//   POST /api/jobs/client   {"host":..,"port":N} -> launch bin/client  (passed as --server/--port)
//   POST /api/jobs/fedserver {"algo":"gcn","opts":{..},"strategy":"fedavgm:0.9","min_clients":N,"rounds":N,"port":N}
//   POST /api/jobs/fedclient {"algo":"gcn","opts":{..},"host":..,"port":N,"id":..}   (any registered algorithm)
//   POST /api/jobs/{id}/kill               -> SIGTERM the job
//
// WebSocket:
//   GET  /ws  (upgrade) -> streams JSON events:
//        {"type":"job_started","job":{...}}
//        {"type":"log","job_id":"...","line":"..."}
//        {"type":"job_exited","job":{...}}
//        {"type":"fed_event","job_id":"...","event":{...}}   (federated rounds / clients,
//             parsed from "@@TFEVT {json}" lines printed by FedCoordinator)
//
// Security: binds 127.0.0.1 by default. To listen elsewhere pass --bind ADDR AND
// set TENSORF_GATEWAY_TOKEN (POST requests then need "Authorization: Bearer <token>").
// CORS only echoes localhost origins (or TENSORF_GATEWAY_ORIGIN).

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
#include <cctype>
#include <stdexcept>
#include <nlohmann/json.hpp>

#include "JobManager.hpp"
#include "WsHub.hpp"

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

static std::string GW_TOKEN;                         // TENSORF_GATEWAY_TOKEN; empty = no auth (loopback only)
static std::string GW_ORIGIN;                        // optional extra allowed CORS origin
static std::string BIN_DIR = "../bin"; // relative to gateway/build by default, overridable via TENSORF_BIN_DIR

static std::string bin_path(const std::string& name) {
    return BIN_DIR + "/" + name;
}

static void on_job_line(const std::string& job_id, const std::string& line) {
    // Structured events from the federated coordinator: "@@TFEVT {json}"
    static const std::string TAG = "@@TFEVT ";
    if (line.compare(0, TAG.size(), TAG) == 0) {
        json ev = json::parse(line.substr(TAG.size()), nullptr, false);
        if (!ev.is_discarded() && ev.is_object()) {
            json m; m["type"] = "fed_event"; m["job_id"] = job_id; m["event"] = ev;
            WsHub::instance().broadcast(m);
        }
    }
    json msg;
    msg["type"] = "log";
    msg["job_id"] = job_id;
    msg["line"] = line;
    WsHub::instance().broadcast(msg);
}

static json launch_and_announce(const std::string& kind, const std::string& binary,
                                 const std::vector<std::string>& args) {
    auto& mgr = JobManager::instance();
    std::string id = mgr.launch(kind, binary, args, on_job_line,
        [](const std::shared_ptr<Job>& j) {
            json m; m["type"] = "job_exited"; m["job"] = JobManager::instance().job_to_json(j, 0);
            WsHub::instance().broadcast(m);
        });
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
    {
        // Only echo origins we trust (localhost dashboards / configured origin);
        // "*" let any web page the user visits drive this process-launching API.
        std::string origin = std::string(req[http::field::origin]);
        auto is_local = [&](const std::string& o) {
            for (const char* pre : {"http://localhost", "http://127.0.0.1", "https://localhost", "https://127.0.0.1"})
                if (o.rfind(pre, 0) == 0 && (o.size() == std::string(pre).size() || o[std::string(pre).size()] == ':')) return true;
            return false;
        };
        if (!origin.empty() && (is_local(origin) || (!GW_ORIGIN.empty() && origin == GW_ORIGIN))) {
            res.set(http::field::access_control_allow_origin, origin);
            res.set(http::field::vary, "Origin");
        }
    }
    res.set(http::field::access_control_allow_headers, "content-type, authorization");
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

// Arguments are passed to execv() (no shell), but still refuse anything that is
// not a plain flag/value token so a caller cannot smuggle odd bytes or huge input.
static bool arg_ok(const std::string& a) {
    if (a.empty() || a.size() > 256) return false;
    for (unsigned char c : a)
        if (!(std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '/' || c == ':' || c == '=' || c == ',' || c == '@' || c == '+')) return false;
    return a.find("..") == std::string::npos;
}

static bool authorized(const http::request<http::string_body>& req) {
    if (GW_TOKEN.empty()) return true;
    std::string h = std::string(req[http::field::authorization]);
    std::string want = "Bearer " + GW_TOKEN;
    if (h.size() != want.size()) return false;
    unsigned char d = 0; for (size_t i = 0; i < h.size(); i++) d |= (unsigned char)(h[i] ^ want[i]);
    return d == 0;
}

static http::response<http::string_body> route(const http::request<http::string_body>& req) {
    std::string target = std::string(req.target());
    std::string path = target;
    if (auto q = target.find('?'); q != std::string::npos) path = target.substr(0, q);

    if (req.method() == http::verb::options) {
        return json_response(req, 204, json::object());
    }

    if (req.method() == http::verb::post && !authorized(req)) {
        return json_response(req, 401, {{"error", "missing or invalid bearer token"}});
    }
    if (path.rfind("/api/jobs/", 0) == 0 && path.size() > 5 && path.substr(path.size() - 5) == "/kill" && !authorized(req)) {
        return json_response(req, 401, {{"error", "missing or invalid bearer token"}});
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

            auto launch_with_args = [&](const std::string& kind, const std::string& binary,
                                        const std::vector<std::string>& default_args) -> json {
                std::vector<std::string> args;
                if (body.contains("args") && body["args"].is_array()) {
                    for (auto& a : body["args"]) {
                        if (!a.is_string() || !arg_ok(a.get<std::string>()))
                            throw std::invalid_argument("rejected argument");
                        args.push_back(a.get<std::string>());
                    }
                } else {
                    args = default_args;
                }
                return launch_and_announce(kind, binary, args);
            };

            if (path == "/api/jobs/benchmark") {
                return json_response(req, 200, launch_with_args("benchmark", bin_path("benchmark"), {}));
            }
            if (path == "/api/jobs/tests") {
                return json_response(req, 200, launch_with_args("tests", bin_path("basic_tests"), {}));
            }
            if (path == "/api/jobs/gpt2") {
                return json_response(req, 200, launch_with_args("gpt2", bin_path("gpt2"), {}));
            }
            if (path == "/api/jobs/smollm") {
                return json_response(req, 200, launch_with_args("smollm", bin_path("smollm"), {}));
            }

            if (path == "/api/jobs/server") {
                std::vector<std::string> default_args;
                if (body.contains("port")) default_args = {"--port", std::to_string(body["port"].get<int>())};
                return json_response(req, 200, launch_with_args("server", bin_path("server"), default_args));
            }

            // Distributed learning, algorithm-agnostic: {"algo":"gcn","opts":{"data":"Datasets/cora","hidden":16},
            //   "strategy":"fedavgm:0.9","min_clients":3,"rounds":20,"port":9740}
            auto add_opts = [&](std::vector<std::string>& a) {
                if (body.contains("algo")) { a.push_back("--algo"); a.push_back(body["algo"].get<std::string>()); }
                if (body.contains("opts") && body["opts"].is_object())
                    for (auto it = body["opts"].begin(); it != body["opts"].end(); ++it) {
                        std::string v = it.value().is_string() ? it.value().get<std::string>() : it.value().dump();
                        a.push_back("--opt"); a.push_back(it.key() + "=" + v);
                    }
            };
            if (path == "/api/jobs/fedserver") {
                std::vector<std::string> a; add_opts(a);
                for (auto& [k, flag] : std::vector<std::pair<std::string, std::string>>{{"port","--port"},{"enroll_grace_ms","--enroll-grace-ms"},{"min_clients","--min-clients"},{"max_clients","--max-clients"},{"rounds","--rounds"},{"deadline_ms","--deadline-ms"},{"params","--params"}})
                    if (body.contains(k)) { a.push_back(flag); a.push_back(std::to_string(body[k].get<long long>())); }
                if (body.contains("strategy")) { a.push_back("--strategy"); a.push_back(body["strategy"].get<std::string>()); }
                return json_response(req, 200, launch_with_args("fedserver", bin_path("fed_mobile_server"), a));
            }
            if (path == "/api/jobs/fedclient") {
                std::vector<std::string> a; add_opts(a);
                if (body.contains("host")) { a.push_back("--server"); a.push_back(body["host"].get<std::string>()); }
                if (body.contains("port")) { a.push_back("--port"); a.push_back(std::to_string(body["port"].get<int>())); }
                if (body.contains("id"))   { a.push_back("--id");   a.push_back(body["id"].get<std::string>()); }
                if (body.contains("rounds")) { a.push_back("--rounds"); a.push_back(std::to_string(body["rounds"].get<long long>())); }
                if (body.contains("params")) { a.push_back("--params"); a.push_back(std::to_string(body["params"].get<long long>())); }
                if (body.contains("score"))  { a.push_back("--score");  a.push_back(std::to_string(body["score"].get<double>())); }
                if (body.contains("mem_mb")) { a.push_back("--mem-mb"); a.push_back(std::to_string(body["mem_mb"].get<long long>())); }
                return json_response(req, 200, launch_with_args("fedclient", bin_path("fed_mobile_client"), a));
            }

            if (path == "/api/jobs/client") {
                std::vector<std::string> default_args;
                if (body.contains("host")) { default_args.push_back("--server"); default_args.push_back(body["host"].get<std::string>()); }
                if (body.contains("port")) { default_args.push_back("--port"); default_args.push_back(std::to_string(body["port"].get<int>())); }
                return json_response(req, 200, launch_with_args("client", bin_path("client"), default_args));
            }
        }

    return json_response(req, 404, {{"error", "not found"}});
}

// Any exception from a handler (bad JSON types, rejected args, fork() failure)
// becomes an HTTP 400/500 instead of escaping into io_context::run().
static http::response<http::string_body> safe_route(const http::request<http::string_body>& req) {
    try { return route(req); }
    catch (const std::invalid_argument& e) { return json_response(req, 400, {{"error", e.what()}}); }
    catch (const nlohmann::json::exception& e) { return json_response(req, 400, {{"error", std::string("bad request body: ") + e.what()}}); }
    catch (const std::exception& e) { return json_response(req, 500, {{"error", e.what()}}); }
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
                auto res = std::make_shared<http::response<http::string_body>>(safe_route(req_));
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
    std::string bind_addr = "127.0.0.1";
    if (const char* t = std::getenv("TENSORF_GATEWAY_TOKEN")) GW_TOKEN = t;
    if (const char* o = std::getenv("TENSORF_GATEWAY_ORIGIN")) GW_ORIGIN = o;
    if (const char* ba = std::getenv("TENSORF_GATEWAY_BIND")) bind_addr = ba;
    if (const char* p = std::getenv("TENSORF_GATEWAY_PORT")) port = std::atoi(p);
    if (const char* b = std::getenv("TENSORF_BIN_DIR")) BIN_DIR = b;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        if (a == "--bin-dir" && i + 1 < argc) BIN_DIR = argv[++i];
        if (a == "--bind" && i + 1 < argc) bind_addr = argv[++i];
    }

    boost::system::error_code addr_ec;
    auto bind_ip = net::ip::make_address(bind_addr, addr_ec);
    if (addr_ec) { std::cerr << "invalid --bind address: " << bind_addr << std::endl; return 1; }
    if (!bind_ip.is_loopback() && GW_TOKEN.empty()) {
        std::cerr << "refusing to listen on " << bind_addr << " without TENSORF_GATEWAY_TOKEN: "
                     "this API launches processes. Set a token or bind to 127.0.0.1." << std::endl;
        return 1;
    }

    std::cout << "TensorF Gateway starting on " << bind_addr << ":" << port
              << " (bin dir: " << BIN_DIR << ")" << std::endl;

    // Single-threaded io_context: this process calls fork()+execv() to launch
    // TensorF jobs, which is unsafe to mix with a multithreaded Boost.Asio
    // event loop (only the calling thread survives fork(), and other threads
    // can leave internal locks held, corrupting the parent). Job stdout is
    // still read on dedicated reader threads spawned per-job in JobManager,
    // which is safe since those threads are created *after* the fork, not
    // running concurrently with it.
    net::io_context ioc{1};
    std::make_shared<Listener>(ioc, tcp::endpoint{bind_ip, static_cast<unsigned short>(port)})->run();

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
