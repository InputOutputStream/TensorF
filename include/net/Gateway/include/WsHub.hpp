#pragma once
// WsHub — tracks connected WebSocket sessions and broadcasts JSON event
// messages to all of them (job started/log-line/exited, node events, etc).
// Kept deliberately simple: one mutex-guarded set of session pointers.

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/strand.hpp>
#include <memory>
#include <set>
#include <mutex>
#include <string>
#include <deque>
#include <nlohmann/json.hpp>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

class WsSession;

class WsHub {
public:
    static WsHub& instance() {
        static WsHub hub;
        return hub;
    }

    void join(std::shared_ptr<WsSession> session) {
        std::lock_guard<std::mutex> lock(mutex_);
        sessions_.insert(session);
    }

    void leave(std::shared_ptr<WsSession> session) {
        std::lock_guard<std::mutex> lock(mutex_);
        sessions_.erase(session);
    }

    void broadcast(const json& msg);

    size_t count() {
        std::lock_guard<std::mutex> lock(mutex_);
        return sessions_.size();
    }

private:
    std::mutex mutex_;
    std::set<std::shared_ptr<WsSession>> sessions_;
};

// One live WebSocket connection to a dashboard client.
class WsSession : public std::enable_shared_from_this<WsSession> {
public:
    explicit WsSession(tcp::socket&& socket) : ws_(std::move(socket)) {}

    // `req` is the HTTP upgrade request already consumed off the socket by
    // the HTTP layer (via http::async_read) before handing off to us — Beast
    // needs it explicitly here, since re-reading it from the socket would
    // hang forever waiting for bytes that were already read.
    template <class Body, class Allocator>
    void run(beast::http::request<Body, beast::http::basic_fields<Allocator>> req) {
        ws_.set_option(websocket::stream_base::timeout::suggested(beast::role_type::server));
        ws_.set_option(websocket::stream_base::decorator(
            [](websocket::response_type& res) {
                res.set(boost::beast::http::field::server, "tensorf-gateway");
            }));
        auto self = shared_from_this();
        ws_.async_accept(req, [this, self](beast::error_code ec) {
            if (ec) return;
            WsHub::instance().join(self);
            do_read();
        });
    }

    void send(const std::string& payload) {
        auto self = shared_from_this();
        net::post(ws_.get_executor(), [this, self, payload]() {
            outgoing_.push_back(payload);
            if (outgoing_.size() == 1) do_write();
        });
    }

private:
    void do_read() {
        auto self = shared_from_this();
        ws_.async_read(buffer_, [this, self](beast::error_code ec, size_t) {
            if (ec) {
                WsHub::instance().leave(self);
                return;
            }
            buffer_.consume(buffer_.size());
            do_read();
        });
    }

    void do_write() {
        auto self = shared_from_this();
        ws_.text(true);
        ws_.async_write(net::buffer(outgoing_.front()),
            [this, self](beast::error_code ec, size_t) {
                outgoing_.pop_front();
                if (ec) {
                    WsHub::instance().leave(self);
                    return;
                }
                if (!outgoing_.empty()) do_write();
            });
    }

    websocket::stream<tcp::socket> ws_;
    beast::flat_buffer buffer_;
    std::deque<std::string> outgoing_;
};

inline void WsHub::broadcast(const json& msg) {
    std::string payload = msg.dump();
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& s : sessions_) s->send(payload);
}
