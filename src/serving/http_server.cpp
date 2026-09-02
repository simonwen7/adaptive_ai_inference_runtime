#include "airuntime/serving/http_session.hpp"

#include <algorithm>
#include <stdexcept>

namespace airuntime::serving {

HttpServer::HttpServer(boost::asio::io_context &ioc, boost::asio::ip::tcp::endpoint endpoint,
                       RequestHandler &handler)
    : ioc_(ioc), acceptor_(ioc), handler_(handler) {
    boost::beast::error_code ec;
    acceptor_.open(endpoint.protocol(), ec);
    if (ec) {
        throw std::runtime_error("failed to open acceptor");
    }
    acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true), ec);
    acceptor_.bind(endpoint, ec);
    if (ec) {
        throw std::runtime_error("failed to bind acceptor");
    }
    acceptor_.listen(boost::asio::socket_base::max_listen_connections, ec);
    if (ec) {
        throw std::runtime_error("failed to listen on acceptor");
    }
}

void HttpServer::run() {
    work_guard_.emplace(boost::asio::make_work_guard(ioc_));
    do_accept();
    ioc_.run();
}

void HttpServer::stop() {
    stopping_.store(true);
    boost::beast::error_code ec;
    acceptor_.close(ec);
    std::vector<std::shared_ptr<HttpSession>> active;
    {
        std::lock_guard lock(sessions_mutex_);
        for (auto &weak : sessions_) {
            if (auto session = weak.lock()) {
                active.push_back(std::move(session));
            }
        }
        sessions_.clear();
    }
    for (auto &session : active) {
        session->shutdown();
    }
    work_guard_.reset();
    ioc_.stop();
}

unsigned short HttpServer::port() const {
    boost::beast::error_code ec;
    const auto endpoint = acceptor_.local_endpoint(ec);
    if (ec) {
        return 0;
    }
    return endpoint.port();
}

bool HttpServer::is_ready() const {
    return ready_.load();
}

void HttpServer::do_accept() {
    if (stopping_.load()) {
        return;
    }
    ready_.store(true);
    acceptor_.async_accept(
        [this](boost::beast::error_code ec, boost::asio::ip::tcp::socket socket) {
            on_accept(ec, std::move(socket));
        });
}

void HttpServer::on_accept(boost::beast::error_code ec, boost::asio::ip::tcp::socket socket) {
    if (!stopping_.load()) {
        do_accept();
    }
    if (ec || stopping_.load()) {
        return;
    }
    auto session = std::make_shared<HttpSession>(std::move(socket), handler_, *this);
    {
        std::lock_guard lock(sessions_mutex_);
        sessions_.push_back(session);
    }
    session->start();
}

void HttpServer::unregister_session(const HttpSession *session) {
    std::lock_guard lock(sessions_mutex_);
    sessions_.erase(std::remove_if(sessions_.begin(), sessions_.end(),
                                   [session](const std::weak_ptr<HttpSession> &weak) {
                                       auto locked = weak.lock();
                                       return !locked || locked.get() == session;
                                   }),
                    sessions_.end());
}

} // namespace airuntime::serving
