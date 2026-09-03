#pragma once

#include "airuntime/serving/request_handler.hpp"

#include <boost/asio/executor_work_guard.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/beast/http/message.hpp>
#include <boost/beast/http/parser.hpp>
#include <boost/beast/http/string_body.hpp>
#include <boost/beast/http/vector_body.hpp>
#include <boost/beast/http/write.hpp>

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace airuntime::serving {

class HttpServer;

class HttpSession : public std::enable_shared_from_this<HttpSession> {
  public:
    HttpSession(boost::asio::ip::tcp::socket socket, RequestHandler &handler, HttpServer &server);
    void start();
    void shutdown();

  private:
    using Request = boost::beast::http::request<boost::beast::http::string_body>;
    using StringResponse = boost::beast::http::response<boost::beast::http::string_body>;

    void do_read();
    void on_read(boost::beast::error_code ec, std::size_t bytes_transferred);
    void handle_request();
    void start_infer(bool stream, const InferRequestSpec &spec);
    void on_state_update(RequestSnapshot snap);
    void arm_disconnect_watch();
    void stop_disconnect_watch();
    void on_peer_wait(boost::beast::error_code wait_ec);
    void on_disconnect();
    void queue_write(std::string data, bool is_terminal);
    void do_write();
    void send_json_response(int status, const std::string &body);
    void close();

    boost::asio::ip::tcp::socket socket_;
    boost::beast::flat_buffer buffer_;
    RequestHandler &handler_;
    HttpServer &server_;
    std::optional<Request> request_;
    std::optional<boost::beast::http::request_parser<boost::beast::http::string_body>> parser_;
    RequestPtr active_request_;
    bool stream_mode_{false};
    bool response_started_{false};
    bool json_response_started_{false};
    bool closed_{false};
    bool disconnect_watch_armed_{false};
    bool disconnect_watch_active_{false};
    std::deque<std::string> write_queue_;
    bool write_in_progress_{false};
    boost::asio::steady_timer deadline_timer_;
    char peer_probe_byte_{0};
};

class HttpServer {
  public:
    HttpServer(boost::asio::io_context &ioc, boost::asio::ip::tcp::endpoint endpoint,
               RequestHandler &handler);
    void run();
    void stop();
    [[nodiscard]] unsigned short port() const;
    [[nodiscard]] bool is_ready() const;
    void unregister_session(const HttpSession *session);

  private:
    void do_accept();
    void on_accept(boost::beast::error_code ec, boost::asio::ip::tcp::socket socket);

    boost::asio::io_context &ioc_;
    boost::asio::ip::tcp::acceptor acceptor_;
    RequestHandler &handler_;
    std::optional<boost::asio::executor_work_guard<boost::asio::io_context::executor_type>>
        work_guard_;
    std::atomic<bool> stopping_{false};
    std::atomic<bool> ready_{false};
    mutable std::mutex sessions_mutex_;
    std::vector<std::weak_ptr<HttpSession>> sessions_;
};

} // namespace airuntime::serving
