#include "airuntime/serving/http_session.hpp"

#include <boost/asio/post.hpp>
#include <boost/beast/http/read.hpp>
#include <boost/beast/http/write.hpp>

#include <sstream>
#include <utility>

namespace airuntime::serving {

namespace {

constexpr std::size_t kBodyLimit = 1024 * 1024;

bool is_terminal_state(RequestState state) {
    return state == RequestState::Completed || state == RequestState::Rejected ||
           state == RequestState::Failed || state == RequestState::Cancelled ||
           state == RequestState::TimedOut;
}

bool is_peer_gone(const boost::beast::error_code &ec) {
    return ec == boost::asio::error::eof || ec == boost::asio::error::connection_reset ||
           ec == boost::asio::error::connection_aborted || ec == boost::asio::error::broken_pipe ||
           ec == boost::asio::error::bad_descriptor || ec == boost::asio::error::network_reset;
}

} // namespace

HttpSession::HttpSession(boost::asio::ip::tcp::socket socket, RequestHandler &handler,
                         HttpServer &server)
    : socket_(std::move(socket)), handler_(handler), server_(server),
      deadline_timer_(socket_.get_executor()) {}

void HttpSession::start() {
    do_read();
}

void HttpSession::shutdown() {
    if (active_request_ && !active_request_->is_terminal()) {
        active_request_->try_cancel();
    }
    close();
}

void HttpSession::do_read() {
    parser_.emplace();
    parser_->body_limit(kBodyLimit);
    boost::beast::http::async_read(
        socket_, buffer_, *parser_,
        [self = shared_from_this()](boost::beast::error_code ec, std::size_t bytes) {
            self->on_read(ec, bytes);
        });
}

void HttpSession::on_read(boost::beast::error_code ec, std::size_t) {
    if (ec == boost::beast::http::error::body_limit) {
        send_json_response(413, R"({"error":"request body too large"})");
        return;
    }
    if (ec) {
        close();
        return;
    }
    request_ = parser_->release();
    handle_request();
}

void HttpSession::handle_request() {
    const auto method = request_->method();
    const std::string target = std::string(request_->target());

    if (method == boost::beast::http::verb::get) {
        auto response = handler_.handle_get(target);
        send_json_response(response.status_code, response.body);
        return;
    }

    if (method != boost::beast::http::verb::post) {
        send_json_response(405, R"({"error":"method not allowed"})");
        return;
    }

    stream_mode_ = (target == "/v1/infer/stream");
    if (target != "/v1/infer" && !stream_mode_) {
        send_json_response(404, R"({"error":"not found"})");
        return;
    }

    std::string error_message;
    auto spec = handler_.parse_infer_request(request_->body(), error_message);
    if (!spec.has_value()) {
        send_json_response(400, std::string(R"({"error":")") + error_message + R"("})");
        return;
    }

    start_infer(stream_mode_, *spec);
}

void HttpSession::start_infer(bool stream, const InferRequestSpec &spec) {
    stream_mode_ = stream;

    const std::string request_id = handler_.next_request_id();
    active_request_ = std::make_shared<InferenceRequest>(request_id, spec.model_id, spec.prompt,
                                                         spec.max_output_tokens);
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(spec.timeout);
    active_request_->set_deadline(deadline);

    std::weak_ptr<HttpSession> weak_self = shared_from_this();
    active_request_->add_observer([weak_self](RequestSnapshot snap) {
        auto self = weak_self.lock();
        if (!self) {
            return;
        }
        boost::asio::post(self->socket_.get_executor(), [self, snap = std::move(snap)]() mutable {
            self->on_state_update(snap);
        });
    });

    deadline_timer_.expires_at(deadline);
    deadline_timer_.async_wait([weak_self, request = active_request_](boost::beast::error_code ec) {
        if (ec) {
            return;
        }
        auto self = weak_self.lock();
        if (!self) {
            return;
        }
        if (request && !request->is_terminal()) {
            request->try_timeout();
        }
    });

    const auto submit_status = handler_.runtime().submit(active_request_);
    if (!submit_status.ok() && active_request_ && !active_request_->is_terminal()) {
        active_request_->try_reject(submit_status);
    }
    if (active_request_->is_terminal()) {
        return;
    }

    arm_disconnect_watch();
}

void HttpSession::arm_disconnect_watch() {
    if (closed_ || disconnect_watch_armed_) {
        return;
    }
    disconnect_watch_active_ = true;
    disconnect_watch_armed_ = true;
    // Wait for read readiness, then probe. Idle sockets do not complete wait_read.
    auto self = shared_from_this();
    socket_.async_wait(boost::asio::socket_base::wait_read,
                       [self](boost::beast::error_code ec) { self->on_peer_wait(ec); });
}

void HttpSession::stop_disconnect_watch() {
    disconnect_watch_active_ = false;
}

void HttpSession::on_peer_wait(boost::beast::error_code wait_ec) {
    disconnect_watch_armed_ = false;
    if (closed_ || !disconnect_watch_active_) {
        return;
    }
    if (wait_ec == boost::asio::error::operation_aborted) {
        return;
    }

    // Mere readiness is not proof of abandon. Probe with a non-blocking read.
    boost::beast::error_code nb_ec;
    const bool was_non_blocking = socket_.non_blocking();
    socket_.non_blocking(true, nb_ec);

    boost::beast::error_code read_ec;
    const std::size_t n = socket_.read_some(boost::asio::buffer(&peer_probe_byte_, 1), read_ec);

    socket_.non_blocking(was_non_blocking, nb_ec);

    if (is_peer_gone(read_ec) || (!read_ec && n == 0) || is_peer_gone(wait_ec)) {
        on_disconnect();
        return;
    }
    if (read_ec == boost::asio::error::would_block || read_ec == boost::asio::error::try_again) {
        // Spurious readiness — re-arm without cancelling.
        arm_disconnect_watch();
        return;
    }
    if (!read_ec && n > 0) {
        // One-request-per-connection: unexpected follow-up bytes → cancel pending work.
        on_disconnect();
        return;
    }
    if (read_ec) {
        on_disconnect();
        return;
    }
    arm_disconnect_watch();
}

void HttpSession::on_state_update(RequestSnapshot snap) {
    if (closed_) {
        return;
    }
    if (stream_mode_) {
        if (!response_started_) {
            response_started_ = true;
            stop_disconnect_watch();
            static const std::string kHeaders = "HTTP/1.1 200 OK\r\n"
                                                "Content-Type: application/x-ndjson\r\n"
                                                "Transfer-Encoding: chunked\r\n"
                                                "Connection: close\r\n\r\n";
            queue_write(kHeaders, false);
        }
        if (is_terminal_state(snap.state)) {
            stop_disconnect_watch();
            queue_write(handler_.serialize_terminal_event(snap), true);
            deadline_timer_.cancel();
            return;
        }
        if (snap.state == RequestState::Queued || snap.state == RequestState::Running) {
            queue_write(handler_.serialize_state_event(snap), false);
        }
        return;
    }

    if (!is_terminal_state(snap.state)) {
        return;
    }
    stop_disconnect_watch();
    deadline_timer_.cancel();
    const auto response = handler_.serialize_infer_response(snap);
    send_json_response(response.status_code, response.body);
}

void HttpSession::on_disconnect() {
    stop_disconnect_watch();
    if (active_request_ && !active_request_->is_terminal()) {
        active_request_->try_cancel();
    }
    if (closed_) {
        return;
    }
    close();
}

void HttpSession::queue_write(std::string data, bool is_terminal) {
    if (!data.empty()) {
        write_queue_.push_back(std::move(data));
    }
    if (is_terminal) {
        write_queue_.push_back(std::string{});
    }
    if (!write_in_progress_) {
        do_write();
    }
}

void HttpSession::do_write() {
    if (write_queue_.empty()) {
        write_in_progress_ = false;
        return;
    }
    write_in_progress_ = true;
    std::string chunk = std::move(write_queue_.front());
    write_queue_.pop_front();

    if (chunk.empty()) {
        auto payload = std::make_shared<std::string>("0\r\n\r\n");
        boost::asio::async_write(
            socket_, boost::asio::buffer(*payload),
            [self = shared_from_this(), payload](boost::beast::error_code ec, std::size_t) {
                if (ec) {
                    self->on_disconnect();
                    return;
                }
                boost::beast::error_code ignored;
                self->socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_send, ignored);
                self->close();
            });
        return;
    }

    std::ostringstream framed;
    if (response_started_ && chunk.rfind("HTTP/1.1", 0) == 0) {
        framed << chunk;
    } else {
        framed << std::hex << chunk.size() << "\r\n" << chunk << "\r\n";
    }
    auto payload = std::make_shared<std::string>(framed.str());
    boost::asio::async_write(
        socket_, boost::asio::buffer(*payload),
        [self = shared_from_this(), payload](boost::beast::error_code ec, std::size_t) {
            if (ec) {
                self->on_disconnect();
                return;
            }
            self->do_write();
        });
}

void HttpSession::send_json_response(int status, const std::string &body) {
    if (closed_ || json_response_started_) {
        return;
    }
    json_response_started_ = true;
    stop_disconnect_watch();
    deadline_timer_.cancel();
    auto response = std::make_shared<StringResponse>();
    response->result(static_cast<boost::beast::http::status>(status));
    response->set(boost::beast::http::field::content_type, "application/json");
    response->set(boost::beast::http::field::connection, "close");
    response->keep_alive(false);
    response->body() = body;
    response->prepare_payload();
    boost::beast::http::async_write(
        socket_, *response,
        [self = shared_from_this(), response](boost::beast::error_code ec, std::size_t) {
            (void)response;
            if (ec) {
                if (self->active_request_ && !self->active_request_->is_terminal()) {
                    self->active_request_->try_cancel();
                }
                self->close();
                return;
            }
            boost::beast::error_code ignored;
            self->socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_send, ignored);
            self->close();
        });
}

void HttpSession::close() {
    if (closed_) {
        return;
    }
    closed_ = true;
    stop_disconnect_watch();
    deadline_timer_.cancel();
    boost::beast::error_code ec;
    socket_.cancel(ec);
    socket_.close(ec);
    server_.unregister_session(this);
}

} // namespace airuntime::serving
