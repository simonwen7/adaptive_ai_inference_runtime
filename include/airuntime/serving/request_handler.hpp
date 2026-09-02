#pragma once

#include "airuntime/request.hpp"
#include "airuntime/runtime.hpp"
#include "airuntime/status.hpp"

#include <chrono>

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>

namespace airuntime::serving {

struct InferRequestSpec {
    std::string model_id;
    std::string prompt;
    std::size_t max_output_tokens{16};
    std::chrono::milliseconds timeout{30000};
};

struct HandlerResponse {
    int status_code{500};
    std::string content_type{"application/json"};
    std::string body;
};

class RequestHandler {
  public:
    explicit RequestHandler(Runtime &runtime);

    [[nodiscard]] std::string next_request_id();
    [[nodiscard]] HandlerResponse handle_get(const std::string &target) const;
    [[nodiscard]] HandlerResponse handle_health() const;
    [[nodiscard]] std::optional<InferRequestSpec>
    parse_infer_request(const std::string &body, std::string &error_message) const;

    [[nodiscard]] HandlerResponse serialize_infer_response(const RequestSnapshot &snap) const;

    [[nodiscard]] std::string serialize_state_event(const RequestSnapshot &snap) const;
    [[nodiscard]] std::string serialize_terminal_event(const RequestSnapshot &snap) const;

    [[nodiscard]] int http_status_for_snapshot(const RequestSnapshot &snap) const;

    Runtime &runtime() {
        return runtime_;
    }

    const Runtime &runtime() const {
        return runtime_;
    }

  private:
    static std::string state_to_string(RequestState state);
    static std::string error_code_to_string(ErrorCode code);

    Runtime &runtime_;
    std::atomic<std::uint64_t> next_id_{1};
};

} // namespace airuntime::serving
