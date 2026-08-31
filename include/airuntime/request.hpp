#pragma once

#include "airuntime/status.hpp"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace airuntime {

enum class RequestState { Received, Queued, Running, Completed, Rejected, Failed };

class InferenceRequest {
  public:
    InferenceRequest(std::string request_id, std::string model_id, std::string prompt,
                     std::size_t max_output_tokens);

    InferenceRequest(const InferenceRequest &) = delete;
    InferenceRequest &operator=(const InferenceRequest &) = delete;

    [[nodiscard]] std::string request_id() const;
    [[nodiscard]] std::string model_id() const;
    [[nodiscard]] std::string prompt() const;
    [[nodiscard]] std::size_t max_output_tokens() const;

    [[nodiscard]] RequestState state() const;
    [[nodiscard]] std::optional<InferenceResult> result() const;
    [[nodiscard]] bool is_terminal() const;

    Status transition_to(RequestState next);
    Status complete(InferenceResult result);
    Status fail(Status error);
    Status reject(Status error);

    // Waits until the request reaches a terminal state.
    // Optional timeout is a safety bound for callers/tests, not product deadline semantics.
    bool wait_for_terminal(std::optional<std::chrono::milliseconds> timeout = std::nullopt) const;

  private:
    static bool is_valid_transition(RequestState from, RequestState to);
    static bool is_terminal_state(RequestState state);

    Status transition_to_locked(RequestState next);
    void notify_if_terminal_locked();

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;

    const std::string request_id_;
    const std::string model_id_;
    const std::string prompt_;
    const std::size_t max_output_tokens_;

    RequestState state_{RequestState::Received};
    std::optional<InferenceResult> result_;
};

using RequestPtr = std::shared_ptr<InferenceRequest>;

} // namespace airuntime
