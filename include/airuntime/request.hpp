#pragma once

#include "airuntime/status.hpp"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace airuntime {

enum class RequestState {
    Received,
    Queued,
    Running,
    Completed,
    Rejected,
    Failed,
    Cancelled,
    TimedOut
};

struct RequestSnapshot {
    std::string request_id;
    std::string model_id;
    RequestState state{RequestState::Received};
    std::optional<InferenceResult> result;
};

using RequestObserver = std::function<void(RequestSnapshot)>;

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
    [[nodiscard]] RequestSnapshot snapshot() const;

    void set_deadline(std::optional<std::chrono::steady_clock::time_point> deadline);
    [[nodiscard]] std::optional<std::chrono::steady_clock::time_point> deadline() const;
    [[nodiscard]] bool is_expired(std::chrono::steady_clock::time_point now) const;

    void add_observer(RequestObserver observer);

    Status transition_to(RequestState next);
    bool try_complete(InferenceResult result);
    bool try_fail(Status error);
    bool try_reject(Status error);
    bool try_cancel();
    bool try_timeout();
    bool try_timeout_if_expired(std::chrono::steady_clock::time_point now);

    // Legacy helpers that delegate to try_* variants.
    Status complete(InferenceResult result);
    Status fail(Status error);
    Status reject(Status error);

    bool wait_for_terminal(std::optional<std::chrono::milliseconds> timeout = std::nullopt) const;

  private:
    static bool is_valid_transition(RequestState from, RequestState to);
    static bool is_terminal_state(RequestState state);

    bool try_transition_to_locked(RequestState next);
    void notify_observers_locked();
    void notify_if_terminal_locked();

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;

    const std::string request_id_;
    const std::string model_id_;
    const std::string prompt_;
    const std::size_t max_output_tokens_;

    RequestState state_{RequestState::Received};
    std::optional<InferenceResult> result_;
    std::optional<std::chrono::steady_clock::time_point> deadline_;
    std::vector<RequestObserver> observers_;
};

using RequestPtr = std::shared_ptr<InferenceRequest>;

} // namespace airuntime
