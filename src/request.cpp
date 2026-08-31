#include "airuntime/request.hpp"

namespace airuntime {

InferenceRequest::InferenceRequest(std::string request_id, std::string model_id, std::string prompt,
                                   std::size_t max_output_tokens)
    : request_id_(std::move(request_id)), model_id_(std::move(model_id)),
      prompt_(std::move(prompt)), max_output_tokens_(max_output_tokens) {}

std::string InferenceRequest::request_id() const {
    std::lock_guard lock(mutex_);
    return request_id_;
}

std::string InferenceRequest::model_id() const {
    std::lock_guard lock(mutex_);
    return model_id_;
}

std::string InferenceRequest::prompt() const {
    std::lock_guard lock(mutex_);
    return prompt_;
}

std::size_t InferenceRequest::max_output_tokens() const {
    std::lock_guard lock(mutex_);
    return max_output_tokens_;
}

RequestState InferenceRequest::state() const {
    std::lock_guard lock(mutex_);
    return state_;
}

std::optional<InferenceResult> InferenceRequest::result() const {
    std::lock_guard lock(mutex_);
    return result_;
}

bool InferenceRequest::is_terminal() const {
    std::lock_guard lock(mutex_);
    return is_terminal_state(state_);
}

bool InferenceRequest::is_terminal_state(RequestState state) {
    return state == RequestState::Completed || state == RequestState::Rejected ||
           state == RequestState::Failed;
}

bool InferenceRequest::is_valid_transition(RequestState from, RequestState to) {
    switch (from) {
    case RequestState::Received:
        return to == RequestState::Queued || to == RequestState::Rejected;
    case RequestState::Queued:
        return to == RequestState::Running || to == RequestState::Rejected;
    case RequestState::Running:
        return to == RequestState::Completed || to == RequestState::Failed;
    case RequestState::Completed:
    case RequestState::Rejected:
    case RequestState::Failed:
        return false;
    }
    return false;
}

Status InferenceRequest::transition_to_locked(RequestState next) {
    if (!is_valid_transition(state_, next)) {
        return Status::error(ErrorCode::InvalidStateTransition, "invalid request state transition");
    }
    state_ = next;
    notify_if_terminal_locked();
    return Status::success();
}

void InferenceRequest::notify_if_terminal_locked() {
    if (is_terminal_state(state_)) {
        cv_.notify_all();
    }
}

Status InferenceRequest::transition_to(RequestState next) {
    std::lock_guard lock(mutex_);
    return transition_to_locked(next);
}

Status InferenceRequest::complete(InferenceResult result) {
    std::lock_guard lock(mutex_);
    auto status = transition_to_locked(RequestState::Completed);
    if (!status.ok()) {
        return status;
    }
    result_ = std::move(result);
    return Status::success();
}

Status InferenceRequest::fail(Status error) {
    std::lock_guard lock(mutex_);
    auto status = transition_to_locked(RequestState::Failed);
    if (!status.ok()) {
        return status;
    }
    result_ = InferenceResult::failure(std::move(error));
    return Status::success();
}

Status InferenceRequest::reject(Status error) {
    std::lock_guard lock(mutex_);
    auto status = transition_to_locked(RequestState::Rejected);
    if (!status.ok()) {
        return status;
    }
    result_ = InferenceResult::failure(std::move(error));
    return Status::success();
}

bool InferenceRequest::wait_for_terminal(std::optional<std::chrono::milliseconds> timeout) const {
    std::unique_lock lock(mutex_);
    if (timeout.has_value()) {
        return cv_.wait_for(lock, *timeout, [this] { return is_terminal_state(state_); });
    }
    cv_.wait(lock, [this] { return is_terminal_state(state_); });
    return true;
}

} // namespace airuntime
