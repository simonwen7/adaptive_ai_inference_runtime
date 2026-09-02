#include "airuntime/request.hpp"

namespace airuntime {

namespace {

void invoke_observers(const std::vector<RequestObserver> &callbacks, const RequestSnapshot &snap) {
    for (const auto &callback : callbacks) {
        try {
            callback(snap);
        } catch (...) {
        }
    }
}

} // namespace

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

RequestSnapshot InferenceRequest::snapshot() const {
    std::lock_guard lock(mutex_);
    RequestSnapshot snap;
    snap.request_id = request_id_;
    snap.model_id = model_id_;
    snap.state = state_;
    snap.result = result_;
    return snap;
}

void InferenceRequest::set_deadline(std::optional<std::chrono::steady_clock::time_point> deadline) {
    std::lock_guard lock(mutex_);
    deadline_ = deadline;
}

std::optional<std::chrono::steady_clock::time_point> InferenceRequest::deadline() const {
    std::lock_guard lock(mutex_);
    return deadline_;
}

bool InferenceRequest::is_expired(std::chrono::steady_clock::time_point now) const {
    std::lock_guard lock(mutex_);
    return deadline_.has_value() && now >= *deadline_;
}

void InferenceRequest::add_observer(RequestObserver observer) {
    if (!observer) {
        return;
    }
    RequestSnapshot snap;
    bool invoke_now = false;
    {
        std::lock_guard lock(mutex_);
        if (is_terminal_state(state_)) {
            snap.request_id = request_id_;
            snap.model_id = model_id_;
            snap.state = state_;
            snap.result = result_;
            invoke_now = true;
        } else {
            observers_.push_back(observer);
            return;
        }
    }
    if (invoke_now) {
        try {
            observer(snap);
        } catch (...) {
        }
    }
}

bool InferenceRequest::is_terminal_state(RequestState state) {
    return state == RequestState::Completed || state == RequestState::Rejected ||
           state == RequestState::Failed || state == RequestState::Cancelled ||
           state == RequestState::TimedOut;
}

bool InferenceRequest::is_valid_transition(RequestState from, RequestState to) {
    if (is_terminal_state(from)) {
        return false;
    }
    switch (from) {
    case RequestState::Received:
        return to == RequestState::Queued || to == RequestState::Rejected ||
               to == RequestState::Cancelled || to == RequestState::TimedOut;
    case RequestState::Queued:
        return to == RequestState::Running || to == RequestState::Rejected ||
               to == RequestState::Cancelled || to == RequestState::TimedOut;
    case RequestState::Running:
        return to == RequestState::Completed || to == RequestState::Failed ||
               to == RequestState::Cancelled || to == RequestState::TimedOut;
    case RequestState::Completed:
    case RequestState::Rejected:
    case RequestState::Failed:
    case RequestState::Cancelled:
    case RequestState::TimedOut:
        return false;
    }
    return false;
}

bool InferenceRequest::try_transition_to_locked(RequestState next) {
    if (!is_valid_transition(state_, next)) {
        return false;
    }
    state_ = next;
    notify_if_terminal_locked();
    return true;
}

void InferenceRequest::notify_if_terminal_locked() {
    if (is_terminal_state(state_)) {
        cv_.notify_all();
    }
}

void InferenceRequest::notify_observers_locked() {
    if (observers_.empty()) {
        return;
    }
    RequestSnapshot snap;
    snap.request_id = request_id_;
    snap.model_id = model_id_;
    snap.state = state_;
    snap.result = result_;

    std::vector<RequestObserver> callbacks;
    if (is_terminal_state(state_)) {
        callbacks.swap(observers_);
    } else {
        callbacks = observers_;
    }

    mutex_.unlock();
    invoke_observers(callbacks, snap);
    mutex_.lock();
}

Status InferenceRequest::transition_to(RequestState next) {
    std::lock_guard lock(mutex_);
    if (!try_transition_to_locked(next)) {
        return Status::error(ErrorCode::InvalidStateTransition, "invalid request state transition");
    }
    notify_observers_locked();
    return Status::success();
}

bool InferenceRequest::try_complete(InferenceResult result_in) {
    std::vector<RequestObserver> callbacks;
    RequestSnapshot snap;
    {
        std::lock_guard lock(mutex_);
        if (!try_transition_to_locked(RequestState::Completed)) {
            return false;
        }
        result_ = std::move(result_in);
        snap.request_id = request_id_;
        snap.model_id = model_id_;
        snap.state = state_;
        snap.result = result_;
        callbacks.swap(observers_);
    }
    invoke_observers(callbacks, snap);
    return true;
}

bool InferenceRequest::try_fail(Status error) {
    std::vector<RequestObserver> callbacks;
    RequestSnapshot snap;
    {
        std::lock_guard lock(mutex_);
        if (!try_transition_to_locked(RequestState::Failed)) {
            return false;
        }
        result_ = InferenceResult::failure(std::move(error));
        snap.request_id = request_id_;
        snap.model_id = model_id_;
        snap.state = state_;
        snap.result = result_;
        callbacks.swap(observers_);
    }
    invoke_observers(callbacks, snap);
    return true;
}

bool InferenceRequest::try_reject(Status error) {
    std::vector<RequestObserver> callbacks;
    RequestSnapshot snap;
    {
        std::lock_guard lock(mutex_);
        if (!try_transition_to_locked(RequestState::Rejected)) {
            return false;
        }
        result_ = InferenceResult::failure(std::move(error));
        snap.request_id = request_id_;
        snap.model_id = model_id_;
        snap.state = state_;
        snap.result = result_;
        callbacks.swap(observers_);
    }
    invoke_observers(callbacks, snap);
    return true;
}

bool InferenceRequest::try_cancel() {
    std::vector<RequestObserver> callbacks;
    RequestSnapshot snap;
    {
        std::lock_guard lock(mutex_);
        if (!try_transition_to_locked(RequestState::Cancelled)) {
            return false;
        }
        result_ =
            InferenceResult::failure(Status::error(ErrorCode::Cancelled, "request cancelled"));
        snap.request_id = request_id_;
        snap.model_id = model_id_;
        snap.state = state_;
        snap.result = result_;
        callbacks.swap(observers_);
    }
    invoke_observers(callbacks, snap);
    return true;
}

bool InferenceRequest::try_timeout() {
    std::vector<RequestObserver> callbacks;
    RequestSnapshot snap;
    {
        std::lock_guard lock(mutex_);
        if (!try_transition_to_locked(RequestState::TimedOut)) {
            return false;
        }
        result_ = InferenceResult::failure(Status::error(ErrorCode::TimedOut, "request timed out"));
        snap.request_id = request_id_;
        snap.model_id = model_id_;
        snap.state = state_;
        snap.result = result_;
        callbacks.swap(observers_);
    }
    invoke_observers(callbacks, snap);
    return true;
}

bool InferenceRequest::try_timeout_if_expired(std::chrono::steady_clock::time_point now) {
    {
        std::lock_guard lock(mutex_);
        if (is_terminal_state(state_)) {
            return false;
        }
        if (!deadline_.has_value() || now < *deadline_) {
            return false;
        }
    }
    return try_timeout();
}

Status InferenceRequest::complete(InferenceResult result_in) {
    return try_complete(std::move(result_in))
               ? Status::success()
               : Status::error(ErrorCode::InvalidStateTransition, "request already terminal");
}

Status InferenceRequest::fail(Status error) {
    return try_fail(std::move(error))
               ? Status::success()
               : Status::error(ErrorCode::InvalidStateTransition, "request already terminal");
}

Status InferenceRequest::reject(Status error) {
    return try_reject(std::move(error))
               ? Status::success()
               : Status::error(ErrorCode::InvalidStateTransition, "request already terminal");
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
