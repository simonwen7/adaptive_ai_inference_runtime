#include "airuntime/runtime.hpp"

#include <stdexcept>
#include <utility>

namespace airuntime {

Runtime::Runtime(std::unique_ptr<IRequestScheduler> scheduler, std::unique_ptr<Worker> worker)
    : scheduler_(std::move(scheduler)), worker_(std::move(worker)) {
    if (!scheduler_ || !worker_) {
        throw std::invalid_argument("Runtime requires non-null scheduler and worker");
    }
    admission_.emplace(*scheduler_);
}

Runtime::~Runtime() {
    stop();
}

Status Runtime::start() {
    std::lock_guard lock(mutex_);
    if (started_) {
        return Status::success();
    }
    accepting_.store(true);
    dispatch_thread_ = detail::JoinThread([this] { dispatch_loop(); });
    started_ = true;
    return Status::success();
}

Status Runtime::submit(const RequestPtr &request) {
    if (!accepting_.load()) {
        if (request && request->state() == RequestState::Received) {
            request->reject(
                Status::error(ErrorCode::RuntimeStopped, "runtime is not accepting work"));
        }
        return Status::error(ErrorCode::RuntimeStopped, "runtime is not accepting work");
    }
    return admission_->admit(request);
}

void Runtime::stop() {
    accepting_.store(false);
    if (scheduler_) {
        scheduler_->close();
    }

    std::optional<detail::JoinThread> local;
    {
        std::lock_guard lock(mutex_);
        local = std::move(dispatch_thread_);
        dispatch_thread_.reset();
    }
    // JoinThread joins on destruction after close drains the queue.
}

bool Runtime::is_running() const {
    return accepting_.load() && started_;
}

void Runtime::dispatch_loop() {
    while (true) {
        auto next = scheduler_->next();
        if (!next.has_value()) {
            break;
        }
        worker_->execute(*next);
    }
}

} // namespace airuntime
