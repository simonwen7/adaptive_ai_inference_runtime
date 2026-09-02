#include "airuntime/scheduler.hpp"

#include <chrono>

namespace airuntime {

namespace {

bool is_dead_request(const RequestPtr &request) {
    if (!request) {
        return true;
    }
    if (request->is_terminal()) {
        return true;
    }
    request->try_timeout_if_expired(std::chrono::steady_clock::now());
    return request->is_terminal();
}

} // namespace

FifoScheduler::FifoScheduler(std::size_t capacity) : queue_(capacity) {}

Status FifoScheduler::enqueue(RequestPtr request) {
    return queue_.try_push(std::move(request));
}

std::optional<RequestPtr> FifoScheduler::next() {
    while (true) {
        auto item = queue_.wait_pop();
        if (!item.has_value()) {
            return std::nullopt;
        }
        if (!is_dead_request(*item)) {
            return item;
        }
    }
}

void FifoScheduler::close() {
    queue_.close();
}

std::size_t FifoScheduler::size() const {
    return queue_.size();
}

std::size_t FifoScheduler::capacity() const {
    return queue_.capacity();
}

} // namespace airuntime
