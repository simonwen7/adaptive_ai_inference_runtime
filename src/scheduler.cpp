#include "airuntime/scheduler.hpp"

namespace airuntime {

FifoScheduler::FifoScheduler(std::size_t capacity) : queue_(capacity) {}

Status FifoScheduler::enqueue(RequestPtr request) {
    return queue_.try_push(std::move(request));
}

std::optional<RequestPtr> FifoScheduler::next() {
    return queue_.wait_pop();
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
