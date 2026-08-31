#pragma once

#include "airuntime/status.hpp"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace airuntime {

template <typename T> class BoundedQueue {
  public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity_ == 0) {
            throw std::invalid_argument("BoundedQueue capacity must be positive");
        }
    }

    BoundedQueue(const BoundedQueue &) = delete;
    BoundedQueue &operator=(const BoundedQueue &) = delete;

    Status try_push(T value) {
        std::lock_guard lock(mutex_);
        if (closed_) {
            return Status::error(ErrorCode::QueueClosed, "queue is closed");
        }
        if (queue_.size() >= capacity_) {
            return Status::error(ErrorCode::QueueFull, "queue is full");
        }
        queue_.push_back(std::move(value));
        cv_.notify_one();
        return Status::success();
    }

    // Blocks until an item is available or the queue is closed and empty.
    // Returns nullopt only when closed and empty.
    std::optional<T> wait_pop() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || closed_; });
        if (queue_.empty()) {
            return std::nullopt;
        }
        T value = std::move(queue_.front());
        queue_.pop_front();
        return value;
    }

    void close() {
        std::lock_guard lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }

    [[nodiscard]] std::size_t size() const {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }

    [[nodiscard]] std::size_t capacity() const {
        return capacity_;
    }

    [[nodiscard]] bool closed() const {
        std::lock_guard lock(mutex_);
        return closed_;
    }

  private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<T> queue_;
    bool closed_{false};
};

} // namespace airuntime
