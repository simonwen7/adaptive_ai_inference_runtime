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
        not_empty_.notify_one();
        return Status::success();
    }

    // Blocks while full and open. Fails if closed before space is available.
    Status wait_push(T value) {
        std::unique_lock lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < capacity_ || closed_; });
        if (closed_) {
            return Status::error(ErrorCode::QueueClosed, "queue is closed");
        }
        queue_.push_back(std::move(value));
        not_empty_.notify_one();
        return Status::success();
    }

    // Blocks until an item is available or the queue is closed and empty.
    // Returns nullopt only when closed and empty.
    std::optional<T> wait_pop() {
        std::unique_lock lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty() || closed_; });
        if (queue_.empty()) {
            return std::nullopt;
        }
        T value = std::move(queue_.front());
        queue_.pop_front();
        not_full_.notify_one();
        return value;
    }

    void close() {
        std::lock_guard lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
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
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T> queue_;
    bool closed_{false};
};

} // namespace airuntime
