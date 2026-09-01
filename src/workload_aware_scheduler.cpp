#include "airuntime/workload_aware_scheduler.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace airuntime {

WorkloadAwareScheduler::WorkloadAwareScheduler(WorkloadAwareSchedulerConfig config)
    : capacity_(config.capacity), max_bypass_(config.max_bypass) {
    if (capacity_ == 0) {
        throw std::invalid_argument("WorkloadAwareScheduler capacity must be > 0");
    }
    if (max_bypass_ < 1) {
        throw std::invalid_argument("WorkloadAwareScheduler max_bypass must be >= 1");
    }
}

Status WorkloadAwareScheduler::enqueue(RequestPtr request) {
    if (!request) {
        return Status::error(ErrorCode::InternalError, "null request");
    }
    std::lock_guard lock(mutex_);
    if (closed_) {
        return Status::error(ErrorCode::QueueClosed, "scheduler is closed");
    }
    if (size_ >= capacity_) {
        return Status::error(ErrorCode::QueueFull, "scheduler is full");
    }

    Entry entry;
    entry.request = std::move(request);
    entry.enqueue_sequence = next_sequence_++;
    entry.bypass_count = 0;
    by_model_[entry.request->model_id()].push_back(std::move(entry));
    ++size_;
    cv_.notify_one();
    return Status::success();
}

std::optional<RequestPtr> WorkloadAwareScheduler::next() {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return size_ > 0 || closed_; });
    if (size_ == 0) {
        return std::nullopt;
    }
    return select_locked();
}

std::optional<RequestPtr> WorkloadAwareScheduler::select_locked() {
    std::uint64_t oldest_global_seq = std::numeric_limits<std::uint64_t>::max();
    for (const auto &[model, q] : by_model_) {
        (void)model;
        if (!q.empty() && q.front().enqueue_sequence < oldest_global_seq) {
            oldest_global_seq = q.front().enqueue_sequence;
        }
    }

    std::optional<std::string> chosen_model;
    std::size_t chosen_index = 0;
    bool starvation = false;

    std::uint64_t best_starved_seq = std::numeric_limits<std::uint64_t>::max();
    for (auto &[model, q] : by_model_) {
        for (std::size_t i = 0; i < q.size(); ++i) {
            if (q[i].bypass_count >= max_bypass_ && q[i].enqueue_sequence < best_starved_seq) {
                best_starved_seq = q[i].enqueue_sequence;
                chosen_model = model;
                chosen_index = i;
                starvation = true;
            }
        }
    }

    if (!starvation) {
        std::size_t best_group_size = 0;
        std::uint64_t best_oldest_seq = std::numeric_limits<std::uint64_t>::max();
        std::string best_model_id;

        for (const auto &[model, q] : by_model_) {
            if (q.empty()) {
                continue;
            }
            const std::size_t group_size = q.size();
            const std::uint64_t oldest_seq = q.front().enqueue_sequence;
            const bool better = !chosen_model.has_value() || group_size > best_group_size ||
                                (group_size == best_group_size && oldest_seq < best_oldest_seq) ||
                                (group_size == best_group_size && oldest_seq == best_oldest_seq &&
                                 model < best_model_id);
            if (better) {
                best_group_size = group_size;
                best_oldest_seq = oldest_seq;
                best_model_id = model;
                chosen_model = model;
                chosen_index = 0;
            }
        }
    }

    if (!chosen_model.has_value()) {
        return std::nullopt;
    }

    auto mit = by_model_.find(*chosen_model);
    if (mit == by_model_.end() || chosen_index >= mit->second.size()) {
        return std::nullopt;
    }

    Entry selected = std::move(mit->second[chosen_index]);
    mit->second.erase(mit->second.begin() + static_cast<std::ptrdiff_t>(chosen_index));
    if (mit->second.empty()) {
        by_model_.erase(mit);
    }
    --size_;

    if (selected.enqueue_sequence != oldest_global_seq) {
        ++metrics_.reordered_dispatches;
    }
    if (starvation) {
        ++metrics_.starvation_promotions;
    }

    for (auto &[model, q] : by_model_) {
        (void)model;
        for (auto &entry : q) {
            ++entry.bypass_count;
        }
    }

    return selected.request;
}

void WorkloadAwareScheduler::close() {
    std::lock_guard lock(mutex_);
    closed_ = true;
    cv_.notify_all();
}

std::size_t WorkloadAwareScheduler::size() const {
    std::lock_guard lock(mutex_);
    return size_;
}

std::size_t WorkloadAwareScheduler::capacity() const {
    return capacity_;
}

WorkloadAwareSchedulerMetrics WorkloadAwareScheduler::metrics() const {
    std::lock_guard lock(mutex_);
    return metrics_;
}

} // namespace airuntime
