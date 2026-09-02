#include "airuntime/workload_aware_scheduler.hpp"

#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

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

WorkloadAwareScheduler::WorkloadAwareScheduler(WorkloadAwareSchedulerConfig config)
    : capacity_(config.capacity), max_bypass_(config.max_bypass) {
    if (capacity_ == 0) {
        throw std::invalid_argument("WorkloadAwareScheduler capacity must be > 0");
    }
    if (max_bypass_ < 1) {
        throw std::invalid_argument("WorkloadAwareScheduler max_bypass must be >= 1");
    }
}

void WorkloadAwareScheduler::prune_dead_entries_locked() {
    for (auto it = by_model_.begin(); it != by_model_.end();) {
        auto &q = it->second;
        for (auto entry_it = q.begin(); entry_it != q.end();) {
            if (is_dead_request(entry_it->request)) {
                entry_it = q.erase(entry_it);
                if (size_ > 0) {
                    --size_;
                }
            } else {
                ++entry_it;
            }
        }
        if (q.empty()) {
            it = by_model_.erase(it);
        } else {
            ++it;
        }
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
    prune_dead_entries_locked();
    if (size_ == 0) {
        return std::nullopt;
    }
    return select_locked();
}

std::optional<RequestPtr> WorkloadAwareScheduler::select_locked() {
    prune_dead_entries_locked();
    if (size_ == 0) {
        return std::nullopt;
    }

    std::uint64_t oldest_global_seq = std::numeric_limits<std::uint64_t>::max();
    for (const auto &[model, q] : by_model_) {
        (void)model;
        if (!q.empty() && !is_dead_request(q.front().request) &&
            q.front().enqueue_sequence < oldest_global_seq) {
            oldest_global_seq = q.front().enqueue_sequence;
        }
    }

    std::optional<std::string> chosen_model;
    std::size_t chosen_index = 0;
    bool starvation = false;

    std::uint64_t best_starved_seq = std::numeric_limits<std::uint64_t>::max();
    for (auto &[model, q] : by_model_) {
        for (std::size_t i = 0; i < q.size(); ++i) {
            if (is_dead_request(q[i].request)) {
                continue;
            }
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
            std::size_t live_group_size = 0;
            std::uint64_t oldest_seq = std::numeric_limits<std::uint64_t>::max();
            for (const auto &entry : q) {
                if (is_dead_request(entry.request)) {
                    continue;
                }
                ++live_group_size;
                oldest_seq = std::min(oldest_seq, entry.enqueue_sequence);
            }
            if (live_group_size == 0) {
                continue;
            }
            const bool better =
                !chosen_model.has_value() || live_group_size > best_group_size ||
                (live_group_size == best_group_size && oldest_seq < best_oldest_seq) ||
                (live_group_size == best_group_size && oldest_seq == best_oldest_seq &&
                 model < best_model_id);
            if (better) {
                best_group_size = live_group_size;
                best_oldest_seq = oldest_seq;
                best_model_id = model;
                chosen_model = model;
                chosen_index = 0;
            }
        }
    }

    if (!chosen_model.has_value()) {
        prune_dead_entries_locked();
        return std::nullopt;
    }

    auto mit = by_model_.find(*chosen_model);
    if (mit == by_model_.end()) {
        return std::nullopt;
    }

    while (chosen_index < mit->second.size() &&
           is_dead_request(mit->second[chosen_index].request)) {
        mit->second.erase(mit->second.begin() + static_cast<std::ptrdiff_t>(chosen_index));
        if (size_ > 0) {
            --size_;
        }
    }
    if (chosen_index >= mit->second.size()) {
        if (mit->second.empty()) {
            by_model_.erase(mit);
        }
        return select_locked();
    }

    Entry selected = std::move(mit->second[chosen_index]);
    mit->second.erase(mit->second.begin() + static_cast<std::ptrdiff_t>(chosen_index));
    if (mit->second.empty()) {
        by_model_.erase(mit);
    }
    --size_;

    if (is_dead_request(selected.request)) {
        return select_locked();
    }

    if (selected.enqueue_sequence != oldest_global_seq) {
        ++metrics_.reordered_dispatches;
    }
    if (starvation) {
        ++metrics_.starvation_promotions;
    }

    for (auto &[model, q] : by_model_) {
        (void)model;
        for (auto &entry : q) {
            if (!is_dead_request(entry.request)) {
                ++entry.bypass_count;
            }
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
