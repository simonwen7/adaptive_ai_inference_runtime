#pragma once

#include "airuntime/request.hpp"
#include "airuntime/scheduler.hpp"
#include "airuntime/status.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace airuntime {

struct WorkloadAwareSchedulerConfig {
    std::size_t capacity{1};
    std::uint64_t max_bypass{1};
};

struct WorkloadAwareSchedulerMetrics {
    std::uint64_t reordered_dispatches{0};
    std::uint64_t starvation_promotions{0};
};

class WorkloadAwareScheduler final : public IRequestScheduler {
  public:
    explicit WorkloadAwareScheduler(WorkloadAwareSchedulerConfig config);

    Status enqueue(RequestPtr request) override;
    std::optional<RequestPtr> next() override;
    void close() override;
    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] std::size_t capacity() const override;

    [[nodiscard]] WorkloadAwareSchedulerMetrics metrics() const;

  private:
    struct Entry {
        RequestPtr request;
        std::uint64_t enqueue_sequence{0};
        std::uint64_t bypass_count{0};
    };

    std::optional<RequestPtr> select_locked();

    const std::size_t capacity_;
    const std::uint64_t max_bypass_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<std::string, std::deque<Entry>> by_model_;
    std::size_t size_{0};
    std::uint64_t next_sequence_{0};
    bool closed_{false};
    WorkloadAwareSchedulerMetrics metrics_;
};

} // namespace airuntime
