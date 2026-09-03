#pragma once

#include "airuntime/backend.hpp"
#include "airuntime/batch_builder.hpp"
#include "airuntime/bounded_queue.hpp"
#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_manager.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/request.hpp"
#include "airuntime/status.hpp"
#include "airuntime/threading.hpp"
#include "airuntime/worker_snapshot.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace airuntime {

enum class WorkerState { Idle, Busy };

struct WorkerBatchMetrics {
    std::uint64_t batches_executed{0};
    std::uint64_t requests_executed_via_batches{0};
    std::uint64_t multi_request_batches{0};
    std::uint64_t max_batch_size_observed{0};
};

struct WorkerConfig {
    WorkerId worker_id{0};
    std::size_t queue_capacity{16};
    std::uint64_t memory_budget_bytes{0};
    BatchBuilderConfig batch_config{};
    std::shared_ptr<const ModelRegistry> registry;
    std::unique_ptr<IModelBackend> backend;
    std::unique_ptr<IEvictionPolicy> eviction_policy;
};

class Worker {
  public:
    explicit Worker(WorkerConfig config);
    ~Worker();

    Worker(const Worker &) = delete;
    Worker &operator=(const Worker &) = delete;

    Status start();
    Status try_enqueue(const RequestPtr &request);
    Status enqueue(const RequestPtr &request);
    Status enqueue_until(const RequestPtr &request, std::chrono::steady_clock::time_point deadline);
    void close();
    void join();

    [[nodiscard]] WorkerSnapshot snapshot() const;
    [[nodiscard]] WorkerId id() const;
    [[nodiscard]] WorkerState state() const;
    [[nodiscard]] ResidencyMetrics residency_metrics() const;
    [[nodiscard]] WorkerBatchMetrics batch_metrics() const;
    [[nodiscard]] ModelManager &model_manager();
    [[nodiscard]] const ModelManager &model_manager() const;

  private:
    void run_loop();
    Status execute_batch(std::vector<RequestPtr> batch);
    void discard_queued_request(const RequestPtr &request);
    static bool is_dead_request(const RequestPtr &request);
    static std::vector<RequestPtr> filter_live_batch(std::vector<RequestPtr> batch);

    const WorkerId worker_id_;
    const std::size_t queue_capacity_;
    BatchBuilder batch_builder_;

    // Declaration order: backend outlives model_manager which references it.
    std::unique_ptr<IModelBackend> backend_;
    ModelManager model_manager_;
    BoundedQueue<RequestPtr> lane_;

    mutable std::mutex mutex_;
    WorkerState state_{WorkerState::Idle};
    std::size_t active_count_{0};
    WorkerBatchMetrics batch_metrics_;
    std::optional<RequestPtr> deferred_;
    std::atomic<std::size_t> queued_count_{0};
    std::atomic<bool> accepting_{false};
    bool started_{false};
    std::optional<detail::JoinThread> thread_;
};

} // namespace airuntime
