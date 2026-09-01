#pragma once

#include "airuntime/backend.hpp"
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

namespace airuntime {

enum class WorkerState { Idle, Busy };

struct WorkerConfig {
    WorkerId worker_id{0};
    std::size_t queue_capacity{16};
    std::uint64_t memory_budget_bytes{0};
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
    Status enqueue(const RequestPtr &request);
    void close();
    void join();

    [[nodiscard]] WorkerSnapshot snapshot() const;
    [[nodiscard]] WorkerId id() const;
    [[nodiscard]] WorkerState state() const;
    [[nodiscard]] ResidencyMetrics residency_metrics() const;
    [[nodiscard]] ModelManager &model_manager();
    [[nodiscard]] const ModelManager &model_manager() const;

  private:
    void run_loop();
    Status execute(const RequestPtr &request);

    const WorkerId worker_id_;
    const std::size_t queue_capacity_;

    // Declaration order: backend outlives model_manager which references it.
    std::unique_ptr<IModelBackend> backend_;
    ModelManager model_manager_;
    BoundedQueue<RequestPtr> lane_;

    mutable std::mutex mutex_;
    WorkerState state_{WorkerState::Idle};
    std::size_t active_count_{0};
    std::atomic<bool> accepting_{false};
    bool started_{false};
    std::optional<detail::JoinThread> thread_;
};

} // namespace airuntime
