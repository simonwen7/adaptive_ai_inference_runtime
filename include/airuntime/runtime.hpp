#pragma once

#include "airuntime/admission_controller.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/request.hpp"
#include "airuntime/router.hpp"
#include "airuntime/scheduler.hpp"
#include "airuntime/status.hpp"
#include "airuntime/threading.hpp"
#include "airuntime/worker.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace airuntime {

class Runtime {
  public:
    Runtime(std::unique_ptr<IRequestScheduler> scheduler, std::unique_ptr<IWorkerRouter> router,
            std::vector<std::unique_ptr<Worker>> workers,
            std::shared_ptr<const ModelRegistry> registry);
    ~Runtime();

    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    Status start();
    Status submit(const RequestPtr &request);
    void stop();

    [[nodiscard]] bool is_running() const;
    [[nodiscard]] Worker *worker(WorkerId id);
    [[nodiscard]] const Worker *worker(WorkerId id) const;

  private:
    void routing_loop();
    std::vector<WorkerSnapshot> collect_snapshots() const;
    Worker *find_worker(WorkerId id);

    mutable std::mutex mutex_;
    std::unique_ptr<IRequestScheduler> scheduler_;
    std::optional<AdmissionController> admission_;
    std::unique_ptr<IWorkerRouter> router_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::unordered_map<WorkerId, Worker *> worker_index_;
    std::shared_ptr<const ModelRegistry> registry_;
    std::optional<detail::JoinThread> routing_thread_;
    std::atomic<bool> accepting_{false};
    bool started_{false};
};

} // namespace airuntime
