#pragma once

#include "airuntime/model.hpp"
#include "airuntime/status.hpp"
#include "airuntime/worker_snapshot.hpp"

#include <optional>
#include <span>

namespace airuntime {

struct RoutingResult {
    Status status;
    std::optional<WorkerId> worker_id;

    static RoutingResult success(WorkerId id) {
        RoutingResult result;
        result.status = Status::success();
        result.worker_id = id;
        return result;
    }

    static RoutingResult failure(Status status_in) {
        RoutingResult result;
        result.status = std::move(status_in);
        return result;
    }

    [[nodiscard]] bool ok() const {
        return status.ok() && worker_id.has_value();
    }
};

class IWorkerRouter {
  public:
    virtual ~IWorkerRouter() = default;

    virtual RoutingResult select(const ModelSpec &model,
                                 std::span<const WorkerSnapshot> workers) = 0;
};

class RoundRobinRouter final : public IWorkerRouter {
  public:
    RoutingResult select(const ModelSpec &model, std::span<const WorkerSnapshot> workers) override;

  private:
    std::size_t next_index_{0};
};

class LeastLoadedRouter final : public IWorkerRouter {
  public:
    RoutingResult select(const ModelSpec &model, std::span<const WorkerSnapshot> workers) override;
};

struct ResidencyAwareRouterMetrics {
    std::uint64_t resident_selections{0};
    std::uint64_t no_eviction_load_selections{0};
    std::uint64_t eviction_required_selections{0};
};

class ResidencyAwareRouter final : public IWorkerRouter {
  public:
    RoutingResult select(const ModelSpec &model, std::span<const WorkerSnapshot> workers) override;

    [[nodiscard]] ResidencyAwareRouterMetrics metrics() const;

  private:
    ResidencyAwareRouterMetrics metrics_;
};

[[nodiscard]] bool is_m2_feasible(const ModelSpec &model, const WorkerSnapshot &worker);

} // namespace airuntime
