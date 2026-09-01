#include "airuntime/router.hpp"

#include <limits>

namespace airuntime {

bool is_m2_feasible(const ModelSpec &model, const WorkerSnapshot &worker) {
    return worker.accepting && model.estimated_memory_bytes <= worker.memory_budget_bytes;
}

RoutingResult RoundRobinRouter::select(const ModelSpec &model,
                                       std::span<const WorkerSnapshot> workers) {
    if (workers.empty()) {
        return RoutingResult::failure(
            Status::error(ErrorCode::NoFeasibleWorker, "no workers available"));
    }

    const std::size_t n = workers.size();
    if (next_index_ >= n) {
        next_index_ = 0;
    }

    for (std::size_t offset = 0; offset < n; ++offset) {
        const std::size_t index = (next_index_ + offset) % n;
        const auto &worker = workers[index];
        if (is_m2_feasible(model, worker)) {
            next_index_ = (index + 1) % n;
            return RoutingResult::success(worker.worker_id);
        }
    }

    return RoutingResult::failure(
        Status::error(ErrorCode::NoFeasibleWorker, "no feasible worker for model"));
}

RoutingResult LeastLoadedRouter::select(const ModelSpec &model,
                                        std::span<const WorkerSnapshot> workers) {
    if (workers.empty()) {
        return RoutingResult::failure(
            Status::error(ErrorCode::NoFeasibleWorker, "no workers available"));
    }

    std::optional<WorkerId> best_id;
    std::size_t best_load = std::numeric_limits<std::size_t>::max();

    for (const auto &worker : workers) {
        if (!is_m2_feasible(model, worker)) {
            continue;
        }
        const std::size_t load = worker.queue_depth + worker.active_count;
        if (!best_id.has_value() || load < best_load ||
            (load == best_load && worker.worker_id < *best_id)) {
            best_id = worker.worker_id;
            best_load = load;
        }
    }

    if (!best_id.has_value()) {
        return RoutingResult::failure(
            Status::error(ErrorCode::NoFeasibleWorker, "no feasible worker for model"));
    }
    return RoutingResult::success(*best_id);
}

} // namespace airuntime
