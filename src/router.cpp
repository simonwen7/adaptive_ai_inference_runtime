#include "airuntime/router.hpp"

#include <algorithm>
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

namespace {

int residency_tier(const ModelSpec &model, const WorkerSnapshot &worker) {
    const bool resident = std::binary_search(worker.resident_model_ids.begin(),
                                             worker.resident_model_ids.end(), model.model_id);
    if (resident) {
        return 0;
    }
    const std::uint64_t free_bytes = worker.memory_budget_bytes >= worker.memory_used_bytes
                                         ? worker.memory_budget_bytes - worker.memory_used_bytes
                                         : 0;
    if (model.estimated_memory_bytes <= free_bytes) {
        return 1;
    }
    return 2;
}

} // namespace

RoutingResult ResidencyAwareRouter::select(const ModelSpec &model,
                                           std::span<const WorkerSnapshot> workers) {
    if (workers.empty()) {
        return RoutingResult::failure(
            Status::error(ErrorCode::NoFeasibleWorker, "no workers available"));
    }

    std::optional<WorkerId> best_id;
    int best_tier = 3;
    std::size_t best_load = std::numeric_limits<std::size_t>::max();

    for (const auto &worker : workers) {
        if (!is_m2_feasible(model, worker)) {
            continue;
        }
        const int tier = residency_tier(model, worker);
        const std::size_t load = worker.queue_depth + worker.active_count;
        if (!best_id.has_value() || tier < best_tier || (tier == best_tier && load < best_load) ||
            (tier == best_tier && load == best_load && worker.worker_id < *best_id)) {
            best_id = worker.worker_id;
            best_tier = tier;
            best_load = load;
        }
    }

    if (!best_id.has_value()) {
        return RoutingResult::failure(
            Status::error(ErrorCode::NoFeasibleWorker, "no feasible worker for model"));
    }

    if (best_tier == 0) {
        ++metrics_.resident_selections;
    } else if (best_tier == 1) {
        ++metrics_.no_eviction_load_selections;
    } else {
        ++metrics_.eviction_required_selections;
    }

    return RoutingResult::success(*best_id);
}

ResidencyAwareRouterMetrics ResidencyAwareRouter::metrics() const {
    return metrics_;
}

} // namespace airuntime
