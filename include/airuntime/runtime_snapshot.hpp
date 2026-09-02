#pragma once

#include "airuntime/model.hpp"
#include "airuntime/worker.hpp"
#include "airuntime/worker_snapshot.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace airuntime {

struct WorkerRuntimeSnapshot {
    WorkerSnapshot worker;
    WorkerBatchMetrics batch_metrics;
    ResidencyMetrics residency_metrics;
};

struct RuntimeSnapshot {
    bool accepting{false};
    bool started{false};
    std::size_t scheduler_depth{0};
    std::size_t scheduler_capacity{0};
    std::size_t worker_count{0};
    std::vector<WorkerRuntimeSnapshot> workers;
};

struct ModelRuntimeSnapshot {
    ModelSpec spec;
    std::vector<WorkerId> resident_worker_ids;
};

struct MetricsSnapshot {
    std::size_t scheduler_depth{0};
    std::size_t scheduler_capacity{0};
    std::vector<WorkerRuntimeSnapshot> workers;
};

} // namespace airuntime
