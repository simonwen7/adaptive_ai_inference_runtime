#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace airuntime {

using WorkerId = std::size_t;

struct WorkerSnapshot {
    WorkerId worker_id{0};
    std::size_t queue_depth{0};
    std::size_t queue_capacity{0};
    std::size_t active_count{0};
    bool accepting{false};
    std::uint64_t memory_budget_bytes{0};
    std::uint64_t memory_used_bytes{0};
    // Sorted immutable copy of currently resident model IDs.
    std::vector<std::string> resident_model_ids;
};

} // namespace airuntime
