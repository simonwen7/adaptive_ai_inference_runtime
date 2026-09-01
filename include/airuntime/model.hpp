#pragma once

#include <cstdint>
#include <string>

namespace airuntime {

struct ModelSpec {
    std::string model_id;
    std::uint64_t estimated_memory_bytes{0};
    // Abstract cost units for residency/eviction policy — not wall-clock milliseconds.
    std::uint64_t estimated_load_cost{1};
};

enum class ModelState { Unloaded, Loading, Resident, Evicting, Failed };

} // namespace airuntime
