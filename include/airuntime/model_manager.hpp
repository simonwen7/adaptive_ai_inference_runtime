#pragma once

#include "airuntime/backend.hpp"
#include "airuntime/eviction_policy.hpp"
#include "airuntime/model.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/resource_manager.hpp"
#include "airuntime/status.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace airuntime {

struct ResidencyMetrics {
    std::uint64_t residency_hits{0};
    std::uint64_t residency_misses{0};
    std::uint64_t loads{0};
    std::uint64_t unloads{0};
    std::uint64_t evictions{0};
    std::uint64_t reloads{0};
};

class ModelManager {
  public:
    ModelManager(std::shared_ptr<const ModelRegistry> registry, IModelBackend &backend,
                 std::uint64_t memory_budget_bytes,
                 std::unique_ptr<IEvictionPolicy> eviction_policy);

    Status ensure_resident(std::string_view model_id);

    [[nodiscard]] ModelState model_state(std::string_view model_id) const;
    [[nodiscard]] ResidencyMetrics metrics() const;
    [[nodiscard]] std::uint64_t memory_budget_bytes() const;
    [[nodiscard]] std::uint64_t memory_used_bytes() const;

  private:
    struct Entry {
        ModelState state{ModelState::Unloaded};
        std::uint64_t estimated_memory_bytes{0};
        std::uint64_t last_used{0};
        bool previously_loaded{false};
    };

    Status free_memory_for(std::uint64_t needed_bytes);
    Status evict_one(const EvictionCandidate &victim);

    std::shared_ptr<const ModelRegistry> registry_;
    IModelBackend &backend_;
    ResourceManager resources_;
    std::unique_ptr<IEvictionPolicy> eviction_policy_;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> entries_;
    std::uint64_t touch_counter_{0};
    ResidencyMetrics metrics_;
};

} // namespace airuntime
