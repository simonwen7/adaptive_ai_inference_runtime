#pragma once

#include "airuntime/model.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace airuntime {

struct EvictionCandidate {
    std::string model_id;
    std::uint64_t estimated_memory_bytes{0};
    std::uint64_t last_used{0};
    std::uint64_t estimated_load_cost{1};
    std::uint64_t use_count{0};
};

class IEvictionPolicy {
  public:
    virtual ~IEvictionPolicy() = default;

    // Returns victims in eviction order (first should be evicted first).
    // Only considers candidates already filtered to Resident by caller.
    [[nodiscard]] virtual std::vector<EvictionCandidate>
    order_victims(const std::vector<EvictionCandidate> &candidates) const = 0;
};

class LruEvictionPolicy final : public IEvictionPolicy {
  public:
    [[nodiscard]] std::vector<EvictionCandidate>
    order_victims(const std::vector<EvictionCandidate> &candidates) const override;
};

class CostAwareEvictionPolicy final : public IEvictionPolicy {
  public:
    [[nodiscard]] std::vector<EvictionCandidate>
    order_victims(const std::vector<EvictionCandidate> &candidates) const override;
};

} // namespace airuntime
