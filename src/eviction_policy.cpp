#include "airuntime/eviction_policy.hpp"

#include <algorithm>

namespace airuntime {

std::vector<EvictionCandidate>
LruEvictionPolicy::order_victims(const std::vector<EvictionCandidate> &candidates) const {
    auto ordered = candidates;
    std::sort(ordered.begin(), ordered.end(),
              [](const EvictionCandidate &a, const EvictionCandidate &b) {
                  if (a.last_used != b.last_used) {
                      return a.last_used < b.last_used;
                  }
                  return a.model_id < b.model_id;
              });
    return ordered;
}

std::vector<EvictionCandidate>
CostAwareEvictionPolicy::order_victims(const std::vector<EvictionCandidate> &candidates) const {
    auto ordered = candidates;
    std::sort(ordered.begin(), ordered.end(),
              [](const EvictionCandidate &a, const EvictionCandidate &b) {
                  if (a.estimated_load_cost != b.estimated_load_cost) {
                      return a.estimated_load_cost < b.estimated_load_cost;
                  }
                  if (a.use_count != b.use_count) {
                      return a.use_count < b.use_count;
                  }
                  if (a.estimated_memory_bytes != b.estimated_memory_bytes) {
                      return a.estimated_memory_bytes > b.estimated_memory_bytes;
                  }
                  if (a.last_used != b.last_used) {
                      return a.last_used < b.last_used;
                  }
                  return a.model_id < b.model_id;
              });
    return ordered;
}

} // namespace airuntime
