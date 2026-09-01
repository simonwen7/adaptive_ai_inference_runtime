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

} // namespace airuntime
