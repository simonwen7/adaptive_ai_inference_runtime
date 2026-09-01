#pragma once

#include "airuntime/request.hpp"

#include <chrono>
#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

namespace airuntime {

struct BatchBuilderConfig {
    std::size_t max_batch_size{1};
    std::chrono::microseconds max_batch_wait{0};
};

struct FormedBatch {
    std::vector<RequestPtr> requests;
    std::optional<RequestPtr> deferred;
};

// Worker-local batch formation only. Does not route, load, or call backends.
class BatchBuilder {
  public:
    using PopUntilFn =
        std::function<std::optional<RequestPtr>(std::chrono::steady_clock::time_point deadline)>;

    explicit BatchBuilder(BatchBuilderConfig config);

    [[nodiscard]] const BatchBuilderConfig &config() const;

    // Forms one contiguous same-model batch starting at `first`.
    // When allow_wait is false (e.g. lane closed), does not wait for max_batch_wait.
    FormedBatch form(RequestPtr first, const PopUntilFn &pop_until, bool allow_wait) const;

  private:
    BatchBuilderConfig config_;
};

} // namespace airuntime
