#include "airuntime/batch_builder.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace airuntime {

BatchBuilder::BatchBuilder(BatchBuilderConfig config) : config_(config) {
    if (config_.max_batch_size < 1) {
        throw std::invalid_argument("BatchBuilder max_batch_size must be >= 1");
    }
    if (config_.max_batch_wait.count() < 0) {
        throw std::invalid_argument("BatchBuilder max_batch_wait must be >= 0");
    }
}

const BatchBuilderConfig &BatchBuilder::config() const {
    return config_;
}

FormedBatch BatchBuilder::form(RequestPtr first, const PopUntilFn &pop_until,
                               bool allow_wait) const {
    FormedBatch formed;
    if (!first) {
        return formed;
    }

    formed.requests.push_back(std::move(first));
    const std::string model_id = formed.requests.front()->model_id();

    const auto deadline = allow_wait ? (std::chrono::steady_clock::now() + config_.max_batch_wait)
                                     : std::chrono::steady_clock::now();

    while (formed.requests.size() < config_.max_batch_size) {
        auto next = pop_until(deadline);
        if (!next.has_value()) {
            break;
        }
        if ((*next)->model_id() != model_id) {
            formed.deferred = std::move(*next);
            break;
        }
        formed.requests.push_back(std::move(*next));
    }

    return formed;
}

} // namespace airuntime
