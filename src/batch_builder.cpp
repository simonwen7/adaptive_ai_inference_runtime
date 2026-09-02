#include "airuntime/batch_builder.hpp"

#include <algorithm>
#include <limits>
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

std::optional<std::chrono::steady_clock::time_point>
BatchBuilder::earliest_deadline(const std::vector<RequestPtr> &requests) {
    std::optional<std::chrono::steady_clock::time_point> earliest;
    for (const auto &request : requests) {
        if (!request) {
            continue;
        }
        const auto dl = request->deadline();
        if (!dl.has_value()) {
            continue;
        }
        if (!earliest.has_value() || *dl < *earliest) {
            earliest = dl;
        }
    }
    return earliest;
}

FormedBatch BatchBuilder::form(RequestPtr first, const PopUntilFn &pop_until, bool allow_wait,
                               std::function<bool(RequestPtr &)> on_extra_request) const {
    FormedBatch formed;
    if (!first) {
        return formed;
    }

    formed.requests.push_back(std::move(first));
    const std::string model_id = formed.requests.front()->model_id();

    auto compute_deadline = [&]() {
        const auto wait_deadline = std::chrono::steady_clock::now() + config_.max_batch_wait;
        const auto request_deadline = earliest_deadline(formed.requests);
        if (!allow_wait) {
            return std::chrono::steady_clock::now();
        }
        if (!request_deadline.has_value()) {
            return wait_deadline;
        }
        return std::min(wait_deadline, *request_deadline);
    };

    while (formed.requests.size() < config_.max_batch_size) {
        const auto deadline = compute_deadline();
        auto next = pop_until(deadline);
        if (!next.has_value()) {
            break;
        }
        if (on_extra_request && !on_extra_request(*next)) {
            formed.deferred = std::move(*next);
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
