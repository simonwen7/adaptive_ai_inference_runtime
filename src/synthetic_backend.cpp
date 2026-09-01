#include "airuntime/synthetic_backend.hpp"

#include <utility>

namespace airuntime {

void SyntheticModelBackend::register_model(std::string model_id, SyntheticModelConfig config) {
    std::lock_guard lock(mutex_);
    models_[std::move(model_id)] = ModelEntry{std::move(config), ModelState::Unloaded};
}

Status SyntheticModelBackend::load(const ModelSpec &model) {
    std::lock_guard lock(mutex_);
    auto it = models_.find(model.model_id);
    if (it == models_.end()) {
        return Status::error(ErrorCode::ModelNotFound, "model profile not registered");
    }

    auto &entry = it->second;
    if (entry.state == ModelState::Resident) {
        return Status::success();
    }

    entry.state = ModelState::Loading;
    metrics_.last_simulated_load_cost = entry.config.load_latency;
    metrics_.total_simulated_load_cost += entry.config.load_latency;
    ++metrics_.load_count;

    if (entry.config.fail_load) {
        entry.state = ModelState::Failed;
        return Status::error(ErrorCode::ModelLoadFailed, "synthetic load failure");
    }

    entry.state = ModelState::Resident;
    return Status::success();
}

Status SyntheticModelBackend::unload(std::string_view model_id) {
    std::lock_guard lock(mutex_);
    auto it = models_.find(std::string(model_id));
    if (it == models_.end()) {
        return Status::error(ErrorCode::ModelNotFound, "model profile not registered");
    }

    if (it->second.state == ModelState::Resident) {
        it->second.state = ModelState::Unloaded;
    } else if (it->second.state == ModelState::Failed) {
        it->second.state = ModelState::Unloaded;
    }
    return Status::success();
}

bool SyntheticModelBackend::is_loaded(std::string_view model_id) const {
    std::lock_guard lock(mutex_);
    auto it = models_.find(std::string(model_id));
    if (it == models_.end()) {
        return false;
    }
    return it->second.state == ModelState::Resident;
}

InferenceResult SyntheticModelBackend::infer(const InferenceRequest &request) {
    const InferenceRequest *ptr = &request;
    auto results = infer_batch(std::span<const InferenceRequest *const>(&ptr, 1));
    if (results.size() != 1) {
        return InferenceResult::failure(
            Status::error(ErrorCode::InternalError, "synthetic infer_batch cardinality error"));
    }
    return std::move(results.front());
}

std::vector<InferenceResult>
SyntheticModelBackend::infer_batch(std::span<const InferenceRequest *const> requests) {
    std::lock_guard lock(mutex_);
    std::vector<InferenceResult> results;
    results.reserve(requests.size());

    if (requests.empty()) {
        return results;
    }

    // Real synthetic batching: one shared prefill + sum of per-token costs.
    // batch_cost(N) = prefill + per_token * sum(max_output_tokens)
    // Not a loop of independent infer() serial prefills.
    const InferenceRequest *first = requests.front();
    if (first == nullptr) {
        for (std::size_t i = 0; i < requests.size(); ++i) {
            results.push_back(InferenceResult::failure(
                Status::error(ErrorCode::InternalError, "null request in batch")));
        }
        return results;
    }

    auto it = models_.find(first->model_id());
    if (it == models_.end()) {
        for (std::size_t i = 0; i < requests.size(); ++i) {
            results.push_back(InferenceResult::failure(
                Status::error(ErrorCode::ModelNotFound, "model profile not registered")));
        }
        return results;
    }

    auto &entry = it->second;
    if (entry.state != ModelState::Resident) {
        for (std::size_t i = 0; i < requests.size(); ++i) {
            results.push_back(InferenceResult::failure(
                Status::error(ErrorCode::InferenceFailed, "model is not resident")));
        }
        return results;
    }

    std::size_t total_tokens = 0;
    for (const InferenceRequest *req : requests) {
        if (req == nullptr || req->model_id() != first->model_id()) {
            for (std::size_t i = 0; i < requests.size(); ++i) {
                results.push_back(InferenceResult::failure(
                    Status::error(ErrorCode::InternalError, "incompatible batch member")));
            }
            return results;
        }
        total_tokens += req->max_output_tokens();
    }

    const auto batch_cost =
        entry.config.prefill_latency + entry.config.per_token_latency * total_tokens;

    metrics_.last_simulated_batch_cost = batch_cost;
    metrics_.total_simulated_batch_cost += batch_cost;
    metrics_.last_batch_size = requests.size();
    ++metrics_.batch_inference_count;

    // Keep single-inference metrics coherent for batch size 1 / derived infer().
    metrics_.last_simulated_inference_cost = batch_cost;
    metrics_.total_simulated_inference_cost += batch_cost;
    metrics_.inference_count += requests.size();

    for (const InferenceRequest *req : requests) {
        if (entry.config.fail_infer) {
            results.push_back(InferenceResult::failure(
                Status::error(ErrorCode::InferenceFailed, "synthetic inference failure")));
            continue;
        }
        InferenceOutput output;
        output.text = "synthetic:" + req->request_id();
        output.output_tokens = req->max_output_tokens();
        results.push_back(InferenceResult::success(std::move(output)));
    }

    return results;
}

ModelState SyntheticModelBackend::model_state(std::string_view model_id) const {
    std::lock_guard lock(mutex_);
    auto it = models_.find(std::string(model_id));
    if (it == models_.end()) {
        return ModelState::Unloaded;
    }
    return it->second.state;
}

SyntheticBackendMetrics SyntheticModelBackend::metrics() const {
    std::lock_guard lock(mutex_);
    return metrics_;
}

} // namespace airuntime
