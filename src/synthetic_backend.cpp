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
    std::lock_guard lock(mutex_);
    auto it = models_.find(request.model_id());
    if (it == models_.end()) {
        return InferenceResult::failure(
            Status::error(ErrorCode::ModelNotFound, "model profile not registered"));
    }

    auto &entry = it->second;
    if (entry.state != ModelState::Resident) {
        return InferenceResult::failure(
            Status::error(ErrorCode::InferenceFailed, "model is not resident"));
    }

    const auto output_tokens = request.max_output_tokens();
    const auto simulated_cost =
        entry.config.prefill_latency + entry.config.per_token_latency * output_tokens;

    metrics_.last_simulated_inference_cost = simulated_cost;
    metrics_.total_simulated_inference_cost += simulated_cost;
    ++metrics_.inference_count;

    if (entry.config.fail_infer) {
        return InferenceResult::failure(
            Status::error(ErrorCode::InferenceFailed, "synthetic inference failure"));
    }

    InferenceOutput output;
    output.text = "synthetic:" + request.request_id();
    output.output_tokens = output_tokens;
    return InferenceResult::success(std::move(output));
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
