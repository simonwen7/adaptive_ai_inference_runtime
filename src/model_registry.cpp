#include "airuntime/model_registry.hpp"

#include <utility>

namespace airuntime {

ModelRegistry::ModelRegistry(std::unordered_map<std::string, ModelSpec> models)
    : models_(std::move(models)) {}

Status ModelRegistry::Builder::add(ModelSpec spec) {
    if (spec.model_id.empty()) {
        return Status::error(ErrorCode::InternalError, "empty model_id");
    }
    if (spec.estimated_memory_bytes == 0) {
        return Status::error(ErrorCode::InternalError, "estimated_memory_bytes must be > 0");
    }
    if (models_.contains(spec.model_id)) {
        return Status::error(ErrorCode::InternalError, "duplicate model_id");
    }
    models_.emplace(spec.model_id, std::move(spec));
    return Status::success();
}

std::unique_ptr<const ModelRegistry> ModelRegistry::Builder::build(Status &status) const {
    if (models_.empty()) {
        status = Status::error(ErrorCode::InternalError, "empty model registry");
        return nullptr;
    }
    status = Status::success();
    return std::unique_ptr<const ModelRegistry>(
        new ModelRegistry(std::unordered_map<std::string, ModelSpec>(models_)));
}

Status ModelRegistry::find(std::string_view model_id, ModelSpec &out) const {
    auto it = models_.find(std::string(model_id));
    if (it == models_.end()) {
        return Status::error(ErrorCode::ModelNotFound, "model not found in registry");
    }
    out = it->second;
    return Status::success();
}

bool ModelRegistry::contains(std::string_view model_id) const {
    return models_.contains(std::string(model_id));
}

std::size_t ModelRegistry::size() const {
    return models_.size();
}

} // namespace airuntime
