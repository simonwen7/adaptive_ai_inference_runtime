#pragma once

#include "airuntime/model.hpp"
#include "airuntime/request.hpp"
#include "airuntime/status.hpp"

#include <string_view>

namespace airuntime {

class IModelBackend {
  public:
    virtual ~IModelBackend() = default;

    virtual Status load(const ModelSpec &model) = 0;
    virtual Status unload(std::string_view model_id) = 0;
    [[nodiscard]] virtual bool is_loaded(std::string_view model_id) const = 0;
    virtual InferenceResult infer(const InferenceRequest &request) = 0;
};

} // namespace airuntime
