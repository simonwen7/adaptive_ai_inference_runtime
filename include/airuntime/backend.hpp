#pragma once

#include "airuntime/model.hpp"
#include "airuntime/request.hpp"
#include "airuntime/status.hpp"

#include <span>
#include <string_view>
#include <vector>

namespace airuntime {

class IModelBackend {
  public:
    virtual ~IModelBackend() = default;

    virtual Status load(const ModelSpec &model) = 0;
    virtual Status unload(std::string_view model_id) = 0;
    [[nodiscard]] virtual bool is_loaded(std::string_view model_id) const = 0;
    virtual InferenceResult infer(const InferenceRequest &request) = 0;

    // Ordered non-owning batch. result[i] corresponds to requests[i].
    // Must implement real batch semantics (not a default loop over infer).
    virtual std::vector<InferenceResult>
    infer_batch(std::span<const InferenceRequest *const> requests) = 0;
};

} // namespace airuntime
