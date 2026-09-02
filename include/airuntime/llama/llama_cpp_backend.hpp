#pragma once

#include "airuntime/backend.hpp"
#include "airuntime/llama/llama_backend_runtime.hpp"
#include "airuntime/llama/llama_cpp_backend_config.hpp"

#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace airuntime {

class LlamaCppBackend final : public IModelBackend {
  public:
    LlamaCppBackend(std::shared_ptr<LlamaBackendRuntime> runtime, LlamaCppBackendConfig config);
    ~LlamaCppBackend() override;

    LlamaCppBackend(const LlamaCppBackend &) = delete;
    LlamaCppBackend &operator=(const LlamaCppBackend &) = delete;

    Status load(const ModelSpec &model) override;
    Status unload(std::string_view model_id) override;
    [[nodiscard]] bool is_loaded(std::string_view model_id) const override;
    InferenceResult infer(const InferenceRequest &request) override;
    std::vector<InferenceResult>
    infer_batch(std::span<const InferenceRequest *const> requests) override;

    [[nodiscard]] bool supports_gpu_offload() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace airuntime
