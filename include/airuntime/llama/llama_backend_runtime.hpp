#pragma once

#include <memory>

namespace airuntime {

// Process-scoped RAII for ggml_backend_load_all + llama_backend_init/free.
// Must outlive all LlamaCppBackend instances that share it.
class LlamaBackendRuntime {
  public:
    LlamaBackendRuntime();
    ~LlamaBackendRuntime();

    LlamaBackendRuntime(const LlamaBackendRuntime &) = delete;
    LlamaBackendRuntime &operator=(const LlamaBackendRuntime &) = delete;

    [[nodiscard]] static std::shared_ptr<LlamaBackendRuntime> create();
};

[[nodiscard]] bool llama_runtime_supports_gpu_offload();

} // namespace airuntime
