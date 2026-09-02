#include "airuntime/llama/llama_backend_runtime.hpp"

#include <ggml-backend.h>
#include <llama.h>

#include <stdexcept>

namespace airuntime {

LlamaBackendRuntime::LlamaBackendRuntime() {
    ggml_backend_load_all();
    llama_backend_init();
}

LlamaBackendRuntime::~LlamaBackendRuntime() {
    llama_backend_free();
}

std::shared_ptr<LlamaBackendRuntime> LlamaBackendRuntime::create() {
    return std::shared_ptr<LlamaBackendRuntime>(new LlamaBackendRuntime());
}

bool llama_runtime_supports_gpu_offload() {
    return llama_supports_gpu_offload();
}

} // namespace airuntime
