#pragma once

#include "airuntime/status.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

namespace airuntime {

struct LlamaModelConfig {
    std::filesystem::path gguf_path;
    std::uint32_t context_tokens_per_sequence{2048};
    std::uint32_t n_batch{512};
    std::uint32_t n_ubatch{512};
    // unset: auto (-1 if GPU offload supported, else 0)
    // 0: CPU; -1: all layers; positive: exact layer count
    std::optional<std::int32_t> n_gpu_layers;
    std::optional<std::int32_t> n_threads;
    std::optional<std::int32_t> n_threads_batch;
};

struct LlamaCppBackendConfig {
    std::unordered_map<std::string, LlamaModelConfig> models;
    std::size_t max_sequences{1};

    [[nodiscard]] Status validate() const;
};

} // namespace airuntime
