#include "airuntime/llama/llama_backend_runtime.hpp"
#include "airuntime/llama/llama_cpp_backend.hpp"
#include "airuntime/llama/llama_cpp_backend_config.hpp"
#include "airuntime/request.hpp"
#include "airuntime/status.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

using airuntime::ErrorCode;
using airuntime::InferenceRequest;
using airuntime::LlamaBackendRuntime;
using airuntime::LlamaCppBackend;
using airuntime::LlamaCppBackendConfig;
using airuntime::LlamaModelConfig;
using airuntime::ModelSpec;

namespace {

std::optional<std::filesystem::path> gguf_path_from_env() {
    const char *env = std::getenv("AIRUNTIME_TEST_GGUF");
    if (env == nullptr || env[0] == '\0') {
        return std::nullopt;
    }
    std::filesystem::path path(env);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || !std::filesystem::is_regular_file(path, ec)) {
        return std::nullopt;
    }
    return path;
}

#define AIRUNTIME_REQUIRE_GGUF(path_var)                                                           \
    const auto path_var##_opt = gguf_path_from_env();                                              \
    if (!(path_var##_opt)) {                                                                       \
        GTEST_SKIP() << "AIRUNTIME_TEST_GGUF not set or path missing";                             \
    }                                                                                              \
    const std::filesystem::path path_var = *(path_var##_opt)

LlamaCppBackendConfig make_config(const std::filesystem::path &gguf, std::size_t max_seq,
                                  std::optional<std::int32_t> gpu_layers = std::nullopt) {
    LlamaCppBackendConfig config;
    config.max_sequences = max_seq;
    LlamaModelConfig model;
    model.gguf_path = gguf;
    model.context_tokens_per_sequence = 512;
    model.n_batch = 256;
    model.n_ubatch = 256;
    if (max_seq > model.n_batch) {
        model.n_batch = static_cast<std::uint32_t>(max_seq);
        model.n_ubatch = model.n_batch;
    }
    model.n_gpu_layers = gpu_layers;
    config.models.emplace("qwen-small", std::move(model));
    return config;
}

} // namespace

TEST(LlamaRealModelTest, LoadInferUnloadReload) {
    AIRUNTIME_REQUIRE_GGUF(gguf);
    auto runtime = LlamaBackendRuntime::create();
    LlamaCppBackend backend(runtime, make_config(gguf, 1));

    std::cout << "llama_supports_gpu_offload="
              << (airuntime::llama_runtime_supports_gpu_offload() ? "true" : "false") << '\n';

    ASSERT_TRUE(backend.load(ModelSpec{"qwen-small", 1, 1}).ok());
    EXPECT_TRUE(backend.is_loaded("qwen-small"));

    auto request = std::make_shared<InferenceRequest>("r1", "qwen-small", "Hello", 8);
    auto result = backend.infer(*request);
    ASSERT_TRUE(result.ok()) << result.status.message;
    ASSERT_TRUE(result.output.has_value());
    EXPECT_LE(result.output->output_tokens, 8u);

    ASSERT_TRUE(backend.unload("qwen-small").ok());
    EXPECT_FALSE(backend.is_loaded("qwen-small"));

    ASSERT_TRUE(backend.load(ModelSpec{"qwen-small", 1, 1}).ok());
    auto request2 = std::make_shared<InferenceRequest>("r2", "qwen-small", "Hi", 4);
    auto result2 = backend.infer(*request2);
    ASSERT_TRUE(result2.ok()) << result2.status.message;
    ASSERT_TRUE(backend.unload("qwen-small").ok());
}

TEST(LlamaRealModelTest, RealMultiSequenceBatchAndCrossBatchCleanup) {
    AIRUNTIME_REQUIRE_GGUF(gguf);
    auto runtime = LlamaBackendRuntime::create();
    LlamaCppBackend backend(runtime, make_config(gguf, 2));
    ASSERT_TRUE(backend.load(ModelSpec{"qwen-small", 1, 1}).ok());

    auto a = std::make_shared<InferenceRequest>("a", "qwen-small", "Name a color.", 6);
    auto b = std::make_shared<InferenceRequest>("b", "qwen-small", "Name an animal.", 6);
    const InferenceRequest *views[2] = {a.get(), b.get()};
    auto results = backend.infer_batch(std::span<const InferenceRequest *const>(views, 2));
    ASSERT_EQ(results.size(), 2u);
    EXPECT_TRUE(results[0].ok()) << results[0].status.message;
    EXPECT_TRUE(results[1].ok()) << results[1].status.message;
    if (results[0].ok() && results[0].output) {
        EXPECT_LE(results[0].output->output_tokens, 6u);
    }
    if (results[1].ok() && results[1].output) {
        EXPECT_LE(results[1].output->output_tokens, 6u);
    }

    auto c = std::make_shared<InferenceRequest>("c", "qwen-small", "Say one word.", 4);
    auto d = std::make_shared<InferenceRequest>("d", "qwen-small", "Count to one.", 4);
    const InferenceRequest *views2[2] = {c.get(), d.get()};
    auto results2 = backend.infer_batch(std::span<const InferenceRequest *const>(views2, 2));
    ASSERT_EQ(results2.size(), 2u);
    EXPECT_TRUE(results2[0].ok()) << results2[0].status.message;
    EXPECT_TRUE(results2[1].ok()) << results2[1].status.message;
    ASSERT_TRUE(backend.unload("qwen-small").ok());
}

TEST(LlamaRealModelTest, ContextLengthExceeded) {
    AIRUNTIME_REQUIRE_GGUF(gguf);
    auto runtime = LlamaBackendRuntime::create();
    auto config = make_config(gguf, 1);
    config.models["qwen-small"].context_tokens_per_sequence = 16;
    LlamaCppBackend backend(runtime, config);
    ASSERT_TRUE(backend.load(ModelSpec{"qwen-small", 1, 1}).ok());

    const std::string long_prompt(256, 'x');
    auto request = std::make_shared<InferenceRequest>("long", "qwen-small", long_prompt, 64);
    auto result = backend.infer(*request);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status.code, ErrorCode::ContextLengthExceeded);
    ASSERT_TRUE(backend.unload("qwen-small").ok());
}

TEST(LlamaRealModelTest, CpuForcedLayersZeroSmoke) {
    AIRUNTIME_REQUIRE_GGUF(gguf);
    auto runtime = LlamaBackendRuntime::create();
    LlamaCppBackend backend(runtime, make_config(gguf, 1, /*gpu_layers=*/0));
    ASSERT_TRUE(backend.load(ModelSpec{"qwen-small", 1, 1}).ok());
    auto request = std::make_shared<InferenceRequest>("cpu", "qwen-small", "Hi", 2);
    auto result = backend.infer(*request);
    ASSERT_TRUE(result.ok()) << result.status.message;
    ASSERT_TRUE(backend.unload("qwen-small").ok());
}
