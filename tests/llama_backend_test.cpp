#include "airuntime/llama/llama_backend_runtime.hpp"
#include "airuntime/llama/llama_cpp_backend.hpp"
#include "airuntime/llama/llama_cpp_backend_config.hpp"
#include "airuntime/status.hpp"

#include <gtest/gtest.h>

#include <filesystem>

using airuntime::ErrorCode;
using airuntime::LlamaBackendRuntime;
using airuntime::LlamaCppBackend;
using airuntime::LlamaCppBackendConfig;
using airuntime::LlamaModelConfig;
using airuntime::ModelSpec;

TEST(LlamaBackendConfigTest, RejectsEmptyModels) {
    LlamaCppBackendConfig config;
    config.max_sequences = 1;
    EXPECT_EQ(config.validate().code, ErrorCode::InternalError);
}

TEST(LlamaBackendConfigTest, RejectsZeroMaxSequences) {
    LlamaCppBackendConfig config;
    config.max_sequences = 0;
    LlamaModelConfig model;
    model.gguf_path = "/tmp/missing.gguf";
    config.models.emplace("m", model);
    EXPECT_EQ(config.validate().code, ErrorCode::InternalError);
}

TEST(LlamaBackendConfigTest, RejectsNBatchLessThanMaxSequences) {
    LlamaCppBackendConfig config;
    config.max_sequences = 4;
    LlamaModelConfig model;
    model.gguf_path = "/tmp/missing.gguf";
    model.n_batch = 2;
    model.n_ubatch = 2;
    config.models.emplace("m", model);
    EXPECT_EQ(config.validate().code, ErrorCode::InternalError);
}

TEST(LlamaBackendConfigTest, RejectsNUbatchGreaterThanNBatch) {
    LlamaCppBackendConfig config;
    config.max_sequences = 1;
    LlamaModelConfig model;
    model.gguf_path = "/tmp/missing.gguf";
    model.n_batch = 8;
    model.n_ubatch = 16;
    config.models.emplace("m", model);
    EXPECT_EQ(config.validate().code, ErrorCode::InternalError);
}

TEST(LlamaBackendConfigTest, RejectsZeroContext) {
    LlamaCppBackendConfig config;
    config.max_sequences = 1;
    LlamaModelConfig model;
    model.gguf_path = "/tmp/missing.gguf";
    model.context_tokens_per_sequence = 0;
    config.models.emplace("m", model);
    EXPECT_EQ(config.validate().code, ErrorCode::InternalError);
}

TEST(LlamaBackendConfigTest, RejectsNonPositiveThreads) {
    LlamaCppBackendConfig config;
    config.max_sequences = 1;
    LlamaModelConfig model;
    model.gguf_path = "/tmp/missing.gguf";
    model.n_threads = 0;
    config.models.emplace("m", model);
    EXPECT_EQ(config.validate().code, ErrorCode::InternalError);
}

TEST(LlamaBackendConfigTest, AcceptsValidConfig) {
    LlamaCppBackendConfig config;
    config.max_sequences = 2;
    LlamaModelConfig model;
    model.gguf_path = "/tmp/missing.gguf";
    model.n_batch = 8;
    model.n_ubatch = 8;
    config.models.emplace("m", model);
    EXPECT_TRUE(config.validate().ok());
}

TEST(LlamaBackendTest, MissingConfiguredModelAndMissingFile) {
    auto runtime = LlamaBackendRuntime::create();
    LlamaCppBackendConfig config;
    config.max_sequences = 1;
    LlamaModelConfig model;
    model.gguf_path = "/tmp/airuntime-definitely-missing-model.gguf";
    config.models.emplace("qwen", model);
    LlamaCppBackend backend(runtime, config);

    EXPECT_FALSE(backend.is_loaded("qwen"));
    EXPECT_EQ(backend.load(ModelSpec{"other", 1, 1}).code, ErrorCode::ModelNotFound);
    EXPECT_FALSE(backend.is_loaded("other"));

    const auto status = backend.load(ModelSpec{"qwen", 1, 1});
    EXPECT_EQ(status.code, ErrorCode::ModelLoadFailed);
    EXPECT_FALSE(backend.is_loaded("qwen"));
}

TEST(LlamaBackendTest, ContextSizingOverflowRejected) {
    auto runtime = LlamaBackendRuntime::create();
    LlamaCppBackendConfig config;
    config.max_sequences = 2;
    LlamaModelConfig model;
    // Any existing regular file is enough to pass the path check before n_ctx overflow.
    model.gguf_path = __FILE__;
    model.context_tokens_per_sequence = 0xFFFFFFFFu;
    model.n_batch = 8;
    model.n_ubatch = 8;
    config.models.emplace("qwen", model);
    ASSERT_TRUE(config.validate().ok());
    LlamaCppBackend backend(runtime, config);
    const auto status = backend.load(ModelSpec{"qwen", 1, 1});
    EXPECT_EQ(status.code, ErrorCode::ModelLoadFailed);
    EXPECT_FALSE(backend.is_loaded("qwen"));
}
