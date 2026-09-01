#include "airuntime/synthetic_backend.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <span>
#include <vector>

using airuntime::ErrorCode;
using airuntime::InferenceRequest;
using airuntime::ModelSpec;
using airuntime::ModelState;
using airuntime::SyntheticModelBackend;
using airuntime::SyntheticModelConfig;

namespace {

SyntheticModelConfig make_config() {
    SyntheticModelConfig config;
    config.load_latency = std::chrono::microseconds{100};
    config.prefill_latency = std::chrono::microseconds{40};
    config.per_token_latency = std::chrono::microseconds{5};
    return config;
}

} // namespace

TEST(SyntheticBackendTest, KnownModelLoadAndState) {
    SyntheticModelBackend backend;
    backend.register_model("m1", make_config());

    EXPECT_EQ(backend.model_state("m1"), ModelState::Unloaded);
    ASSERT_TRUE(backend.load(ModelSpec{"m1"}).ok());
    EXPECT_EQ(backend.model_state("m1"), ModelState::Resident);
    EXPECT_TRUE(backend.is_loaded("m1"));
}

TEST(SyntheticBackendTest, UnknownModel) {
    SyntheticModelBackend backend;
    auto status = backend.load(ModelSpec{"missing"});
    EXPECT_EQ(status.code, ErrorCode::ModelNotFound);
}

TEST(SyntheticBackendTest, Unload) {
    SyntheticModelBackend backend;
    backend.register_model("m1", make_config());
    ASSERT_TRUE(backend.load(ModelSpec{"m1"}).ok());
    ASSERT_TRUE(backend.unload("m1").ok());
    EXPECT_FALSE(backend.is_loaded("m1"));
    EXPECT_EQ(backend.model_state("m1"), ModelState::Unloaded);
}

TEST(SyntheticBackendTest, DeterministicInferenceAndCost) {
    SyntheticModelBackend backend;
    backend.register_model("m1", make_config());
    ASSERT_TRUE(backend.load(ModelSpec{"m1"}).ok());

    InferenceRequest request("req-42", "m1", "hello", 3);
    auto first = backend.infer(request);
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(first.output.has_value());
    EXPECT_EQ(first.output->text, "synthetic:req-42");
    EXPECT_EQ(first.output->output_tokens, 3u);

    auto metrics = backend.metrics();
    EXPECT_EQ(metrics.load_count, 1u);
    EXPECT_EQ(metrics.inference_count, 1u);
    EXPECT_EQ(metrics.batch_inference_count, 1u);
    EXPECT_EQ(metrics.last_simulated_load_cost, std::chrono::microseconds{100});
    EXPECT_EQ(metrics.last_simulated_inference_cost, std::chrono::microseconds{55});
    EXPECT_EQ(metrics.last_simulated_batch_cost, std::chrono::microseconds{55});

    auto second = backend.infer(request);
    ASSERT_TRUE(second.ok());
    EXPECT_EQ(second.output->text, first.output->text);
    EXPECT_EQ(backend.metrics().inference_count, 2u);
    EXPECT_EQ(backend.metrics().total_simulated_inference_cost, std::chrono::microseconds{110});
}

TEST(SyntheticBackendTest, ExplicitLoadFailure) {
    SyntheticModelBackend backend;
    auto config = make_config();
    config.fail_load = true;
    backend.register_model("m1", config);

    auto status = backend.load(ModelSpec{"m1"});
    EXPECT_EQ(status.code, ErrorCode::ModelLoadFailed);
    EXPECT_EQ(backend.model_state("m1"), ModelState::Failed);
    EXPECT_FALSE(backend.is_loaded("m1"));
}

TEST(SyntheticBackendTest, ExplicitInferenceFailure) {
    SyntheticModelBackend backend;
    auto config = make_config();
    config.fail_infer = true;
    backend.register_model("m1", config);
    ASSERT_TRUE(backend.load(ModelSpec{"m1"}).ok());

    InferenceRequest request("r1", "m1", "p", 2);
    auto result = backend.infer(request);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status.code, ErrorCode::InferenceFailed);
}

TEST(SyntheticBackendTest, BatchCostAmortizesPrefill) {
    SyntheticModelBackend backend;
    backend.register_model("m1", make_config());
    ASSERT_TRUE(backend.load(ModelSpec{"m1"}).ok());

    InferenceRequest r1("r1", "m1", "p", 2);
    InferenceRequest r2("r2", "m1", "p", 3);
    InferenceRequest r3("r3", "m1", "p", 1);
    // tokens = 2+3+1 = 6
    // batch_cost = 40 + 5*6 = 70
    // serial would be 3*40 + 5*6 = 150
    const InferenceRequest *ptrs[] = {&r1, &r2, &r3};
    auto results = backend.infer_batch(ptrs);
    ASSERT_EQ(results.size(), 3u);
    EXPECT_TRUE(results[0].ok());
    EXPECT_TRUE(results[1].ok());
    EXPECT_TRUE(results[2].ok());
    EXPECT_EQ(results[0].output->text, "synthetic:r1");
    EXPECT_EQ(results[1].output->text, "synthetic:r2");
    EXPECT_EQ(results[2].output->text, "synthetic:r3");

    auto metrics = backend.metrics();
    EXPECT_EQ(metrics.last_batch_size, 3u);
    EXPECT_EQ(metrics.batch_inference_count, 1u);
    EXPECT_EQ(metrics.last_simulated_batch_cost, std::chrono::microseconds{70});

    const auto serial = std::chrono::microseconds{3 * 40 + 5 * 6};
    EXPECT_LT(metrics.last_simulated_batch_cost, serial);
}

TEST(SyntheticBackendTest, BatchSizeOneMatchesSingleCost) {
    SyntheticModelBackend backend;
    backend.register_model("m1", make_config());
    ASSERT_TRUE(backend.load(ModelSpec{"m1"}).ok());
    InferenceRequest r("r1", "m1", "p", 3);
    const InferenceRequest *ptr = &r;
    auto results = backend.infer_batch(std::span<const InferenceRequest *const>(&ptr, 1));
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(backend.metrics().last_simulated_batch_cost, std::chrono::microseconds{55});
}
