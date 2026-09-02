#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/request.hpp"
#include "airuntime/router.hpp"
#include "airuntime/runtime.hpp"
#include "airuntime/scheduler.hpp"
#include "airuntime/serving/request_handler.hpp"
#include "airuntime/synthetic_backend.hpp"
#include "airuntime/worker.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

using airuntime::ErrorCode;
using airuntime::FifoScheduler;
using airuntime::InferenceRequest;
using airuntime::ModelRegistry;
using airuntime::ModelSpec;
using airuntime::ResidencyAwareRouter;
using airuntime::Runtime;
using airuntime::Status;
using airuntime::SyntheticModelBackend;
using airuntime::SyntheticModelConfig;
using airuntime::Worker;
using airuntime::WorkerConfig;
using airuntime::serving::RequestHandler;

namespace {

std::shared_ptr<Runtime> make_runtime() {
    ModelRegistry::Builder builder;
    EXPECT_TRUE(builder.add(ModelSpec{"model-a", 1024, 1}).ok());
    Status status;
    auto registry = builder.build(status);
    EXPECT_TRUE(status.ok());
    auto shared_registry = std::shared_ptr<const ModelRegistry>(std::move(registry));

    auto backend = std::make_unique<SyntheticModelBackend>();
    backend->register_model("model-a", SyntheticModelConfig{});

    WorkerConfig worker_config;
    worker_config.worker_id = 0;
    worker_config.queue_capacity = 4;
    worker_config.memory_budget_bytes = 4096;
    worker_config.registry = shared_registry;
    worker_config.backend = std::move(backend);
    worker_config.eviction_policy = std::make_unique<airuntime::LruEvictionPolicy>();

    std::vector<std::unique_ptr<Worker>> workers;
    workers.push_back(std::make_unique<Worker>(std::move(worker_config)));

    auto runtime = std::make_shared<Runtime>(std::make_unique<FifoScheduler>(4),
                                             std::make_unique<ResidencyAwareRouter>(),
                                             std::move(workers), shared_registry);
    EXPECT_TRUE(runtime->start().ok());
    return runtime;
}

} // namespace

TEST(RequestHandlerTest, ParsesValidInferRequest) {
    auto runtime = make_runtime();
    RequestHandler handler(*runtime);
    std::string error;
    const auto spec = handler.parse_infer_request(
        R"({"model_id":"model-a","prompt":"hello","max_output_tokens":8,"timeout_ms":1000})",
        error);
    ASSERT_TRUE(spec.has_value());
    EXPECT_EQ(spec->model_id, "model-a");
    runtime->stop();
}

TEST(RequestHandlerTest, RejectsUnknownFields) {
    auto runtime = make_runtime();
    RequestHandler handler(*runtime);
    std::string error;
    EXPECT_FALSE(
        handler.parse_infer_request(R"({"model_id":"model-a","prompt":"x","stream":true})", error)
            .has_value());
    runtime->stop();
}

TEST(RequestHandlerTest, RequestIdGeneration) {
    auto runtime = make_runtime();
    RequestHandler handler(*runtime);
    EXPECT_EQ(handler.next_request_id(), "req-1");
    EXPECT_EQ(handler.next_request_id(), "req-2");
    runtime->stop();
}

TEST(RequestHandlerTest, HttpStatusMapping) {
    auto runtime = make_runtime();
    RequestHandler handler(*runtime);
    auto request = std::make_shared<InferenceRequest>("req-1", "model-a", "p", 1);
    request->try_reject(Status::error(ErrorCode::ModelNotFound, "missing"));
    EXPECT_EQ(handler.http_status_for_snapshot(request->snapshot()), 404);

    auto too_long = std::make_shared<InferenceRequest>("req-2", "model-a", "p", 1);
    too_long->try_reject(Status::error(ErrorCode::ContextLengthExceeded, "prompt exceeds context"));
    EXPECT_EQ(handler.http_status_for_snapshot(too_long->snapshot()), 400);
    runtime->stop();
}
