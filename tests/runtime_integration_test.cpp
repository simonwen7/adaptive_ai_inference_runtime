#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/router.hpp"
#include "airuntime/runtime.hpp"
#include "airuntime/scheduler.hpp"
#include "airuntime/synthetic_backend.hpp"
#include "airuntime/worker.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

using airuntime::ErrorCode;
using airuntime::FifoScheduler;
using airuntime::InferenceRequest;
using airuntime::LeastLoadedRouter;
using airuntime::LruEvictionPolicy;
using airuntime::ModelRegistry;
using airuntime::RequestPtr;
using airuntime::RequestState;
using airuntime::RoundRobinRouter;
using airuntime::Runtime;
using airuntime::SyntheticModelBackend;
using airuntime::SyntheticModelConfig;
using airuntime::Worker;
using airuntime::WorkerConfig;
using airuntime::WorkerId;

namespace {

std::shared_ptr<const ModelRegistry> make_registry() {
    ModelRegistry::Builder builder;
    EXPECT_TRUE(builder.add({"m1", 8}).ok());
    EXPECT_TRUE(builder.add({"big", 100}).ok());
    airuntime::Status status;
    auto registry = builder.build(status);
    EXPECT_TRUE(status.ok());
    return std::shared_ptr<const ModelRegistry>(std::move(registry));
}

std::unique_ptr<SyntheticModelBackend> make_backend() {
    auto backend = std::make_unique<SyntheticModelBackend>();
    SyntheticModelConfig config;
    config.load_latency = std::chrono::microseconds{1};
    backend->register_model("m1", config);
    backend->register_model("big", config);
    return backend;
}

WorkerConfig make_worker_config(WorkerId id, std::shared_ptr<const ModelRegistry> registry,
                                std::uint64_t budget = 64) {
    WorkerConfig config;
    config.worker_id = id;
    config.queue_capacity = 8;
    config.memory_budget_bytes = budget;
    config.registry = std::move(registry);
    config.backend = make_backend();
    config.eviction_policy = std::make_unique<LruEvictionPolicy>();
    return config;
}

std::unique_ptr<Runtime> make_runtime(std::size_t worker_count = 2) {
    auto registry = make_registry();
    std::vector<std::unique_ptr<Worker>> workers;
    for (std::size_t i = 0; i < worker_count; ++i) {
        workers.push_back(std::make_unique<Worker>(make_worker_config(i, registry)));
    }
    return std::make_unique<Runtime>(std::make_unique<FifoScheduler>(32),
                                     std::make_unique<RoundRobinRouter>(), std::move(workers),
                                     registry);
}

RequestPtr make_request(const std::string &id, const std::string &model = "m1") {
    return std::make_shared<InferenceRequest>(id, model, "prompt", 2);
}

} // namespace

TEST(RuntimeIntegrationTest, StartsAndCompletesRequest) {
    auto runtime = make_runtime();
    ASSERT_TRUE(runtime->start().ok());
    auto request = make_request("r1");
    ASSERT_TRUE(runtime->submit(request).ok());
    ASSERT_TRUE(request->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(request->state(), RequestState::Completed);
    EXPECT_EQ(request->result()->output->text, "synthetic:r1");
    runtime->stop();
}

TEST(RuntimeIntegrationTest, RoundRobinDistributesAcrossWorkers) {
    auto runtime = make_runtime(2);
    ASSERT_TRUE(runtime->start().ok());

    std::vector<RequestPtr> requests;
    for (int i = 0; i < 4; ++i) {
        auto request = make_request("r" + std::to_string(i));
        ASSERT_TRUE(runtime->submit(request).ok());
        requests.push_back(request);
    }
    for (auto &request : requests) {
        ASSERT_TRUE(request->wait_for_terminal(std::chrono::seconds{2}));
        EXPECT_EQ(request->state(), RequestState::Completed);
    }

    EXPECT_GE(runtime->worker(0)->residency_metrics().loads +
                  runtime->worker(0)->residency_metrics().residency_hits,
              1u);
    EXPECT_GE(runtime->worker(1)->residency_metrics().loads +
                  runtime->worker(1)->residency_metrics().residency_hits,
              1u);
    runtime->stop();
}

TEST(RuntimeIntegrationTest, StopDrainsQueuedWork) {
    auto runtime = make_runtime();
    ASSERT_TRUE(runtime->start().ok());
    std::vector<RequestPtr> requests;
    for (int i = 0; i < 4; ++i) {
        auto request = make_request("drain-" + std::to_string(i));
        ASSERT_TRUE(runtime->submit(request).ok());
        requests.push_back(std::move(request));
    }
    runtime->stop();
    for (auto &request : requests) {
        ASSERT_TRUE(request->wait_for_terminal(std::chrono::seconds{2}));
        EXPECT_TRUE(request->is_terminal());
    }
}

TEST(RuntimeIntegrationTest, SubmitAfterStopRejects) {
    auto runtime = make_runtime();
    ASSERT_TRUE(runtime->start().ok());
    runtime->stop();
    auto request = make_request("late");
    auto status = runtime->submit(request);
    EXPECT_EQ(status.code, ErrorCode::RuntimeStopped);
    EXPECT_EQ(request->state(), RequestState::Rejected);
}

TEST(RuntimeIntegrationTest, NoFeasibleWorkerRejectsQueued) {
    auto registry = make_registry();
    std::vector<std::unique_ptr<Worker>> workers;
    workers.push_back(std::make_unique<Worker>(make_worker_config(0, registry, 10)));
    Runtime runtime(std::make_unique<FifoScheduler>(8), std::make_unique<RoundRobinRouter>(),
                    std::move(workers), registry);
    ASSERT_TRUE(runtime.start().ok());
    auto request = make_request("too-big", "big");
    ASSERT_TRUE(runtime.submit(request).ok());
    ASSERT_TRUE(request->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(request->state(), RequestState::Rejected);
    EXPECT_EQ(request->result()->status.code, ErrorCode::NoFeasibleWorker);
    runtime.stop();
}

TEST(RuntimeIntegrationTest, LeastLoadedRouterWorksEndToEnd) {
    auto registry = make_registry();
    std::vector<std::unique_ptr<Worker>> workers;
    workers.push_back(std::make_unique<Worker>(make_worker_config(0, registry)));
    workers.push_back(std::make_unique<Worker>(make_worker_config(1, registry)));
    Runtime runtime(std::make_unique<FifoScheduler>(16), std::make_unique<LeastLoadedRouter>(),
                    std::move(workers), registry);
    ASSERT_TRUE(runtime.start().ok());
    auto request = make_request("ll-1");
    ASSERT_TRUE(runtime.submit(request).ok());
    ASSERT_TRUE(request->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(request->state(), RequestState::Completed);
    runtime.stop();
}
