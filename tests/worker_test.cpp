#include "airuntime/backend.hpp"
#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/synthetic_backend.hpp"
#include "airuntime/worker.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

using airuntime::ErrorCode;
using airuntime::IModelBackend;
using airuntime::InferenceRequest;
using airuntime::InferenceResult;
using airuntime::LruEvictionPolicy;
using airuntime::ModelRegistry;
using airuntime::ModelSpec;
using airuntime::RequestPtr;
using airuntime::RequestState;
using airuntime::Status;
using airuntime::SyntheticModelBackend;
using airuntime::SyntheticModelConfig;
using airuntime::Worker;
using airuntime::WorkerConfig;
using airuntime::WorkerState;

namespace {

class ThrowingBackend final : public IModelBackend {
  public:
    Status load(const ModelSpec &) override {
        return Status::success();
    }
    Status unload(std::string_view) override {
        return Status::success();
    }
    bool is_loaded(std::string_view) const override {
        return true;
    }
    InferenceResult infer(const InferenceRequest &) override {
        throw std::runtime_error("boom");
    }
};

std::shared_ptr<const ModelRegistry> make_registry() {
    ModelRegistry::Builder builder;
    EXPECT_TRUE(builder.add({"m1", 8}).ok());
    airuntime::Status status;
    auto registry = builder.build(status);
    EXPECT_TRUE(status.ok());
    return std::shared_ptr<const ModelRegistry>(std::move(registry));
}

std::unique_ptr<SyntheticModelBackend> make_backend(bool fail_load = false,
                                                    bool fail_infer = false) {
    auto backend = std::make_unique<SyntheticModelBackend>();
    SyntheticModelConfig config;
    config.load_latency = std::chrono::microseconds{1};
    config.fail_load = fail_load;
    config.fail_infer = fail_infer;
    backend->register_model("m1", config);
    return backend;
}

WorkerConfig make_config(std::unique_ptr<IModelBackend> backend,
                         std::shared_ptr<const ModelRegistry> registry = nullptr) {
    WorkerConfig config;
    config.worker_id = 7;
    config.queue_capacity = 4;
    config.memory_budget_bytes = 64;
    config.registry = registry ? std::move(registry) : make_registry();
    config.backend = std::move(backend);
    config.eviction_policy = std::make_unique<LruEvictionPolicy>();
    return config;
}

RequestPtr queued_request(const std::string &id) {
    auto request = std::make_shared<InferenceRequest>(id, "m1", "prompt", 2);
    EXPECT_TRUE(request->transition_to(RequestState::Queued).ok());
    return request;
}

} // namespace

TEST(WorkerTest, StableIdAndStartsIdle) {
    Worker worker(make_config(make_backend()));
    EXPECT_EQ(worker.id(), 7u);
    EXPECT_EQ(worker.state(), WorkerState::Idle);
    auto snap = worker.snapshot();
    EXPECT_EQ(snap.worker_id, 7u);
    EXPECT_EQ(snap.queue_capacity, 4u);
    EXPECT_EQ(snap.memory_budget_bytes, 64u);
}

TEST(WorkerTest, LaneStartEnqueueComplete) {
    Worker worker(make_config(make_backend()));
    ASSERT_TRUE(worker.start().ok());
    auto request = queued_request("r1");
    ASSERT_TRUE(worker.enqueue(request).ok());
    ASSERT_TRUE(request->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(request->state(), RequestState::Completed);
    EXPECT_EQ(request->result()->output->text, "synthetic:r1");
    EXPECT_EQ(worker.state(), WorkerState::Idle);
    worker.close();
    worker.join();
}

TEST(WorkerTest, SnapshotQueueAndActive) {
    Worker worker(make_config(make_backend()));
    ASSERT_TRUE(worker.start().ok());
    EXPECT_TRUE(worker.snapshot().accepting);
    auto request = queued_request("r1");
    ASSERT_TRUE(worker.enqueue(request).ok());
    ASSERT_TRUE(request->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(worker.snapshot().queue_depth, 0u);
    EXPECT_EQ(worker.snapshot().active_count, 0u);
    worker.close();
    worker.join();
}

TEST(WorkerTest, ResidencyHitAvoidsReload) {
    auto backend_ptr = make_backend();
    auto *backend = backend_ptr.get();
    Worker worker(make_config(std::move(backend_ptr)));
    ASSERT_TRUE(worker.start().ok());

    auto first = queued_request("r1");
    ASSERT_TRUE(worker.enqueue(first).ok());
    ASSERT_TRUE(first->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(backend->metrics().load_count, 1u);

    auto second = queued_request("r2");
    ASSERT_TRUE(worker.enqueue(second).ok());
    ASSERT_TRUE(second->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(backend->metrics().load_count, 1u);
    EXPECT_EQ(worker.residency_metrics().residency_hits, 1u);

    worker.close();
    worker.join();
}

TEST(WorkerTest, LoadFailureMarksFailedAndIdle) {
    Worker worker(make_config(make_backend(true, false)));
    ASSERT_TRUE(worker.start().ok());
    auto request = queued_request("r1");
    ASSERT_TRUE(worker.enqueue(request).ok());
    ASSERT_TRUE(request->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(request->state(), RequestState::Failed);
    EXPECT_EQ(request->result()->status.code, ErrorCode::ModelLoadFailed);
    EXPECT_EQ(worker.state(), WorkerState::Idle);
    worker.close();
    worker.join();
}

TEST(WorkerTest, InferenceFailureMarksFailedAndIdle) {
    Worker worker(make_config(make_backend(false, true)));
    ASSERT_TRUE(worker.start().ok());
    auto request = queued_request("r1");
    ASSERT_TRUE(worker.enqueue(request).ok());
    ASSERT_TRUE(request->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(request->state(), RequestState::Failed);
    EXPECT_EQ(worker.state(), WorkerState::Idle);
    worker.close();
    worker.join();
}

TEST(WorkerTest, UnexpectedExceptionDoesNotCrash) {
    Worker worker(make_config(std::make_unique<ThrowingBackend>()));
    ASSERT_TRUE(worker.start().ok());
    auto request = queued_request("r1");
    ASSERT_TRUE(worker.enqueue(request).ok());
    ASSERT_TRUE(request->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(request->state(), RequestState::Failed);
    EXPECT_EQ(request->result()->status.code, ErrorCode::InternalError);
    EXPECT_EQ(worker.state(), WorkerState::Idle);
    worker.close();
    worker.join();
}
