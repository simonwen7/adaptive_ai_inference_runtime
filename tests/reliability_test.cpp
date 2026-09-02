#include "airuntime/backend.hpp"
#include "airuntime/batch_builder.hpp"
#include "airuntime/bounded_queue.hpp"
#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/request.hpp"
#include "airuntime/router.hpp"
#include "airuntime/runtime.hpp"
#include "airuntime/scheduler.hpp"
#include "airuntime/synthetic_backend.hpp"
#include "airuntime/worker.hpp"
#include "airuntime/workload_aware_scheduler.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <span>
#include <thread>

using airuntime::BatchBuilder;
using airuntime::BatchBuilderConfig;
using airuntime::BoundedQueue;
using airuntime::ErrorCode;
using airuntime::FifoScheduler;
using airuntime::IModelBackend;
using airuntime::InferenceRequest;
using airuntime::InferenceResult;
using airuntime::ModelRegistry;
using airuntime::ModelSpec;
using airuntime::RequestPtr;
using airuntime::RequestSnapshot;
using airuntime::RequestState;
using airuntime::ResidencyAwareRouter;
using airuntime::Runtime;
using airuntime::Status;
using airuntime::SyntheticModelBackend;
using airuntime::SyntheticModelConfig;
using airuntime::Worker;
using airuntime::WorkerConfig;

namespace {

class BlockingInferBackend final : public IModelBackend {
  public:
    BlockingInferBackend(std::shared_ptr<std::mutex> mutex,
                         std::shared_ptr<std::condition_variable> cv, std::shared_ptr<bool> entered,
                         std::shared_ptr<bool> release,
                         std::unique_ptr<SyntheticModelBackend> inner)
        : mutex_(std::move(mutex)), cv_(std::move(cv)), entered_(std::move(entered)),
          release_(std::move(release)), inner_(std::move(inner)) {}

    Status load(const ModelSpec &model) override {
        return inner_->load(model);
    }
    Status unload(std::string_view model_id) override {
        return inner_->unload(model_id);
    }
    bool is_loaded(std::string_view model_id) const override {
        return inner_->is_loaded(model_id);
    }
    InferenceResult infer(const InferenceRequest &request) override {
        const InferenceRequest *ptr = &request;
        return infer_batch(std::span<const InferenceRequest *const>(&ptr, 1)).front();
    }
    std::vector<InferenceResult>
    infer_batch(std::span<const InferenceRequest *const> requests) override {
        {
            std::unique_lock lock(*mutex_);
            *entered_ = true;
            cv_->notify_all();
            cv_->wait(lock, [&] { return *release_; });
        }
        return inner_->infer_batch(requests);
    }

  private:
    std::shared_ptr<std::mutex> mutex_;
    std::shared_ptr<std::condition_variable> cv_;
    std::shared_ptr<bool> entered_;
    std::shared_ptr<bool> release_;
    std::unique_ptr<SyntheticModelBackend> inner_;
};

std::shared_ptr<const ModelRegistry> make_registry() {
    ModelRegistry::Builder builder;
    EXPECT_TRUE(builder.add(ModelSpec{"model-a", 1024, 1}).ok());
    Status status;
    auto registry = builder.build(status);
    EXPECT_TRUE(status.ok());
    return std::shared_ptr<const ModelRegistry>(std::move(registry));
}

std::unique_ptr<SyntheticModelBackend> make_synth() {
    auto backend = std::make_unique<SyntheticModelBackend>();
    SyntheticModelConfig config;
    backend->register_model("model-a", config);
    return backend;
}

WorkerConfig make_worker_config(std::shared_ptr<const ModelRegistry> registry,
                                std::unique_ptr<IModelBackend> backend,
                                BatchBuilderConfig batch_config = {}) {
    WorkerConfig config;
    config.worker_id = 0;
    config.queue_capacity = 8;
    config.memory_budget_bytes = 4096;
    config.batch_config = batch_config;
    config.registry = std::move(registry);
    config.backend = std::move(backend);
    config.eviction_policy = std::make_unique<airuntime::LruEvictionPolicy>();
    return config;
}

} // namespace

TEST(ReliabilityTest, CancelledAndTimedOutTerminalStates) {
    InferenceRequest cancelled("r1", "m", "p", 1);
    EXPECT_TRUE(cancelled.try_cancel());
    EXPECT_EQ(cancelled.state(), RequestState::Cancelled);

    InferenceRequest timed_out("r2", "m", "p", 1);
    EXPECT_TRUE(timed_out.try_timeout());
    EXPECT_EQ(timed_out.state(), RequestState::TimedOut);
}

TEST(ReliabilityTest, TerminalStickiness) {
    InferenceRequest request("r1", "m", "p", 1);
    ASSERT_TRUE(request.transition_to(RequestState::Queued).ok());
    ASSERT_TRUE(request.transition_to(RequestState::Running).ok());
    ASSERT_TRUE(request.try_complete(InferenceResult::success({"ok", 1})));
    EXPECT_FALSE(request.try_timeout());
    EXPECT_EQ(request.state(), RequestState::Completed);
}

TEST(ReliabilityTest, WaitPushUntilTimesOut) {
    BoundedQueue<int> queue(1);
    ASSERT_TRUE(queue.try_push(1).ok());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
    auto status = queue.wait_push_until(2, deadline);
    EXPECT_EQ(status.code, ErrorCode::TimedOut);
}

TEST(ReliabilityTest, PositiveMaxBatchWaitFormsBatch) {
    BatchBuilderConfig config;
    config.max_batch_size = 4;
    config.max_batch_wait = std::chrono::milliseconds(50);
    BatchBuilder builder(config);

    auto first = std::make_shared<InferenceRequest>("a", "model-a", "p", 1);
    std::optional<RequestPtr> second;
    std::thread consumer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        second = std::make_shared<InferenceRequest>("b", "model-a", "p", 1);
    });

    auto formed = builder.form(
        first,
        [&](std::chrono::steady_clock::time_point wait_deadline) -> std::optional<RequestPtr> {
            while (std::chrono::steady_clock::now() < wait_deadline) {
                if (second.has_value()) {
                    return second;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return std::nullopt;
        },
        true);

    consumer.join();
    EXPECT_GE(formed.requests.size(), 1u);
}

TEST(ReliabilityTest, CancelQueuedRequest) {
    auto registry = make_registry();
    std::vector<std::unique_ptr<Worker>> workers;
    workers.push_back(std::make_unique<Worker>(make_worker_config(registry, make_synth())));
    Runtime runtime(std::make_unique<FifoScheduler>(4), std::make_unique<ResidencyAwareRouter>(),
                    std::move(workers), registry);
    ASSERT_TRUE(runtime.start().ok());

    auto request = std::make_shared<InferenceRequest>("r1", "model-a", "p", 1);
    ASSERT_TRUE(runtime.submit(request).ok());
    EXPECT_TRUE(request->try_cancel());
    EXPECT_EQ(request->state(), RequestState::Cancelled);
    runtime.stop();
}

TEST(ReliabilityTest, LateBackendSuccessDiscardedAfterTimeout) {
    auto mutex = std::make_shared<std::mutex>();
    auto cv = std::make_shared<std::condition_variable>();
    auto entered = std::make_shared<bool>(false);
    auto release = std::make_shared<bool>(false);

    auto registry = make_registry();
    std::vector<std::unique_ptr<Worker>> workers;
    workers.push_back(std::make_unique<Worker>(make_worker_config(
        registry,
        std::make_unique<BlockingInferBackend>(mutex, cv, entered, release, make_synth()))));
    Runtime runtime(std::make_unique<FifoScheduler>(4), std::make_unique<ResidencyAwareRouter>(),
                    std::move(workers), registry);
    ASSERT_TRUE(runtime.start().ok());

    auto request = std::make_shared<InferenceRequest>("r1", "model-a", "p", 1);
    request->set_deadline(std::chrono::steady_clock::now() + std::chrono::milliseconds(20));
    ASSERT_TRUE(runtime.submit(request).ok());

    {
        std::unique_lock lock(*mutex);
        ASSERT_TRUE(cv->wait_for(lock, std::chrono::seconds(2), [&] { return *entered; }));
    }
    request->try_timeout();
    *release = true;
    cv->notify_all();
    ASSERT_TRUE(request->wait_for_terminal(std::chrono::seconds(2)));
    EXPECT_EQ(request->state(), RequestState::TimedOut);
    runtime.stop();
}

TEST(ReliabilityTest, ObserverSeesOrderedTransitions) {
    auto request = std::make_shared<InferenceRequest>("r1", "m", "p", 1);
    std::vector<RequestState> seen;
    std::mutex mutex;
    request->add_observer([&](RequestSnapshot snap) {
        std::lock_guard lock(mutex);
        seen.push_back(snap.state);
    });
    ASSERT_TRUE(request->transition_to(RequestState::Queued).ok());
    ASSERT_TRUE(request->transition_to(RequestState::Running).ok());
    ASSERT_TRUE(request->try_complete(InferenceResult::success({"ok", 1})));
    std::lock_guard lock(mutex);
    ASSERT_GE(seen.size(), 3u);
    EXPECT_EQ(seen.front(), RequestState::Queued);
    EXPECT_EQ(seen.back(), RequestState::Completed);
}
