#include "airuntime/backend.hpp"
#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/router.hpp"
#include "airuntime/runtime.hpp"
#include "airuntime/scheduler.hpp"
#include "airuntime/synthetic_backend.hpp"
#include "airuntime/worker.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

using airuntime::ErrorCode;
using airuntime::FifoScheduler;
using airuntime::IModelBackend;
using airuntime::InferenceRequest;
using airuntime::InferenceResult;
using airuntime::LruEvictionPolicy;
using airuntime::ModelRegistry;
using airuntime::RequestPtr;
using airuntime::RequestState;
using airuntime::RoundRobinRouter;
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

    Status load(const airuntime::ModelSpec &model) override {
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
    EXPECT_TRUE(builder.add({"m1", 8}).ok());
    Status status;
    auto registry = builder.build(status);
    EXPECT_TRUE(status.ok());
    return std::shared_ptr<const ModelRegistry>(std::move(registry));
}

std::unique_ptr<SyntheticModelBackend> make_synth() {
    auto backend = std::make_unique<SyntheticModelBackend>();
    backend->register_model("m1", SyntheticModelConfig{});
    return backend;
}

RequestPtr make_request(const std::string &id) {
    return std::make_shared<InferenceRequest>(id, "m1", "prompt", 1);
}

} // namespace

TEST(WorkerBackpressureTest, TryEnqueueRejectsWhenLaneFull) {
    auto mutex = std::make_shared<std::mutex>();
    auto cv = std::make_shared<std::condition_variable>();
    auto entered = std::make_shared<bool>(false);
    auto release = std::make_shared<bool>(false);

    WorkerConfig config;
    config.worker_id = 0;
    config.queue_capacity = 2;
    config.memory_budget_bytes = 64;
    config.batch_config.max_batch_size = 1;
    config.batch_config.max_batch_wait = std::chrono::microseconds{0};
    config.registry = make_registry();
    config.backend =
        std::make_unique<BlockingInferBackend>(mutex, cv, entered, release, make_synth());
    config.eviction_policy = std::make_unique<LruEvictionPolicy>();

    Worker worker(std::move(config));
    ASSERT_TRUE(worker.start().ok());

    auto r1 = make_request("r1");
    ASSERT_TRUE(r1->transition_to(RequestState::Queued).ok());
    ASSERT_TRUE(worker.try_enqueue(r1).ok());

    {
        std::unique_lock lock(*mutex);
        ASSERT_TRUE(cv->wait_for(lock, std::chrono::seconds{2}, [&] { return *entered; }));
    }

    auto r2 = make_request("r2");
    ASSERT_TRUE(r2->transition_to(RequestState::Queued).ok());
    ASSERT_TRUE(worker.try_enqueue(r2).ok());

    auto r3 = make_request("r3");
    ASSERT_TRUE(r3->transition_to(RequestState::Queued).ok());
    ASSERT_TRUE(worker.try_enqueue(r3).ok());

    auto r4 = make_request("r4");
    ASSERT_TRUE(r4->transition_to(RequestState::Queued).ok());
    auto full = worker.try_enqueue(r4);
    EXPECT_EQ(full.code, ErrorCode::QueueFull);
    EXPECT_FALSE(r4->is_terminal());

    {
        std::lock_guard lock(*mutex);
        *release = true;
    }
    cv->notify_all();

    ASSERT_TRUE(r1->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(r2->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(r3->wait_for_terminal(std::chrono::seconds{2}));
    worker.close();
    worker.join();
}

TEST(WorkerBackpressureTest, RoutingRejectsQueueFullWithoutReleasingWorker) {
    auto mutex = std::make_shared<std::mutex>();
    auto cv = std::make_shared<std::condition_variable>();
    auto entered = std::make_shared<bool>(false);
    auto release = std::make_shared<bool>(false);

    auto registry = make_registry();
    WorkerConfig config;
    config.worker_id = 0;
    config.queue_capacity = 2;
    config.memory_budget_bytes = 64;
    config.batch_config.max_batch_size = 1;
    config.batch_config.max_batch_wait = std::chrono::microseconds{0};
    config.registry = registry;
    config.backend =
        std::make_unique<BlockingInferBackend>(mutex, cv, entered, release, make_synth());
    config.eviction_policy = std::make_unique<LruEvictionPolicy>();

    std::vector<std::unique_ptr<Worker>> workers;
    workers.push_back(std::make_unique<Worker>(std::move(config)));

    Runtime runtime(std::make_unique<FifoScheduler>(32), std::make_unique<RoundRobinRouter>(),
                    std::move(workers), registry);
    ASSERT_TRUE(runtime.start().ok());

    auto r1 = make_request("r1");
    ASSERT_TRUE(runtime.submit(r1).ok());

    {
        std::unique_lock lock(*mutex);
        ASSERT_TRUE(cv->wait_for(lock, std::chrono::seconds{2}, [&] { return *entered; }));
    }

    auto r2 = make_request("r2");
    auto r3 = make_request("r3");
    ASSERT_TRUE(runtime.submit(r2).ok());
    ASSERT_TRUE(runtime.submit(r3).ok());

    auto r4 = make_request("r4");
    ASSERT_TRUE(runtime.submit(r4).ok());

    ASSERT_TRUE(r4->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(r4->state(), RequestState::Rejected);
    EXPECT_EQ(r4->result()->status.code, ErrorCode::QueueFull);
    EXPECT_FALSE(*release);

    {
        std::lock_guard lock(*mutex);
        *release = true;
    }
    cv->notify_all();

    ASSERT_TRUE(r1->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(r2->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(r3->wait_for_terminal(std::chrono::seconds{2}));
    runtime.stop();
}
