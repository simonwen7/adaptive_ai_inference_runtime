#include "airuntime/backend.hpp"
#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_manager.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/router.hpp"
#include "airuntime/runtime.hpp"
#include "airuntime/scheduler.hpp"
#include "airuntime/synthetic_backend.hpp"
#include "airuntime/worker.hpp"
#include "airuntime/workload_aware_scheduler.hpp"

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

using airuntime::CostAwareEvictionPolicy;
using airuntime::ErrorCode;
using airuntime::FifoScheduler;
using airuntime::IModelBackend;
using airuntime::InferenceRequest;
using airuntime::InferenceResult;
using airuntime::LruEvictionPolicy;
using airuntime::ModelManager;
using airuntime::ModelRegistry;
using airuntime::ModelSpec;
using airuntime::ModelState;
using airuntime::RequestPtr;
using airuntime::RequestState;
using airuntime::ResidencyAwareRouter;
using airuntime::Runtime;
using airuntime::Status;
using airuntime::SyntheticModelBackend;
using airuntime::SyntheticModelConfig;
using airuntime::Worker;
using airuntime::WorkerConfig;
using airuntime::WorkerId;
using airuntime::WorkloadAwareScheduler;
using airuntime::WorkloadAwareSchedulerConfig;

namespace {

class FailUnloadBackend final : public IModelBackend {
  public:
    explicit FailUnloadBackend(std::unique_ptr<SyntheticModelBackend> inner)
        : inner_(std::move(inner)) {}

    Status load(const ModelSpec &model) override {
        return inner_->load(model);
    }
    Status unload(std::string_view model_id) override {
        if (fail_unload_) {
            return Status::error(ErrorCode::InternalError, "synthetic unload failure");
        }
        return inner_->unload(model_id);
    }
    bool is_loaded(std::string_view model_id) const override {
        return inner_->is_loaded(model_id);
    }
    InferenceResult infer(const InferenceRequest &request) override {
        return inner_->infer(request);
    }
    std::vector<InferenceResult>
    infer_batch(std::span<const InferenceRequest *const> requests) override {
        return inner_->infer_batch(requests);
    }

    void set_fail_unload(bool value) {
        fail_unload_ = value;
    }

  private:
    std::unique_ptr<SyntheticModelBackend> inner_;
    bool fail_unload_{true};
};

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

std::shared_ptr<const ModelRegistry> make_registry(std::initializer_list<ModelSpec> specs) {
    ModelRegistry::Builder builder;
    for (const auto &spec : specs) {
        EXPECT_TRUE(builder.add(spec).ok());
    }
    Status status;
    auto registry = builder.build(status);
    EXPECT_TRUE(status.ok());
    return std::shared_ptr<const ModelRegistry>(std::move(registry));
}

std::unique_ptr<SyntheticModelBackend> make_synth(std::initializer_list<const char *> ids) {
    auto backend = std::make_unique<SyntheticModelBackend>();
    SyntheticModelConfig config;
    config.prefill_latency = std::chrono::microseconds{10};
    config.per_token_latency = std::chrono::microseconds{1};
    for (const char *id : ids) {
        backend->register_model(id, config);
    }
    return backend;
}

RequestPtr make_queued(const std::string &id, const std::string &model) {
    auto request = std::make_shared<InferenceRequest>(id, model, "p", 1);
    return request;
}

} // namespace

TEST(ModelManagerUnloadFailureTest, UnloadFailureRestoresResidentAndMemory) {
    auto registry = make_registry({{"A", 5, 10}, {"B", 6, 1}, {"C", 4, 1}});
    auto fail_backend = std::make_unique<FailUnloadBackend>(make_synth({"A", "B", "C"}));
    auto *backend = fail_backend.get();
    ModelManager manager(registry, *fail_backend, 11, std::make_unique<LruEvictionPolicy>());

    ASSERT_TRUE(manager.ensure_resident("A").ok());
    ASSERT_TRUE(manager.ensure_resident("B").ok());
    EXPECT_EQ(manager.memory_used_bytes(), 11u);

    auto before_evictions = manager.metrics().evictions;
    auto before_unloads = manager.metrics().unloads;

    backend->set_fail_unload(true);
    auto status = manager.ensure_resident("C"); // needs eviction
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code, ErrorCode::InternalError);
    EXPECT_EQ(manager.metrics().evictions, before_evictions);
    EXPECT_EQ(manager.metrics().unloads, before_unloads);
    EXPECT_EQ(manager.memory_used_bytes(), 11u);
    // Victim should be restored to Resident (LRU oldest among A,B).
    EXPECT_TRUE(manager.model_state("A") == ModelState::Resident ||
                manager.model_state("B") == ModelState::Resident);
    EXPECT_EQ(manager.model_state("C"), ModelState::Unloaded);
}

TEST(M3IntegrationTest, WorkloadAwareAndResidencyAwareProduceMultiBatch) {
    auto registry = make_registry({{"A", 8, 1}, {"B", 8, 1}});

    WorkerConfig cfg;
    cfg.worker_id = 0;
    cfg.queue_capacity = 32;
    cfg.memory_budget_bytes = 64;
    cfg.batch_config.max_batch_size = 8;
    cfg.batch_config.max_batch_wait = std::chrono::microseconds{0};
    cfg.registry = registry;
    cfg.backend = make_synth({"A", "B"});
    cfg.eviction_policy = std::make_unique<LruEvictionPolicy>();

    std::vector<std::unique_ptr<Worker>> workers;
    workers.push_back(std::make_unique<Worker>(std::move(cfg)));

    WorkloadAwareSchedulerConfig sched;
    sched.capacity = 64;
    sched.max_bypass = 16;

    Runtime runtime(std::make_unique<WorkloadAwareScheduler>(sched),
                    std::make_unique<ResidencyAwareRouter>(), std::move(workers), registry);
    ASSERT_TRUE(runtime.start().ok());

    std::vector<RequestPtr> reqs;
    for (int i = 0; i < 4; ++i) {
        reqs.push_back(make_queued("a" + std::to_string(i), "A"));
    }
    reqs.push_back(make_queued("b0", "B"));

    for (auto &r : reqs) {
        ASSERT_TRUE(runtime.submit(r).ok());
    }
    for (auto &r : reqs) {
        ASSERT_TRUE(r->wait_for_terminal(std::chrono::seconds{3}));
        EXPECT_EQ(r->state(), RequestState::Completed);
    }

    auto *worker = runtime.worker(0);
    ASSERT_NE(worker, nullptr);
    EXPECT_GE(worker->batch_metrics().multi_request_batches, 1u);
    EXPECT_GE(worker->batch_metrics().max_batch_size_observed, 2u);
    runtime.stop();
}

TEST(M3IntegrationTest, CostAwareAndLruChooseDifferentVictims) {
    auto run = [&](std::unique_ptr<airuntime::IEvictionPolicy> policy) {
        auto backend = make_synth({"cheap", "expensive", "newm"});
        ModelRegistry::Builder builder;
        EXPECT_TRUE(builder.add({"expensive", 6, 100}).ok());
        EXPECT_TRUE(builder.add({"cheap", 6, 1}).ok());
        EXPECT_TRUE(builder.add({"newm", 6, 1}).ok());
        Status st;
        auto reg = std::shared_ptr<const ModelRegistry>(builder.build(st));
        ModelManager manager(reg, *backend, 12, std::move(policy));
        // Load expensive first (older), then cheap (newer).
        EXPECT_TRUE(manager.ensure_resident("expensive").ok());
        EXPECT_TRUE(manager.ensure_resident("cheap").ok());
        EXPECT_TRUE(manager.ensure_resident("newm").ok());
        return std::make_pair(manager.model_state("cheap"), manager.model_state("expensive"));
    };

    auto lru = run(std::make_unique<LruEvictionPolicy>());
    auto cost = run(std::make_unique<CostAwareEvictionPolicy>());

    // LRU evicts oldest: expensive
    EXPECT_EQ(lru.second, ModelState::Unloaded);
    EXPECT_EQ(lru.first, ModelState::Resident);

    // CostAware evicts lower load cost: cheap
    EXPECT_EQ(cost.first, ModelState::Unloaded);
    EXPECT_EQ(cost.second, ModelState::Resident);
}

TEST(M3IntegrationTest, CostAwareProtectsHighLoadCostWhenUseEqual) {
    ModelRegistry::Builder builder;
    EXPECT_TRUE(builder.add({"low", 5, 1}).ok());
    EXPECT_TRUE(builder.add({"high", 5, 50}).ok());
    EXPECT_TRUE(builder.add({"need", 5, 1}).ok());
    Status st;
    auto reg = std::shared_ptr<const ModelRegistry>(builder.build(st));

    auto backend_cost = make_synth({"low", "high", "need"});
    ModelManager cost_mgr(reg, *backend_cost, 10, std::make_unique<CostAwareEvictionPolicy>());
    ASSERT_TRUE(cost_mgr.ensure_resident("high").ok());
    ASSERT_TRUE(cost_mgr.ensure_resident("low").ok());
    ASSERT_TRUE(cost_mgr.ensure_resident("need").ok());
    EXPECT_EQ(cost_mgr.model_state("low"), ModelState::Unloaded);
    EXPECT_EQ(cost_mgr.model_state("high"), ModelState::Resident);
}

TEST(M3IntegrationTest, ShutdownFlushesPartialBatch) {
    auto registry = make_registry({{"A", 8, 1}});
    WorkerConfig cfg;
    cfg.worker_id = 0;
    cfg.queue_capacity = 16;
    cfg.memory_budget_bytes = 64;
    cfg.batch_config.max_batch_size = 8;
    cfg.batch_config.max_batch_wait =
        std::chrono::seconds{30}; // would hang if wait ignored wrongly
    cfg.registry = registry;
    cfg.backend = make_synth({"A"});
    cfg.eviction_policy = std::make_unique<LruEvictionPolicy>();

    std::vector<std::unique_ptr<Worker>> workers;
    workers.push_back(std::make_unique<Worker>(std::move(cfg)));

    Runtime runtime(std::make_unique<FifoScheduler>(32),
                    std::make_unique<airuntime::RoundRobinRouter>(), std::move(workers), registry);
    ASSERT_TRUE(runtime.start().ok());

    auto r1 = make_queued("r1", "A");
    ASSERT_TRUE(runtime.submit(r1).ok());
    runtime.stop();
    ASSERT_TRUE(r1->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(r1->state(), RequestState::Completed);
}

TEST(M3IntegrationTest, ShutdownWhileWaitPushNoDeadlock) {
    auto registry = make_registry({{"A", 8, 1}});
    auto mutex = std::make_shared<std::mutex>();
    auto cv = std::make_shared<std::condition_variable>();
    auto entered = std::make_shared<bool>(false);
    auto release = std::make_shared<bool>(false);

    WorkerConfig cfg;
    cfg.worker_id = 0;
    cfg.queue_capacity = 1;
    cfg.memory_budget_bytes = 64;
    cfg.batch_config.max_batch_size = 1;
    cfg.batch_config.max_batch_wait = std::chrono::microseconds{0};
    cfg.registry = registry;
    cfg.backend =
        std::make_unique<BlockingInferBackend>(mutex, cv, entered, release, make_synth({"A"}));
    cfg.eviction_policy = std::make_unique<LruEvictionPolicy>();

    std::vector<std::unique_ptr<Worker>> workers;
    workers.push_back(std::make_unique<Worker>(std::move(cfg)));

    Runtime runtime(std::make_unique<FifoScheduler>(32),
                    std::make_unique<airuntime::RoundRobinRouter>(), std::move(workers), registry);
    ASSERT_TRUE(runtime.start().ok());

    auto r1 = make_queued("r1", "A");
    auto r2 = make_queued("r2", "A");
    auto r3 = make_queued("r3", "A");
    ASSERT_TRUE(runtime.submit(r1).ok());

    {
        std::unique_lock lock(*mutex);
        ASSERT_TRUE(cv->wait_for(lock, std::chrono::seconds{2}, [&] { return *entered; }));
    }

    ASSERT_TRUE(runtime.submit(r2).ok());
    ASSERT_TRUE(runtime.submit(r3).ok());

    // Give routing time to block on wait_push for r3.
    for (int i = 0; i < 200; ++i) {
        if (runtime.worker(0)->snapshot().queue_depth >= 1) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }

    std::atomic<bool> stop_done{false};
    std::thread stopper([&] {
        runtime.stop();
        stop_done.store(true);
    });

    {
        std::lock_guard lock(*mutex);
        *release = true;
    }
    cv->notify_all();

    stopper.join();
    EXPECT_TRUE(stop_done.load());
    ASSERT_TRUE(r1->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(r2->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(r3->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_TRUE(r1->is_terminal());
    EXPECT_TRUE(r2->is_terminal());
    EXPECT_TRUE(r3->is_terminal());
}

TEST(RuntimeDestructorTest, DestructorWithoutExplicitStopIsSafe) {
    auto registry = make_registry({{"A", 8, 1}});
    WorkerConfig cfg;
    cfg.worker_id = 0;
    cfg.queue_capacity = 8;
    cfg.memory_budget_bytes = 64;
    cfg.registry = registry;
    cfg.backend = make_synth({"A"});
    cfg.eviction_policy = std::make_unique<LruEvictionPolicy>();

    auto r = make_queued("r1", "A");
    {
        std::vector<std::unique_ptr<Worker>> workers;
        workers.push_back(std::make_unique<Worker>(std::move(cfg)));
        Runtime runtime(std::make_unique<FifoScheduler>(8),
                        std::make_unique<airuntime::RoundRobinRouter>(), std::move(workers),
                        registry);
        ASSERT_TRUE(runtime.start().ok());
        ASSERT_TRUE(runtime.submit(r).ok());
        ASSERT_TRUE(r->wait_for_terminal(std::chrono::seconds{2}));
    }
    EXPECT_EQ(r->state(), RequestState::Completed);
}
