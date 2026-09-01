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
#include <string>
#include <vector>

using airuntime::ErrorCode;
using airuntime::FifoScheduler;
using airuntime::IModelBackend;
using airuntime::InferenceRequest;
using airuntime::InferenceResult;
using airuntime::LruEvictionPolicy;
using airuntime::ModelRegistry;
using airuntime::ModelSpec;
using airuntime::RequestPtr;
using airuntime::RequestState;
using airuntime::RoundRobinRouter;
using airuntime::Runtime;
using airuntime::Status;
using airuntime::SyntheticModelBackend;
using airuntime::SyntheticModelConfig;
using airuntime::Worker;
using airuntime::WorkerConfig;
using airuntime::WorkerId;

namespace {

class BarrierBackend final : public IModelBackend {
  public:
    BarrierBackend(std::shared_ptr<std::mutex> mutex, std::shared_ptr<std::condition_variable> cv,
                   std::shared_ptr<int> arrived, std::shared_ptr<bool> released, int party_count,
                   std::unique_ptr<SyntheticModelBackend> inner)
        : mutex_(std::move(mutex)), cv_(std::move(cv)), arrived_(std::move(arrived)),
          released_(std::move(released)), party_count_(party_count), inner_(std::move(inner)) {}

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
        {
            std::unique_lock lock(*mutex_);
            ++(*arrived_);
            if (*arrived_ >= party_count_) {
                *released_ = true;
                cv_->notify_all();
            } else {
                cv_->wait(lock, [&] { return *released_; });
            }
        }
        return inner_->infer(request);
    }

  private:
    std::shared_ptr<std::mutex> mutex_;
    std::shared_ptr<std::condition_variable> cv_;
    std::shared_ptr<int> arrived_;
    std::shared_ptr<bool> released_;
    int party_count_;
    std::unique_ptr<SyntheticModelBackend> inner_;
};

std::shared_ptr<const ModelRegistry> make_registry_with(std::initializer_list<ModelSpec> specs) {
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
    config.load_latency = std::chrono::microseconds{1};
    for (const char *id : ids) {
        backend->register_model(id, config);
    }
    return backend;
}

WorkerConfig make_config(WorkerId id, std::shared_ptr<const ModelRegistry> registry,
                         std::unique_ptr<IModelBackend> backend, std::uint64_t budget) {
    WorkerConfig config;
    config.worker_id = id;
    config.queue_capacity = 8;
    config.memory_budget_bytes = budget;
    config.registry = std::move(registry);
    config.backend = std::move(backend);
    config.eviction_policy = std::make_unique<LruEvictionPolicy>();
    return config;
}

} // namespace

TEST(MultiWorkerIntegrationTest, OverlappingWorkerExecution) {
    auto mutex = std::make_shared<std::mutex>();
    auto cv = std::make_shared<std::condition_variable>();
    auto arrived = std::make_shared<int>(0);
    auto released = std::make_shared<bool>(false);

    auto registry = make_registry_with({{"m1", 8}});
    std::vector<std::unique_ptr<Worker>> workers;
    for (WorkerId id = 0; id < 2; ++id) {
        auto barrier =
            std::make_unique<BarrierBackend>(mutex, cv, arrived, released, 2, make_synth({"m1"}));
        workers.push_back(
            std::make_unique<Worker>(make_config(id, registry, std::move(barrier), 64)));
    }

    Runtime runtime(std::make_unique<FifoScheduler>(8), std::make_unique<RoundRobinRouter>(),
                    std::move(workers), registry);
    ASSERT_TRUE(runtime.start().ok());

    auto r0 = std::make_shared<InferenceRequest>("r0", "m1", "p", 1);
    auto r1 = std::make_shared<InferenceRequest>("r1", "m1", "p", 1);
    ASSERT_TRUE(runtime.submit(r0).ok());
    ASSERT_TRUE(runtime.submit(r1).ok());

    ASSERT_TRUE(r0->wait_for_terminal(std::chrono::seconds{3}));
    ASSERT_TRUE(r1->wait_for_terminal(std::chrono::seconds{3}));
    EXPECT_EQ(r0->state(), RequestState::Completed);
    EXPECT_EQ(r1->state(), RequestState::Completed);
    EXPECT_TRUE(*released);
    EXPECT_GE(*arrived, 2);

    runtime.stop();
}

TEST(MultiWorkerIntegrationTest, MemoryPressureEvictionScenario) {
    constexpr std::uint64_t kGiB = 1024ull * 1024 * 1024;
    auto registry = make_registry_with({{"A", 5 * kGiB}, {"B", 6 * kGiB}, {"C", 4 * kGiB}});

    auto backend = make_synth({"A", "B", "C"});
    auto *backend_ptr = backend.get();
    std::vector<std::unique_ptr<Worker>> workers;
    workers.push_back(
        std::make_unique<Worker>(make_config(0, registry, std::move(backend), 12 * kGiB)));

    Runtime runtime(std::make_unique<FifoScheduler>(8), std::make_unique<RoundRobinRouter>(),
                    std::move(workers), registry);
    ASSERT_TRUE(runtime.start().ok());

    auto ra = std::make_shared<InferenceRequest>("ra", "A", "p", 1);
    auto rb = std::make_shared<InferenceRequest>("rb", "B", "p", 1);
    auto rc = std::make_shared<InferenceRequest>("rc", "C", "p", 1);

    ASSERT_TRUE(runtime.submit(ra).ok());
    ASSERT_TRUE(ra->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(runtime.submit(rb).ok());
    ASSERT_TRUE(rb->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(runtime.submit(rc).ok());
    ASSERT_TRUE(rc->wait_for_terminal(std::chrono::seconds{2}));

    EXPECT_EQ(ra->state(), RequestState::Completed);
    EXPECT_EQ(rb->state(), RequestState::Completed);
    EXPECT_EQ(rc->state(), RequestState::Completed);

    auto *worker = runtime.worker(0);
    ASSERT_NE(worker, nullptr);
    EXPECT_LE(worker->snapshot().memory_used_bytes, 12 * kGiB);
    EXPECT_GE(worker->residency_metrics().evictions, 1u);
    EXPECT_EQ(worker->model_manager().model_state("C"), airuntime::ModelState::Resident);
    EXPECT_TRUE(backend_ptr->is_loaded("C"));

    runtime.stop();
}

TEST(MultiWorkerIntegrationTest, HeterogeneousBudgetsRouteToFeasibleWorker) {
    auto registry = make_registry_with({{"mid", 50}});
    std::vector<std::unique_ptr<Worker>> workers;
    workers.push_back(std::make_unique<Worker>(make_config(0, registry, make_synth({"mid"}), 10)));
    workers.push_back(std::make_unique<Worker>(make_config(1, registry, make_synth({"mid"}), 100)));

    Runtime runtime(std::make_unique<FifoScheduler>(8), std::make_unique<RoundRobinRouter>(),
                    std::move(workers), registry);
    ASSERT_TRUE(runtime.start().ok());

    auto request = std::make_shared<InferenceRequest>("r", "mid", "p", 1);
    ASSERT_TRUE(runtime.submit(request).ok());
    ASSERT_TRUE(request->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(request->state(), RequestState::Completed);
    EXPECT_GE(runtime.worker(1)->residency_metrics().loads, 1u);
    EXPECT_EQ(runtime.worker(0)->residency_metrics().loads, 0u);

    runtime.stop();
}
