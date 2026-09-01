#include "airuntime/backend.hpp"
#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_manager.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/synthetic_backend.hpp"
#include "airuntime/worker.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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

class CountingBatchBackend final : public IModelBackend {
  public:
    explicit CountingBatchBackend(std::unique_ptr<SyntheticModelBackend> inner)
        : inner_(std::move(inner)) {}

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
        auto results = infer_batch(std::span<const InferenceRequest *const>(&ptr, 1));
        return std::move(results.front());
    }
    std::vector<InferenceResult>
    infer_batch(std::span<const InferenceRequest *const> requests) override {
        ++batch_calls_;
        last_batch_size_ = requests.size();
        return inner_->infer_batch(requests);
    }

    std::size_t batch_calls() const {
        return batch_calls_;
    }
    std::size_t last_batch_size() const {
        return last_batch_size_;
    }

  private:
    std::unique_ptr<SyntheticModelBackend> inner_;
    std::atomic<std::size_t> batch_calls_{0};
    std::atomic<std::size_t> last_batch_size_{0};
};

class BadCardinalityBackend final : public IModelBackend {
  public:
    explicit BadCardinalityBackend(std::size_t result_count) : result_count_(result_count) {}
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
        return InferenceResult::success({"x", 1});
    }
    std::vector<InferenceResult> infer_batch(std::span<const InferenceRequest *const>) override {
        return std::vector<InferenceResult>(result_count_, InferenceResult::success({"x", 1}));
    }

  private:
    std::size_t result_count_;
};

class PartialFailBackend final : public IModelBackend {
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
    InferenceResult infer(const InferenceRequest &r) override {
        const InferenceRequest *ptr = &r;
        return infer_batch(std::span<const InferenceRequest *const>(&ptr, 1)).front();
    }
    std::vector<InferenceResult>
    infer_batch(std::span<const InferenceRequest *const> requests) override {
        std::vector<InferenceResult> out;
        for (std::size_t i = 0; i < requests.size(); ++i) {
            if (i == 1) {
                out.push_back(
                    InferenceResult::failure(Status::error(ErrorCode::InferenceFailed, "partial")));
            } else {
                out.push_back(InferenceResult::success({"ok:" + requests[i]->request_id(), 1}));
            }
        }
        return out;
    }
};

std::shared_ptr<const ModelRegistry> make_registry() {
    ModelRegistry::Builder builder;
    EXPECT_TRUE(builder.add({"m1", 8}).ok());
    EXPECT_TRUE(builder.add({"m2", 8}).ok());
    Status status;
    auto registry = builder.build(status);
    EXPECT_TRUE(status.ok());
    return std::shared_ptr<const ModelRegistry>(std::move(registry));
}

std::unique_ptr<SyntheticModelBackend> make_synth() {
    auto backend = std::make_unique<SyntheticModelBackend>();
    SyntheticModelConfig config;
    config.prefill_latency = std::chrono::microseconds{40};
    config.per_token_latency = std::chrono::microseconds{5};
    backend->register_model("m1", config);
    backend->register_model("m2", config);
    return backend;
}

WorkerConfig make_config(std::unique_ptr<IModelBackend> backend, std::size_t max_batch = 4) {
    WorkerConfig config;
    config.worker_id = 1;
    config.queue_capacity = 16;
    config.memory_budget_bytes = 64;
    config.batch_config.max_batch_size = max_batch;
    config.batch_config.max_batch_wait = std::chrono::microseconds{0};
    config.registry = make_registry();
    config.backend = std::move(backend);
    config.eviction_policy = std::make_unique<LruEvictionPolicy>();
    return config;
}

RequestPtr queued(const std::string &id, const std::string &model = "m1") {
    auto request = std::make_shared<InferenceRequest>(id, model, "p", 2);
    EXPECT_TRUE(request->transition_to(RequestState::Queued).ok());
    return request;
}

} // namespace

TEST(WorkerBatchTest, OneEnsureAndOneInferBatchPerBatch) {
    auto synth = make_synth();
    auto *synth_ptr = synth.get();
    auto counting = std::make_unique<CountingBatchBackend>(std::move(synth));
    auto *counter = counting.get();
    Worker worker(make_config(std::move(counting), 4));
    ASSERT_TRUE(worker.start().ok());

    std::vector<RequestPtr> reqs{queued("r1"), queued("r2"), queued("r3")};
    for (auto &r : reqs) {
        ASSERT_TRUE(worker.enqueue(r).ok());
    }
    for (auto &r : reqs) {
        ASSERT_TRUE(r->wait_for_terminal(std::chrono::seconds{2}));
        EXPECT_EQ(r->state(), RequestState::Completed);
    }

    EXPECT_EQ(counter->batch_calls(), 1u);
    EXPECT_EQ(counter->last_batch_size(), 3u);
    EXPECT_EQ(synth_ptr->metrics().batch_inference_count, 1u);
    EXPECT_EQ(worker.residency_metrics().loads, 1u);
    EXPECT_EQ(worker.model_manager().use_count("m1"), 1u);
    EXPECT_EQ(worker.batch_metrics().batches_executed, 1u);
    EXPECT_EQ(worker.batch_metrics().multi_request_batches, 1u);
    EXPECT_EQ(worker.batch_metrics().max_batch_size_observed, 3u);
    EXPECT_EQ(worker.state(), WorkerState::Idle);

    worker.close();
    worker.join();
}

TEST(WorkerBatchTest, SeparatesDifferentModelsContiguouslyTracked) {
    auto counting = std::make_unique<CountingBatchBackend>(make_synth());
    auto *counter = counting.get();
    Worker worker(make_config(std::move(counting), 8));
    ASSERT_TRUE(worker.start().ok());

    auto a1 = queued("a1", "m1");
    auto b1 = queued("b1", "m2");
    auto a2 = queued("a2", "m1");
    ASSERT_TRUE(worker.enqueue(a1).ok());
    ASSERT_TRUE(worker.enqueue(b1).ok());
    ASSERT_TRUE(worker.enqueue(a2).ok());

    ASSERT_TRUE(a1->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(b1->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(a2->wait_for_terminal(std::chrono::seconds{2}));

    EXPECT_EQ(counter->batch_calls(), 3u);
    EXPECT_EQ(worker.batch_metrics().batches_executed, 3u);

    worker.close();
    worker.join();
}

TEST(WorkerBatchTest, CardinalityMismatchFailsAll) {
    Worker worker(make_config(std::make_unique<BadCardinalityBackend>(0), 2));
    ASSERT_TRUE(worker.start().ok());
    auto r1 = queued("r1");
    auto r2 = queued("r2");
    ASSERT_TRUE(worker.enqueue(r1).ok());
    ASSERT_TRUE(worker.enqueue(r2).ok());
    ASSERT_TRUE(r1->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(r2->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(r1->state(), RequestState::Failed);
    EXPECT_EQ(r2->state(), RequestState::Failed);
    EXPECT_EQ(r1->result()->status.code, ErrorCode::InternalError);
    EXPECT_EQ(worker.state(), WorkerState::Idle);
    worker.close();
    worker.join();
}

TEST(WorkerBatchTest, PartialPerRequestFailure) {
    Worker worker(make_config(std::make_unique<PartialFailBackend>(), 3));
    ASSERT_TRUE(worker.start().ok());
    auto r1 = queued("r1");
    auto r2 = queued("r2");
    auto r3 = queued("r3");
    ASSERT_TRUE(worker.enqueue(r1).ok());
    ASSERT_TRUE(worker.enqueue(r2).ok());
    ASSERT_TRUE(worker.enqueue(r3).ok());
    ASSERT_TRUE(r1->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(r2->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(r3->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(r1->state(), RequestState::Completed);
    EXPECT_EQ(r2->state(), RequestState::Failed);
    EXPECT_EQ(r3->state(), RequestState::Completed);
    EXPECT_EQ(r2->result()->status.code, ErrorCode::InferenceFailed);
    worker.close();
    worker.join();
}

TEST(WorkerBatchTest, BatchSizeOneWorks) {
    auto counting = std::make_unique<CountingBatchBackend>(make_synth());
    auto *counter = counting.get();
    Worker worker(make_config(std::move(counting), 1));
    ASSERT_TRUE(worker.start().ok());
    auto r1 = queued("r1");
    auto r2 = queued("r2");
    ASSERT_TRUE(worker.enqueue(r1).ok());
    ASSERT_TRUE(r1->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(worker.enqueue(r2).ok());
    ASSERT_TRUE(r2->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(counter->batch_calls(), 2u);
    EXPECT_EQ(counter->last_batch_size(), 1u);
    worker.close();
    worker.join();
}

TEST(WorkerBatchTest, SnapshotQueuedCountIncludesBufferedWork) {
    // Slow backend: hold first batch so more enqueues stay queued.
    class HoldBackend final : public IModelBackend {
      public:
        HoldBackend(std::shared_ptr<std::mutex> m, std::shared_ptr<std::condition_variable> cv,
                    std::shared_ptr<bool> go, std::unique_ptr<SyntheticModelBackend> inner)
            : mutex_(std::move(m)), cv_(std::move(cv)), go_(std::move(go)),
              inner_(std::move(inner)) {}
        Status load(const ModelSpec &model) override {
            return inner_->load(model);
        }
        Status unload(std::string_view id) override {
            return inner_->unload(id);
        }
        bool is_loaded(std::string_view id) const override {
            return inner_->is_loaded(id);
        }
        InferenceResult infer(const InferenceRequest &r) override {
            const InferenceRequest *p = &r;
            return infer_batch(std::span<const InferenceRequest *const>(&p, 1)).front();
        }
        std::vector<InferenceResult>
        infer_batch(std::span<const InferenceRequest *const> requests) override {
            std::unique_lock lock(*mutex_);
            cv_->wait(lock, [&] { return *go_; });
            return inner_->infer_batch(requests);
        }

      private:
        std::shared_ptr<std::mutex> mutex_;
        std::shared_ptr<std::condition_variable> cv_;
        std::shared_ptr<bool> go_;
        std::unique_ptr<SyntheticModelBackend> inner_;
    };

    auto mutex = std::make_shared<std::mutex>();
    auto cv = std::make_shared<std::condition_variable>();
    auto go = std::make_shared<bool>(false);
    Worker worker(make_config(std::make_unique<HoldBackend>(mutex, cv, go, make_synth()), 1));
    ASSERT_TRUE(worker.start().ok());

    auto r1 = queued("r1");
    auto r2 = queued("r2");
    ASSERT_TRUE(worker.enqueue(r1).ok());
    // Wait until r1 is active (left queued accounting).
    for (int i = 0; i < 1000 && worker.snapshot().active_count == 0; ++i) {
        std::this_thread::yield();
    }
    ASSERT_TRUE(worker.enqueue(r2).ok());
    EXPECT_EQ(worker.snapshot().queue_depth, 1u);
    EXPECT_EQ(worker.snapshot().active_count, 1u);

    {
        std::lock_guard lock(*mutex);
        *go = true;
    }
    cv->notify_all();
    ASSERT_TRUE(r1->wait_for_terminal(std::chrono::seconds{2}));
    ASSERT_TRUE(r2->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(worker.snapshot().queue_depth, 0u);
    EXPECT_EQ(worker.snapshot().active_count, 0u);
    worker.close();
    worker.join();
}

TEST(WorkerDestructorTest, DestructorWithoutExplicitCloseIsSafe) {
    auto r = queued("r1");
    {
        Worker worker(make_config(make_synth(), 1));
        ASSERT_TRUE(worker.start().ok());
        ASSERT_TRUE(worker.enqueue(r).ok());
        ASSERT_TRUE(r->wait_for_terminal(std::chrono::seconds{2}));
    }
    EXPECT_EQ(r->state(), RequestState::Completed);
}
