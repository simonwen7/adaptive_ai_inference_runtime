#include "airuntime/runtime.hpp"
#include "airuntime/scheduler.hpp"
#include "airuntime/synthetic_backend.hpp"
#include "airuntime/worker.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

using airuntime::ErrorCode;
using airuntime::FifoScheduler;
using airuntime::IModelBackend;
using airuntime::InferenceRequest;
using airuntime::InferenceResult;
using airuntime::ModelSpec;
using airuntime::RequestPtr;
using airuntime::RequestState;
using airuntime::Runtime;
using airuntime::Status;
using airuntime::SyntheticModelBackend;
using airuntime::SyntheticModelConfig;
using airuntime::Worker;

namespace {

class OrderTrackingBackend final : public IModelBackend {
  public:
    explicit OrderTrackingBackend(std::unique_ptr<SyntheticModelBackend> inner)
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
        {
            std::lock_guard lock(mutex_);
            execution_order_.push_back(request.request_id());
        }
        return inner_->infer(request);
    }

    std::vector<std::string> execution_order() const {
        std::lock_guard lock(mutex_);
        return execution_order_;
    }

  private:
    std::unique_ptr<SyntheticModelBackend> inner_;
    mutable std::mutex mutex_;
    std::vector<std::string> execution_order_;
};

std::unique_ptr<SyntheticModelBackend> make_synthetic_backend() {
    auto backend = std::make_unique<SyntheticModelBackend>();
    SyntheticModelConfig config;
    config.load_latency = std::chrono::microseconds{1};
    config.prefill_latency = std::chrono::microseconds{1};
    config.per_token_latency = std::chrono::microseconds{1};
    backend->register_model("m1", config);
    return backend;
}

std::unique_ptr<Runtime> make_runtime(std::size_t capacity = 8) {
    return std::make_unique<Runtime>(std::make_unique<FifoScheduler>(capacity),
                                     std::make_unique<Worker>(make_synthetic_backend()));
}

RequestPtr make_request(const std::string &id) {
    return std::make_shared<InferenceRequest>(id, "m1", "prompt", 2);
}

} // namespace

TEST(RuntimeIntegrationTest, StartsAndCompletesRequest) {
    auto runtime = make_runtime();
    ASSERT_TRUE(runtime->start().ok());

    auto request = make_request("r1");
    ASSERT_TRUE(runtime->submit(request).ok());
    ASSERT_TRUE(request->wait_for_terminal(std::chrono::seconds{2}));
    EXPECT_EQ(request->state(), RequestState::Completed);
    ASSERT_TRUE(request->result().has_value());
    EXPECT_EQ(request->result()->output->text, "synthetic:r1");

    runtime->stop();
}

TEST(RuntimeIntegrationTest, FifoExecutionOrder) {
    auto tracking = std::make_unique<OrderTrackingBackend>(make_synthetic_backend());
    auto *tracking_ptr = tracking.get();
    auto runtime = std::make_unique<Runtime>(std::make_unique<FifoScheduler>(16),
                                             std::make_unique<Worker>(std::move(tracking)));
    ASSERT_TRUE(runtime->start().ok());

    std::vector<RequestPtr> requests;
    for (int i = 0; i < 5; ++i) {
        auto request = make_request("r" + std::to_string(i));
        ASSERT_TRUE(runtime->submit(request).ok());
        requests.push_back(request);
    }

    for (auto &request : requests) {
        ASSERT_TRUE(request->wait_for_terminal(std::chrono::seconds{2}));
        EXPECT_EQ(request->state(), RequestState::Completed);
    }

    const auto order = tracking_ptr->execution_order();
    ASSERT_EQ(order.size(), 5u);
    for (std::size_t i = 0; i < order.size(); ++i) {
        EXPECT_EQ(order[i], "r" + std::to_string(i));
    }

    runtime->stop();
}

TEST(RuntimeIntegrationTest, StopDrainsQueuedWork) {
    auto runtime = make_runtime(16);
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
        EXPECT_EQ(request->state(), RequestState::Completed);
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
