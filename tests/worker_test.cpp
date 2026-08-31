#include "airuntime/backend.hpp"
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
using airuntime::ModelSpec;
using airuntime::RequestPtr;
using airuntime::RequestState;
using airuntime::Status;
using airuntime::SyntheticModelBackend;
using airuntime::SyntheticModelConfig;
using airuntime::Worker;
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

RequestPtr queued_request(const std::string &id, const std::string &model) {
    auto request = std::make_shared<InferenceRequest>(id, model, "prompt", 2);
    EXPECT_TRUE(request->transition_to(RequestState::Queued).ok());
    return request;
}

std::unique_ptr<SyntheticModelBackend> make_backend(bool fail_load = false,
                                                    bool fail_infer = false) {
    auto backend = std::make_unique<SyntheticModelBackend>();
    SyntheticModelConfig config;
    config.load_latency = std::chrono::microseconds{10};
    config.prefill_latency = std::chrono::microseconds{5};
    config.per_token_latency = std::chrono::microseconds{1};
    config.fail_load = fail_load;
    config.fail_infer = fail_infer;
    backend->register_model("m1", config);
    return backend;
}

} // namespace

TEST(WorkerTest, StartsIdle) {
    Worker worker(make_backend());
    EXPECT_EQ(worker.state(), WorkerState::Idle);
}

TEST(WorkerTest, ExecuteCompletesRequest) {
    Worker worker(make_backend());
    auto request = queued_request("r1", "m1");
    ASSERT_TRUE(worker.execute(request).ok());
    EXPECT_EQ(request->state(), RequestState::Completed);
    ASSERT_TRUE(request->result().has_value());
    EXPECT_EQ(request->result()->output->text, "synthetic:r1");
    EXPECT_EQ(worker.state(), WorkerState::Idle);
}

TEST(WorkerTest, AutoLoadsModel) {
    auto backend_ptr = make_backend();
    auto *backend = backend_ptr.get();
    Worker worker(std::move(backend_ptr));
    EXPECT_FALSE(backend->is_loaded("m1"));

    auto request = queued_request("r1", "m1");
    ASSERT_TRUE(worker.execute(request).ok());
    EXPECT_TRUE(backend->is_loaded("m1"));
}

TEST(WorkerTest, AlreadyLoadedModel) {
    auto backend_ptr = make_backend();
    auto *backend = backend_ptr.get();
    ASSERT_TRUE(backend->load(ModelSpec{"m1"}).ok());
    Worker worker(std::move(backend_ptr));

    auto request = queued_request("r1", "m1");
    ASSERT_TRUE(worker.execute(request).ok());
    EXPECT_EQ(request->state(), RequestState::Completed);
    EXPECT_EQ(backend->metrics().load_count, 1u);
}

TEST(WorkerTest, LoadFailureMarksRequestFailed) {
    Worker worker(make_backend(true, false));
    auto request = queued_request("r1", "m1");
    auto status = worker.execute(request);
    EXPECT_EQ(status.code, ErrorCode::ModelLoadFailed);
    EXPECT_EQ(request->state(), RequestState::Failed);
    EXPECT_EQ(worker.state(), WorkerState::Idle);
}

TEST(WorkerTest, InferenceFailureMarksRequestFailed) {
    Worker worker(make_backend(false, true));
    auto request = queued_request("r1", "m1");
    auto status = worker.execute(request);
    EXPECT_EQ(status.code, ErrorCode::InferenceFailed);
    EXPECT_EQ(request->state(), RequestState::Failed);
    EXPECT_EQ(worker.state(), WorkerState::Idle);
}

TEST(WorkerTest, UnexpectedExceptionDoesNotCrash) {
    Worker worker(std::make_unique<ThrowingBackend>());
    auto request = queued_request("r1", "m1");
    auto status = worker.execute(request);
    EXPECT_EQ(status.code, ErrorCode::InternalError);
    EXPECT_EQ(request->state(), RequestState::Failed);
    EXPECT_EQ(worker.state(), WorkerState::Idle);
}
