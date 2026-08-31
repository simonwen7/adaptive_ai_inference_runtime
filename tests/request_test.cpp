#include "airuntime/request.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>

using airuntime::ErrorCode;
using airuntime::InferenceOutput;
using airuntime::InferenceRequest;
using airuntime::InferenceResult;
using airuntime::RequestState;
using airuntime::Status;

TEST(RequestTest, InitialStateIsReceived) {
    InferenceRequest request("r1", "m1", "prompt", 8);
    EXPECT_EQ(request.state(), RequestState::Received);
    EXPECT_FALSE(request.is_terminal());
    EXPECT_FALSE(request.result().has_value());
}

TEST(RequestTest, ReceivedToQueued) {
    InferenceRequest request("r1", "m1", "prompt", 8);
    ASSERT_TRUE(request.transition_to(RequestState::Queued).ok());
    EXPECT_EQ(request.state(), RequestState::Queued);
}

TEST(RequestTest, ReceivedToRejected) {
    InferenceRequest request("r1", "m1", "prompt", 8);
    ASSERT_TRUE(request.reject(Status::error(ErrorCode::QueueFull, "full")).ok());
    EXPECT_EQ(request.state(), RequestState::Rejected);
    EXPECT_TRUE(request.is_terminal());
    ASSERT_TRUE(request.result().has_value());
    EXPECT_EQ(request.result()->status.code, ErrorCode::QueueFull);
}

TEST(RequestTest, QueuedToRunning) {
    InferenceRequest request("r1", "m1", "prompt", 8);
    ASSERT_TRUE(request.transition_to(RequestState::Queued).ok());
    ASSERT_TRUE(request.transition_to(RequestState::Running).ok());
    EXPECT_EQ(request.state(), RequestState::Running);
}

TEST(RequestTest, QueuedToRejected) {
    InferenceRequest request("r1", "m1", "prompt", 8);
    ASSERT_TRUE(request.transition_to(RequestState::Queued).ok());
    ASSERT_TRUE(request.reject(Status::error(ErrorCode::RuntimeStopped, "stop")).ok());
    EXPECT_EQ(request.state(), RequestState::Rejected);
}

TEST(RequestTest, RunningToCompleted) {
    InferenceRequest request("r1", "m1", "prompt", 8);
    ASSERT_TRUE(request.transition_to(RequestState::Queued).ok());
    ASSERT_TRUE(request.transition_to(RequestState::Running).ok());

    InferenceOutput output{"synthetic:r1", 8};
    ASSERT_TRUE(request.complete(InferenceResult::success(output)).ok());
    EXPECT_EQ(request.state(), RequestState::Completed);
    ASSERT_TRUE(request.result().has_value());
    EXPECT_EQ(request.result()->output->text, "synthetic:r1");
}

TEST(RequestTest, RunningToFailed) {
    InferenceRequest request("r1", "m1", "prompt", 8);
    ASSERT_TRUE(request.transition_to(RequestState::Queued).ok());
    ASSERT_TRUE(request.transition_to(RequestState::Running).ok());
    ASSERT_TRUE(request.fail(Status::error(ErrorCode::InferenceFailed, "boom")).ok());
    EXPECT_EQ(request.state(), RequestState::Failed);
}

TEST(RequestTest, TerminalCannotTransition) {
    InferenceRequest completed("c1", "m1", "p", 1);
    ASSERT_TRUE(completed.transition_to(RequestState::Queued).ok());
    ASSERT_TRUE(completed.transition_to(RequestState::Running).ok());
    ASSERT_TRUE(completed.complete(InferenceResult::success({"ok", 1})).ok());
    EXPECT_EQ(completed.transition_to(RequestState::Queued).code,
              ErrorCode::InvalidStateTransition);

    InferenceRequest rejected("rj", "m1", "p", 1);
    ASSERT_TRUE(rejected.reject(Status::error(ErrorCode::QueueFull, "full")).ok());
    EXPECT_EQ(rejected.transition_to(RequestState::Running).code,
              ErrorCode::InvalidStateTransition);

    InferenceRequest failed("f1", "m1", "p", 1);
    ASSERT_TRUE(failed.transition_to(RequestState::Queued).ok());
    ASSERT_TRUE(failed.transition_to(RequestState::Running).ok());
    ASSERT_TRUE(failed.fail(Status::error(ErrorCode::InferenceFailed, "x")).ok());
    EXPECT_EQ(failed.transition_to(RequestState::Completed).code,
              ErrorCode::InvalidStateTransition);
}

TEST(RequestTest, CannotEnterRunningTwice) {
    InferenceRequest request("r1", "m1", "prompt", 8);
    ASSERT_TRUE(request.transition_to(RequestState::Queued).ok());
    ASSERT_TRUE(request.transition_to(RequestState::Running).ok());
    EXPECT_EQ(request.transition_to(RequestState::Running).code, ErrorCode::InvalidStateTransition);
    EXPECT_EQ(request.state(), RequestState::Running);
}

TEST(RequestTest, WaitForTerminalSignalsCompletion) {
    auto request = std::make_shared<InferenceRequest>("r1", "m1", "p", 1);
    ASSERT_TRUE(request->transition_to(RequestState::Queued).ok());
    ASSERT_TRUE(request->transition_to(RequestState::Running).ok());
    ASSERT_TRUE(request->complete(InferenceResult::success({"synthetic:r1", 1})).ok());
    EXPECT_TRUE(request->wait_for_terminal(std::chrono::milliseconds{100}));
}
