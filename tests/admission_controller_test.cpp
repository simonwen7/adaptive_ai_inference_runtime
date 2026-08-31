#include "airuntime/admission_controller.hpp"
#include "airuntime/scheduler.hpp"

#include <gtest/gtest.h>

using airuntime::AdmissionController;
using airuntime::ErrorCode;
using airuntime::FifoScheduler;
using airuntime::InferenceRequest;
using airuntime::RequestPtr;
using airuntime::RequestState;

namespace {

RequestPtr make_request(const std::string &id) {
    return std::make_shared<InferenceRequest>(id, "model", "prompt", 2);
}

} // namespace

TEST(AdmissionControllerTest, SuccessfulAdmissionQueuesRequest) {
    FifoScheduler scheduler(4);
    AdmissionController admission(scheduler);
    auto request = make_request("r1");

    ASSERT_TRUE(admission.admit(request).ok());
    EXPECT_EQ(request->state(), RequestState::Queued);
    EXPECT_EQ(scheduler.size(), 1u);
}

TEST(AdmissionControllerTest, QueueFullRejects) {
    FifoScheduler scheduler(1);
    AdmissionController admission(scheduler);
    ASSERT_TRUE(admission.admit(make_request("r1")).ok());

    auto request = make_request("r2");
    auto status = admission.admit(request);
    EXPECT_EQ(status.code, ErrorCode::QueueFull);
    EXPECT_EQ(request->state(), RequestState::Rejected);
}

TEST(AdmissionControllerTest, ClosedSchedulerRejects) {
    FifoScheduler scheduler(4);
    AdmissionController admission(scheduler);
    scheduler.close();

    auto request = make_request("r1");
    auto status = admission.admit(request);
    EXPECT_EQ(status.code, ErrorCode::RuntimeStopped);
    EXPECT_EQ(request->state(), RequestState::Rejected);
}

TEST(AdmissionControllerTest, InvalidLifecycleRejected) {
    FifoScheduler scheduler(4);
    AdmissionController admission(scheduler);
    auto request = make_request("r1");
    ASSERT_TRUE(request->transition_to(RequestState::Queued).ok());

    auto status = admission.admit(request);
    EXPECT_EQ(status.code, ErrorCode::InvalidStateTransition);
    EXPECT_EQ(request->state(), RequestState::Queued);
}
