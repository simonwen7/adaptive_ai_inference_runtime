#include "airuntime/scheduler.hpp"

#include <gtest/gtest.h>

using airuntime::ErrorCode;
using airuntime::FifoScheduler;
using airuntime::InferenceRequest;
using airuntime::RequestPtr;

namespace {

RequestPtr make_request(const std::string &id) {
    return std::make_shared<InferenceRequest>(id, "model", "prompt", 2);
}

} // namespace

TEST(SchedulerTest, FifoOrdering) {
    FifoScheduler scheduler(8);
    ASSERT_TRUE(scheduler.enqueue(make_request("a")).ok());
    ASSERT_TRUE(scheduler.enqueue(make_request("b")).ok());
    ASSERT_TRUE(scheduler.enqueue(make_request("c")).ok());

    EXPECT_EQ((*scheduler.next())->request_id(), "a");
    EXPECT_EQ((*scheduler.next())->request_id(), "b");
    EXPECT_EQ((*scheduler.next())->request_id(), "c");
}

TEST(SchedulerTest, CapacityPropagation) {
    FifoScheduler scheduler(2);
    EXPECT_EQ(scheduler.capacity(), 2u);
    ASSERT_TRUE(scheduler.enqueue(make_request("a")).ok());
    ASSERT_TRUE(scheduler.enqueue(make_request("b")).ok());
    EXPECT_EQ(scheduler.size(), 2u);
    EXPECT_EQ(scheduler.enqueue(make_request("c")).code, ErrorCode::QueueFull);
}

TEST(SchedulerTest, CloseBehaviorAndDrain) {
    FifoScheduler scheduler(4);
    ASSERT_TRUE(scheduler.enqueue(make_request("a")).ok());
    scheduler.close();
    EXPECT_EQ(scheduler.enqueue(make_request("b")).code, ErrorCode::QueueClosed);

    auto first = scheduler.next();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ((*first)->request_id(), "a");
    EXPECT_FALSE(scheduler.next().has_value());
}

TEST(SchedulerTest, NextAfterDrainWhenClosed) {
    FifoScheduler scheduler(1);
    scheduler.close();
    EXPECT_FALSE(scheduler.next().has_value());
}
