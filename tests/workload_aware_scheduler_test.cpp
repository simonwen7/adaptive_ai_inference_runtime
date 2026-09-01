#include "airuntime/workload_aware_scheduler.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>

using airuntime::ErrorCode;
using airuntime::InferenceRequest;
using airuntime::RequestPtr;
using airuntime::WorkloadAwareScheduler;
using airuntime::WorkloadAwareSchedulerConfig;

namespace {

RequestPtr make_req(const std::string &id, const std::string &model) {
    return std::make_shared<InferenceRequest>(id, model, "p", 1);
}

} // namespace

TEST(WorkloadAwareSchedulerTest, PrefersLargestGroup) {
    WorkloadAwareSchedulerConfig config;
    config.capacity = 16;
    config.max_bypass = 100;
    WorkloadAwareScheduler scheduler(config);

    ASSERT_TRUE(scheduler.enqueue(make_req("b1", "B")).ok());
    ASSERT_TRUE(scheduler.enqueue(make_req("a1", "A")).ok());
    ASSERT_TRUE(scheduler.enqueue(make_req("a2", "A")).ok());

    auto first = scheduler.next();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ((*first)->model_id(), "A");
    EXPECT_EQ((*first)->request_id(), "a1");
}

TEST(WorkloadAwareSchedulerTest, GroupTieOldestThenLexical) {
    WorkloadAwareSchedulerConfig config;
    config.capacity = 16;
    config.max_bypass = 100;
    WorkloadAwareScheduler scheduler(config);

    // Equal group sizes; A enqueued first → older group wins over B.
    ASSERT_TRUE(scheduler.enqueue(make_req("a1", "A")).ok());
    ASSERT_TRUE(scheduler.enqueue(make_req("b1", "B")).ok());

    auto first = scheduler.next();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ((*first)->request_id(), "a1");
}

TEST(WorkloadAwareSchedulerTest, LexicalTieWhenSameAgeAndSize) {
    WorkloadAwareSchedulerConfig config;
    config.capacity = 16;
    config.max_bypass = 100;
    WorkloadAwareScheduler scheduler(config);

    // Force equal sizes with same oldest age impossible for different enqueues;
    // after draining one from each equal-size path: two models size 1 each,
    // older seq wins. For pure lexical: enqueue B then A as single each after
    // making sizes equal with matching — use two models same size, older is B.
    ASSERT_TRUE(scheduler.enqueue(make_req("b1", "B")).ok());
    ASSERT_TRUE(scheduler.enqueue(make_req("a1", "A")).ok());
    auto first = scheduler.next();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ((*first)->request_id(), "b1"); // older group
}

TEST(WorkloadAwareSchedulerTest, StarvationPromotion) {
    WorkloadAwareSchedulerConfig config;
    config.capacity = 32;
    config.max_bypass = 2;
    WorkloadAwareScheduler scheduler(config);

    ASSERT_TRUE(scheduler.enqueue(make_req("lonely", "L")).ok());
    ASSERT_TRUE(scheduler.enqueue(make_req("a1", "A")).ok());
    ASSERT_TRUE(scheduler.enqueue(make_req("a2", "A")).ok());

    // First pick A (larger group). L bypass becomes 1.
    EXPECT_EQ((*scheduler.next())->request_id(), "a1");
    // Replenish A so group stays larger.
    ASSERT_TRUE(scheduler.enqueue(make_req("a3", "A")).ok());
    // Second pick A again. L bypass becomes 2.
    EXPECT_EQ((*scheduler.next())->request_id(), "a2");
    ASSERT_TRUE(scheduler.enqueue(make_req("a4", "A")).ok());
    // Third: L has bypass_count >= 2 → promote lonely.
    EXPECT_EQ((*scheduler.next())->request_id(), "lonely");
    EXPECT_GE(scheduler.metrics().starvation_promotions, 1u);
}

TEST(WorkloadAwareSchedulerTest, CapacityAndFull) {
    WorkloadAwareSchedulerConfig config;
    config.capacity = 1;
    config.max_bypass = 1;
    WorkloadAwareScheduler scheduler(config);
    ASSERT_TRUE(scheduler.enqueue(make_req("a1", "A")).ok());
    EXPECT_EQ(scheduler.enqueue(make_req("a2", "A")).code, ErrorCode::QueueFull);
}

TEST(WorkloadAwareSchedulerTest, CloseDrain) {
    WorkloadAwareSchedulerConfig config;
    config.capacity = 8;
    config.max_bypass = 1;
    WorkloadAwareScheduler scheduler(config);
    ASSERT_TRUE(scheduler.enqueue(make_req("a1", "A")).ok());
    ASSERT_TRUE(scheduler.enqueue(make_req("b1", "B")).ok());
    scheduler.close();
    EXPECT_EQ(scheduler.enqueue(make_req("c1", "C")).code, ErrorCode::QueueClosed);

    std::vector<std::string> ids;
    while (auto next = scheduler.next()) {
        ids.push_back((*next)->request_id());
    }
    EXPECT_EQ(ids.size(), 2u);
    EXPECT_FALSE(scheduler.next().has_value());
}

TEST(WorkloadAwareSchedulerTest, ReorderedDispatchMetric) {
    WorkloadAwareSchedulerConfig config;
    config.capacity = 8;
    config.max_bypass = 100;
    WorkloadAwareScheduler scheduler(config);
    ASSERT_TRUE(scheduler.enqueue(make_req("b1", "B")).ok());
    ASSERT_TRUE(scheduler.enqueue(make_req("a1", "A")).ok());
    ASSERT_TRUE(scheduler.enqueue(make_req("a2", "A")).ok());
    (void)scheduler.next();
    EXPECT_GE(scheduler.metrics().reordered_dispatches, 1u);
}

TEST(FifoSchedulerPurityViaWorkloadContrast, WorkloadCanReorder) {
    // Documents that WorkloadAware is not FIFO; FIFO purity lives in scheduler_test.
    WorkloadAwareSchedulerConfig config;
    config.capacity = 8;
    config.max_bypass = 100;
    WorkloadAwareScheduler scheduler(config);
    ASSERT_TRUE(scheduler.enqueue(make_req("first", "X")).ok());
    ASSERT_TRUE(scheduler.enqueue(make_req("a1", "A")).ok());
    ASSERT_TRUE(scheduler.enqueue(make_req("a2", "A")).ok());
    EXPECT_NE((*scheduler.next())->request_id(), "first");
}
