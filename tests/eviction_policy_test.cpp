#include "airuntime/eviction_policy.hpp"

#include <gtest/gtest.h>

using airuntime::EvictionCandidate;
using airuntime::LruEvictionPolicy;

TEST(EvictionPolicyTest, OrdersByRecencyThenModelId) {
    LruEvictionPolicy policy;
    std::vector<EvictionCandidate> candidates{
        {"b", 10, 5},
        {"a", 10, 5},
        {"c", 10, 1},
    };
    auto ordered = policy.order_victims(candidates);
    ASSERT_EQ(ordered.size(), 3u);
    EXPECT_EQ(ordered[0].model_id, "c");
    EXPECT_EQ(ordered[1].model_id, "a");
    EXPECT_EQ(ordered[2].model_id, "b");
}

TEST(EvictionPolicyTest, EmptyInput) {
    LruEvictionPolicy policy;
    EXPECT_TRUE(policy.order_victims({}).empty());
}
