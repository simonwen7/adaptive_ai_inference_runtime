#include "airuntime/eviction_policy.hpp"

#include <gtest/gtest.h>

using airuntime::CostAwareEvictionPolicy;
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

TEST(EvictionPolicyTest, LruIgnoresCostUseAndMemory) {
    LruEvictionPolicy policy;
    EvictionCandidate cheap_old{"cheap", 100, 1, /*load_cost*/ 1, /*use*/ 0};
    EvictionCandidate expensive_newer{"expensive", 1, 2, /*load_cost*/ 1000, /*use*/ 99};
    auto ordered = policy.order_victims({expensive_newer, cheap_old});
    ASSERT_EQ(ordered.size(), 2u);
    EXPECT_EQ(ordered[0].model_id, "cheap"); // older last_used only
    EXPECT_EQ(ordered[1].model_id, "expensive");
}

TEST(CostAwareEvictionTest, ExactLexicographicOrder) {
    CostAwareEvictionPolicy policy;
    // Differ only by load cost
    EvictionCandidate high_cost{"h", 10, 1, 50, 1};
    EvictionCandidate low_cost{"l", 10, 1, 5, 1};
    auto by_cost = policy.order_victims({high_cost, low_cost});
    EXPECT_EQ(by_cost[0].model_id, "l");

    // Same cost: lower use_count first
    EvictionCandidate low_use{"u0", 10, 1, 5, 0};
    EvictionCandidate high_use{"u1", 10, 1, 5, 3};
    auto by_use = policy.order_victims({high_use, low_use});
    EXPECT_EQ(by_use[0].model_id, "u0");

    // Same cost+use: larger memory first
    EvictionCandidate small{"s", 5, 1, 5, 1};
    EvictionCandidate large{"g", 50, 1, 5, 1};
    auto by_mem = policy.order_victims({small, large});
    EXPECT_EQ(by_mem[0].model_id, "g");

    // Same cost+use+mem: older last_used first
    EvictionCandidate older{"old", 10, 1, 5, 1};
    EvictionCandidate newer{"new", 10, 9, 5, 1};
    auto by_recency = policy.order_victims({newer, older});
    EXPECT_EQ(by_recency[0].model_id, "old");

    // Final tie: lexical model_id
    EvictionCandidate a{"a", 10, 1, 5, 1};
    EvictionCandidate b{"b", 10, 1, 5, 1};
    auto by_id = policy.order_victims({b, a});
    EXPECT_EQ(by_id[0].model_id, "a");
}

TEST(CostAwareEvictionTest, DiffersFromLru) {
    LruEvictionPolicy lru;
    CostAwareEvictionPolicy cost;

    EvictionCandidate old_expensive{"exp", 10, 1, 1000, 5};
    EvictionCandidate new_cheap{"cheap", 10, 9, 1, 0};

    auto lru_order = lru.order_victims({old_expensive, new_cheap});
    auto cost_order = cost.order_victims({old_expensive, new_cheap});

    EXPECT_EQ(lru_order[0].model_id, "exp");    // older
    EXPECT_EQ(cost_order[0].model_id, "cheap"); // cheaper / lower use
}

TEST(CostAwareEvictionTest, EmptyAndMultiple) {
    CostAwareEvictionPolicy policy;
    EXPECT_TRUE(policy.order_victims({}).empty());
    auto ordered = policy.order_victims({{"c", 3, 3, 2, 1}, {"a", 9, 1, 2, 1}, {"b", 5, 2, 2, 1}});
    ASSERT_EQ(ordered.size(), 3u);
    // same cost+use: larger memory first → a(9), b(5), c(3)
    EXPECT_EQ(ordered[0].model_id, "a");
    EXPECT_EQ(ordered[1].model_id, "b");
    EXPECT_EQ(ordered[2].model_id, "c");
}
