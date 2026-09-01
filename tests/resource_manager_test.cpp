#include "airuntime/resource_manager.hpp"

#include <gtest/gtest.h>
#include <limits>

using airuntime::ErrorCode;
using airuntime::ResourceManager;

TEST(ResourceManagerTest, ReserveReleaseAndBudget) {
    ResourceManager resources(100);
    EXPECT_EQ(resources.budget(), 100u);
    EXPECT_EQ(resources.used(), 0u);
    EXPECT_EQ(resources.available(), 100u);

    ASSERT_TRUE(resources.reserve(40).ok());
    EXPECT_EQ(resources.used(), 40u);
    EXPECT_EQ(resources.available(), 60u);
    EXPECT_TRUE(resources.can_reserve(60));
    EXPECT_FALSE(resources.can_reserve(61));

    EXPECT_EQ(resources.reserve(61).code, ErrorCode::InsufficientMemory);
    ASSERT_TRUE(resources.reserve(60).ok());
    EXPECT_EQ(resources.used(), 100u);

    ASSERT_TRUE(resources.release(30).ok());
    EXPECT_EQ(resources.used(), 70u);
    EXPECT_EQ(resources.release(80).code, ErrorCode::InternalError);
    EXPECT_EQ(resources.used(), 70u);
}

TEST(ResourceManagerTest, ExactFullReservation) {
    ResourceManager resources(50);
    ASSERT_TRUE(resources.reserve(50).ok());
    EXPECT_FALSE(resources.can_reserve(1));
}

TEST(ResourceManagerTest, OverflowSafeNearMax) {
    ResourceManager resources(std::numeric_limits<std::uint64_t>::max());
    ASSERT_TRUE(resources.reserve(10).ok());
    EXPECT_TRUE(resources.can_reserve(std::numeric_limits<std::uint64_t>::max() - 10));
    EXPECT_FALSE(resources.can_reserve(std::numeric_limits<std::uint64_t>::max()));
}
