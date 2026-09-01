#include "airuntime/model_registry.hpp"

#include <gtest/gtest.h>

using airuntime::ErrorCode;
using airuntime::ModelRegistry;
using airuntime::ModelSpec;
using airuntime::Status;

TEST(ModelRegistryTest, LookupAndUnknown) {
    ModelRegistry::Builder builder;
    ASSERT_TRUE(builder.add({"m1", 10}).ok());
    Status status;
    auto registry = builder.build(status);
    ASSERT_TRUE(status.ok());
    ASSERT_NE(registry, nullptr);

    ModelSpec found;
    ASSERT_TRUE(registry->find("m1", found).ok());
    EXPECT_EQ(found.model_id, "m1");
    EXPECT_EQ(found.estimated_memory_bytes, 10u);
    EXPECT_TRUE(registry->contains("m1"));
    EXPECT_FALSE(registry->contains("missing"));
    EXPECT_EQ(registry->find("missing", found).code, ErrorCode::ModelNotFound);
}

TEST(ModelRegistryTest, RejectsInvalidEntries) {
    ModelRegistry::Builder builder;
    EXPECT_EQ(builder.add({"", 10}).code, ErrorCode::InternalError);
    EXPECT_EQ(builder.add({"m1", 0}).code, ErrorCode::InternalError);
    ASSERT_TRUE(builder.add({"m1", 10}).ok());
    EXPECT_EQ(builder.add({"m1", 20}).code, ErrorCode::InternalError);
}

TEST(ModelRegistryTest, EmptyBuildRejected) {
    ModelRegistry::Builder builder;
    Status status;
    auto registry = builder.build(status);
    EXPECT_EQ(status.code, ErrorCode::InternalError);
    EXPECT_EQ(registry, nullptr);
}
