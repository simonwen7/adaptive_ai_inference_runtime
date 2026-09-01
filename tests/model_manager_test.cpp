#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_manager.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/synthetic_backend.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

using airuntime::ErrorCode;
using airuntime::LruEvictionPolicy;
using airuntime::ModelManager;
using airuntime::ModelRegistry;
using airuntime::ModelState;
using airuntime::SyntheticModelBackend;
using airuntime::SyntheticModelConfig;

namespace {

std::shared_ptr<const ModelRegistry> make_registry() {
    ModelRegistry::Builder builder;
    EXPECT_TRUE(builder.add({"A", 5}).ok());
    EXPECT_TRUE(builder.add({"B", 6}).ok());
    EXPECT_TRUE(builder.add({"C", 4}).ok());
    EXPECT_TRUE(builder.add({"huge", 100}).ok());
    airuntime::Status status;
    auto registry = builder.build(status);
    EXPECT_TRUE(status.ok());
    return std::shared_ptr<const ModelRegistry>(std::move(registry));
}

std::unique_ptr<SyntheticModelBackend> make_backend(bool fail_load = false) {
    auto backend = std::make_unique<SyntheticModelBackend>();
    SyntheticModelConfig config;
    config.load_latency = std::chrono::microseconds{1};
    config.fail_load = fail_load;
    for (const char *id : {"A", "B", "C", "huge"}) {
        backend->register_model(id, config);
    }
    return backend;
}

} // namespace

TEST(ModelManagerTest, HitMissLoadAndNoDuplicateLoad) {
    auto backend_ptr = make_backend();
    auto *backend = backend_ptr.get();
    ModelManager manager(make_registry(), *backend_ptr, 12, std::make_unique<LruEvictionPolicy>());

    ASSERT_TRUE(manager.ensure_resident("A").ok());
    EXPECT_EQ(manager.model_state("A"), ModelState::Resident);
    EXPECT_EQ(manager.metrics().residency_misses, 1u);
    EXPECT_EQ(manager.metrics().loads, 1u);
    EXPECT_EQ(backend->metrics().load_count, 1u);

    ASSERT_TRUE(manager.ensure_resident("A").ok());
    EXPECT_EQ(manager.metrics().residency_hits, 1u);
    EXPECT_EQ(manager.metrics().loads, 1u);
    EXPECT_EQ(backend->metrics().load_count, 1u);
    EXPECT_EQ(manager.memory_used_bytes(), 5u);
}

TEST(ModelManagerTest, MultipleEvictionForMemoryPressure) {
    auto backend = make_backend();
    ModelManager manager(make_registry(), *backend, 12, std::make_unique<LruEvictionPolicy>());

    ASSERT_TRUE(manager.ensure_resident("A").ok());
    ASSERT_TRUE(manager.ensure_resident("B").ok());
    EXPECT_EQ(manager.memory_used_bytes(), 11u);

    ASSERT_TRUE(manager.ensure_resident("C").ok());
    EXPECT_LE(manager.memory_used_bytes(), 12u);
    EXPECT_EQ(manager.model_state("C"), ModelState::Resident);
    EXPECT_GE(manager.metrics().evictions, 1u);
    EXPECT_EQ(manager.model_state("A") == ModelState::Unloaded ||
                  manager.model_state("B") == ModelState::Unloaded,
              true);
}

TEST(ModelManagerTest, ModelLargerThanBudget) {
    auto backend = make_backend();
    ModelManager manager(make_registry(), *backend, 12, std::make_unique<LruEvictionPolicy>());
    ASSERT_TRUE(manager.ensure_resident("A").ok());
    auto status = manager.ensure_resident("huge");
    EXPECT_EQ(status.code, ErrorCode::InsufficientMemory);
    EXPECT_EQ(manager.model_state("A"), ModelState::Resident);
    EXPECT_EQ(manager.metrics().evictions, 0u);
}

TEST(ModelManagerTest, LoadFailureRollback) {
    auto backend = make_backend(true);
    ModelManager manager(make_registry(), *backend, 12, std::make_unique<LruEvictionPolicy>());
    auto status = manager.ensure_resident("A");
    EXPECT_EQ(status.code, ErrorCode::ModelLoadFailed);
    EXPECT_EQ(manager.model_state("A"), ModelState::Failed);
    EXPECT_EQ(manager.memory_used_bytes(), 0u);
}

TEST(ModelManagerTest, ReloadMetric) {
    auto backend = make_backend();
    ModelManager manager(make_registry(), *backend, 5, std::make_unique<LruEvictionPolicy>());
    ASSERT_TRUE(manager.ensure_resident("A").ok());
    ASSERT_TRUE(manager.ensure_resident("C").ok()); // evicts A (budget 5, C=4)
    ASSERT_TRUE(manager.ensure_resident("A").ok());
    EXPECT_GE(manager.metrics().reloads, 1u);
    EXPECT_GE(manager.metrics().evictions, 1u);
}

TEST(ModelManagerTest, UnknownModel) {
    auto backend = make_backend();
    ModelManager manager(make_registry(), *backend, 12, std::make_unique<LruEvictionPolicy>());
    EXPECT_EQ(manager.ensure_resident("missing").code, ErrorCode::ModelNotFound);
}

TEST(ModelManagerTest, UseCountIncrementsOncePerSuccessfulEnsure) {
    auto backend = make_backend();
    ModelManager manager(make_registry(), *backend, 12, std::make_unique<LruEvictionPolicy>());
    ASSERT_TRUE(manager.ensure_resident("A").ok());
    EXPECT_EQ(manager.use_count("A"), 1u);
    ASSERT_TRUE(manager.ensure_resident("A").ok());
    EXPECT_EQ(manager.use_count("A"), 2u);
    EXPECT_EQ(manager.resident_model_ids(), (std::vector<std::string>{"A"}));
}
