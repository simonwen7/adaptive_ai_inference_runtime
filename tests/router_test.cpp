#include "airuntime/router.hpp"
#include "airuntime/worker_snapshot.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using airuntime::ErrorCode;
using airuntime::LeastLoadedRouter;
using airuntime::ModelSpec;
using airuntime::ResidencyAwareRouter;
using airuntime::RoundRobinRouter;
using airuntime::WorkerSnapshot;

namespace {

WorkerSnapshot make_snap(airuntime::WorkerId id, std::uint64_t budget, std::size_t depth = 0,
                         std::size_t active = 0, bool accepting = true, std::uint64_t used = 0,
                         std::vector<std::string> resident = {}) {
    WorkerSnapshot snap;
    snap.worker_id = id;
    snap.queue_depth = depth;
    snap.queue_capacity = 8;
    snap.active_count = active;
    snap.accepting = accepting;
    snap.memory_budget_bytes = budget;
    snap.memory_used_bytes = used;
    snap.resident_model_ids = std::move(resident);
    return snap;
}

} // namespace

TEST(RouterTest, RoundRobinDeterministicSequenceAndWrap) {
    RoundRobinRouter router;
    std::vector<WorkerSnapshot> workers{make_snap(0, 100), make_snap(1, 100), make_snap(2, 100)};
    ModelSpec model{"m", 10};

    EXPECT_EQ(*router.select(model, workers).worker_id, 0u);
    EXPECT_EQ(*router.select(model, workers).worker_id, 1u);
    EXPECT_EQ(*router.select(model, workers).worker_id, 2u);
    EXPECT_EQ(*router.select(model, workers).worker_id, 0u);
}

TEST(RouterTest, RoundRobinSkipsInfeasibleBudget) {
    RoundRobinRouter router;
    std::vector<WorkerSnapshot> workers{make_snap(0, 5), make_snap(1, 100), make_snap(2, 5)};
    ModelSpec model{"m", 10};

    EXPECT_EQ(*router.select(model, workers).worker_id, 1u);
    EXPECT_EQ(*router.select(model, workers).worker_id, 1u);
}

TEST(RouterTest, RoundRobinNoFeasibleWorker) {
    RoundRobinRouter router;
    std::vector<WorkerSnapshot> workers{make_snap(0, 5), make_snap(1, 5)};
    auto result = router.select(ModelSpec{"m", 10}, workers);
    EXPECT_EQ(result.status.code, ErrorCode::NoFeasibleWorker);
    EXPECT_FALSE(result.worker_id.has_value());
}

TEST(RouterTest, RoundRobinIgnoresLoadAndUsedMemory) {
    RoundRobinRouter router;
    auto busy = make_snap(0, 100, 99, 1, true, 99, {"m"});
    std::vector<WorkerSnapshot> workers{busy, make_snap(1, 100)};
    EXPECT_EQ(*router.select(ModelSpec{"m", 10}, workers).worker_id, 0u);
}

TEST(RouterTest, RoundRobinIgnoresResidencySnapshot) {
    RoundRobinRouter router;
    std::vector<WorkerSnapshot> workers{make_snap(0, 100, 0, 0, true, 0, {}),
                                        make_snap(1, 100, 0, 0, true, 0, {"m"})};
    // RR still picks worker 0 first despite residency on worker 1.
    EXPECT_EQ(*router.select(ModelSpec{"m", 10}, workers).worker_id, 0u);
}

TEST(RouterTest, LeastLoadedPrefersLowerLoad) {
    LeastLoadedRouter router;
    std::vector<WorkerSnapshot> workers{make_snap(0, 100, 3, 1), make_snap(1, 100, 0, 0),
                                        make_snap(2, 100, 1, 0)};
    EXPECT_EQ(*router.select(ModelSpec{"m", 10}, workers).worker_id, 1u);
}

TEST(RouterTest, LeastLoadedActiveCountsAsOne) {
    LeastLoadedRouter router;
    std::vector<WorkerSnapshot> workers{make_snap(0, 100, 0, 1), make_snap(1, 100, 0, 0)};
    EXPECT_EQ(*router.select(ModelSpec{"m", 10}, workers).worker_id, 1u);
}

TEST(RouterTest, LeastLoadedTieBreaksLowestId) {
    LeastLoadedRouter router;
    std::vector<WorkerSnapshot> workers{make_snap(2, 100, 1, 0), make_snap(0, 100, 1, 0),
                                        make_snap(1, 100, 1, 0)};
    EXPECT_EQ(*router.select(ModelSpec{"m", 10}, workers).worker_id, 0u);
}

TEST(RouterTest, LeastLoadedSkipsInfeasible) {
    LeastLoadedRouter router;
    std::vector<WorkerSnapshot> workers{make_snap(0, 5, 0, 0), make_snap(1, 100, 5, 1)};
    EXPECT_EQ(*router.select(ModelSpec{"m", 10}, workers).worker_id, 1u);
}

TEST(RouterTest, LeastLoadedNoFeasible) {
    LeastLoadedRouter router;
    auto result = router.select(ModelSpec{"m", 50},
                                std::vector<WorkerSnapshot>{make_snap(0, 10), make_snap(1, 20)});
    EXPECT_EQ(result.status.code, ErrorCode::NoFeasibleWorker);
}

TEST(RouterTest, LeastLoadedIgnoresResidency) {
    LeastLoadedRouter router;
    std::vector<WorkerSnapshot> workers{make_snap(0, 100, 0, 0, true, 90, {}),
                                        make_snap(1, 100, 0, 0, true, 0, {"m"})};
    // Equal load → lowest id, ignoring residency and used memory.
    EXPECT_EQ(*router.select(ModelSpec{"m", 10}, workers).worker_id, 0u);
}

TEST(ResidencyAwareRouterTest, PrefersResident) {
    ResidencyAwareRouter router;
    std::vector<WorkerSnapshot> workers{make_snap(0, 100, 0, 0, true, 0, {}),
                                        make_snap(1, 100, 5, 0, true, 10, {"m"})};
    EXPECT_EQ(*router.select(ModelSpec{"m", 10}, workers).worker_id, 1u);
    EXPECT_GE(router.metrics().resident_selections, 1u);
}

TEST(ResidencyAwareRouterTest, PrefersNoEvictionOverEvictionRequired) {
    ResidencyAwareRouter router;
    // Worker 0: not resident, needs eviction (used 90, need 20, free 10)
    // Worker 1: not resident, free fits (used 0)
    std::vector<WorkerSnapshot> workers{make_snap(0, 100, 0, 0, true, 90, {"other"}),
                                        make_snap(1, 100, 2, 0, true, 0, {})};
    EXPECT_EQ(*router.select(ModelSpec{"m", 20}, workers).worker_id, 1u);
    EXPECT_GE(router.metrics().no_eviction_load_selections, 1u);
}

TEST(ResidencyAwareRouterTest, SameTierLowerLoadThenLowestId) {
    ResidencyAwareRouter router;
    std::vector<WorkerSnapshot> workers{make_snap(2, 100, 3, 0, true, 0, {"m"}),
                                        make_snap(0, 100, 1, 0, true, 0, {"m"}),
                                        make_snap(1, 100, 1, 0, true, 0, {"m"})};
    EXPECT_EQ(*router.select(ModelSpec{"m", 10}, workers).worker_id, 0u);
}

TEST(ResidencyAwareRouterTest, NoFeasible) {
    ResidencyAwareRouter router;
    auto result = router.select(ModelSpec{"m", 50},
                                std::vector<WorkerSnapshot>{make_snap(0, 10), make_snap(1, 20)});
    EXPECT_EQ(result.status.code, ErrorCode::NoFeasibleWorker);
}

TEST(ResidencyAwareRouterTest, EvictionRequiredTierWhenOnlyOption) {
    ResidencyAwareRouter router;
    std::vector<WorkerSnapshot> workers{make_snap(0, 100, 0, 0, true, 90, {"other"})};
    EXPECT_EQ(*router.select(ModelSpec{"m", 20}, workers).worker_id, 0u);
    EXPECT_GE(router.metrics().eviction_required_selections, 1u);
}
