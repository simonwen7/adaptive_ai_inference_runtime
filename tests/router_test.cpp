#include "airuntime/router.hpp"
#include "airuntime/worker_snapshot.hpp"

#include <gtest/gtest.h>
#include <vector>

using airuntime::ErrorCode;
using airuntime::LeastLoadedRouter;
using airuntime::ModelSpec;
using airuntime::RoundRobinRouter;
using airuntime::WorkerSnapshot;

namespace {

WorkerSnapshot make_snap(airuntime::WorkerId id, std::uint64_t budget, std::size_t depth = 0,
                         std::size_t active = 0, bool accepting = true) {
    WorkerSnapshot snap;
    snap.worker_id = id;
    snap.queue_depth = depth;
    snap.queue_capacity = 8;
    snap.active_count = active;
    snap.accepting = accepting;
    snap.memory_budget_bytes = budget;
    snap.memory_used_bytes = 0;
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
    auto busy = make_snap(0, 100, 99, 1);
    busy.memory_used_bytes = 99;
    std::vector<WorkerSnapshot> workers{busy, make_snap(1, 100)};
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
