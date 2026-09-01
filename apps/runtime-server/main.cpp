#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/router.hpp"
#include "airuntime/runtime.hpp"
#include "airuntime/synthetic_backend.hpp"
#include "airuntime/worker.hpp"
#include "airuntime/workload_aware_scheduler.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {

std::shared_ptr<const airuntime::ModelRegistry> make_registry() {
    airuntime::ModelRegistry::Builder builder;
    airuntime::ModelSpec a{"model-a", 5ull * 1024 * 1024 * 1024, 10};
    airuntime::ModelSpec b{"model-b", 6ull * 1024 * 1024 * 1024, 20};
    if (!builder.add(std::move(a)).ok() || !builder.add(std::move(b)).ok()) {
        return nullptr;
    }
    airuntime::Status status;
    auto registry = builder.build(status);
    if (!status.ok()) {
        return nullptr;
    }
    return std::shared_ptr<const airuntime::ModelRegistry>(std::move(registry));
}

airuntime::WorkerConfig make_worker_config(airuntime::WorkerId id,
                                           std::shared_ptr<const airuntime::ModelRegistry> registry,
                                           std::uint64_t budget) {
    auto backend = std::make_unique<airuntime::SyntheticModelBackend>();
    airuntime::SyntheticModelConfig config;
    config.load_latency = std::chrono::microseconds{10};
    config.prefill_latency = std::chrono::microseconds{5};
    config.per_token_latency = std::chrono::microseconds{1};
    backend->register_model("model-a", config);
    backend->register_model("model-b", config);

    airuntime::WorkerConfig worker_config;
    worker_config.worker_id = id;
    worker_config.queue_capacity = 8;
    worker_config.memory_budget_bytes = budget;
    worker_config.batch_config.max_batch_size = 4;
    worker_config.batch_config.max_batch_wait = std::chrono::microseconds{0};
    worker_config.registry = std::move(registry);
    worker_config.backend = std::move(backend);
    worker_config.eviction_policy = std::make_unique<airuntime::CostAwareEvictionPolicy>();
    return worker_config;
}

} // namespace

int main() {
    using namespace airuntime;

    std::cout << "Adaptive AI Inference Runtime\n";

    auto registry = make_registry();
    if (!registry) {
        std::cerr << "failed to build model registry\n";
        return 1;
    }

    constexpr std::uint64_t kBudget = 12ull * 1024 * 1024 * 1024;
    std::vector<std::unique_ptr<Worker>> workers;
    workers.push_back(std::make_unique<Worker>(make_worker_config(0, registry, kBudget)));
    workers.push_back(std::make_unique<Worker>(make_worker_config(1, registry, kBudget)));

    WorkloadAwareSchedulerConfig sched_config;
    sched_config.capacity = 32;
    sched_config.max_bypass = 8;

    Runtime runtime(std::make_unique<WorkloadAwareScheduler>(sched_config),
                    std::make_unique<ResidencyAwareRouter>(), std::move(workers), registry);

    if (!runtime.start().ok()) {
        std::cerr << "failed to start runtime\n";
        return 1;
    }

    std::vector<RequestPtr> requests;
    requests.push_back(std::make_shared<InferenceRequest>("req-1", "model-a", "hello a1", 2));
    requests.push_back(std::make_shared<InferenceRequest>("req-2", "model-a", "hello a2", 2));
    requests.push_back(std::make_shared<InferenceRequest>("req-3", "model-b", "hello b1", 2));
    requests.push_back(std::make_shared<InferenceRequest>("req-4", "model-a", "hello a3", 2));

    for (const auto &request : requests) {
        if (!runtime.submit(request).ok()) {
            std::cerr << "submit failed\n";
            runtime.stop();
            return 1;
        }
    }

    for (const auto &request : requests) {
        if (!request->wait_for_terminal(std::chrono::seconds{5})) {
            std::cerr << "timed out waiting for requests\n";
            runtime.stop();
            return 1;
        }
        if (request->state() != RequestState::Completed) {
            std::cerr << "request did not complete\n";
            runtime.stop();
            return 1;
        }
        std::cout << "request " << request->request_id() << "\n";
        std::cout << "→ completed\n";
        std::cout << "→ output=" << request->result()->output->text << '\n';
    }

    for (WorkerId id : {WorkerId{0}, WorkerId{1}}) {
        auto *worker = runtime.worker(id);
        if (!worker) {
            continue;
        }
        auto snap = worker->snapshot();
        auto residency = worker->residency_metrics();
        auto batch = worker->batch_metrics();
        std::cout << "worker " << id << " memory_used=" << snap.memory_used_bytes
                  << " loads=" << residency.loads << " hits=" << residency.residency_hits
                  << " batches=" << batch.batches_executed
                  << " multi_batches=" << batch.multi_request_batches
                  << " max_batch=" << batch.max_batch_size_observed << '\n';
    }

    runtime.stop();
    return 0;
}
