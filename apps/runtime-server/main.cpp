#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/router.hpp"
#include "airuntime/runtime.hpp"
#include "airuntime/scheduler.hpp"
#include "airuntime/synthetic_backend.hpp"
#include "airuntime/worker.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {

std::shared_ptr<const airuntime::ModelRegistry> make_registry() {
    airuntime::ModelRegistry::Builder builder;
    if (!builder.add({"model-a", 5ull * 1024 * 1024 * 1024}).ok()) {
        return nullptr;
    }
    if (!builder.add({"model-b", 6ull * 1024 * 1024 * 1024}).ok()) {
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
    worker_config.registry = std::move(registry);
    worker_config.backend = std::move(backend);
    worker_config.eviction_policy = std::make_unique<airuntime::LruEvictionPolicy>();
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

    Runtime runtime(std::make_unique<FifoScheduler>(32), std::make_unique<RoundRobinRouter>(),
                    std::move(workers), registry);

    if (!runtime.start().ok()) {
        std::cerr << "failed to start runtime\n";
        return 1;
    }

    auto req1 = std::make_shared<InferenceRequest>("req-1", "model-a", "hello a", 2);
    auto req2 = std::make_shared<InferenceRequest>("req-2", "model-b", "hello b", 2);

    if (!runtime.submit(req1).ok() || !runtime.submit(req2).ok()) {
        std::cerr << "submit failed\n";
        runtime.stop();
        return 1;
    }

    if (!req1->wait_for_terminal(std::chrono::seconds{5}) ||
        !req2->wait_for_terminal(std::chrono::seconds{5})) {
        std::cerr << "timed out waiting for requests\n";
        runtime.stop();
        return 1;
    }

    if (req1->state() != RequestState::Completed || req2->state() != RequestState::Completed) {
        std::cerr << "request did not complete\n";
        runtime.stop();
        return 1;
    }

    std::cout << "request req-1\n";
    std::cout << "→ completed\n";
    std::cout << "→ output=" << req1->result()->output->text << '\n';
    std::cout << "request req-2\n";
    std::cout << "→ completed\n";
    std::cout << "→ output=" << req2->result()->output->text << '\n';

    for (WorkerId id : {WorkerId{0}, WorkerId{1}}) {
        auto *worker = runtime.worker(id);
        if (!worker) {
            continue;
        }
        auto snap = worker->snapshot();
        auto metrics = worker->residency_metrics();
        std::cout << "worker " << id << " memory_used=" << snap.memory_used_bytes
                  << " loads=" << metrics.loads << " hits=" << metrics.residency_hits << '\n';
    }

    runtime.stop();
    return 0;
}
