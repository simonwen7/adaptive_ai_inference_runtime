#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/router.hpp"
#include "airuntime/runtime.hpp"
#include "airuntime/serving/http_session.hpp"
#include "airuntime/serving/request_handler.hpp"
#include "airuntime/synthetic_backend.hpp"
#include "airuntime/worker.hpp"
#include "airuntime/workload_aware_scheduler.hpp"

#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/signal_set.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
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

void print_usage() {
    std::cerr << "Usage: runtime-server [--host HOST] [--port PORT]\n";
}

bool parse_args(int argc, char **argv, std::string &host, unsigned short &port) {
    host = "127.0.0.1";
    port = 8080;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return false;
        }
        if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
            continue;
        }
        if (arg == "--port" && i + 1 < argc) {
            port = static_cast<unsigned short>(std::stoi(argv[++i]));
            continue;
        }
        std::cerr << "unknown argument: " << arg << '\n';
        print_usage();
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    std::string host;
    unsigned short port = 0;
    if (!parse_args(argc, argv, host, port)) {
        return 0;
    }

    std::cout << "Adaptive AI Inference Runtime\n";

    auto registry = make_registry();
    if (!registry) {
        std::cerr << "failed to build model registry\n";
        return 1;
    }

    constexpr std::uint64_t kBudget = 12ull * 1024 * 1024 * 1024;
    std::vector<std::unique_ptr<airuntime::Worker>> workers;
    workers.push_back(
        std::make_unique<airuntime::Worker>(make_worker_config(0, registry, kBudget)));
    workers.push_back(
        std::make_unique<airuntime::Worker>(make_worker_config(1, registry, kBudget)));

    airuntime::WorkloadAwareSchedulerConfig sched_config;
    sched_config.capacity = 32;
    sched_config.max_bypass = 8;

    airuntime::Runtime runtime(std::make_unique<airuntime::WorkloadAwareScheduler>(sched_config),
                               std::make_unique<airuntime::ResidencyAwareRouter>(),
                               std::move(workers), registry);

    if (!runtime.start().ok()) {
        std::cerr << "failed to start runtime\n";
        return 1;
    }

    boost::asio::io_context ioc;
    airuntime::serving::RequestHandler handler(runtime);
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::make_address(host), port);
    airuntime::serving::HttpServer server(ioc, endpoint, handler);

    boost::asio::signal_set signals(ioc, SIGINT, SIGTERM);
    signals.async_wait([&](const boost::system::error_code &, int) {
        std::cerr << "shutdown signal received\n";
        server.stop();
    });

    std::cout << "Listening on http://" << host << ':' << server.port() << std::endl;
    std::cout.flush();
    server.run();
    runtime.stop();
    return 0;
}
