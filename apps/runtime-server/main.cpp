#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/router.hpp"
#include "airuntime/runtime.hpp"
#include "airuntime/serving/http_session.hpp"
#include "airuntime/serving/request_handler.hpp"
#include "airuntime/synthetic_backend.hpp"
#include "airuntime/worker.hpp"
#include "airuntime/workload_aware_scheduler.hpp"

#if defined(AIRUNTIME_ENABLE_LLAMA)
#include "airuntime/llama/llama_backend_runtime.hpp"
#include "airuntime/llama/llama_cpp_backend.hpp"
#include "airuntime/llama/llama_cpp_backend_config.hpp"
#endif

#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/signal_set.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

struct ServerOptions {
    std::string host{"127.0.0.1"};
    unsigned short port{8080};
    std::string backend{"synthetic"};
    std::string model_id;
    std::filesystem::path model_path;
    std::uint32_t ctx_size{2048};
    std::optional<std::int32_t> gpu_layers;
    std::optional<std::int32_t> threads;
    std::optional<std::size_t> workers;
};

void print_usage() {
    std::cerr << "Usage: runtime-server [--host HOST] [--port PORT]\n"
              << "                      [--backend synthetic|llama]\n"
              << "                      [--model-id ID] [--model-path PATH]\n"
              << "                      [--ctx-size N] [--gpu-layers N] [--threads N]\n"
              << "                      [--workers N]\n";
}

bool parse_args(int argc, char **argv, ServerOptions &opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << '\n';
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return false;
        }
        if (arg == "--host") {
            const char *v = need_value("--host");
            if (!v) {
                return false;
            }
            opts.host = v;
            continue;
        }
        if (arg == "--port") {
            const char *v = need_value("--port");
            if (!v) {
                return false;
            }
            opts.port = static_cast<unsigned short>(std::stoi(v));
            continue;
        }
        if (arg == "--backend") {
            const char *v = need_value("--backend");
            if (!v) {
                return false;
            }
            opts.backend = v;
            continue;
        }
        if (arg == "--model-id") {
            const char *v = need_value("--model-id");
            if (!v) {
                return false;
            }
            opts.model_id = v;
            continue;
        }
        if (arg == "--model-path") {
            const char *v = need_value("--model-path");
            if (!v) {
                return false;
            }
            opts.model_path = v;
            continue;
        }
        if (arg == "--ctx-size") {
            const char *v = need_value("--ctx-size");
            if (!v) {
                return false;
            }
            opts.ctx_size = static_cast<std::uint32_t>(std::stoul(v));
            continue;
        }
        if (arg == "--gpu-layers") {
            const char *v = need_value("--gpu-layers");
            if (!v) {
                return false;
            }
            opts.gpu_layers = std::stoi(v);
            continue;
        }
        if (arg == "--threads") {
            const char *v = need_value("--threads");
            if (!v) {
                return false;
            }
            opts.threads = std::stoi(v);
            continue;
        }
        if (arg == "--workers") {
            const char *v = need_value("--workers");
            if (!v) {
                return false;
            }
            opts.workers = static_cast<std::size_t>(std::stoull(v));
            continue;
        }
        std::cerr << "unknown argument: " << arg << '\n';
        print_usage();
        return false;
    }
    return true;
}

std::shared_ptr<const airuntime::ModelRegistry> make_synthetic_registry() {
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

airuntime::WorkerConfig
make_synthetic_worker(airuntime::WorkerId id,
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

#if defined(AIRUNTIME_ENABLE_LLAMA)
std::shared_ptr<const airuntime::ModelRegistry> make_llama_registry(const ServerOptions &opts,
                                                                    std::uint64_t estimate_bytes) {
    airuntime::ModelRegistry::Builder builder;
    airuntime::ModelSpec spec{opts.model_id, estimate_bytes, 1};
    if (!builder.add(std::move(spec)).ok()) {
        return nullptr;
    }
    airuntime::Status status;
    auto registry = builder.build(status);
    if (!status.ok()) {
        return nullptr;
    }
    return std::shared_ptr<const airuntime::ModelRegistry>(std::move(registry));
}

airuntime::WorkerConfig
make_llama_worker(airuntime::WorkerId id, std::shared_ptr<const airuntime::ModelRegistry> registry,
                  std::uint64_t budget,
                  const std::shared_ptr<airuntime::LlamaBackendRuntime> &runtime,
                  const ServerOptions &opts, std::size_t max_sequences) {
    airuntime::LlamaCppBackendConfig backend_config;
    backend_config.max_sequences = max_sequences;
    airuntime::LlamaModelConfig model_config;
    model_config.gguf_path = opts.model_path;
    model_config.context_tokens_per_sequence = opts.ctx_size;
    model_config.n_batch = std::max<std::uint32_t>(512, static_cast<std::uint32_t>(max_sequences));
    model_config.n_ubatch = model_config.n_batch;
    if (opts.gpu_layers.has_value()) {
        model_config.n_gpu_layers = *opts.gpu_layers;
    }
    if (opts.threads.has_value()) {
        model_config.n_threads = *opts.threads;
        model_config.n_threads_batch = *opts.threads;
    }
    backend_config.models.emplace(opts.model_id, std::move(model_config));

    airuntime::WorkerConfig worker_config;
    worker_config.worker_id = id;
    worker_config.queue_capacity = 8;
    worker_config.memory_budget_bytes = budget;
    worker_config.batch_config.max_batch_size = max_sequences;
    worker_config.batch_config.max_batch_wait = std::chrono::microseconds{0};
    worker_config.registry = std::move(registry);
    worker_config.backend =
        std::make_unique<airuntime::LlamaCppBackend>(runtime, std::move(backend_config));
    worker_config.eviction_policy = std::make_unique<airuntime::CostAwareEvictionPolicy>();
    return worker_config;
}
#endif

} // namespace

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    ServerOptions opts;
    if (!parse_args(argc, argv, opts)) {
        return 0;
    }

    std::cout << "Adaptive AI Inference Runtime\n";

    std::shared_ptr<const airuntime::ModelRegistry> registry;
    std::vector<std::unique_ptr<airuntime::Worker>> workers;
    constexpr std::uint64_t kBudget = 12ull * 1024 * 1024 * 1024;

    if (opts.backend == "synthetic") {
        registry = make_synthetic_registry();
        if (!registry) {
            std::cerr << "failed to build model registry\n";
            return 1;
        }
        const std::size_t worker_count = opts.workers.value_or(2);
        for (std::size_t i = 0; i < worker_count; ++i) {
            workers.push_back(std::make_unique<airuntime::Worker>(
                make_synthetic_worker(static_cast<airuntime::WorkerId>(i), registry, kBudget)));
        }
    } else if (opts.backend == "llama") {
#if !defined(AIRUNTIME_ENABLE_LLAMA)
        std::cerr << "runtime-server was built without AIRUNTIME_ENABLE_LLAMA\n";
        return 1;
#else
        if (opts.model_id.empty() || opts.model_path.empty()) {
            std::cerr << "llama backend requires --model-id and --model-path\n";
            return 1;
        }
        if (opts.ctx_size == 0) {
            std::cerr << "--ctx-size must be > 0\n";
            return 1;
        }
        std::error_code ec;
        if (!std::filesystem::exists(opts.model_path, ec) ||
            !std::filesystem::is_regular_file(opts.model_path, ec)) {
            std::cerr << "model path missing or not a regular file\n";
            return 1;
        }
        const auto estimate =
            static_cast<std::uint64_t>(std::filesystem::file_size(opts.model_path, ec));
        if (ec || estimate == 0) {
            std::cerr << "failed to read GGUF file size for policy estimate\n";
            return 1;
        }
        registry = make_llama_registry(opts, estimate);
        if (!registry) {
            std::cerr << "failed to build llama model registry\n";
            return 1;
        }
        auto llama_runtime = airuntime::LlamaBackendRuntime::create();
        const std::size_t worker_count = opts.workers.value_or(1);
        const std::size_t max_sequences = 4;
        for (std::size_t i = 0; i < worker_count; ++i) {
            workers.push_back(std::make_unique<airuntime::Worker>(
                make_llama_worker(static_cast<airuntime::WorkerId>(i), registry, kBudget,
                                  llama_runtime, opts, max_sequences)));
        }
        std::cout << "llama backend enabled (model_id=" << opts.model_id
                  << ", gpu_offload_supported="
                  << (airuntime::llama_runtime_supports_gpu_offload() ? "true" : "false") << ")\n";
#endif
    } else {
        std::cerr << "unknown --backend (expected synthetic|llama)\n";
        return 1;
    }

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
    boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::make_address(opts.host), opts.port);
    airuntime::serving::HttpServer server(ioc, endpoint, handler);

    boost::asio::signal_set signals(ioc, SIGINT, SIGTERM);
    signals.async_wait([&](const boost::system::error_code &, int) {
        std::cerr << "shutdown signal received\n";
        server.stop();
    });

    std::cout << "Listening on http://" << opts.host << ':' << server.port() << std::endl;
    std::cout.flush();
    server.run();
    runtime.stop();
    return 0;
}
