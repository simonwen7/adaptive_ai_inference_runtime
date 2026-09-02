#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/router.hpp"
#include "airuntime/runtime.hpp"
#include "airuntime/scheduler.hpp"
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
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
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

    std::size_t max_batch_size{4};
    std::uint64_t max_batch_wait_ms{0};
    std::size_t scheduler_capacity{32};
    std::size_t worker_queue_capacity{8};
    std::string scheduler{"workload"};
};

void print_usage() {
    std::cerr << "Usage: runtime-server [--host HOST] [--port PORT]\n"
              << "                      [--backend synthetic|llama]\n"
              << "                      [--model-id ID] [--model-path PATH]\n"
              << "                      [--ctx-size N] [--gpu-layers N] [--threads N]\n"
              << "                      [--workers N]\n"
              << "                      [--max-batch-size N] [--max-batch-wait-ms N]\n"
              << "                      [--scheduler-capacity N] [--worker-queue-capacity N]\n"
              << "                      [--scheduler fifo|workload]\n";
}

bool parse_positive_size(const char *name, const char *text, std::size_t &out) {
    try {
        if (text == nullptr || text[0] == '\0' || text[0] == '-') {
            std::cerr << name << " must be a positive integer\n";
            return false;
        }
        const unsigned long long value = std::stoull(text);
        if (value == 0 ||
            value > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
            std::cerr << name << " must be a positive integer\n";
            return false;
        }
        out = static_cast<std::size_t>(value);
        return true;
    } catch (const std::exception &) {
        std::cerr << "malformed value for " << name << '\n';
        return false;
    }
}

bool parse_nonnegative_u64(const char *name, const char *text, std::uint64_t &out) {
    try {
        if (text == nullptr || text[0] == '\0' || text[0] == '-') {
            std::cerr << name << " must be a nonnegative integer\n";
            return false;
        }
        out = std::stoull(text);
        return true;
    } catch (const std::exception &) {
        std::cerr << "malformed value for " << name << '\n';
        return false;
    }
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
            try {
                const int port = std::stoi(v);
                if (port < 0 || port > 65535) {
                    std::cerr << "--port out of range\n";
                    return false;
                }
                opts.port = static_cast<unsigned short>(port);
            } catch (const std::exception &) {
                std::cerr << "malformed value for --port\n";
                return false;
            }
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
            std::size_t ctx = 0;
            if (!parse_positive_size("--ctx-size", v, ctx)) {
                return false;
            }
            opts.ctx_size = static_cast<std::uint32_t>(ctx);
            continue;
        }
        if (arg == "--gpu-layers") {
            const char *v = need_value("--gpu-layers");
            if (!v) {
                return false;
            }
            try {
                opts.gpu_layers = std::stoi(v);
            } catch (const std::exception &) {
                std::cerr << "malformed value for --gpu-layers\n";
                return false;
            }
            continue;
        }
        if (arg == "--threads") {
            const char *v = need_value("--threads");
            if (!v) {
                return false;
            }
            try {
                const int threads = std::stoi(v);
                if (threads <= 0) {
                    std::cerr << "--threads must be > 0\n";
                    return false;
                }
                opts.threads = threads;
            } catch (const std::exception &) {
                std::cerr << "malformed value for --threads\n";
                return false;
            }
            continue;
        }
        if (arg == "--workers") {
            const char *v = need_value("--workers");
            if (!v) {
                return false;
            }
            std::size_t workers = 0;
            if (!parse_positive_size("--workers", v, workers)) {
                return false;
            }
            opts.workers = workers;
            continue;
        }
        if (arg == "--max-batch-size") {
            const char *v = need_value("--max-batch-size");
            if (!v) {
                return false;
            }
            if (!parse_positive_size("--max-batch-size", v, opts.max_batch_size)) {
                return false;
            }
            continue;
        }
        if (arg == "--max-batch-wait-ms") {
            const char *v = need_value("--max-batch-wait-ms");
            if (!v) {
                return false;
            }
            if (!parse_nonnegative_u64("--max-batch-wait-ms", v, opts.max_batch_wait_ms)) {
                return false;
            }
            continue;
        }
        if (arg == "--scheduler-capacity") {
            const char *v = need_value("--scheduler-capacity");
            if (!v) {
                return false;
            }
            if (!parse_positive_size("--scheduler-capacity", v, opts.scheduler_capacity)) {
                return false;
            }
            continue;
        }
        if (arg == "--worker-queue-capacity") {
            const char *v = need_value("--worker-queue-capacity");
            if (!v) {
                return false;
            }
            if (!parse_positive_size("--worker-queue-capacity", v, opts.worker_queue_capacity)) {
                return false;
            }
            continue;
        }
        if (arg == "--scheduler") {
            const char *v = need_value("--scheduler");
            if (!v) {
                return false;
            }
            opts.scheduler = v;
            continue;
        }
        std::cerr << "unknown argument: " << arg << '\n';
        print_usage();
        return false;
    }

    if (opts.scheduler != "fifo" && opts.scheduler != "workload") {
        std::cerr << "unknown --scheduler (expected fifo|workload)\n";
        return false;
    }
    return true;
}

airuntime::BatchBuilderConfig make_batch_config(const ServerOptions &opts) {
    airuntime::BatchBuilderConfig config;
    config.max_batch_size = opts.max_batch_size;
    if (opts.max_batch_wait_ms >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() / 1000)) {
        throw std::invalid_argument("--max-batch-wait-ms too large");
    }
    config.max_batch_wait =
        std::chrono::microseconds{static_cast<std::int64_t>(opts.max_batch_wait_ms) * 1000};
    return config;
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
                      std::uint64_t budget, const ServerOptions &opts) {
    auto backend = std::make_unique<airuntime::SyntheticModelBackend>();
    airuntime::SyntheticModelConfig config;
    config.load_latency = std::chrono::microseconds{10};
    config.prefill_latency = std::chrono::microseconds{5};
    config.per_token_latency = std::chrono::microseconds{1};
    backend->register_model("model-a", config);
    backend->register_model("model-b", config);

    airuntime::WorkerConfig worker_config;
    worker_config.worker_id = id;
    worker_config.queue_capacity = opts.worker_queue_capacity;
    worker_config.memory_budget_bytes = budget;
    worker_config.batch_config = make_batch_config(opts);
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
                  const ServerOptions &opts) {
    const std::size_t max_sequences = opts.max_batch_size;
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
    worker_config.queue_capacity = opts.worker_queue_capacity;
    worker_config.memory_budget_bytes = budget;
    worker_config.batch_config = make_batch_config(opts);
    worker_config.registry = std::move(registry);
    worker_config.backend =
        std::make_unique<airuntime::LlamaCppBackend>(runtime, std::move(backend_config));
    worker_config.eviction_policy = std::make_unique<airuntime::CostAwareEvictionPolicy>();
    return worker_config;
}
#endif

std::unique_ptr<airuntime::IRequestScheduler> make_scheduler(const ServerOptions &opts) {
    if (opts.scheduler == "fifo") {
        return std::make_unique<airuntime::FifoScheduler>(opts.scheduler_capacity);
    }
    airuntime::WorkloadAwareSchedulerConfig sched_config;
    sched_config.capacity = opts.scheduler_capacity;
    sched_config.max_bypass = 8;
    return std::make_unique<airuntime::WorkloadAwareScheduler>(sched_config);
}

} // namespace

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    ServerOptions opts;
    bool help_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            help_only = true;
            break;
        }
    }
    if (!parse_args(argc, argv, opts)) {
        return help_only ? 0 : 1;
    }

    std::cout << "Adaptive AI Inference Runtime\n";

    std::shared_ptr<const airuntime::ModelRegistry> registry;
    std::vector<std::unique_ptr<airuntime::Worker>> workers;
    constexpr std::uint64_t kBudget = 12ull * 1024 * 1024 * 1024;

    try {
        if (opts.backend == "synthetic") {
            registry = make_synthetic_registry();
            if (!registry) {
                std::cerr << "failed to build model registry\n";
                return 1;
            }
            const std::size_t worker_count = opts.workers.value_or(2);
            for (std::size_t i = 0; i < worker_count; ++i) {
                workers.push_back(std::make_unique<airuntime::Worker>(make_synthetic_worker(
                    static_cast<airuntime::WorkerId>(i), registry, kBudget, opts)));
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
            for (std::size_t i = 0; i < worker_count; ++i) {
                workers.push_back(std::make_unique<airuntime::Worker>(make_llama_worker(
                    static_cast<airuntime::WorkerId>(i), registry, kBudget, llama_runtime, opts)));
            }
            std::cout << "llama backend enabled (model_id=" << opts.model_id
                      << ", max_batch_size=" << opts.max_batch_size << ", gpu_offload_supported="
                      << (airuntime::llama_runtime_supports_gpu_offload() ? "true" : "false")
                      << ")\n";
#endif
        } else {
            std::cerr << "unknown --backend (expected synthetic|llama)\n";
            return 1;
        }
    } catch (const std::exception &ex) {
        std::cerr << "failed to construct runtime workers: " << ex.what() << '\n';
        return 1;
    }

    airuntime::Runtime runtime(make_scheduler(opts),
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
