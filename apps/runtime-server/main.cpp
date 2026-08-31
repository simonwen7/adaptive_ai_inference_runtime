#include "airuntime/runtime.hpp"
#include "airuntime/scheduler.hpp"
#include "airuntime/synthetic_backend.hpp"
#include "airuntime/worker.hpp"

#include <chrono>
#include <iostream>
#include <memory>

int main() {
    using namespace airuntime;

    std::cout << "Adaptive AI Inference Runtime\n";

    auto backend = std::make_unique<SyntheticModelBackend>();
    SyntheticModelConfig config;
    config.load_latency = std::chrono::microseconds{100};
    config.prefill_latency = std::chrono::microseconds{50};
    config.per_token_latency = std::chrono::microseconds{10};
    backend->register_model("demo-model", config);

    auto worker = std::make_unique<Worker>(std::move(backend));
    auto scheduler = std::make_unique<FifoScheduler>(16);
    Runtime runtime(std::move(scheduler), std::move(worker));

    auto start_status = runtime.start();
    if (!start_status.ok()) {
        std::cerr << "failed to start runtime\n";
        return 1;
    }

    auto request =
        std::make_shared<InferenceRequest>("req-1", "demo-model", "hello from milestone 1", 4);

    auto submit_status = runtime.submit(request);
    if (!submit_status.ok()) {
        std::cerr << "submit failed\n";
        runtime.stop();
        return 1;
    }

    if (!request->wait_for_terminal(std::chrono::seconds{5})) {
        std::cerr << "timed out waiting for request completion\n";
        runtime.stop();
        return 1;
    }

    auto result = request->result();
    if (!result.has_value() || !result->ok() || !result->output.has_value()) {
        std::cerr << "request failed\n";
        runtime.stop();
        return 1;
    }

    std::cout << "request_id=" << request->request_id() << '\n';
    std::cout << "state=Completed\n";
    std::cout << "output=" << result->output->text << '\n';
    std::cout << "output_tokens=" << result->output->output_tokens << '\n';

    runtime.stop();
    return 0;
}
