#include "airuntime/backend.hpp"
#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/router.hpp"
#include "airuntime/runtime.hpp"
#include "airuntime/scheduler.hpp"
#include "airuntime/serving/http_server.hpp"
#include "airuntime/serving/request_handler.hpp"
#include "airuntime/synthetic_backend.hpp"
#include "airuntime/worker.hpp"

#include <boost/asio/connect.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/asio/write.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

using airuntime::FifoScheduler;
using airuntime::IModelBackend;
using airuntime::InferenceRequest;
using airuntime::InferenceResult;
using airuntime::ModelRegistry;
using airuntime::ModelSpec;
using airuntime::ResidencyAwareRouter;
using airuntime::Runtime;
using airuntime::Status;
using airuntime::SyntheticModelBackend;
using airuntime::SyntheticModelConfig;
using airuntime::Worker;
using airuntime::WorkerConfig;
using airuntime::serving::HttpServer;
using airuntime::serving::RequestHandler;

namespace {

class BlockingInferBackend final : public IModelBackend {
  public:
    BlockingInferBackend(std::shared_ptr<std::mutex> mutex,
                         std::shared_ptr<std::condition_variable> cv, std::shared_ptr<bool> entered,
                         std::shared_ptr<bool> release,
                         std::unique_ptr<SyntheticModelBackend> inner)
        : mutex_(std::move(mutex)), cv_(std::move(cv)), entered_(std::move(entered)),
          release_(std::move(release)), inner_(std::move(inner)) {}

    Status load(const ModelSpec &model) override {
        return inner_->load(model);
    }
    Status unload(std::string_view model_id) override {
        return inner_->unload(model_id);
    }
    bool is_loaded(std::string_view model_id) const override {
        return inner_->is_loaded(model_id);
    }
    InferenceResult infer(const InferenceRequest &request) override {
        const InferenceRequest *ptr = &request;
        return infer_batch(std::span<const InferenceRequest *const>(&ptr, 1)).front();
    }
    std::vector<InferenceResult>
    infer_batch(std::span<const InferenceRequest *const> requests) override {
        {
            std::unique_lock lock(*mutex_);
            *entered_ = true;
            cv_->notify_all();
            cv_->wait(lock, [&] { return *release_; });
        }
        return inner_->infer_batch(requests);
    }

  private:
    std::shared_ptr<std::mutex> mutex_;
    std::shared_ptr<std::condition_variable> cv_;
    std::shared_ptr<bool> entered_;
    std::shared_ptr<bool> release_;
    std::unique_ptr<SyntheticModelBackend> inner_;
};

struct OverloadHarness {
    std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>();
    std::shared_ptr<std::condition_variable> cv = std::make_shared<std::condition_variable>();
    std::shared_ptr<bool> entered = std::make_shared<bool>(false);
    std::shared_ptr<bool> release = std::make_shared<bool>(false);

    std::shared_ptr<Runtime> runtime;
    std::unique_ptr<RequestHandler> handler;
    boost::asio::io_context ioc;
    std::unique_ptr<HttpServer> server;
    std::thread server_thread;
    unsigned short port{0};
    bool stopped_{false};

    ~OverloadHarness() {
        release_backend();
        stop();
    }

    bool start() {
        ModelRegistry::Builder builder;
        if (!builder.add(ModelSpec{"model-a", 1024, 1}).ok()) {
            return false;
        }
        Status status;
        auto registry = builder.build(status);
        if (!status.ok()) {
            return false;
        }
        auto shared_registry = std::shared_ptr<const ModelRegistry>(std::move(registry));

        auto synth = std::make_unique<SyntheticModelBackend>();
        synth->register_model("model-a", SyntheticModelConfig{});

        WorkerConfig worker_config;
        worker_config.worker_id = 0;
        worker_config.queue_capacity = 2;
        worker_config.memory_budget_bytes = 4096;
        worker_config.batch_config.max_batch_size = 1;
        worker_config.batch_config.max_batch_wait = std::chrono::microseconds{0};
        worker_config.registry = shared_registry;
        worker_config.backend =
            std::make_unique<BlockingInferBackend>(mutex, cv, entered, release, std::move(synth));
        worker_config.eviction_policy = std::make_unique<airuntime::LruEvictionPolicy>();

        std::vector<std::unique_ptr<Worker>> workers;
        workers.push_back(std::make_unique<Worker>(std::move(worker_config)));

        runtime = std::make_shared<Runtime>(std::make_unique<FifoScheduler>(4),
                                            std::make_unique<ResidencyAwareRouter>(),
                                            std::move(workers), shared_registry);
        if (!runtime->start().ok()) {
            return false;
        }

        handler = std::make_unique<RequestHandler>(*runtime);
        server = std::make_unique<HttpServer>(
            ioc, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0), *handler);
        server_thread = std::thread([this] { server->run(); });

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
        while (!server->is_ready() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        if (!server->is_ready()) {
            return false;
        }
        port = server->port();
        return port != 0;
    }

    void stop() {
        if (stopped_) {
            return;
        }
        stopped_ = true;
        if (server) {
            server->stop();
        }
        if (server_thread.joinable()) {
            server_thread.join();
        }
        if (runtime) {
            runtime->stop();
        }
    }

    void release_backend() {
        {
            std::lock_guard lock(*mutex);
            *release = true;
        }
        cv->notify_all();
    }
};

std::string http_get(unsigned short port, const std::string &path) {
    boost::asio::io_context ioc;
    boost::asio::ip::tcp::socket socket(ioc);
    socket.connect(
        boost::asio::ip::tcp::endpoint(boost::asio::ip::make_address("127.0.0.1"), port));

    const std::string request = "GET " + path +
                                " HTTP/1.1\r\n"
                                "Host: 127.0.0.1:" +
                                std::to_string(port) +
                                "\r\n"
                                "Connection: close\r\n\r\n";
    boost::asio::write(socket, boost::asio::buffer(request));

    boost::asio::streambuf response_buf;
    boost::system::error_code ec;
    do {
        boost::asio::read(socket, response_buf, boost::asio::transfer_at_least(1), ec);
    } while (!ec);

    return {std::istreambuf_iterator<char>{&response_buf}, std::istreambuf_iterator<char>{}};
}

std::string http_post_infer(unsigned short port, int timeout_ms = 120000) {
    boost::asio::io_context ioc;
    boost::asio::ip::tcp::socket socket(ioc);
    socket.connect(
        boost::asio::ip::tcp::endpoint(boost::asio::ip::make_address("127.0.0.1"), port));

    const std::string body =
        R"({"model_id":"model-a","prompt":"hello","max_output_tokens":1,"timeout_ms":)" +
        std::to_string(timeout_ms) + "}";
    const std::string request = "POST /v1/infer HTTP/1.1\r\n"
                                "Host: 127.0.0.1:" +
                                std::to_string(port) +
                                "\r\n"
                                "Content-Type: application/json\r\n"
                                "Content-Length: " +
                                std::to_string(body.size()) +
                                "\r\n"
                                "Connection: close\r\n\r\n" +
                                body;
    boost::asio::write(socket, boost::asio::buffer(request));

    boost::asio::streambuf response_buf;
    boost::system::error_code ec;
    do {
        boost::asio::read(socket, response_buf, boost::asio::transfer_at_least(1), ec);
    } while (!ec);

    return {std::istreambuf_iterator<char>{&response_buf}, std::istreambuf_iterator<char>{}};
}

} // namespace

TEST(HttpOverloadTest, SaturatedWorkerReturns503AndHealthStaysResponsive) {
    OverloadHarness harness;
    ASSERT_TRUE(harness.start());

    std::thread starter([&] { (void)http_post_infer(harness.port); });

    {
        std::unique_lock lock(*harness.mutex);
        ASSERT_TRUE(
            harness.cv->wait_for(lock, std::chrono::seconds{2}, [&] { return *harness.entered; }));
    }

    std::mutex response_mutex;
    std::vector<std::string> overload_responses;
    std::vector<std::thread> flood;
    for (int i = 0; i < 8; ++i) {
        flood.emplace_back([&] {
            const std::string response = http_post_infer(harness.port, 5000);
            std::lock_guard lock(response_mutex);
            overload_responses.push_back(response);
        });
    }

    const std::string health = http_get(harness.port, "/health");
    EXPECT_NE(health.find("200"), std::string::npos);
    EXPECT_NE(health.find("healthy"), std::string::npos);
    EXPECT_FALSE(*harness.release);

    for (auto &thread : flood) {
        thread.join();
    }

    bool saw_503 = false;
    {
        std::lock_guard lock(response_mutex);
        for (const auto &response : overload_responses) {
            if (response.find("503") != std::string::npos) {
                saw_503 = true;
                break;
            }
        }
    }
    EXPECT_TRUE(saw_503);

    harness.release_backend();
    starter.join();

    harness.stop();
}
