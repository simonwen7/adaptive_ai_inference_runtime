#include "airuntime/backend.hpp"
#include "airuntime/eviction_policy.hpp"
#include "airuntime/model_registry.hpp"
#include "airuntime/request.hpp"
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
#include <iterator>
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
using airuntime::RequestSnapshot;
using airuntime::RequestState;
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

class TrackingBlockingBackend final : public IModelBackend {
  public:
    TrackingBlockingBackend(std::shared_ptr<std::mutex> mutex,
                            std::shared_ptr<std::condition_variable> cv,
                            std::shared_ptr<bool> entered, std::shared_ptr<bool> release,
                            std::shared_ptr<bool> finished,
                            std::shared_ptr<std::vector<const InferenceRequest *>> seen,
                            std::unique_ptr<SyntheticModelBackend> inner)
        : mutex_(std::move(mutex)), cv_(std::move(cv)), entered_(std::move(entered)),
          release_(std::move(release)), finished_(std::move(finished)), seen_(std::move(seen)),
          inner_(std::move(inner)) {}

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
            seen_->assign(requests.begin(), requests.end());
            *entered_ = true;
            cv_->notify_all();
            cv_->wait(lock, [&] { return *release_; });
            *finished_ = true;
            cv_->notify_all();
        }
        return inner_->infer_batch(requests);
    }

  private:
    std::shared_ptr<std::mutex> mutex_;
    std::shared_ptr<std::condition_variable> cv_;
    std::shared_ptr<bool> entered_;
    std::shared_ptr<bool> release_;
    std::shared_ptr<bool> finished_;
    std::shared_ptr<std::vector<const InferenceRequest *>> seen_;
    std::unique_ptr<SyntheticModelBackend> inner_;
};

struct TestHarness {
    std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>();
    std::shared_ptr<std::condition_variable> cv = std::make_shared<std::condition_variable>();
    std::shared_ptr<bool> entered = std::make_shared<bool>(false);
    std::shared_ptr<bool> release = std::make_shared<bool>(false);
    std::shared_ptr<bool> finished = std::make_shared<bool>(false);
    std::shared_ptr<std::vector<const InferenceRequest *>> seen =
        std::make_shared<std::vector<const InferenceRequest *>>();

    std::shared_ptr<Runtime> runtime;
    std::unique_ptr<RequestHandler> handler;
    boost::asio::io_context ioc;
    std::unique_ptr<HttpServer> server;
    std::thread server_thread;
    unsigned short port{0};
    bool stopped_{false};

    ~TestHarness() {
        release_backend();
        stop();
    }

    bool start_with_backend(std::unique_ptr<IModelBackend> backend) {
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

        WorkerConfig worker_config;
        worker_config.worker_id = 0;
        worker_config.queue_capacity = 8;
        worker_config.memory_budget_bytes = 4096;
        worker_config.registry = shared_registry;
        worker_config.backend = std::move(backend);
        worker_config.eviction_policy = std::make_unique<airuntime::LruEvictionPolicy>();

        std::vector<std::unique_ptr<Worker>> workers;
        workers.push_back(std::make_unique<Worker>(std::move(worker_config)));

        runtime = std::make_shared<Runtime>(std::make_unique<FifoScheduler>(8),
                                            std::make_unique<ResidencyAwareRouter>(),
                                            std::move(workers), shared_registry);
        if (!runtime->start().ok()) {
            return false;
        }

        handler = std::make_unique<RequestHandler>(*runtime);
        server = std::make_unique<HttpServer>(
            ioc, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0), *handler);
        server_thread = std::thread([this] { server->run(); });

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!server->is_ready() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!server->is_ready()) {
            return false;
        }
        port = server->port();
        return port != 0;
    }

    bool start_blocking() {
        auto synth = std::make_unique<SyntheticModelBackend>();
        synth->register_model("model-a", SyntheticModelConfig{});
        return start_with_backend(std::make_unique<TrackingBlockingBackend>(
            mutex, cv, entered, release, finished, seen, std::move(synth)));
    }

    bool start_fast() {
        auto synth = std::make_unique<SyntheticModelBackend>();
        synth->register_model("model-a", SyntheticModelConfig{});
        return start_with_backend(std::move(synth));
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

void post_infer_request(unsigned short port, boost::asio::ip::tcp::socket &socket) {
    const std::string body =
        R"({"model_id":"model-a","prompt":"hello","max_output_tokens":8,"timeout_ms":30000})";
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
}

} // namespace

TEST(HttpDisconnectTest, PeerCloseCancelsPendingInferBeforeBackendRelease) {
    TestHarness harness;
    ASSERT_TRUE(harness.start_blocking());

    boost::asio::io_context client_ioc;
    boost::asio::ip::tcp::socket client(client_ioc);
    client.connect(
        boost::asio::ip::tcp::endpoint(boost::asio::ip::make_address("127.0.0.1"), harness.port));

    post_infer_request(harness.port, client);

    {
        std::unique_lock lock(*harness.mutex);
        ASSERT_TRUE(
            harness.cv->wait_for(lock, std::chrono::seconds(2), [&] { return *harness.entered; }));
        ASSERT_FALSE(harness.seen->empty());
    }

    InferenceRequest *pending = nullptr;
    {
        std::lock_guard lock(*harness.mutex);
        pending = const_cast<InferenceRequest *>(harness.seen->front());
    }
    ASSERT_NE(pending, nullptr);
    EXPECT_FALSE(pending->is_terminal());

    // Retain terminal observations without keeping a RequestPtr (Worker owns it).
    auto observed = std::make_shared<std::atomic<RequestState>>(pending->state());
    pending->add_observer([observed](RequestSnapshot snap) {
        observed->store(snap.state, std::memory_order_release);
    });

    auto sibling = std::make_shared<InferenceRequest>("sibling", "model-a", "p", 1);
    sibling->set_deadline(std::chrono::steady_clock::now() + std::chrono::seconds(30));
    ASSERT_TRUE(harness.runtime->submit(sibling).ok());
    EXPECT_NE(sibling->state(), RequestState::Cancelled);

    boost::system::error_code ec;
    // Half-close send side so the server observes peer EOF on its pending read watch.
    client.shutdown(boost::asio::ip::tcp::socket::shutdown_send, ec);
    client.close(ec);

    const auto cancel_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < cancel_deadline) {
        if (observed->load(std::memory_order_acquire) == RequestState::Cancelled) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(observed->load(std::memory_order_acquire), RequestState::Cancelled);
    ASSERT_EQ(pending->state(), RequestState::Cancelled);
    EXPECT_NE(sibling->state(), RequestState::Cancelled);

    harness.release_backend();

    {
        std::unique_lock lock(*harness.mutex);
        ASSERT_TRUE(
            harness.cv->wait_for(lock, std::chrono::seconds(2), [&] { return *harness.finished; }));
    }
    // Late backend completion must not overwrite Cancelled (first-terminal-wins).
    EXPECT_EQ(observed->load(std::memory_order_acquire), RequestState::Cancelled);

    ASSERT_TRUE(sibling->wait_for_terminal(std::chrono::seconds(2)));
    EXPECT_NE(sibling->state(), RequestState::Cancelled);
    EXPECT_EQ(sibling->state(), RequestState::Completed);

    harness.stop();
}

TEST(HttpDisconnectTest, IdleConnectedClientIsNotSpuriouslyCancelled) {
    TestHarness harness;
    ASSERT_TRUE(harness.start_fast());

    boost::asio::io_context client_ioc;
    boost::asio::ip::tcp::socket client(client_ioc);
    client.connect(
        boost::asio::ip::tcp::endpoint(boost::asio::ip::make_address("127.0.0.1"), harness.port));

    post_infer_request(harness.port, client);

    boost::asio::streambuf response_buf;
    boost::system::error_code ec;
    do {
        boost::asio::read(client, response_buf, boost::asio::transfer_at_least(1), ec);
    } while (!ec);

    const std::string response{std::istreambuf_iterator<char>{&response_buf},
                               std::istreambuf_iterator<char>{}};
    EXPECT_NE(response.find("200"), std::string::npos);
    EXPECT_NE(response.find("Completed"), std::string::npos);
    EXPECT_EQ(response.find("Cancelled"), std::string::npos);

    harness.stop();
}
