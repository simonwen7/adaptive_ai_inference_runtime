#pragma once

#include "airuntime/admission_controller.hpp"
#include "airuntime/request.hpp"
#include "airuntime/scheduler.hpp"
#include "airuntime/status.hpp"
#include "airuntime/worker.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace airuntime {

namespace detail {

// Prefer std::jthread when available. Apple Clang's default libc++ currently
// omits jthread; provide an equivalent join-on-destroy wrapper so Runtime still
// owns one background consumer thread with RAII join semantics.
#if defined(__cpp_lib_jthread)
using JoinThread = std::jthread;
#else
class JoinThread {
  public:
    JoinThread() = default;

    template <typename Function>
    explicit JoinThread(Function &&function) : thread_(std::forward<Function>(function)) {}

    JoinThread(JoinThread &&other) noexcept : thread_(std::move(other.thread_)) {}

    JoinThread &operator=(JoinThread &&other) noexcept {
        if (joinable()) {
            thread_.join();
        }
        thread_ = std::move(other.thread_);
        return *this;
    }

    ~JoinThread() {
        if (joinable()) {
            thread_.join();
        }
    }

    JoinThread(const JoinThread &) = delete;
    JoinThread &operator=(const JoinThread &) = delete;

    [[nodiscard]] bool joinable() const noexcept {
        return thread_.joinable();
    }

  private:
    std::thread thread_;
};
#endif

} // namespace detail

class Runtime {
  public:
    Runtime(std::unique_ptr<IRequestScheduler> scheduler, std::unique_ptr<Worker> worker);
    ~Runtime();

    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    Status start();
    Status submit(const RequestPtr &request);
    void stop();

    [[nodiscard]] bool is_running() const;

  private:
    void dispatch_loop();

    mutable std::mutex mutex_;
    std::unique_ptr<IRequestScheduler> scheduler_;
    std::optional<AdmissionController> admission_;
    std::unique_ptr<Worker> worker_;
    std::optional<detail::JoinThread> dispatch_thread_;
    std::atomic<bool> accepting_{false};
    bool started_{false};
};

} // namespace airuntime
