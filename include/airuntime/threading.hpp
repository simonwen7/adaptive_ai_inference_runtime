#pragma once

#include <cstddef>
#include <thread>
#include <utility>

namespace airuntime::detail {

// Prefer std::jthread when available. Apple Clang's default libc++ currently
// omits jthread; provide an equivalent join-on-destroy wrapper.
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

} // namespace airuntime::detail
