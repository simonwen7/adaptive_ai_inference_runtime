#pragma once

#include "airuntime/bounded_queue.hpp"
#include "airuntime/request.hpp"
#include "airuntime/status.hpp"

#include <cstddef>
#include <memory>
#include <optional>

namespace airuntime {

class IRequestScheduler {
  public:
    virtual ~IRequestScheduler() = default;

    virtual Status enqueue(RequestPtr request) = 0;
    virtual std::optional<RequestPtr> next() = 0;
    virtual void close() = 0;
    [[nodiscard]] virtual std::size_t size() const = 0;
    [[nodiscard]] virtual std::size_t capacity() const = 0;
};

class FifoScheduler final : public IRequestScheduler {
  public:
    explicit FifoScheduler(std::size_t capacity);

    Status enqueue(RequestPtr request) override;
    std::optional<RequestPtr> next() override;
    void close() override;
    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] std::size_t capacity() const override;

  private:
    BoundedQueue<RequestPtr> queue_;
};

} // namespace airuntime
