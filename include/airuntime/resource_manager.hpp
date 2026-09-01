#pragma once

#include "airuntime/status.hpp"

#include <cstdint>
#include <mutex>

namespace airuntime {

class ResourceManager {
  public:
    explicit ResourceManager(std::uint64_t memory_budget_bytes);

    [[nodiscard]] std::uint64_t budget() const;
    [[nodiscard]] std::uint64_t used() const;
    [[nodiscard]] std::uint64_t available() const;

    [[nodiscard]] bool can_reserve(std::uint64_t bytes) const;
    Status reserve(std::uint64_t bytes);
    Status release(std::uint64_t bytes);

  private:
    const std::uint64_t budget_;
    mutable std::mutex mutex_;
    std::uint64_t used_{0};
};

} // namespace airuntime
