#include "airuntime/resource_manager.hpp"

namespace airuntime {

ResourceManager::ResourceManager(std::uint64_t memory_budget_bytes)
    : budget_(memory_budget_bytes) {}

std::uint64_t ResourceManager::budget() const {
    return budget_;
}

std::uint64_t ResourceManager::used() const {
    std::lock_guard lock(mutex_);
    return used_;
}

std::uint64_t ResourceManager::available() const {
    std::lock_guard lock(mutex_);
    return budget_ - used_;
}

bool ResourceManager::can_reserve(std::uint64_t bytes) const {
    std::lock_guard lock(mutex_);
    return bytes <= (budget_ - used_);
}

Status ResourceManager::reserve(std::uint64_t bytes) {
    std::lock_guard lock(mutex_);
    if (bytes > (budget_ - used_)) {
        return Status::error(ErrorCode::InsufficientMemory, "cannot reserve memory");
    }
    used_ += bytes;
    return Status::success();
}

Status ResourceManager::release(std::uint64_t bytes) {
    std::lock_guard lock(mutex_);
    if (bytes > used_) {
        return Status::error(ErrorCode::InternalError, "over-release of memory");
    }
    used_ -= bytes;
    return Status::success();
}

} // namespace airuntime
