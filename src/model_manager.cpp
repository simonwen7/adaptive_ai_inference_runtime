#include "airuntime/model_manager.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace airuntime {

ModelManager::ModelManager(std::shared_ptr<const ModelRegistry> registry, IModelBackend &backend,
                           std::uint64_t memory_budget_bytes,
                           std::unique_ptr<IEvictionPolicy> eviction_policy)
    : registry_(std::move(registry)), backend_(backend), resources_(memory_budget_bytes),
      eviction_policy_(std::move(eviction_policy)) {
    if (!registry_) {
        throw std::invalid_argument("ModelManager requires a ModelRegistry");
    }
    if (!eviction_policy_) {
        throw std::invalid_argument("ModelManager requires an eviction policy");
    }
}

std::uint64_t ModelManager::memory_budget_bytes() const {
    return resources_.budget();
}

std::uint64_t ModelManager::memory_used_bytes() const {
    return resources_.used();
}

ResidencyMetrics ModelManager::metrics() const {
    std::lock_guard lock(mutex_);
    return metrics_;
}

ModelState ModelManager::model_state(std::string_view model_id) const {
    std::lock_guard lock(mutex_);
    auto it = entries_.find(std::string(model_id));
    if (it == entries_.end()) {
        return ModelState::Unloaded;
    }
    return it->second.state;
}

std::vector<std::string> ModelManager::resident_model_ids() const {
    std::lock_guard lock(mutex_);
    std::vector<std::string> ids;
    for (const auto &[id, entry] : entries_) {
        if (entry.state == ModelState::Resident) {
            ids.push_back(id);
        }
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

std::uint64_t ModelManager::use_count(std::string_view model_id) const {
    std::lock_guard lock(mutex_);
    auto it = entries_.find(std::string(model_id));
    if (it == entries_.end()) {
        return 0;
    }
    return it->second.use_count;
}

Status ModelManager::evict_one(const EvictionCandidate &victim) {
    {
        std::lock_guard lock(mutex_);
        auto it = entries_.find(victim.model_id);
        if (it == entries_.end() || it->second.state != ModelState::Resident) {
            return Status::error(ErrorCode::InternalError, "eviction candidate not resident");
        }
        it->second.state = ModelState::Evicting;
    }

    auto unload_status = backend_.unload(victim.model_id);

    std::lock_guard lock(mutex_);
    auto it = entries_.find(victim.model_id);
    if (it == entries_.end()) {
        return Status::error(ErrorCode::InternalError, "missing entry after eviction");
    }

    if (!unload_status.ok()) {
        it->second.state = ModelState::Resident;
        return unload_status;
    }

    it->second.state = ModelState::Unloaded;
    auto release_status = resources_.release(victim.estimated_memory_bytes);
    if (!release_status.ok()) {
        return release_status;
    }
    ++metrics_.unloads;
    ++metrics_.evictions;
    return Status::success();
}

Status ModelManager::free_memory_for(std::uint64_t needed_bytes) {
    while (!resources_.can_reserve(needed_bytes)) {
        std::vector<EvictionCandidate> candidates;
        {
            std::lock_guard lock(mutex_);
            candidates.reserve(entries_.size());
            for (const auto &[id, entry] : entries_) {
                if (entry.state != ModelState::Resident) {
                    continue;
                }
                EvictionCandidate candidate;
                candidate.model_id = id;
                candidate.estimated_memory_bytes = entry.estimated_memory_bytes;
                candidate.last_used = entry.last_used;
                candidate.use_count = entry.use_count;
                candidate.estimated_load_cost = 1;
                ModelSpec spec;
                if (registry_->find(id, spec).ok()) {
                    candidate.estimated_load_cost = spec.estimated_load_cost;
                }
                candidates.push_back(std::move(candidate));
            }
        }

        if (candidates.empty()) {
            return Status::error(ErrorCode::InsufficientMemory, "no eviction candidates");
        }

        auto ordered = eviction_policy_->order_victims(candidates);
        if (ordered.empty()) {
            return Status::error(ErrorCode::InsufficientMemory,
                                 "eviction policy returned no victims");
        }

        auto status = evict_one(ordered.front());
        if (!status.ok()) {
            return status;
        }
    }
    return Status::success();
}

Status ModelManager::ensure_resident(std::string_view model_id) {
    const std::string id(model_id);

    {
        std::lock_guard lock(mutex_);
        auto it = entries_.find(id);
        if (it != entries_.end() && it->second.state == ModelState::Resident) {
            ++metrics_.residency_hits;
            it->second.last_used = ++touch_counter_;
            ++it->second.use_count;
            return Status::success();
        }
        ++metrics_.residency_misses;
    }

    ModelSpec spec;
    auto find_status = registry_->find(id, spec);
    if (!find_status.ok()) {
        return find_status;
    }

    if (spec.estimated_memory_bytes > resources_.budget()) {
        return Status::error(ErrorCode::InsufficientMemory,
                             "model larger than worker memory budget");
    }

    auto free_status = free_memory_for(spec.estimated_memory_bytes);
    if (!free_status.ok()) {
        return free_status;
    }

    bool previously_loaded = false;
    {
        std::lock_guard lock(mutex_);
        auto &entry = entries_[id];
        previously_loaded = entry.previously_loaded;
        auto reserve_status = resources_.reserve(spec.estimated_memory_bytes);
        if (!reserve_status.ok()) {
            return reserve_status;
        }
        entry.state = ModelState::Loading;
        entry.estimated_memory_bytes = spec.estimated_memory_bytes;
    }

    auto load_status = backend_.load(spec);

    std::lock_guard lock(mutex_);
    auto &entry = entries_[id];
    if (!load_status.ok()) {
        (void)resources_.release(spec.estimated_memory_bytes);
        entry.state = ModelState::Failed;
        return load_status;
    }

    entry.state = ModelState::Resident;
    entry.last_used = ++touch_counter_;
    entry.previously_loaded = true;
    ++entry.use_count;
    ++metrics_.loads;
    if (previously_loaded) {
        ++metrics_.reloads;
    }
    return Status::success();
}

} // namespace airuntime
