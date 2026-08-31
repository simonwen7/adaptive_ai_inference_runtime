#pragma once

#include "airuntime/backend.hpp"
#include "airuntime/request.hpp"
#include "airuntime/status.hpp"

#include <memory>
#include <mutex>

namespace airuntime {

enum class WorkerState { Idle, Busy };

class Worker {
  public:
    explicit Worker(std::unique_ptr<IModelBackend> backend);

    Status execute(const RequestPtr &request);
    [[nodiscard]] WorkerState state() const;

  private:
    Status ensure_model_loaded(const InferenceRequest &request);

    mutable std::mutex mutex_;
    WorkerState state_{WorkerState::Idle};
    std::unique_ptr<IModelBackend> backend_;
};

} // namespace airuntime
