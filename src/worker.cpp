#include "airuntime/worker.hpp"

#include "airuntime/model.hpp"

#include <exception>
#include <stdexcept>
#include <utility>

namespace airuntime {

Worker::Worker(std::unique_ptr<IModelBackend> backend) : backend_(std::move(backend)) {
    if (!backend_) {
        throw std::invalid_argument("Worker requires a non-null backend");
    }
}

WorkerState Worker::state() const {
    std::lock_guard lock(mutex_);
    return state_;
}

Status Worker::ensure_model_loaded(const InferenceRequest &request) {
    if (backend_->is_loaded(request.model_id())) {
        return Status::success();
    }
    ModelSpec spec;
    spec.model_id = request.model_id();
    return backend_->load(spec);
}

Status Worker::execute(const RequestPtr &request) {
    if (!request) {
        return Status::error(ErrorCode::InternalError, "null request");
    }

    {
        std::lock_guard lock(mutex_);
        state_ = WorkerState::Busy;
    }

    auto restore_idle = [this]() {
        std::lock_guard lock(mutex_);
        state_ = WorkerState::Idle;
    };

    try {
        auto running = request->transition_to(RequestState::Running);
        if (!running.ok()) {
            restore_idle();
            return running;
        }

        auto load_status = ensure_model_loaded(*request);
        if (!load_status.ok()) {
            request->fail(load_status);
            restore_idle();
            return load_status;
        }

        auto inference = backend_->infer(*request);
        if (!inference.ok()) {
            request->fail(inference.status);
            restore_idle();
            return inference.status;
        }

        auto completed = request->complete(std::move(inference));
        restore_idle();
        return completed;
    } catch (const std::exception &ex) {
        request->fail(Status::error(ErrorCode::InternalError, ex.what()));
        restore_idle();
        return Status::error(ErrorCode::InternalError, ex.what());
    } catch (...) {
        request->fail(Status::error(ErrorCode::InternalError, "unknown backend exception"));
        restore_idle();
        return Status::error(ErrorCode::InternalError, "unknown backend exception");
    }
}

} // namespace airuntime
