#include "airuntime/worker.hpp"

#include <exception>
#include <stdexcept>
#include <utility>

namespace airuntime {
namespace {

IModelBackend &require_backend(std::unique_ptr<IModelBackend> &backend) {
    if (!backend) {
        throw std::invalid_argument("Worker requires a non-null backend");
    }
    return *backend;
}

} // namespace

Worker::Worker(WorkerConfig config)
    : worker_id_(config.worker_id),
      queue_capacity_((config.queue_capacity == 0 || config.memory_budget_bytes == 0)
                          ? throw std::invalid_argument(
                                "Worker requires positive queue capacity and memory budget")
                          : config.queue_capacity),
      backend_(std::move(config.backend)),
      model_manager_(std::move(config.registry), require_backend(backend_),
                     config.memory_budget_bytes, std::move(config.eviction_policy)),
      lane_(queue_capacity_) {}

Worker::~Worker() {
    close();
    join();
}

WorkerId Worker::id() const {
    return worker_id_;
}

WorkerState Worker::state() const {
    std::lock_guard lock(mutex_);
    return state_;
}

ResidencyMetrics Worker::residency_metrics() const {
    return model_manager_.metrics();
}

ModelManager &Worker::model_manager() {
    return model_manager_;
}

const ModelManager &Worker::model_manager() const {
    return model_manager_;
}

Status Worker::start() {
    std::lock_guard lock(mutex_);
    if (started_) {
        return Status::success();
    }
    accepting_.store(true);
    thread_ = detail::JoinThread([this] { run_loop(); });
    started_ = true;
    return Status::success();
}

Status Worker::enqueue(const RequestPtr &request) {
    if (!request) {
        return Status::error(ErrorCode::InternalError, "null request");
    }
    if (!accepting_.load()) {
        return Status::error(ErrorCode::QueueClosed, "worker is not accepting work");
    }
    return lane_.wait_push(request);
}

void Worker::close() {
    accepting_.store(false);
    lane_.close();
}

void Worker::join() {
    std::optional<detail::JoinThread> local;
    {
        std::lock_guard lock(mutex_);
        local = std::move(thread_);
        thread_.reset();
    }
}

WorkerSnapshot Worker::snapshot() const {
    WorkerSnapshot snap;
    snap.worker_id = worker_id_;
    snap.queue_depth = lane_.size();
    snap.queue_capacity = queue_capacity_;
    snap.accepting = accepting_.load();
    snap.memory_budget_bytes = model_manager_.memory_budget_bytes();
    snap.memory_used_bytes = model_manager_.memory_used_bytes();
    {
        std::lock_guard lock(mutex_);
        snap.active_count = active_count_;
    }
    return snap;
}

void Worker::run_loop() {
    while (true) {
        auto next = lane_.wait_pop();
        if (!next.has_value()) {
            break;
        }
        (void)execute(*next);
    }
}

Status Worker::execute(const RequestPtr &request) {
    {
        std::lock_guard lock(mutex_);
        state_ = WorkerState::Busy;
        active_count_ = 1;
    }

    auto restore_idle = [this]() {
        std::lock_guard lock(mutex_);
        state_ = WorkerState::Idle;
        active_count_ = 0;
    };

    try {
        auto running = request->transition_to(RequestState::Running);
        if (!running.ok()) {
            restore_idle();
            return running;
        }

        auto resident = model_manager_.ensure_resident(request->model_id());
        if (!resident.ok()) {
            request->fail(resident);
            restore_idle();
            return resident;
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
