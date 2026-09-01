#include "airuntime/worker.hpp"

#include <chrono>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

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
      batch_builder_(config.batch_config), backend_(std::move(config.backend)),
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

WorkerBatchMetrics Worker::batch_metrics() const {
    std::lock_guard lock(mutex_);
    return batch_metrics_;
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
    auto status = lane_.wait_push(request);
    if (status.ok()) {
        queued_count_.fetch_add(1, std::memory_order_relaxed);
    }
    return status;
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
    snap.queue_depth = queued_count_.load(std::memory_order_relaxed);
    snap.queue_capacity = queue_capacity_;
    snap.accepting = accepting_.load();
    snap.memory_budget_bytes = model_manager_.memory_budget_bytes();
    snap.memory_used_bytes = model_manager_.memory_used_bytes();
    snap.resident_model_ids = model_manager_.resident_model_ids();
    {
        std::lock_guard lock(mutex_);
        snap.active_count = active_count_;
    }
    return snap;
}

void Worker::run_loop() {
    while (true) {
        RequestPtr first;
        if (deferred_.has_value()) {
            first = std::move(*deferred_);
            deferred_.reset();
        } else {
            auto next = lane_.wait_pop();
            if (!next.has_value()) {
                break;
            }
            first = std::move(*next);
        }

        const bool allow_wait = !lane_.closed();
        auto formed = batch_builder_.form(
            std::move(first),
            [this](std::chrono::steady_clock::time_point deadline) {
                return lane_.wait_pop_until(deadline);
            },
            allow_wait);

        if (formed.deferred.has_value()) {
            deferred_ = std::move(formed.deferred);
        }

        if (!formed.requests.empty()) {
            (void)execute_batch(std::move(formed.requests));
        }
    }
}

Status Worker::execute_batch(std::vector<RequestPtr> batch) {
    if (batch.empty()) {
        return Status::success();
    }

    {
        std::lock_guard lock(mutex_);
        state_ = WorkerState::Busy;
        active_count_ = batch.size();
    }

    // Leave queued accounting as these requests are entering execution.
    queued_count_.fetch_sub(batch.size(), std::memory_order_relaxed);

    auto restore_idle = [this]() {
        std::lock_guard lock(mutex_);
        state_ = WorkerState::Idle;
        active_count_ = 0;
    };

    auto fail_all = [&](Status error) {
        for (auto &request : batch) {
            if (!request || request->is_terminal()) {
                continue;
            }
            if (request->state() == RequestState::Queued) {
                (void)request->transition_to(RequestState::Running);
            }
            if (!request->is_terminal()) {
                request->fail(error);
            }
        }
        restore_idle();
        return error;
    };

    auto record_batch_metrics = [&]() {
        std::lock_guard lock(mutex_);
        ++batch_metrics_.batches_executed;
        batch_metrics_.requests_executed_via_batches += batch.size();
        if (batch.size() > 1) {
            ++batch_metrics_.multi_request_batches;
        }
        if (batch.size() > batch_metrics_.max_batch_size_observed) {
            batch_metrics_.max_batch_size_observed = batch.size();
        }
    };

    try {
        for (auto &request : batch) {
            auto running = request->transition_to(RequestState::Running);
            if (!running.ok()) {
                return fail_all(running);
            }
        }

        const std::string model_id = batch.front()->model_id();
        auto resident = model_manager_.ensure_resident(model_id);
        if (!resident.ok()) {
            for (auto &request : batch) {
                request->fail(resident);
            }
            record_batch_metrics();
            restore_idle();
            return resident;
        }

        std::vector<const InferenceRequest *> views;
        views.reserve(batch.size());
        for (const auto &request : batch) {
            views.push_back(request.get());
        }

        std::vector<InferenceResult> results;
        try {
            results = backend_->infer_batch(views);
        } catch (const std::exception &ex) {
            return fail_all(Status::error(ErrorCode::InternalError, ex.what()));
        } catch (...) {
            return fail_all(Status::error(ErrorCode::InternalError, "unknown backend exception"));
        }

        if (results.size() != batch.size()) {
            auto err =
                Status::error(ErrorCode::InternalError, "infer_batch result cardinality mismatch");
            for (auto &request : batch) {
                request->fail(err);
            }
            record_batch_metrics();
            restore_idle();
            return err;
        }

        Status last = Status::success();
        for (std::size_t i = 0; i < batch.size(); ++i) {
            if (!results[i].ok()) {
                auto failed = batch[i]->fail(results[i].status);
                last = failed.ok() ? results[i].status : failed;
            } else {
                auto completed = batch[i]->complete(std::move(results[i]));
                if (!completed.ok()) {
                    last = completed;
                }
            }
        }

        record_batch_metrics();
        restore_idle();
        return last;
    } catch (const std::exception &ex) {
        return fail_all(Status::error(ErrorCode::InternalError, ex.what()));
    } catch (...) {
        return fail_all(Status::error(ErrorCode::InternalError, "unknown backend exception"));
    }
}

} // namespace airuntime
