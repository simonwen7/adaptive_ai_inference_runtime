#include "airuntime/runtime.hpp"

#include <stdexcept>
#include <utility>

namespace airuntime {

Runtime::Runtime(std::unique_ptr<IRequestScheduler> scheduler,
                 std::unique_ptr<IWorkerRouter> router,
                 std::vector<std::unique_ptr<Worker>> workers,
                 std::shared_ptr<const ModelRegistry> registry)
    : scheduler_(std::move(scheduler)), router_(std::move(router)), workers_(std::move(workers)),
      registry_(std::move(registry)) {
    if (!scheduler_ || !router_ || !registry_) {
        throw std::invalid_argument("Runtime requires scheduler, router, and registry");
    }
    if (workers_.empty()) {
        throw std::invalid_argument("Runtime requires at least one worker");
    }
    admission_.emplace(*scheduler_);
    for (auto &worker : workers_) {
        if (!worker) {
            throw std::invalid_argument("Runtime workers must be non-null");
        }
        if (worker_index_.contains(worker->id())) {
            throw std::invalid_argument("duplicate WorkerId");
        }
        worker_index_[worker->id()] = worker.get();
    }
}

Runtime::~Runtime() {
    stop();
}

Status Runtime::start() {
    std::lock_guard lock(mutex_);
    if (started_) {
        return Status::success();
    }
    for (auto &worker : workers_) {
        auto status = worker->start();
        if (!status.ok()) {
            return status;
        }
    }
    accepting_.store(true);
    routing_thread_ = detail::JoinThread([this] { routing_loop(); });
    started_ = true;
    return Status::success();
}

Status Runtime::submit(const RequestPtr &request) {
    if (!accepting_.load()) {
        if (request && request->state() == RequestState::Received) {
            request->reject(
                Status::error(ErrorCode::RuntimeStopped, "runtime is not accepting work"));
        }
        return Status::error(ErrorCode::RuntimeStopped, "runtime is not accepting work");
    }
    return admission_->admit(request);
}

void Runtime::stop() {
    accepting_.store(false);
    if (scheduler_) {
        scheduler_->close();
    }

    std::optional<detail::JoinThread> routing;
    {
        std::lock_guard lock(mutex_);
        routing = std::move(routing_thread_);
        routing_thread_.reset();
    }
    // Join routing thread before closing workers.

    for (auto &worker : workers_) {
        if (worker) {
            worker->close();
        }
    }
    for (auto &worker : workers_) {
        if (worker) {
            worker->join();
        }
    }
}

bool Runtime::is_running() const {
    return accepting_.load() && started_;
}

Worker *Runtime::worker(WorkerId id) {
    return find_worker(id);
}

const Worker *Runtime::worker(WorkerId id) const {
    auto it = worker_index_.find(id);
    if (it == worker_index_.end()) {
        return nullptr;
    }
    return it->second;
}

Worker *Runtime::find_worker(WorkerId id) {
    auto it = worker_index_.find(id);
    if (it == worker_index_.end()) {
        return nullptr;
    }
    return it->second;
}

std::vector<WorkerSnapshot> Runtime::collect_snapshots() const {
    std::vector<WorkerSnapshot> snapshots;
    snapshots.reserve(workers_.size());
    for (const auto &worker : workers_) {
        snapshots.push_back(worker->snapshot());
    }
    return snapshots;
}

void Runtime::routing_loop() {
    while (true) {
        auto next = scheduler_->next();
        if (!next.has_value()) {
            break;
        }
        RequestPtr request = *next;

        ModelSpec spec;
        auto find_status = registry_->find(request->model_id(), spec);
        if (!find_status.ok()) {
            request->reject(find_status);
            continue;
        }

        auto snapshots = collect_snapshots();
        auto selected = router_->select(spec, snapshots);
        if (!selected.ok()) {
            request->reject(selected.status);
            continue;
        }

        Worker *target = find_worker(*selected.worker_id);
        if (!target) {
            request->reject(Status::error(ErrorCode::InternalError, "selected worker missing"));
            continue;
        }

        auto enqueue_status = target->enqueue(request);
        if (!enqueue_status.ok()) {
            request->reject(enqueue_status);
        }
    }
}

} // namespace airuntime
