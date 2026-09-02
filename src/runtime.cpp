#include "airuntime/runtime.hpp"

#include <algorithm>
#include <chrono>
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
    if (!request) {
        return Status::error(ErrorCode::InternalError, "null request");
    }
    if (request->is_terminal()) {
        return Status::success();
    }
    request->try_timeout_if_expired(std::chrono::steady_clock::now());
    if (request->is_terminal()) {
        return Status::success();
    }
    if (!accepting_.load()) {
        if (request->state() == RequestState::Received) {
            request->try_reject(
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

    {
        std::optional<detail::JoinThread> routing;
        {
            std::lock_guard lock(mutex_);
            routing = std::move(routing_thread_);
            routing_thread_.reset();
        }
    }

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

bool Runtime::is_accepting() const {
    return accepting_.load();
}

bool Runtime::is_healthy() const {
    if (!accepting_.load() || !started_) {
        return false;
    }
    if (workers_.empty()) {
        return false;
    }
    for (const auto &worker : workers_) {
        if (worker && worker->snapshot().accepting) {
            return true;
        }
    }
    return false;
}

RuntimeSnapshot Runtime::snapshot() const {
    RuntimeSnapshot snap;
    snap.accepting = accepting_.load();
    snap.started = started_;
    if (scheduler_) {
        snap.scheduler_depth = scheduler_->size();
        snap.scheduler_capacity = scheduler_->capacity();
    }
    snap.worker_count = workers_.size();
    for (const auto &worker : workers_) {
        WorkerRuntimeSnapshot worker_snap;
        worker_snap.worker = worker->snapshot();
        worker_snap.batch_metrics = worker->batch_metrics();
        worker_snap.residency_metrics = worker->residency_metrics();
        snap.workers.push_back(std::move(worker_snap));
    }
    return snap;
}

MetricsSnapshot Runtime::metrics_snapshot() const {
    MetricsSnapshot snap;
    if (scheduler_) {
        snap.scheduler_depth = scheduler_->size();
        snap.scheduler_capacity = scheduler_->capacity();
    }
    for (const auto &worker : workers_) {
        WorkerRuntimeSnapshot worker_snap;
        worker_snap.worker = worker->snapshot();
        worker_snap.batch_metrics = worker->batch_metrics();
        worker_snap.residency_metrics = worker->residency_metrics();
        snap.workers.push_back(std::move(worker_snap));
    }
    return snap;
}

std::vector<ModelRuntimeSnapshot> Runtime::model_snapshots() const {
    std::vector<ModelRuntimeSnapshot> models;
    for (const auto &spec : registry_->models()) {
        ModelRuntimeSnapshot model_snap;
        model_snap.spec = spec;
        for (const auto &worker : workers_) {
            const auto resident = worker->snapshot().resident_model_ids;
            if (std::find(resident.begin(), resident.end(), spec.model_id) != resident.end()) {
                model_snap.resident_worker_ids.push_back(worker->id());
            }
        }
        std::sort(model_snap.resident_worker_ids.begin(), model_snap.resident_worker_ids.end());
        models.push_back(std::move(model_snap));
    }
    std::sort(models.begin(), models.end(),
              [](const ModelRuntimeSnapshot &a, const ModelRuntimeSnapshot &b) {
                  return a.spec.model_id < b.spec.model_id;
              });
    return models;
}

const ModelRegistry &Runtime::registry() const {
    return *registry_;
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

Status Runtime::route_request(const RequestPtr &request) {
    if (!request || request->is_terminal()) {
        return Status::success();
    }
    request->try_timeout_if_expired(std::chrono::steady_clock::now());
    if (request->is_terminal()) {
        return Status::success();
    }

    ModelSpec spec;
    auto find_status = registry_->find(request->model_id(), spec);
    if (!find_status.ok()) {
        request->try_reject(find_status);
        return find_status;
    }

    auto snapshots = collect_snapshots();
    auto selected = router_->select(spec, snapshots);
    if (!selected.ok()) {
        request->try_reject(selected.status);
        return selected.status;
    }

    Worker *target = find_worker(*selected.worker_id);
    if (!target) {
        auto err = Status::error(ErrorCode::InternalError, "selected worker missing");
        request->try_reject(err);
        return err;
    }

    const auto deadline = request->deadline();
    Status enqueue_status;
    if (deadline.has_value()) {
        enqueue_status = target->enqueue_until(request, *deadline);
    } else {
        enqueue_status = target->enqueue(request);
    }
    if (!enqueue_status.ok()) {
        if (enqueue_status.code == ErrorCode::TimedOut) {
            return enqueue_status;
        }
        request->try_reject(enqueue_status);
        return enqueue_status;
    }
    return Status::success();
}

void Runtime::routing_loop() {
    while (true) {
        auto next = scheduler_->next();
        if (!next.has_value()) {
            break;
        }
        (void)route_request(*next);
    }
}

} // namespace airuntime
