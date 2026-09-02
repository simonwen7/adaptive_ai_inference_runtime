#include "airuntime/serving/request_handler.hpp"

#include <nlohmann/json.hpp>

#include <sstream>

namespace airuntime::serving {

namespace {

constexpr std::size_t kMaxOutputTokensDefault = 16;
constexpr std::size_t kMaxOutputTokensMin = 1;
constexpr std::size_t kMaxOutputTokensMax = 4096;
constexpr std::int64_t kTimeoutMsDefault = 30000;
constexpr std::int64_t kTimeoutMsMin = 1;
constexpr std::int64_t kTimeoutMsMax = 300000;

} // namespace

RequestHandler::RequestHandler(Runtime &runtime) : runtime_(runtime) {}

std::string RequestHandler::next_request_id() {
    const auto id = next_id_.fetch_add(1, std::memory_order_relaxed);
    return "req-" + std::to_string(id);
}

std::string RequestHandler::state_to_string(RequestState state) {
    switch (state) {
    case RequestState::Received:
        return "Received";
    case RequestState::Queued:
        return "Queued";
    case RequestState::Running:
        return "Running";
    case RequestState::Completed:
        return "Completed";
    case RequestState::Rejected:
        return "Rejected";
    case RequestState::Failed:
        return "Failed";
    case RequestState::Cancelled:
        return "Cancelled";
    case RequestState::TimedOut:
        return "TimedOut";
    }
    return "Unknown";
}

std::string RequestHandler::error_code_to_string(ErrorCode code) {
    switch (code) {
    case ErrorCode::None:
        return "None";
    case ErrorCode::InvalidStateTransition:
        return "InvalidStateTransition";
    case ErrorCode::QueueFull:
        return "QueueFull";
    case ErrorCode::QueueClosed:
        return "QueueClosed";
    case ErrorCode::ModelNotFound:
        return "ModelNotFound";
    case ErrorCode::ModelLoadFailed:
        return "ModelLoadFailed";
    case ErrorCode::InferenceFailed:
        return "InferenceFailed";
    case ErrorCode::RuntimeStopped:
        return "RuntimeStopped";
    case ErrorCode::InternalError:
        return "InternalError";
    case ErrorCode::InsufficientMemory:
        return "InsufficientMemory";
    case ErrorCode::NoFeasibleWorker:
        return "NoFeasibleWorker";
    case ErrorCode::Cancelled:
        return "Cancelled";
    case ErrorCode::TimedOut:
        return "TimedOut";
    case ErrorCode::ContextLengthExceeded:
        return "ContextLengthExceeded";
    }
    return "Unknown";
}

HandlerResponse RequestHandler::handle_health() const {
    nlohmann::json body;
    if (runtime_.is_healthy()) {
        body["status"] = "healthy";
        return {200, "application/json", body.dump()};
    }
    if (!runtime_.is_accepting()) {
        body["status"] = "shutting_down";
        return {503, "application/json", body.dump()};
    }
    body["status"] = "unhealthy";
    return {503, "application/json", body.dump()};
}

HandlerResponse RequestHandler::handle_get(const std::string &target) const {
    if (target == "/health") {
        return handle_health();
    }
    if (target == "/v1/models") {
        nlohmann::json models = nlohmann::json::array();
        for (const auto &model : runtime_.model_snapshots()) {
            nlohmann::json entry;
            entry["model_id"] = model.spec.model_id;
            entry["estimated_memory_bytes"] = model.spec.estimated_memory_bytes;
            entry["estimated_load_cost"] = model.spec.estimated_load_cost;
            entry["resident_worker_ids"] = model.resident_worker_ids;
            models.push_back(std::move(entry));
        }
        nlohmann::json body;
        body["models"] = std::move(models);
        return {200, "application/json", body.dump()};
    }
    if (target == "/v1/runtime") {
        const auto snap = runtime_.snapshot();
        nlohmann::json body;
        body["accepting"] = snap.accepting;
        body["started"] = snap.started;
        body["scheduler_depth"] = snap.scheduler_depth;
        body["scheduler_capacity"] = snap.scheduler_capacity;
        body["worker_count"] = snap.worker_count;
        nlohmann::json workers = nlohmann::json::array();
        for (const auto &worker : snap.workers) {
            nlohmann::json w;
            w["worker_id"] = worker.worker.worker_id;
            w["queue_depth"] = worker.worker.queue_depth;
            w["queue_capacity"] = worker.worker.queue_capacity;
            w["active_count"] = worker.worker.active_count;
            w["accepting"] = worker.worker.accepting;
            w["memory_budget_bytes"] = worker.worker.memory_budget_bytes;
            w["memory_used_bytes"] = worker.worker.memory_used_bytes;
            w["resident_model_ids"] = worker.worker.resident_model_ids;
            w["batch_metrics"] = {
                {"batches_executed", worker.batch_metrics.batches_executed},
                {"requests_executed_via_batches",
                 worker.batch_metrics.requests_executed_via_batches},
                {"multi_request_batches", worker.batch_metrics.multi_request_batches},
                {"max_batch_size_observed", worker.batch_metrics.max_batch_size_observed}};
            w["residency_metrics"] = {
                {"loads", worker.residency_metrics.loads},
                {"reloads", worker.residency_metrics.reloads},
                {"evictions", worker.residency_metrics.evictions},
                {"residency_hits", worker.residency_metrics.residency_hits},
                {"residency_misses", worker.residency_metrics.residency_misses}};
            workers.push_back(std::move(w));
        }
        body["workers"] = std::move(workers);
        return {200, "application/json", body.dump()};
    }
    if (target == "/metrics") {
        const auto metrics = runtime_.metrics_snapshot();
        nlohmann::json body;
        body["scheduler_depth"] = metrics.scheduler_depth;
        body["scheduler_capacity"] = metrics.scheduler_capacity;
        nlohmann::json workers = nlohmann::json::array();
        for (const auto &worker : metrics.workers) {
            nlohmann::json w;
            w["worker_id"] = worker.worker.worker_id;
            w["queue_depth"] = worker.worker.queue_depth;
            w["active_count"] = worker.worker.active_count;
            w["memory_budget_bytes"] = worker.worker.memory_budget_bytes;
            w["memory_used_bytes"] = worker.worker.memory_used_bytes;
            w["batch_metrics"] = {
                {"batches_executed", worker.batch_metrics.batches_executed},
                {"requests_executed_via_batches",
                 worker.batch_metrics.requests_executed_via_batches},
                {"multi_request_batches", worker.batch_metrics.multi_request_batches},
                {"max_batch_size_observed", worker.batch_metrics.max_batch_size_observed}};
            w["residency_metrics"] = {
                {"loads", worker.residency_metrics.loads},
                {"reloads", worker.residency_metrics.reloads},
                {"evictions", worker.residency_metrics.evictions},
                {"residency_hits", worker.residency_metrics.residency_hits},
                {"residency_misses", worker.residency_metrics.residency_misses}};
            workers.push_back(std::move(w));
        }
        body["workers"] = std::move(workers);
        return {200, "application/json", body.dump()};
    }
    return {404, "application/json", R"({"error":"not found"})"};
}

std::optional<InferRequestSpec>
RequestHandler::parse_infer_request(const std::string &body, std::string &error_message) const {
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(body);
    } catch (const std::exception &ex) {
        error_message = ex.what();
        return std::nullopt;
    }
    if (!json.is_object()) {
        error_message = "request body must be a JSON object";
        return std::nullopt;
    }
    for (auto it = json.begin(); it != json.end(); ++it) {
        const auto &key = it.key();
        if (key != "model_id" && key != "prompt" && key != "max_output_tokens" &&
            key != "timeout_ms") {
            error_message = "unknown field: " + key;
            return std::nullopt;
        }
    }
    if (!json.contains("model_id") || !json["model_id"].is_string()) {
        error_message = "model_id is required";
        return std::nullopt;
    }
    if (!json.contains("prompt") || !json["prompt"].is_string()) {
        error_message = "prompt is required";
        return std::nullopt;
    }
    InferRequestSpec spec;
    spec.model_id = json["model_id"].get<std::string>();
    spec.prompt = json["prompt"].get<std::string>();
    if (spec.model_id.empty()) {
        error_message = "model_id must be non-empty";
        return std::nullopt;
    }
    if (spec.prompt.empty()) {
        error_message = "prompt must be non-empty";
        return std::nullopt;
    }
    if (json.contains("max_output_tokens")) {
        if (!json["max_output_tokens"].is_number_integer()) {
            error_message = "max_output_tokens must be an integer";
            return std::nullopt;
        }
        const auto value = json["max_output_tokens"].get<std::int64_t>();
        if (value < static_cast<std::int64_t>(kMaxOutputTokensMin) ||
            value > static_cast<std::int64_t>(kMaxOutputTokensMax)) {
            error_message = "max_output_tokens out of range";
            return std::nullopt;
        }
        spec.max_output_tokens = static_cast<std::size_t>(value);
    } else {
        spec.max_output_tokens = kMaxOutputTokensDefault;
    }
    if (json.contains("timeout_ms")) {
        if (!json["timeout_ms"].is_number_integer()) {
            error_message = "timeout_ms must be an integer";
            return std::nullopt;
        }
        const auto value = json["timeout_ms"].get<std::int64_t>();
        if (value < kTimeoutMsMin || value > kTimeoutMsMax) {
            error_message = "timeout_ms out of range";
            return std::nullopt;
        }
        spec.timeout = std::chrono::milliseconds(value);
    } else {
        spec.timeout = std::chrono::milliseconds(kTimeoutMsDefault);
    }
    return spec;
}

HandlerResponse RequestHandler::serialize_infer_response(const RequestSnapshot &snap) const {
    nlohmann::json body;
    body["request_id"] = snap.request_id;
    body["model_id"] = snap.model_id;
    body["state"] = state_to_string(snap.state);
    if (snap.state == RequestState::Completed && snap.result.has_value() && snap.result->output) {
        body["output"] = {{"text", snap.result->output->text},
                          {"output_tokens", snap.result->output->output_tokens}};
        body["error"] = nullptr;
    } else {
        body["output"] = nullptr;
        if (snap.result.has_value() && !snap.result->status.ok()) {
            body["error"] = {{"code", error_code_to_string(snap.result->status.code)},
                             {"message", snap.result->status.message}};
        } else {
            body["error"] = nullptr;
        }
    }
    HandlerResponse response;
    response.status_code = http_status_for_snapshot(snap);
    response.body = body.dump();
    return response;
}

std::string RequestHandler::serialize_state_event(const RequestSnapshot &snap) const {
    nlohmann::json event;
    event["event"] = "state";
    event["request_id"] = snap.request_id;
    event["state"] = state_to_string(snap.state);
    return event.dump() + "\n";
}

std::string RequestHandler::serialize_terminal_event(const RequestSnapshot &snap) const {
    nlohmann::json event;
    event["event"] = "terminal";
    event["request_id"] = snap.request_id;
    event["model_id"] = snap.model_id;
    event["state"] = state_to_string(snap.state);
    if (snap.state == RequestState::Completed && snap.result.has_value() && snap.result->output) {
        event["output"] = {{"text", snap.result->output->text},
                           {"output_tokens", snap.result->output->output_tokens}};
        event["error"] = nullptr;
    } else {
        event["output"] = nullptr;
        if (snap.result.has_value() && !snap.result->status.ok()) {
            event["error"] = {{"code", error_code_to_string(snap.result->status.code)},
                              {"message", snap.result->status.message}};
        } else {
            event["error"] = nullptr;
        }
    }
    return event.dump() + "\n";
}

int RequestHandler::http_status_for_snapshot(const RequestSnapshot &snap) const {
    if (snap.state == RequestState::Completed) {
        return 200;
    }
    if (snap.state == RequestState::Cancelled) {
        return 409;
    }
    if (snap.state == RequestState::TimedOut) {
        return 504;
    }
    if (!snap.result.has_value()) {
        return 500;
    }
    switch (snap.result->status.code) {
    case ErrorCode::ModelNotFound:
        return 404;
    case ErrorCode::ContextLengthExceeded:
        return 400;
    case ErrorCode::QueueFull:
    case ErrorCode::RuntimeStopped:
    case ErrorCode::NoFeasibleWorker:
    case ErrorCode::InsufficientMemory:
        return 503;
    case ErrorCode::ModelLoadFailed:
    case ErrorCode::InferenceFailed:
    case ErrorCode::InternalError:
        return 500;
    default:
        return 500;
    }
}

} // namespace airuntime::serving
