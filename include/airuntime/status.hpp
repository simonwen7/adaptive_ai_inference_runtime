#pragma once

#include <optional>
#include <string>
#include <utility>

namespace airuntime {

enum class ErrorCode {
    None,
    InvalidStateTransition,
    QueueFull,
    QueueClosed,
    ModelNotFound,
    ModelLoadFailed,
    InferenceFailed,
    RuntimeStopped,
    InternalError,
    InsufficientMemory,
    NoFeasibleWorker
};

struct Status {
    ErrorCode code{ErrorCode::None};
    std::string message;

    Status() = default;

    Status(ErrorCode code_in, std::string message_in = {})
        : code(code_in), message(std::move(message_in)) {}

    [[nodiscard]] bool ok() const {
        return code == ErrorCode::None;
    }

    static Status success() {
        return Status{};
    }

    static Status error(ErrorCode code_in, std::string message_in = {}) {
        return Status{code_in, std::move(message_in)};
    }
};

struct InferenceOutput {
    std::string text;
    std::size_t output_tokens{0};
};

struct InferenceResult {
    Status status;
    std::optional<InferenceOutput> output;

    static InferenceResult success(InferenceOutput output_in) {
        InferenceResult result;
        result.status = Status::success();
        result.output = std::move(output_in);
        return result;
    }

    static InferenceResult failure(Status status_in) {
        InferenceResult result;
        result.status = std::move(status_in);
        return result;
    }

    [[nodiscard]] bool ok() const {
        return status.ok();
    }
};

} // namespace airuntime
