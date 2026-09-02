#include "airuntime/admission_controller.hpp"

#include <chrono>

namespace airuntime {

AdmissionController::AdmissionController(IRequestScheduler &scheduler) : scheduler_(scheduler) {}

Status AdmissionController::admit(const RequestPtr &request) {
    if (!request) {
        return Status::error(ErrorCode::InternalError, "null request");
    }

    request->try_timeout_if_expired(std::chrono::steady_clock::now());
    if (request->is_terminal()) {
        return Status::success();
    }

    auto queued = request->transition_to(RequestState::Queued);
    if (!queued.ok()) {
        return queued;
    }

    auto enqueue_status = scheduler_.enqueue(request);
    if (!enqueue_status.ok()) {
        ErrorCode reject_code = enqueue_status.code;
        if (reject_code == ErrorCode::QueueClosed) {
            reject_code = ErrorCode::RuntimeStopped;
        }
        auto reject_status = Status::error(reject_code, enqueue_status.message.empty()
                                                            ? "request rejected by admission"
                                                            : enqueue_status.message);
        auto transition = request->try_reject(reject_status);
        if (!transition) {
            return Status::error(ErrorCode::InvalidStateTransition, "request already terminal");
        }
        return reject_status;
    }

    return Status::success();
}

} // namespace airuntime
