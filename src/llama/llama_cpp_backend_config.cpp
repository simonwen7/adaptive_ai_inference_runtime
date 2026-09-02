#include "airuntime/llama/llama_cpp_backend_config.hpp"

namespace airuntime {

Status LlamaCppBackendConfig::validate() const {
    if (max_sequences < 1) {
        return Status::error(ErrorCode::InternalError, "max_sequences must be >= 1");
    }
    if (models.empty()) {
        return Status::error(ErrorCode::InternalError, "llama backend config has no models");
    }
    for (const auto &[model_id, model] : models) {
        if (model_id.empty()) {
            return Status::error(ErrorCode::InternalError, "empty model_id in llama config");
        }
        if (model.gguf_path.empty()) {
            return Status::error(ErrorCode::InternalError, "empty gguf_path in llama config");
        }
        if (model.context_tokens_per_sequence == 0) {
            return Status::error(ErrorCode::InternalError,
                                 "context_tokens_per_sequence must be > 0");
        }
        if (model.n_batch == 0) {
            return Status::error(ErrorCode::InternalError, "n_batch must be > 0");
        }
        if (model.n_ubatch == 0) {
            return Status::error(ErrorCode::InternalError, "n_ubatch must be > 0");
        }
        if (model.n_ubatch > model.n_batch) {
            return Status::error(ErrorCode::InternalError, "n_ubatch must be <= n_batch");
        }
        if (model.n_batch < max_sequences) {
            return Status::error(ErrorCode::InternalError, "n_batch must be >= max_sequences");
        }
        if (model.n_threads.has_value() && *model.n_threads <= 0) {
            return Status::error(ErrorCode::InternalError, "n_threads must be > 0");
        }
        if (model.n_threads_batch.has_value() && *model.n_threads_batch <= 0) {
            return Status::error(ErrorCode::InternalError, "n_threads_batch must be > 0");
        }
    }
    return Status::success();
}

} // namespace airuntime
