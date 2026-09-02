#include "airuntime/llama/llama_cpp_backend.hpp"

#include <llama-cpp.h>
#include <llama.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace airuntime {
namespace {

struct LlamaBatchGuard {
    llama_batch batch{};
    explicit LlamaBatchGuard(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max)
        : batch(llama_batch_init(n_tokens_alloc, embd, n_seq_max)) {}
    ~LlamaBatchGuard() {
        llama_batch_free(batch);
    }
    LlamaBatchGuard(const LlamaBatchGuard &) = delete;
    LlamaBatchGuard &operator=(const LlamaBatchGuard &) = delete;
};

void batch_clear(llama_batch &batch) {
    batch.n_tokens = 0;
}

void batch_add(llama_batch &batch, llama_token token, llama_pos pos, llama_seq_id seq_id,
               bool logits) {
    const int32_t i = batch.n_tokens;
    batch.token[i] = token;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = seq_id;
    batch.logits[i] = logits ? 1 : 0;
    ++batch.n_tokens;
}

std::vector<llama_token> tokenize_prompt(const llama_vocab *vocab, const std::string &prompt) {
    const int32_t n_prompt = -llama_tokenize(
        vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    if (n_prompt < 0) {
        throw std::runtime_error("llama_tokenize size probe failed");
    }
    std::vector<llama_token> tokens(static_cast<std::size_t>(n_prompt));
    if (n_prompt == 0) {
        return tokens;
    }
    const int32_t written =
        llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(),
                       static_cast<int32_t>(tokens.size()), true, true);
    if (written < 0) {
        throw std::runtime_error("llama_tokenize failed");
    }
    tokens.resize(static_cast<std::size_t>(written));
    return tokens;
}

std::string token_to_piece(const llama_vocab *vocab, llama_token token) {
    std::vector<char> buf(256);
    int32_t n =
        llama_token_to_piece(vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
    if (n < 0) {
        buf.resize(static_cast<std::size_t>(-n));
        n = llama_token_to_piece(vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0,
                                 true);
    }
    if (n < 0) {
        throw std::runtime_error("llama_token_to_piece failed");
    }
    return std::string(buf.data(), static_cast<std::size_t>(n));
}

bool checked_mul_u32(std::uint32_t a, std::uint32_t b, std::uint32_t &out) {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > std::numeric_limits<std::uint32_t>::max() / b) {
        return false;
    }
    out = a * b;
    return true;
}

} // namespace

struct LlamaCppBackend::Impl {
    std::shared_ptr<LlamaBackendRuntime> runtime;
    LlamaCppBackendConfig config;
    mutable std::mutex mutex;

    struct LoadedModel {
        LlamaModelConfig model_config;
        llama_model_ptr model;
        llama_context_ptr context;
        const llama_vocab *vocab{nullptr};
    };

    std::unordered_map<std::string, LoadedModel> loaded;

    void clear_memory_locked(LoadedModel &entry) {
        if (!entry.context) {
            return;
        }
        llama_memory_t mem = llama_get_memory(entry.context.get());
        if (mem) {
            llama_memory_clear(mem, true);
        }
    }

    Status load_locked(const ModelSpec &model) {
        auto cfg_it = config.models.find(model.model_id);
        if (cfg_it == config.models.end()) {
            return Status::error(ErrorCode::ModelNotFound,
                                 "model not configured for llama backend");
        }
        if (loaded.contains(model.model_id)) {
            return Status::success();
        }

        const LlamaModelConfig &mcfg = cfg_it->second;
        std::error_code ec;
        if (!std::filesystem::exists(mcfg.gguf_path, ec) ||
            !std::filesystem::is_regular_file(mcfg.gguf_path, ec)) {
            return Status::error(ErrorCode::ModelLoadFailed,
                                 "gguf path missing or not a regular file");
        }

        std::uint32_t n_ctx = 0;
        if (!checked_mul_u32(mcfg.context_tokens_per_sequence,
                             static_cast<std::uint32_t>(config.max_sequences), n_ctx)) {
            return Status::error(ErrorCode::ModelLoadFailed, "n_ctx overflow");
        }

        llama_model_params model_params = llama_model_default_params();
        if (mcfg.n_gpu_layers.has_value()) {
            model_params.n_gpu_layers = *mcfg.n_gpu_layers;
        } else if (llama_supports_gpu_offload()) {
            model_params.n_gpu_layers = -1;
        } else {
            model_params.n_gpu_layers = 0;
        }

        llama_model_ptr model_ptr(
            llama_model_load_from_file(mcfg.gguf_path.string().c_str(), model_params));
        if (!model_ptr) {
            return Status::error(ErrorCode::ModelLoadFailed, "llama_model_load_from_file failed");
        }

        if (!llama_model_has_decoder(model_ptr.get()) ||
            llama_model_is_diffusion(model_ptr.get())) {
            return Status::error(ErrorCode::ModelLoadFailed,
                                 "unsupported model architecture for M5 text generation");
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = mcfg.n_batch;
        ctx_params.n_ubatch = mcfg.n_ubatch;
        ctx_params.n_seq_max = static_cast<std::uint32_t>(config.max_sequences);
        ctx_params.kv_unified = false;
        if (mcfg.n_threads.has_value()) {
            ctx_params.n_threads = *mcfg.n_threads;
        }
        if (mcfg.n_threads_batch.has_value()) {
            ctx_params.n_threads_batch = *mcfg.n_threads_batch;
        }

        llama_context_ptr ctx_ptr(llama_init_from_model(model_ptr.get(), ctx_params));
        if (!ctx_ptr) {
            return Status::error(ErrorCode::ModelLoadFailed, "llama_init_from_model failed");
        }

        LoadedModel entry;
        entry.model_config = mcfg;
        entry.vocab = llama_model_get_vocab(model_ptr.get());
        if (!entry.vocab) {
            return Status::error(ErrorCode::ModelLoadFailed, "llama_model_get_vocab failed");
        }
        entry.context = std::move(ctx_ptr);
        entry.model = std::move(model_ptr);
        clear_memory_locked(entry);
        loaded.emplace(model.model_id, std::move(entry));
        return Status::success();
    }

    std::vector<InferenceResult>
    infer_batch_locked(std::span<const InferenceRequest *const> requests) {
        std::vector<InferenceResult> results;
        results.reserve(requests.size());
        if (requests.empty()) {
            return results;
        }

        for (const auto *request : requests) {
            if (!request) {
                results.push_back(InferenceResult::failure(
                    Status::error(ErrorCode::InternalError, "null request in batch")));
            } else {
                results.push_back(InferenceResult{});
            }
        }

        bool any_null = false;
        for (std::size_t i = 0; i < requests.size(); ++i) {
            if (!requests[i]) {
                any_null = true;
            }
        }
        if (any_null) {
            return results;
        }

        const std::string model_id = requests[0]->model_id();
        for (std::size_t i = 1; i < requests.size(); ++i) {
            if (requests[i]->model_id() != model_id) {
                for (auto &result : results) {
                    result = InferenceResult::failure(
                        Status::error(ErrorCode::InternalError, "incompatible batch member"));
                }
                return results;
            }
        }

        if (requests.size() > config.max_sequences) {
            for (auto &result : results) {
                result = InferenceResult::failure(
                    Status::error(ErrorCode::InternalError, "batch exceeds max_sequences"));
            }
            return results;
        }

        auto it = loaded.find(model_id);
        if (it == loaded.end()) {
            for (auto &result : results) {
                result = InferenceResult::failure(
                    Status::error(ErrorCode::InferenceFailed, "model is not loaded"));
            }
            return results;
        }

        LoadedModel &entry = it->second;
        struct MemoryGuard {
            Impl *self;
            LoadedModel *model;
            ~MemoryGuard() {
                if (self && model) {
                    self->clear_memory_locked(*model);
                }
            }
        } memory_guard{this, &entry};
        clear_memory_locked(entry);

        struct LiveSeq {
            std::size_t result_index{0};
            llama_seq_id seq_id{0};
            std::vector<llama_token> prompt_tokens;
            llama_pos n_past{0};
            std::size_t generated{0};
            std::size_t max_output{0};
            std::string text;
            llama_sampler_ptr sampler;
            bool active{true};
            int32_t logits_batch_index{-1};
        };

        std::vector<LiveSeq> live;
        live.reserve(requests.size());

        for (std::size_t i = 0; i < requests.size(); ++i) {
            const InferenceRequest &req = *requests[i];
            std::vector<llama_token> tokens;
            try {
                tokens = tokenize_prompt(entry.vocab, req.prompt());
            } catch (const std::exception &ex) {
                results[i] =
                    InferenceResult::failure(Status::error(ErrorCode::InferenceFailed, ex.what()));
                continue;
            }

            const std::uint64_t needed = static_cast<std::uint64_t>(tokens.size()) +
                                         static_cast<std::uint64_t>(req.max_output_tokens());
            if (needed > entry.model_config.context_tokens_per_sequence) {
                results[i] = InferenceResult::failure(Status::error(
                    ErrorCode::ContextLengthExceeded,
                    "prompt + max_output_tokens exceeds context_tokens_per_sequence"));
                continue;
            }
            if (tokens.empty()) {
                results[i] = InferenceResult::failure(
                    Status::error(ErrorCode::InferenceFailed, "empty tokenization result"));
                continue;
            }

            LiveSeq seq;
            seq.result_index = i;
            seq.seq_id = static_cast<llama_seq_id>(live.size());
            seq.prompt_tokens = std::move(tokens);
            seq.max_output = req.max_output_tokens();
            auto sparams = llama_sampler_chain_default_params();
            sparams.no_perf = true;
            seq.sampler.reset(llama_sampler_chain_init(sparams));
            if (!seq.sampler) {
                results[i] = InferenceResult::failure(
                    Status::error(ErrorCode::InferenceFailed, "sampler init failed"));
                continue;
            }
            llama_sampler_chain_add(seq.sampler.get(), llama_sampler_init_greedy());
            live.push_back(std::move(seq));
        }

        if (live.empty()) {
            return results;
        }

        const int32_t n_batch_cap = static_cast<int32_t>(entry.model_config.n_batch);
        const int32_t n_seq_max = static_cast<int32_t>(config.max_sequences);
        LlamaBatchGuard batch_guard(n_batch_cap, 0, n_seq_max);
        llama_batch &batch = batch_guard.batch;

        auto fail_live = [&](const Status &status) {
            for (auto &seq : live) {
                if (!seq.active) {
                    continue;
                }
                seq.active = false;
                results[seq.result_index] = InferenceResult::failure(status);
            }
        };

        // Prefix stage: all tokens except the last for each live sequence.
        bool more_prefix = true;
        while (more_prefix) {
            more_prefix = false;
            batch_clear(batch);
            for (auto &seq : live) {
                if (!seq.active) {
                    continue;
                }
                while (seq.n_past + 1 < static_cast<llama_pos>(seq.prompt_tokens.size())) {
                    if (batch.n_tokens >= n_batch_cap) {
                        more_prefix = true;
                        break;
                    }
                    const llama_token tok = seq.prompt_tokens[static_cast<std::size_t>(seq.n_past)];
                    batch_add(batch, tok, seq.n_past, seq.seq_id, false);
                    ++seq.n_past;
                }
            }
            if (batch.n_tokens == 0) {
                break;
            }
            if (llama_decode(entry.context.get(), batch) != 0) {
                fail_live(Status::error(ErrorCode::InferenceFailed, "llama_decode prefix failed"));
                return results;
            }
        }

        // Bootstrap: final prompt token for each live sequence, with logits.
        batch_clear(batch);
        for (auto &seq : live) {
            if (!seq.active) {
                continue;
            }
            const llama_token last =
                seq.prompt_tokens[static_cast<std::size_t>(seq.prompt_tokens.size() - 1)];
            seq.logits_batch_index = batch.n_tokens;
            batch_add(batch, last, seq.n_past, seq.seq_id, true);
            ++seq.n_past;
        }
        if (batch.n_tokens == 0) {
            return results;
        }
        if (llama_decode(entry.context.get(), batch) != 0) {
            fail_live(Status::error(ErrorCode::InferenceFailed, "llama_decode bootstrap failed"));
            return results;
        }

        // First sample from bootstrap logits, then generation loop.
        for (;;) {
            batch_clear(batch);
            bool any_active = false;

            for (auto &seq : live) {
                if (!seq.active) {
                    continue;
                }
                llama_token id = llama_sampler_sample(seq.sampler.get(), entry.context.get(),
                                                      seq.logits_batch_index);
                llama_sampler_accept(seq.sampler.get(), id);

                if (llama_vocab_is_eog(entry.vocab, id)) {
                    seq.active = false;
                    results[seq.result_index] = InferenceResult::success(
                        InferenceOutput{std::move(seq.text), seq.generated});
                    continue;
                }

                try {
                    seq.text += token_to_piece(entry.vocab, id);
                } catch (const std::exception &ex) {
                    seq.active = false;
                    results[seq.result_index] = InferenceResult::failure(
                        Status::error(ErrorCode::InferenceFailed, ex.what()));
                    continue;
                }
                ++seq.generated;
                if (seq.generated >= seq.max_output) {
                    seq.active = false;
                    results[seq.result_index] = InferenceResult::success(
                        InferenceOutput{std::move(seq.text), seq.generated});
                    continue;
                }

                any_active = true;
                seq.logits_batch_index = batch.n_tokens;
                batch_add(batch, id, seq.n_past, seq.seq_id, true);
                ++seq.n_past;
            }

            if (!any_active) {
                break;
            }
            if (llama_decode(entry.context.get(), batch) != 0) {
                fail_live(
                    Status::error(ErrorCode::InferenceFailed, "llama_decode generation failed"));
                return results;
            }
        }

        for (auto &seq : live) {
            if (seq.active) {
                results[seq.result_index] =
                    InferenceResult::success(InferenceOutput{std::move(seq.text), seq.generated});
                seq.active = false;
            }
        }
        return results;
    }
};

LlamaCppBackend::LlamaCppBackend(std::shared_ptr<LlamaBackendRuntime> runtime,
                                 LlamaCppBackendConfig config)
    : impl_(std::make_unique<Impl>()) {
    if (!runtime) {
        throw std::invalid_argument("LlamaCppBackend requires LlamaBackendRuntime");
    }
    const Status status = config.validate();
    if (!status.ok()) {
        throw std::invalid_argument(status.message.empty() ? "invalid LlamaCppBackendConfig"
                                                           : status.message);
    }
    impl_->runtime = std::move(runtime);
    impl_->config = std::move(config);
}

LlamaCppBackend::~LlamaCppBackend() = default;

Status LlamaCppBackend::load(const ModelSpec &model) {
    std::lock_guard lock(impl_->mutex);
    return impl_->load_locked(model);
}

Status LlamaCppBackend::unload(std::string_view model_id) {
    std::lock_guard lock(impl_->mutex);
    auto it = impl_->loaded.find(std::string(model_id));
    if (it == impl_->loaded.end()) {
        return Status::error(ErrorCode::ModelNotFound, "model not loaded");
    }
    // Destroy context before model via member order in LoadedModel + unique_ptr.
    it->second.context.reset();
    it->second.model.reset();
    it->second.vocab = nullptr;
    impl_->loaded.erase(it);
    return Status::success();
}

bool LlamaCppBackend::is_loaded(std::string_view model_id) const {
    std::lock_guard lock(impl_->mutex);
    return impl_->loaded.contains(std::string(model_id));
}

InferenceResult LlamaCppBackend::infer(const InferenceRequest &request) {
    const InferenceRequest *ptr = &request;
    auto results = infer_batch(std::span<const InferenceRequest *const>(&ptr, 1));
    if (results.size() != 1) {
        return InferenceResult::failure(
            Status::error(ErrorCode::InternalError, "infer_batch cardinality error"));
    }
    return std::move(results.front());
}

std::vector<InferenceResult>
LlamaCppBackend::infer_batch(std::span<const InferenceRequest *const> requests) {
    std::lock_guard lock(impl_->mutex);
    try {
        return impl_->infer_batch_locked(requests);
    } catch (const std::exception &ex) {
        std::vector<InferenceResult> results;
        results.reserve(requests.size());
        for (std::size_t i = 0; i < requests.size(); ++i) {
            results.push_back(
                InferenceResult::failure(Status::error(ErrorCode::InferenceFailed, ex.what())));
        }
        return results;
    } catch (...) {
        std::vector<InferenceResult> results;
        results.reserve(requests.size());
        for (std::size_t i = 0; i < requests.size(); ++i) {
            results.push_back(InferenceResult::failure(
                Status::error(ErrorCode::InferenceFailed, "unknown llama backend exception")));
        }
        return results;
    }
}

bool LlamaCppBackend::supports_gpu_offload() const {
    return llama_supports_gpu_offload();
}

} // namespace airuntime
