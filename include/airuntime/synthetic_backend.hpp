#pragma once

#include "airuntime/backend.hpp"

#include <chrono>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace airuntime {

struct SyntheticModelConfig {
    std::chrono::microseconds load_latency{0};
    std::chrono::microseconds prefill_latency{0};
    std::chrono::microseconds per_token_latency{0};
    bool fail_load{false};
    bool fail_infer{false};
};

struct SyntheticBackendMetrics {
    std::size_t load_count{0};
    std::size_t inference_count{0};
    std::size_t batch_inference_count{0};
    std::size_t last_batch_size{0};
    std::chrono::microseconds last_simulated_load_cost{0};
    std::chrono::microseconds total_simulated_load_cost{0};
    std::chrono::microseconds last_simulated_inference_cost{0};
    std::chrono::microseconds total_simulated_inference_cost{0};
    std::chrono::microseconds last_simulated_batch_cost{0};
    std::chrono::microseconds total_simulated_batch_cost{0};
};

class SyntheticModelBackend final : public IModelBackend {
  public:
    void register_model(std::string model_id, SyntheticModelConfig config);

    Status load(const ModelSpec &model) override;
    Status unload(std::string_view model_id) override;
    [[nodiscard]] bool is_loaded(std::string_view model_id) const override;
    InferenceResult infer(const InferenceRequest &request) override;
    std::vector<InferenceResult>
    infer_batch(std::span<const InferenceRequest *const> requests) override;

    [[nodiscard]] ModelState model_state(std::string_view model_id) const;
    [[nodiscard]] SyntheticBackendMetrics metrics() const;

  private:
    struct ModelEntry {
        SyntheticModelConfig config;
        ModelState state{ModelState::Unloaded};
    };

    mutable std::mutex mutex_;
    std::unordered_map<std::string, ModelEntry> models_;
    SyntheticBackendMetrics metrics_;
};

} // namespace airuntime
