#pragma once

#include "airuntime/model.hpp"
#include "airuntime/status.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace airuntime {

class ModelRegistry {
  public:
    class Builder {
      public:
        Status add(ModelSpec spec);

        // Returns nullptr and sets status on failure.
        std::unique_ptr<const ModelRegistry> build(Status &status) const;

      private:
        std::unordered_map<std::string, ModelSpec> models_;
    };

    [[nodiscard]] Status find(std::string_view model_id, ModelSpec &out) const;
    [[nodiscard]] bool contains(std::string_view model_id) const;
    [[nodiscard]] std::size_t size() const;

  private:
    explicit ModelRegistry(std::unordered_map<std::string, ModelSpec> models);

    const std::unordered_map<std::string, ModelSpec> models_;
};

} // namespace airuntime
