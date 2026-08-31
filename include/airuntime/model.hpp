#pragma once

#include <string>

namespace airuntime {

struct ModelSpec {
    std::string model_id;
};

enum class ModelState { Unloaded, Loading, Resident, Failed };

} // namespace airuntime
