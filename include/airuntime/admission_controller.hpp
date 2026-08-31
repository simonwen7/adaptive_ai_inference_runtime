#pragma once

#include "airuntime/request.hpp"
#include "airuntime/scheduler.hpp"
#include "airuntime/status.hpp"

namespace airuntime {

class AdmissionController {
  public:
    explicit AdmissionController(IRequestScheduler &scheduler);

    Status admit(const RequestPtr &request);

  private:
    IRequestScheduler &scheduler_;
};

} // namespace airuntime
