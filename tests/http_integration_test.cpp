#include <gtest/gtest.h>

// HTTP integration is executed by the separate airuntime_http_integration_runner
// executable to avoid fork/popen deadlocks inside the multi-threaded gtest process.

TEST(HttpIntegrationTest, DelegatedToStandaloneRunner) {
    SUCCEED();
}
