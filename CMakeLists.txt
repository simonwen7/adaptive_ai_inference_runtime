cmake_minimum_required(VERSION 3.20)

project(adaptive_ai_inference_runtime
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(MSVC)
    add_compile_options(/W4)
endif()

add_library(airuntime_core STATIC
    src/request.cpp
    src/synthetic_backend.cpp
    src/scheduler.cpp
    src/workload_aware_scheduler.cpp
    src/admission_controller.cpp
    src/worker.cpp
    src/runtime.cpp
    src/model_registry.cpp
    src/resource_manager.cpp
    src/eviction_policy.cpp
    src/model_manager.cpp
    src/router.cpp
    src/batch_builder.cpp
)

target_include_directories(airuntime_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(Threads REQUIRED)
target_link_libraries(airuntime_core PUBLIC Threads::Threads)

add_executable(runtime-server apps/runtime-server/main.cpp)
target_link_libraries(runtime-server PRIVATE airuntime_core)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(airuntime_tests
    tests/runtime_server_smoke_test.cpp
    tests/request_test.cpp
    tests/bounded_queue_test.cpp
    tests/scheduler_test.cpp
    tests/workload_aware_scheduler_test.cpp
    tests/admission_controller_test.cpp
    tests/synthetic_backend_test.cpp
    tests/worker_test.cpp
    tests/worker_batch_test.cpp
    tests/batch_builder_test.cpp
    tests/runtime_integration_test.cpp
    tests/model_registry_test.cpp
    tests/resource_manager_test.cpp
    tests/eviction_policy_test.cpp
    tests/router_test.cpp
    tests/model_manager_test.cpp
    tests/multi_worker_integration_test.cpp
    tests/m3_integration_test.cpp
)
target_link_libraries(airuntime_tests PRIVATE airuntime_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(airuntime_tests)
