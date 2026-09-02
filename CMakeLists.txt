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

find_package(Boost 1.83 REQUIRED COMPONENTS system)

include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

add_library(airuntime_serving STATIC
    src/serving/request_handler.cpp
    src/serving/http_session.cpp
    src/serving/http_server.cpp
)

target_include_directories(airuntime_serving
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(airuntime_serving
    PUBLIC
        airuntime_core
        Boost::system
        nlohmann_json::nlohmann_json
        Threads::Threads
)

add_executable(runtime-server apps/runtime-server/main.cpp)
target_link_libraries(runtime-server PRIVATE airuntime_serving)

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
    tests/reliability_test.cpp
)
target_link_libraries(airuntime_tests PRIVATE airuntime_core GTest::gtest_main)

add_executable(airuntime_serving_tests
    tests/request_handler_test.cpp
    tests/http_integration_test.cpp
)
target_link_libraries(airuntime_serving_tests PRIVATE airuntime_serving GTest::gtest_main)
target_compile_definitions(airuntime_serving_tests PRIVATE
    RUNTIME_SERVER_BIN="$<TARGET_FILE:runtime-server>")

add_executable(airuntime_http_integration_runner tests/http_integration_runner.cpp)
target_compile_definitions(airuntime_http_integration_runner PRIVATE
    RUNTIME_SERVER_BIN="$<TARGET_FILE:runtime-server>")

include(GoogleTest)
gtest_discover_tests(airuntime_tests)
gtest_discover_tests(airuntime_serving_tests)
add_test(NAME HttpIntegrationRunner COMMAND airuntime_http_integration_runner)
