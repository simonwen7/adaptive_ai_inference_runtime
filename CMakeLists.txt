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

add_executable(runtime-server apps/runtime-server/main.cpp)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(runtime_server_smoke_test tests/runtime_server_smoke_test.cpp)
target_link_libraries(runtime_server_smoke_test PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(runtime_server_smoke_test)
