#include "airuntime/batch_builder.hpp"
#include "airuntime/request.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <string>

using airuntime::BatchBuilder;
using airuntime::BatchBuilderConfig;
using airuntime::InferenceRequest;
using airuntime::RequestPtr;
using airuntime::RequestState;

namespace {

RequestPtr make_req(const std::string &id, const std::string &model) {
    auto request = std::make_shared<InferenceRequest>(id, model, "p", 1);
    EXPECT_TRUE(request->transition_to(RequestState::Queued).ok());
    return request;
}

} // namespace

TEST(BatchBuilderTest, MaxBatchSizeAndPartial) {
    BatchBuilderConfig config;
    config.max_batch_size = 3;
    config.max_batch_wait = std::chrono::microseconds{0};
    BatchBuilder builder(config);

    std::deque<RequestPtr> pending{make_req("a2", "A"), make_req("a3", "A")};
    auto formed = builder.form(
        make_req("a1", "A"),
        [&](std::chrono::steady_clock::time_point) -> std::optional<RequestPtr> {
            if (pending.empty()) {
                return std::nullopt;
            }
            auto next = std::move(pending.front());
            pending.pop_front();
            return next;
        },
        true);

    ASSERT_EQ(formed.requests.size(), 3u);
    EXPECT_EQ(formed.requests[0]->request_id(), "a1");
    EXPECT_EQ(formed.requests[1]->request_id(), "a2");
    EXPECT_EQ(formed.requests[2]->request_id(), "a3");
    EXPECT_FALSE(formed.deferred.has_value());
}

TEST(BatchBuilderTest, MaxBatchSizeOne) {
    BatchBuilderConfig config;
    config.max_batch_size = 1;
    config.max_batch_wait = std::chrono::microseconds{1000};
    BatchBuilder builder(config);

    bool popped = false;
    auto formed = builder.form(
        make_req("a1", "A"),
        [&](std::chrono::steady_clock::time_point) -> std::optional<RequestPtr> {
            popped = true;
            return make_req("a2", "A");
        },
        true);

    ASSERT_EQ(formed.requests.size(), 1u);
    EXPECT_FALSE(popped);
    EXPECT_FALSE(formed.deferred.has_value());
}

TEST(BatchBuilderTest, DifferentModelSeparationPreservesLookahead) {
    BatchBuilderConfig config;
    config.max_batch_size = 8;
    config.max_batch_wait = std::chrono::microseconds{0};
    BatchBuilder builder(config);

    std::deque<RequestPtr> pending{make_req("b1", "B"), make_req("a2", "A")};
    auto formed = builder.form(
        make_req("a1", "A"),
        [&](std::chrono::steady_clock::time_point) -> std::optional<RequestPtr> {
            if (pending.empty()) {
                return std::nullopt;
            }
            auto next = std::move(pending.front());
            pending.pop_front();
            return next;
        },
        true);

    ASSERT_EQ(formed.requests.size(), 1u);
    EXPECT_EQ(formed.requests[0]->request_id(), "a1");
    ASSERT_TRUE(formed.deferred.has_value());
    EXPECT_EQ((*formed.deferred)->request_id(), "b1");
}

TEST(BatchBuilderTest, ContiguousSameModelDoesNotSkip) {
    // A1 B1 A2 A3 must NOT create A1+A2+A3
    BatchBuilderConfig config;
    config.max_batch_size = 8;
    config.max_batch_wait = std::chrono::microseconds{0};
    BatchBuilder builder(config);

    std::deque<RequestPtr> pending{make_req("b1", "B"), make_req("a2", "A"), make_req("a3", "A")};
    auto first = builder.form(
        make_req("a1", "A"),
        [&](std::chrono::steady_clock::time_point) -> std::optional<RequestPtr> {
            if (pending.empty()) {
                return std::nullopt;
            }
            auto next = std::move(pending.front());
            pending.pop_front();
            return next;
        },
        true);
    ASSERT_EQ(first.requests.size(), 1u);
    ASSERT_TRUE(first.deferred.has_value());

    auto second = builder.form(
        std::move(*first.deferred),
        [&](std::chrono::steady_clock::time_point) -> std::optional<RequestPtr> {
            if (pending.empty()) {
                return std::nullopt;
            }
            auto next = std::move(pending.front());
            pending.pop_front();
            return next;
        },
        true);
    ASSERT_EQ(second.requests.size(), 1u);
    EXPECT_EQ(second.requests[0]->request_id(), "b1");
    ASSERT_TRUE(second.deferred.has_value());
    EXPECT_EQ((*second.deferred)->request_id(), "a2");

    auto third = builder.form(
        std::move(*second.deferred),
        [&](std::chrono::steady_clock::time_point) -> std::optional<RequestPtr> {
            if (pending.empty()) {
                return std::nullopt;
            }
            auto next = std::move(pending.front());
            pending.pop_front();
            return next;
        },
        true);
    ASSERT_EQ(third.requests.size(), 2u);
    EXPECT_EQ(third.requests[0]->request_id(), "a2");
    EXPECT_EQ(third.requests[1]->request_id(), "a3");
}

TEST(BatchBuilderTest, ImmediateFlushWhenWaitDisallowed) {
    BatchBuilderConfig config;
    config.max_batch_size = 4;
    config.max_batch_wait = std::chrono::seconds{60};
    BatchBuilder builder(config);

    auto deadline_seen = std::chrono::steady_clock::time_point::max();
    auto formed = builder.form(
        make_req("a1", "A"),
        [&](std::chrono::steady_clock::time_point deadline) -> std::optional<RequestPtr> {
            deadline_seen = deadline;
            return std::nullopt;
        },
        false);

    ASSERT_EQ(formed.requests.size(), 1u);
    EXPECT_LE(deadline_seen, std::chrono::steady_clock::now() + std::chrono::milliseconds{50});
}

TEST(BatchBuilderTest, RejectsInvalidConfig) {
    BatchBuilderConfig config;
    config.max_batch_size = 0;
    EXPECT_THROW(BatchBuilder{config}, std::invalid_argument);
}
