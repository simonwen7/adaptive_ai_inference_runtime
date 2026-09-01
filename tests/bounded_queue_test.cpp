#include "airuntime/bounded_queue.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

using airuntime::BoundedQueue;
using airuntime::ErrorCode;

TEST(BoundedQueueTest, CapacityAndPushPop) {
    BoundedQueue<int> queue(2);
    EXPECT_EQ(queue.capacity(), 2u);
    EXPECT_EQ(queue.size(), 0u);
    EXPECT_TRUE(queue.try_push(1).ok());
    EXPECT_TRUE(queue.try_push(2).ok());
    EXPECT_EQ(queue.size(), 2u);

    auto full = queue.try_push(3);
    EXPECT_EQ(full.code, ErrorCode::QueueFull);

    auto first = queue.wait_pop();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(*first, 1);
    auto second = queue.wait_pop();
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(*second, 2);
}

TEST(BoundedQueueTest, FifoOrder) {
    BoundedQueue<std::string> queue(4);
    ASSERT_TRUE(queue.try_push("a").ok());
    ASSERT_TRUE(queue.try_push("b").ok());
    ASSERT_TRUE(queue.try_push("c").ok());
    EXPECT_EQ(*queue.wait_pop(), "a");
    EXPECT_EQ(*queue.wait_pop(), "b");
    EXPECT_EQ(*queue.wait_pop(), "c");
}

TEST(BoundedQueueTest, CloseRejectsPushes) {
    BoundedQueue<int> queue(2);
    queue.close();
    EXPECT_TRUE(queue.closed());
    EXPECT_EQ(queue.try_push(1).code, ErrorCode::QueueClosed);
    EXPECT_EQ(queue.wait_push(1).code, ErrorCode::QueueClosed);
}

TEST(BoundedQueueTest, WaitPopDrainsAfterClose) {
    BoundedQueue<int> queue(4);
    ASSERT_TRUE(queue.try_push(10).ok());
    ASSERT_TRUE(queue.try_push(20).ok());
    queue.close();

    EXPECT_EQ(*queue.wait_pop(), 10);
    EXPECT_EQ(*queue.wait_pop(), 20);
    EXPECT_FALSE(queue.wait_pop().has_value());
}

TEST(BoundedQueueTest, ClosedEmptyEndsConsumer) {
    BoundedQueue<int> queue(1);
    queue.close();
    EXPECT_FALSE(queue.wait_pop().has_value());
}

TEST(BoundedQueueTest, WaitPushBlocksUntilSpace) {
    BoundedQueue<int> queue(1);
    ASSERT_TRUE(queue.try_push(1).ok());

    std::atomic<bool> pushed{false};
    std::thread producer([&] {
        ASSERT_TRUE(queue.wait_push(2).ok());
        pushed.store(true);
    });

    while (queue.size() != 1u) {
        std::this_thread::yield();
    }
    EXPECT_FALSE(pushed.load());
    EXPECT_EQ(*queue.wait_pop(), 1);
    producer.join();
    EXPECT_TRUE(pushed.load());
    EXPECT_EQ(*queue.wait_pop(), 2);
}

TEST(BoundedQueueTest, WaitPopUntilReturnsAvailableImmediately) {
    BoundedQueue<int> queue(2);
    ASSERT_TRUE(queue.try_push(7).ok());
    auto past = std::chrono::steady_clock::now() - std::chrono::milliseconds{1};
    auto item = queue.wait_pop_until(past);
    ASSERT_TRUE(item.has_value());
    EXPECT_EQ(*item, 7);
}

TEST(BoundedQueueTest, WaitPopUntilTimesOutWhenEmpty) {
    BoundedQueue<int> queue(2);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{5};
    auto item = queue.wait_pop_until(deadline);
    EXPECT_FALSE(item.has_value());
}

TEST(BoundedQueueTest, WaitPopUntilDrainsAfterClose) {
    BoundedQueue<int> queue(2);
    ASSERT_TRUE(queue.try_push(1).ok());
    queue.close();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    EXPECT_EQ(*queue.wait_pop_until(deadline), 1);
    EXPECT_FALSE(queue.wait_pop_until(deadline).has_value());
}

TEST(BoundedQueueTest, ProducerConsumerSafety) {
    BoundedQueue<int> queue(8);
    constexpr int kCount = 100;
    std::vector<int> consumed;
    consumed.reserve(kCount);

    std::thread producer([&] {
        for (int i = 0; i < kCount; ++i) {
            ASSERT_TRUE(queue.wait_push(i).ok());
        }
        queue.close();
    });

    std::thread consumer([&] {
        while (true) {
            auto item = queue.wait_pop();
            if (!item.has_value()) {
                break;
            }
            consumed.push_back(*item);
        }
    });

    producer.join();
    consumer.join();

    ASSERT_EQ(consumed.size(), static_cast<std::size_t>(kCount));
    for (int i = 0; i < kCount; ++i) {
        EXPECT_EQ(consumed[static_cast<std::size_t>(i)], i);
    }
}
