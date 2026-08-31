#include "airuntime/bounded_queue.hpp"

#include <gtest/gtest.h>

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

TEST(BoundedQueueTest, ProducerConsumerSafety) {
    BoundedQueue<int> queue(8);
    constexpr int kCount = 100;
    std::vector<int> consumed;
    consumed.reserve(kCount);

    std::thread producer([&] {
        for (int i = 0; i < kCount; ++i) {
            while (!queue.try_push(i).ok()) {
                std::this_thread::yield();
            }
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
