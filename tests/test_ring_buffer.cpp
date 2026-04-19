#include <gtest/gtest.h>

#include "core/ring_buffer.h"

using namespace drone_tracker;

TEST(RingBuffer, PushPop) {
    RingBuffer<int, 4> buf;
    EXPECT_TRUE(buf.empty());

    EXPECT_TRUE(buf.try_push(1));
    EXPECT_TRUE(buf.try_push(2));
    EXPECT_TRUE(buf.try_push(3));
    EXPECT_FALSE(buf.try_push(4));  // Full (capacity is N-1 = 3)
    EXPECT_EQ(buf.size(), 3u);

    int val;
    EXPECT_TRUE(buf.try_pop(val));
    EXPECT_EQ(val, 1);
    EXPECT_TRUE(buf.try_pop(val));
    EXPECT_EQ(val, 2);
    EXPECT_TRUE(buf.try_pop(val));
    EXPECT_EQ(val, 3);
    EXPECT_FALSE(buf.try_pop(val));
    EXPECT_TRUE(buf.empty());
}

TEST(RingBuffer, PushOverwrite) {
    RingBuffer<int, 4> buf;
    buf.push_overwrite(1);
    buf.push_overwrite(2);
    buf.push_overwrite(3);
    buf.push_overwrite(4);  // Overwrites oldest

    int val;
    EXPECT_TRUE(buf.try_pop(val));
    EXPECT_EQ(val, 2);  // 1 was overwritten
}

TEST(RingBuffer, TryPopOptional) {
    RingBuffer<int, 4> buf;
    EXPECT_FALSE(buf.try_pop().has_value());

    buf.push_overwrite(42);
    auto val = buf.try_pop();
    EXPECT_TRUE(val.has_value());
    EXPECT_EQ(*val, 42);
}
