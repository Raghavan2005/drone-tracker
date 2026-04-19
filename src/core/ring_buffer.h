#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <type_traits>

namespace drone_tracker {

// Lock-free single-producer single-consumer ring buffer.
// N must be a power of 2.
template <typename T, size_t N>
class RingBuffer {
    static_assert((N & (N - 1)) == 0, "N must be a power of 2");

public:
    bool try_push(T&& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next = (head + 1) & (N - 1);
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        buffer_[head] = std::move(item);
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Push that overwrites oldest if full (never blocks producer)
    void push_overwrite(T&& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next = (head + 1) & (N - 1);
        if (next == tail_.load(std::memory_order_acquire)) {
            tail_.store((next + 1) & (N - 1), std::memory_order_release);
        }
        buffer_[head] = std::move(item);
        head_.store(next, std::memory_order_release);
    }

    bool try_pop(T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return false;
        }
        item = std::move(buffer_[tail]);
        tail_.store((tail + 1) & (N - 1), std::memory_order_release);
        return true;
    }

    std::optional<T> try_pop() {
        T item;
        if (try_pop(item)) {
            return std::move(item);
        }
        return std::nullopt;
    }

    size_t size() const {
        size_t head = head_.load(std::memory_order_acquire);
        size_t tail = tail_.load(std::memory_order_acquire);
        return (head - tail) & (N - 1);
    }

    bool empty() const { return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire); }

private:
    std::array<T, N> buffer_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};

}  // namespace drone_tracker
