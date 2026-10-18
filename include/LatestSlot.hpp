#pragma once
#include <mutex>
#include <condition_variable>

// Хранит только самый свежий элемент: если новый кладут раньше,
// чем забрали предыдущий, предыдущий теряется. Это и есть
// "leaky queue depth=1" — не даём задержке накапливаться в очереди.
template <typename T>
class LatestSlot {
public:
    void put(T v) {
        {
            std::lock_guard<std::mutex> lk(m_);
            slot_ = std::move(v);
            has_ = true;
        }
        cv_.notify_one();
    }

    // Блокируется, пока не появится новый элемент. Возвращает false
    // только после stop().
    bool take(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return has_ || stop_; });
        if (stop_) return false;
        out = std::move(slot_);
        has_ = false;
        return true;
    }

    void stop() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
        cv_.notify_all();
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    T slot_;
    bool has_ = false;
    bool stop_ = false;
};
