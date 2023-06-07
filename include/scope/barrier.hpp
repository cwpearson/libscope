#pragma once

#include <condition_variable>
#include <memory>

// https://stackoverflow.com/questions/24465533/implementing-boostbarrier-in-c11
class Barrier {
public:
  explicit Barrier(std::size_t iCount)
      : threshold_(iCount), count_(iCount), generation_(0) {}

  void wait() {
    std::unique_lock<std::mutex> lLock{mutex_};
    auto lGen = generation_;
    if (!--count_) {
      generation_++;
      count_ = threshold_;
      cv_.notify_all();
    } else {
      cv_.wait(lLock, [this, lGen] { return lGen != generation_; });
    }
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::size_t threshold_;
  std::size_t count_;
  std::size_t generation_;
};
