// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_ASYNC_H
#define SHORTFIN_LOCAL_ASYNC_H

#include <any>
#include <functional>

#include "shortfin/support/api.h"
#include "shortfin/support/iree_concurrency.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin::local {

class Worker;

// CompletionEvents are the most basic form of awaitable object. They
// encapsulate a native iree_wait_source_t (which multiplexes any supported
// system level wait primitive) with a resource baton which keeps any needed
// references alive for the duration of all copies.
//
// Depending on the system wait source used, there may be a limited exception
// side-band (i.e. a way to signal that the wait handle has failed and have
// that propagate to consumers). However, in general, this is a very coarse
// mechanism. For rich result and error propagation, see the higher level
// Promise/Future types, which can be signalled with either a result or
// exception.
class SHORTFIN_API CompletionEvent {
 public:
  CompletionEvent();
  CompletionEvent(iree::shared_event::ref event);
  CompletionEvent(iree::hal_semaphore_ptr sem, uint64_t payload);
  CompletionEvent(CompletionEvent &&other)
      : wait_source_(other.wait_source_),
        resource_baton_(std::move(other.resource_baton_)) {
    other.wait_source_ = iree_wait_source_immediate();
  }
  CompletionEvent(const CompletionEvent &other)
      : wait_source_(other.wait_source_),
        resource_baton_(other.resource_baton_) {}
  CompletionEvent &operator=(const CompletionEvent &other) {
    wait_source_ = other.wait_source_;
    resource_baton_ = other.resource_baton_;
    return *this;
  }
  ~CompletionEvent();

  // Returns true if this CompletionEvent is ready.
  bool is_ready();
  // Block the current thread for up to |timeout|. If a non-infinite timeout
  // was given and the timeout expires while waiting, returns false. In all
  // other cases, returns true.
  // This should not be used in worker loops.
  bool BlockingWait(iree_timeout_t timeout = iree_infinite_timeout());

  // Access the raw wait source.
  operator const iree_wait_source_t &() { return wait_source_; }

 private:
  iree_wait_source_t wait_source_;
  // A baton used to keep any needed backing resource alive.
  std::any resource_baton_;
};

// Object that will eventually be set to some completion state, either a result
// value or an exception status. Like CompletionEvents, Futures are copyable,
// and all such copies share the same state. Future objects are bound to the
// worker on which they are created. When signaled from the same worker,
// they use a fast path, but when signaled from elsewhere, cross-worker
// signaling is used (which has more overhead).
class SHORTFIN_API Future {
 public:
  using FutureCallback = std::function<void(Future &)>;

  Future(const Future &other) = delete;
  Future(Future &&other) = delete;
  Future &operator=(const Future &other) = delete;
  virtual ~Future();

  void set_failure(iree_status_t failure_status);

  // Returns whether this future is done.
  bool is_done() {
    iree::slim_mutex_lock_guard g(state_->lock_);
    return state_->done_;
  }
  bool is_failure() {
    iree::slim_mutex_lock_guard g(state_->lock_);
    return !iree_status_is_ok(state_->failure_status_.status());
  }
  void ThrowFailure() {
    iree::slim_mutex_lock_guard g(state_->lock_);
    ThrowFailureWithLockHeld();
  }

  // Adds a callback that will be made when the future is satisfied (either
  // with a value or a failure). If the future is already satisfied, they
  // will be queued for delivery on a future cycle of the event loop. If
  // running on the same worker as owns this Future, then the callback will
  // never be executed within the fiber of this call. If adding a callback
  // from another thread, then it is possible that the callback runs concurrent
  // with returning from this function.
  void AddCallback(FutureCallback callback);

 protected:
  struct SHORTFIN_API BaseState {
    BaseState(Worker *worker) : worker_(worker) {}
    virtual ~BaseState();
    iree::slim_mutex lock_;
    Worker *worker_;
    int ref_count_ SHORTFIN_GUARDED_BY(lock_) = 1;
    iree::ignorable_status failure_status_ SHORTFIN_GUARDED_BY(lock_);
    bool done_ SHORTFIN_GUARDED_BY(lock_) = false;
    std::vector<FutureCallback> callbacks_ SHORTFIN_GUARDED_BY(lock_);
  };

  Future(BaseState *state) : state_(state) {}
  void Retain() const;
  void Release() const;
  static Worker *GetRequiredWorker();
  void SetSuccessWithLockHeld() SHORTFIN_REQUIRES_LOCK(state_->lock_) {
    state_->done_ = true;
  }
  // Posts a message to the worker to issue callbacks. Lock must be held.
  void IssueCallbacksWithLockHeld() SHORTFIN_REQUIRES_LOCK(state_->lock_);
  static iree_status_t RawHandleWorkerCallback(void *state_vp, iree_loop_t loop,
                                               iree_status_t status) noexcept;
  void HandleWorkerCallback();
  void ThrowFailureWithLockHeld() SHORTFIN_REQUIRES_LOCK(state_->lock_);

  mutable BaseState *state_;
};

// Future that has no result type. It can be done without result or have
// a failure set.
class SHORTFIN_API VoidFuture : public Future {
 public:
  VoidFuture() : Future(new BaseState(GetRequiredWorker())) {}
  VoidFuture(Worker *worker) : Future(new BaseState(worker)) {}
  ~VoidFuture() override = default;
  VoidFuture(const VoidFuture &other) : Future(other.state_) { Retain(); }
  VoidFuture &operator=(const VoidFuture &other) {
    other.Retain();
    Release();
    state_ = other.state_;
    return *this;
  }

  void set_success() {
    iree::slim_mutex_lock_guard g(state_->lock_);
    SetSuccessWithLockHeld();
    IssueCallbacksWithLockHeld();
  }
};

// Value containing Future.
template <typename ResultTy>
class SHORTFIN_API TypedFuture : public Future {
 public:
  TypedFuture() : Future(new TypedState(GetRequiredWorker())) {}
  TypedFuture(Worker *worker) : Future(new TypedState(worker)) {}
  ~TypedFuture() override = default;
  TypedFuture(const TypedFuture &other) : Future(other.state_) { Retain(); }
  TypedFuture &operator=(const TypedFuture &other) {
    other.Retain();
    Release();
    state_ = other.state_;
    return *this;
  }

  // Futures are non-nullable, so construct/assign from an rvalue reference
  // is just a copy and does not clear the original.
  TypedFuture(TypedFuture &&other) : Future(other.state_) { Retain(); }
  TypedFuture &operator=(TypedFuture &&other) {
    other.Retain();
    Release();
    state_ = other.state_;
    return *this;
  }

  void set_result(ResultTy result) {
    iree::slim_mutex_lock_guard g(state_->lock_);
    if (state_->done_) {
      throw std::logic_error(
          "Cannot 'set_failure' on a Future that is already done");
    }
    static_cast<TypedState *>(state_)->result_ = std::move(result);
    SetSuccessWithLockHeld();
    IssueCallbacksWithLockHeld();
  }

  ResultTy &result() {
    iree::slim_mutex_lock_guard g(state_->lock_);
    ThrowFailureWithLockHeld();
    return static_cast<TypedState *>(state_)->result_;
  }

 private:
  struct SHORTFIN_API TypedState : public BaseState {
    using BaseState::BaseState;
    ResultTy result_;
  };
};

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_ASYNC_H
