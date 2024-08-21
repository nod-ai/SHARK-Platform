// Copyright 2024 Advanced Micro Devices, Inc
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
// and all such copies share the same state.
class SHORTFIN_API Future {
 public:
  Future(const Future &other) = delete;
  Future(Future &&other) = delete;
  Future &operator=(const Future &other) = delete;
  virtual ~Future();

  void set_failure(iree_status_t failure_status) {
    state_->failure_status_ = failure_status;
    state_->done_ = true;
    state_->done_event_.set();
  }

  // Returns whether this future is done.
  bool is_done() { return state_->done_; }
  bool is_failure() { return !iree_status_is_ok(state_->failure_status_); }
  void ThrowFailure() {
    if (!state_->done_) {
      throw std::logic_error("Cannot get result from Future that is not done");
    }
    SHORTFIN_THROW_IF_ERROR(state_->failure_status_);
  }

  // Access the raw done wait source.
  operator const iree_wait_source_t() { return state_->done_event_.await(); }

 protected:
  struct BaseState {
    BaseState() : done_event_(false) {}
    virtual ~BaseState() = default;
    int ref_count_ = 1;
    iree::event done_event_;
    iree_status_t failure_status_ = iree_ok_status();
    bool done_ = false;
  };
  Future(BaseState *state) : state_(state) {}
  void set_success() {
    state_->done_ = true;
    state_->done_event_.set();
  }
  BaseState *state_;
};

// Future that has no done result.
class SHORTFIN_API VoidFuture : public Future {
 public:
  VoidFuture() : Future(new BaseState()) {}
  ~VoidFuture() override = default;
  VoidFuture(const VoidFuture &other) : Future(other.state_) {
    state_->ref_count_ += 1;
  }
  VoidFuture &operator=(const VoidFuture &other) {
    other.state_->ref_count_ += 1;
    if (--state_->ref_count_ == 0) delete state_;
    state_ = other.state_;
    return *this;
  }

  using Future::set_success;
};

// Value containing Future.
template <typename ResultTy>
class SHORTFIN_API TypedFuture : public Future {
 public:
  TypedFuture() : Future(new TypedState()) {}
  ~TypedFuture() override = default;
  TypedFuture(const TypedFuture &other) : Future(other.state_) {
    state_->ref_count_ += 1;
  }
  TypedFuture &operator=(const TypedFuture &other) {
    other.state_->ref_count_ += 1;
    if (--state_->ref_count_ == 0) delete state_;
    state_ = other.state_;
    return *this;
  }

  void set_result(ResultTy result) {
    static_cast<TypedState *>(state_)->result_ = std::move(result);
    set_success();
  }

  ResultTy &result() {
    if (!is_done()) {
      throw std::logic_error("Cannot get result from Future that is not done");
    }
    ThrowFailure();
    return static_cast<TypedState *>(state_)->result_;
  }

 private:
  struct TypedState : public BaseState {
    ResultTy result_;
  };
};

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_ASYNC_H
