// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_MESSAGING_H
#define SHORTFIN_LOCAL_MESSAGING_H

#include <string>

#include "shortfin/support/api.h"
#include "shortfin/support/iree_concurrency.h"

namespace shortfin::local {

// -------------------------------------------------------------------------- //
// Message
// -------------------------------------------------------------------------- //

class SHORTFIN_API Message;
namespace detail {

struct MessageRefOwner {
  MessageRefOwner() : Control(nullptr) {}
  enum class Request { RETAIN, RELEASE };
  MessageRefOwner(void (*Control)(Request req, const Message &msg))
      : Control(Control) {}
  void (*Control)(Request req, const Message &msg);
  operator bool() { return Control != nullptr; }
  static intptr_t &access_ref_data(const Message &msg);
  intptr_t set_owner(const Message &msg, intptr_t ref_data);
};

}  // namespace detail

// Message base class.
// Messages have ref-counted semantics and can have vague ownership (i.e.
// they can be aggregated into objects that manage their own reference count).
// Because they play such a central role in the system, they are a bit special.
//
// On the C++ side, they are primarily moved around via Message::Ref, which
// provides an RAII container for a reference. In other languages, they will
// typically be in place constructed in the host language object and interop
// with its GC mechanism.
//
// While this base class is somewhat bare-bones, it is expected that as
// remoting and other concerns come into play, more features will be added.
//
// Clients are expected to subclass Message to suit their needs.
class SHORTFIN_API Message {
 public:
  Message() = default;
  Message(const Message &) = delete;
  Message(Message &&) = delete;
  Message &operator=(const Message &) = delete;
  virtual ~Message();

 protected:
  // Guard a scope with the fine grained lock.
  iree::slim_mutex_lock_guard lock_guard() const {
    return iree::slim_mutex_lock_guard(lock_);
  }
  // Manual retain and release. Callers must assume that the Message is no
  // longer valid after any call to Release() where they do not hold a known
  // reference.
  void Retain() const;
  void Release() const;

 private:
  // Messages are intrusively reference counted, either using an internal
  // reference count in ref_data_ when allocator is null or externally
  // when not null. When externally managed, ref_data_ is just a pointer
  // sized field that the allocator can use at it sees fit. Both fields
  // are managed within a lock_ scope and are optimized for single threaded
  // access and cross-thread transfers with coarse references.
  mutable intptr_t ref_data_ = 1;
  mutable detail::MessageRefOwner owner_;
  mutable iree::slim_mutex lock_;
  friend struct detail::MessageRefOwner;
};

// -------------------------------------------------------------------------- //
// Queue
// -------------------------------------------------------------------------- //

// Queues are the primary form of communication in shortfin for exchanging
// messages. They are inherently thread safe and coupled with the async/worker
// system for enqueue/dequeue operations.
class SHORTFIN_API Queue {
 public:
  struct Options {
    // Queues are generally managed by the system with a global name.
    std::string name;
  };
  Queue(Options options) : options_(std::move(options)) {}
  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;
  Queue(Queue &&) = delete;
  ~Queue() = default;

  const Options &options() const { return options_; }
  std::string to_s() const;

 private:
  Options options_;
};

// -------------------------------------------------------------------------- //
// Message allocation detail
// -------------------------------------------------------------------------- //

inline intptr_t &detail::MessageRefOwner::access_ref_data(const Message &msg) {
  return msg.ref_data_;
}

inline intptr_t detail::MessageRefOwner::set_owner(const Message &msg,
                                                   intptr_t ref_data) {
  auto g = msg.lock_guard();
  assert(!msg.owner_ && "Message ref owner transfer more than once");
  msg.owner_ = *this;
  intptr_t orig_ref_data = msg.ref_data_;
  msg.ref_data_ = ref_data;
  return orig_ref_data;
}

inline void Message::Retain() const {
  auto g = lock_guard();
  if (owner_) {
    owner_.Control(detail::MessageRefOwner::Request::RETAIN, *this);
  } else {
    ref_data_ += 1;
  }
}

inline void Message::Release() const {
  auto g = lock_guard();
  if (owner_) {
    owner_.Control(detail::MessageRefOwner::Request::RELEASE, *this);
  } else {
    if (--ref_data_ == 0) {
      delete this;
    }
  }
}

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_MESSAGING_H
