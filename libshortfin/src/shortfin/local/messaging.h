// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_MESSAGING_H
#define SHORTFIN_LOCAL_MESSAGING_H

#include <deque>
#include <optional>
#include <string>

#include "shortfin/local/async.h"
#include "shortfin/local/worker.h"
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

  // RAII class for holding a reference to a Message.
  class Ref {
   public:
    explicit Ref(Message &msg) : msg_(&msg) { msg.Retain(); }
    Ref() : msg_(nullptr) {}
    ~Ref() {
      if (msg_) {
        msg_->Release();
      }
    }
    Ref(const Ref &other) : msg_(other.msg_) {
      if (msg_) msg_->Retain();
    }
    Ref(Ref &&other) : msg_(other.msg_) { other.msg_ = nullptr; }
    Ref &operator=(const Ref &other) {
      if (other.msg_) other.msg_->Retain();
      reset();
      msg_ = other.msg_;
      return *this;
    }

    operator bool() { return msg_ != nullptr; }
    operator Message &() { return *msg_; }
    Message *operator->() { return msg_; }
    Message *get() { return msg_; }

    void reset() {
      if (msg_) {
        msg_->Release();
        msg_ = nullptr;
      }
    }

    Message *release() {
      Message *ret = msg_;
      msg_ = nullptr;
      return ret;
    }

   private:
    Message *msg_ = nullptr;
  };

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

// Future specialization for Message::Ref.
extern template class TypedFuture<Message::Ref>;
using MessageFuture = TypedFuture<Message::Ref>;

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
  Queue(Options options);
  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;
  Queue(Queue &&) = delete;
  ~Queue() = default;

  const Options &options() const { return options_; }
  std::string to_s() const;

 private:
  mutable iree::slim_mutex lock_;
  Options options_;
  std::deque<Message::Ref> contents_;
  // For now we just have simple signalling: If the queue has elements to read,
  // then signalled.
  iree::shared_event::ref signalled_;

  friend class QueueReader;
  friend class QueueWriter;
};

// Writes messages to a queue.
// Writers must be unique to logical thread of execution and do not support
// concurrency (if you need to write from multiple places, create multiple
// writer).
// Typically, various flow control and priority options are set per writer.
class SHORTFIN_API QueueWriter {
 public:
  struct Options {};
  QueueWriter(Queue &queue, Options options = {});
  ~QueueWriter();

  // Writes a message to the queue.
  // The write must be awaited as it can produce backpressure and failures.
  // TODO: This should be a Future<void> so that exceptions can propagate.
  CompletionEvent Write(Message::Ref mr);

 private:
  Queue &queue_;
  Options options_;
};

class SHORTFIN_API QueueReader {
 public:
  struct Options {};
  QueueReader(Queue &queue, Options options = {});
  ~QueueReader();

  // Reads a message from the queue.
  MessageFuture Read();

 private:
  iree_status_t BeginWaitPump() noexcept;
  static iree_status_t HandleWaitResult(void *user_data, iree_loop_t loop,
                                        iree_status_t status) noexcept;
  Queue &queue_;
  Options options_;

  // Reader state machine.
  Worker *worker_ = nullptr;
  std::optional<MessageFuture> read_result_future_;
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
