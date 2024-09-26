// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_MESSAGING_H
#define SHORTFIN_LOCAL_MESSAGING_H

#include <deque>
#include <memory>
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

class Message;
namespace detail {

// Message lifetime by default is managed by an internal reference count
// system. However, since Messages often need to be owned by some third
// party system with its own notion of lifetime, it is possible to provide
// a custom lifetime controller. This can only be done once, typically on
// construction by a proxy system.
struct MessageLifetimeController {
  MessageLifetimeController() : Control(nullptr) {}
  enum class Request { RETAIN, RELEASE };
  MessageLifetimeController(void (*Control)(Request req, const Message &msg))
      : Control(Control) {}
  void (*Control)(Request req, const Message &msg);
  operator bool() { return Control != nullptr; }
  // Takes ownership of the Message using this ownership controller, providing
  // new ref_data that will be stored in the message and accessed from then
  // on without internal locking. Returns the existing reference count at the
  // time of transfer.
  intptr_t TakeOwnership(const Message &msg, intptr_t ref_data);
  // Accessed the ref_data memory within the Message. This is only valid
  // if ownership has been transferred to a lifetime controller, and it is
  // accessed without locking. This method purely exists to add some static
  // thread/access safety.
  static intptr_t &AccessOwnedRefData(const Message &msg)
      SHORTFIN_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis);
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
  virtual ~Message() = default;

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
  mutable iree::slim_mutex lock_;
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
  mutable intptr_t ref_data_ SHORTFIN_GUARDED_BY(lock_) = 1;
  mutable detail::MessageLifetimeController lifetime_controller_
      SHORTFIN_GUARDED_BY(lock_);
  friend struct detail::MessageLifetimeController;
};

// Future specialization for Message::Ref.
extern template class TypedFuture<Message::Ref>;
using MessageFuture = TypedFuture<Message::Ref>;

// -------------------------------------------------------------------------- //
// Queue
// -------------------------------------------------------------------------- //

class Queue;
class QueueReader;
class QueueWriter;

namespace {
struct QueueCreator;
}

using QueuePtr = std::shared_ptr<Queue>;

// Queues are the primary form of communication in shortfin for exchanging
// messages. They are inherently thread safe and coupled with the async/worker
// system for enqueue/dequeue operations.
class SHORTFIN_API Queue : public std::enable_shared_from_this<Queue> {
 public:
  struct Options {
    // Queues are generally managed by the system with a global name. The
    // the name is empty, then this is an anonymous queue.
    std::string name;
  };
  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;
  Queue(Queue &&) = delete;
  ~Queue() = default;

  operator QueuePtr() { return shared_from_this(); }

  const Options &options() const { return options_; }
  std::string to_s() const;

  // Returns whether the queue is still open.
  bool is_closed();

  // Writes a message to the queue without any possible delay, possibly
  // overriding capacity and throttling policy.
  void WriteNoDelay(Message::Ref mr);

  // Closes the queue. All readers will return with a null message from here
  // on. Writers that attempt to write to the queue will throw an exception.
  void Close();

 protected:
  Queue(Options options);

 private:
  // Queues can only be created as shared by the System.
  static QueuePtr Create(Options options);
  mutable iree::slim_mutex lock_;
  Options options_;
  // Backlog of messages not yet sent to a reader. Messages are pushed on the
  // back and popped from the front.
  std::deque<Message::Ref> backlog_;
  // Deque of all readers that are waiting for messages. An attempt is made
  // to dispatch to readers in FIFO order of having entered a wait state.
  // Readers are pushed on the back and popped from the front.
  std::deque<QueueReader *> pending_readers_;
  // Whether the queue has been closed.
  bool closed_ = false;

  friend class QueueReader;
  friend class QueueWriter;
  friend struct QueueCreator;
  friend class System;
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

  Queue &queue() { return *queue_; }

  // Writes a message to the queue.
  // The write must be awaited as it can produce backpressure and failures.
  // TODO: This should be a Future<void> so that exceptions can propagate.
  CompletionEvent Write(Message::Ref mr);

  // Calls Close() on the backing queue.
  void Close() { queue_->Close(); }

 private:
  std::shared_ptr<Queue> queue_;
  Options options_;
};

class SHORTFIN_API QueueReader {
 public:
  struct Options {};
  QueueReader(Queue &queue, Options options = {});
  ~QueueReader();

  Queue &queue() { return *queue_; }

  // Reads a message from the queue.
  MessageFuture Read();

 private:
  std::shared_ptr<Queue> queue_;
  Options options_;

  // Reader state machine. If worker_ is non null, then there must be a
  // read_result_future_ of the current outstanding read.
  Worker *worker_ = nullptr;
  std::optional<MessageFuture> read_result_future_;

  friend class Queue;
  friend class QueueWriter;
};

// -------------------------------------------------------------------------- //
// Message allocation detail
// -------------------------------------------------------------------------- //

inline intptr_t &detail::MessageLifetimeController::AccessOwnedRefData(
    const Message &msg) {
  return msg.ref_data_;
}

inline intptr_t detail::MessageLifetimeController::TakeOwnership(
    const Message &msg, intptr_t ref_data) {
  iree::slim_mutex_lock_guard g(msg.lock_);
  assert(!msg.lifetime_controller_ &&
         "Message ref owner transfer more than once");
  msg.lifetime_controller_ = *this;
  intptr_t orig_ref_data = msg.ref_data_;
  msg.ref_data_ = ref_data;
  return orig_ref_data;
}

inline void Message::Retain() const {
  iree::slim_mutex_lock_guard g(lock_);
  if (lifetime_controller_) {
    lifetime_controller_.Control(
        detail::MessageLifetimeController::Request::RETAIN, *this);
  } else {
    ref_data_ += 1;
  }
}

inline void Message::Release() const {
  // Since the destructor of the lock asserts that it is not held, we must
  // manually release the lock prior to an action that may result in
  // destruction. As such, just manage lock manually/carefully vs using RAII.
  lock_.Lock();
  auto *local_controller = &lifetime_controller_;
  if (*local_controller) {
    lock_.Unlock();
    local_controller->Control(
        detail::MessageLifetimeController::Request::RELEASE, *this);
    return;
  } else if (--ref_data_ == 0) {
    lock_.Unlock();
    delete this;
    return;
  } else {
    lock_.Unlock();
    return;
  }
}

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_MESSAGING_H
