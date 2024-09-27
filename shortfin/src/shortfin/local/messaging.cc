// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/messaging.h"

#include "shortfin/support/logging.h"

namespace shortfin::local {

// -------------------------------------------------------------------------- //
// Message
// -------------------------------------------------------------------------- //

template class TypedFuture<Message::Ref>;

// -------------------------------------------------------------------------- //
// Queue
// -------------------------------------------------------------------------- //

namespace {

struct QueueCreator : public Queue {
  QueueCreator(Options options) : Queue(std::move(options)) {}
};

}  // namespace

Queue::Queue(Options options) : options_(std::move(options)) {}

std::shared_ptr<Queue> Queue::Create(Options options) {
  return std::make_shared<QueueCreator>(std::move(options));
}

std::string Queue::to_s() const {
  return fmt::format("Queue(name={})", options().name);
}

bool Queue::is_closed() {
  iree::slim_mutex_lock_guard g(lock_);
  return closed_;
}

void Queue::WriteNoDelay(Message::Ref mr) {
  std::optional<MessageFuture> future;
  {
    iree::slim_mutex_lock_guard g(lock_);
    if (pending_readers_.empty()) {
      // No readers. Just add to the backlog.
      backlog_.push_back(std::move(mr));
      return;
    } else {
      // Signal a reader. We must do this within the queue lock to avoid
      // a QueueReader lifetime hazard. But we defer actually setting the
      // future until out of the lock.
      QueueReader *reader = pending_readers_.front();
      pending_readers_.pop_front();
      future = *reader->read_result_future_;
      // Reset the worker for a new read.
      reader->worker_ = nullptr;
      reader->read_result_future_.reset();
    }
  }

  // Signal the future outside of our lock.
  future->set_result(std::move(mr));
}

void Queue::Close() {
  std::vector<QueueReader *> async_close_readers;
  {
    iree::slim_mutex_lock_guard g(lock_);
    if (closed_) return;
    closed_ = true;

    // If there is a backlog then the queue readers will handle any close action
    // on their own.
    if (!backlog_.empty()) {
      assert(pending_readers_.empty() &&
             "Attempt to close queue with backlog and pending readers");
      return;
    }

    async_close_readers.insert(async_close_readers.end(),
                               pending_readers_.begin(),
                               pending_readers_.end());
    pending_readers_.clear();
  }

  // Asynchronously resolve any pending readers with a null message.
  for (QueueReader *reader : async_close_readers) {
    reader->read_result_future_->set_result(Message::Ref());
    reader->worker_ = nullptr;
    reader->read_result_future_.reset();
  }
}

QueueWriter::QueueWriter(Queue &queue, Options options)
    : queue_(queue.shared_from_this()), options_(std::move(options)) {}

QueueWriter::~QueueWriter() = default;

CompletionEvent QueueWriter::Write(Message::Ref mr) {
  queue().WriteNoDelay(std::move(mr));
  return CompletionEvent();
}

QueueReader::QueueReader(Queue &queue, Options options)
    : queue_(queue.shared_from_this()), options_(std::move(options)) {}

QueueReader::~QueueReader() {
  iree::slim_mutex_lock_guard g(queue().lock_);
  if (read_result_future_) {
    logging::warn("QueueReader destroyed while pending");
    // Reader is in progress: Cancel it from the queue.
    auto it = std::find(queue().pending_readers_.begin(),
                        queue().pending_readers_.end(), this);
    if (it != queue().pending_readers_.end()) {
      queue().pending_readers_.erase(it);
    }
  }
}

MessageFuture QueueReader::Read() {
  // TODO: It should be possible to further constrain the scope of this lock,
  // but it is set here to be conservatively safe pending a full analysis.
  iree::slim_mutex_lock_guard g(queue().lock_);
  if (worker_) {
    throw std::logic_error(
        "Cannot read concurrently from a single QueueReader");
  }

  // Make reader current.
  worker_ = Worker::GetCurrent();
  if (!worker_) {
    throw std::logic_error("Cannot wait on QueueReader outside of worker");
  }

  // See if there is a backlog that we can immediately satisfy.
  if (!queue().backlog_.empty()) {
    // Service from the backlog.
    MessageFuture imm_future(worker_);
    imm_future.set_result(std::move(queue().backlog_.front()));
    queue().backlog_.pop_front();
    worker_ = nullptr;
    return imm_future;
  }

  // Handle close.
  if (queue().closed_) {
    MessageFuture imm_future(worker_);
    imm_future.set_result(Message::Ref());
    worker_ = nullptr;
    return imm_future;
  }

  // Settle in for a wait.
  queue().pending_readers_.push_back(this);
  read_result_future_ = MessageFuture(worker_);
  return *read_result_future_;
}

}  // namespace shortfin::local
