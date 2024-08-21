// Copyright 2024 Advanced Micro Devices, Inc
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

Message::~Message() = default;

// -------------------------------------------------------------------------- //
// Queue
// -------------------------------------------------------------------------- //

Queue::Queue(Options options)
    : options_(std::move(options)), signalled_(false) {}

std::string Queue::to_s() const {
  return fmt::format("Queue(name={})", options().name);
}

QueueWriter::QueueWriter(Queue &queue, Options options)
    : queue_(queue), options_(std::move(options)) {}

QueueWriter::~QueueWriter() = default;

CompletionEvent QueueWriter::Write(Message::Ref mr) {
  iree::slim_mutex_lock_guard g(queue_.lock_);
  queue_.contents_.push_front(std::move(mr));
  queue_.signalled_->set();
  // TODO: Real back-pressure and not just immediate satisfaction.
  return CompletionEvent();
}

QueueReader::QueueReader(Queue &queue, Options options)
    : queue_(queue), options_(std::move(options)) {}

QueueReader::~QueueReader() = default;

MessageFuture QueueReader::Read() {
  if (worker_) {
    throw std::logic_error(
        "Cannot read concurrently from a single QueueReader");
  }

  // Prime the wait.
  worker_ = Worker::GetCurrent();
  if (!worker_) {
    throw std::logic_error("Cannot wait on QueueReader outside of worker");
  }
  // TODO: Set the future.
  read_result_future_ = MessageFuture();
  SHORTFIN_THROW_IF_ERROR(BeginWaitPump());
  return *read_result_future_;
}

iree_status_t QueueReader::BeginWaitPump() noexcept {
  return worker_->WaitOneLowLevel(queue_.signalled_->await(),
                                  iree_infinite_timeout(), &HandleWaitResult,
                                  this);
}

iree_status_t QueueReader::HandleWaitResult(void *user_data, iree_loop_t loop,
                                            iree_status_t status) noexcept {
  IREE_RETURN_IF_ERROR(status);
  auto *self = static_cast<QueueReader *>(user_data);
  Queue &queue = self->queue_;
  Message::Ref mr;
  {
    iree::slim_mutex_lock_guard g(queue.lock_);
    if (queue.contents_.empty()) {
      // Spurious wake up. Try again.
      queue.signalled_->reset();
      self->BeginWaitPump();
      return iree_ok_status();
    }

    // Pop.
    self->read_result_future_->set_result(std::move(queue.contents_.back()));
    queue.contents_.pop_back();

    // Update signal status.
    if (queue.contents_.empty()) {
      queue.signalled_->reset();
    } else {
      queue.signalled_->set();
    }
  }

  // Signal completion.
  self->worker_ = nullptr;
  self->read_result_future_.reset();

  return iree_ok_status();
}

}  // namespace shortfin::local
