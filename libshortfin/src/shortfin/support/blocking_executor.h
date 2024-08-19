// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_SUPPORT_BLOCKING_EXECUTOR_H
#define SHORTFIN_SUPPORT_BLOCKING_EXECUTOR_H

#include <functional>

#include "shortfin/support/iree_concurrency.h"

namespace shortfin {

// An executor for offloading potentially blocking operations from event loops
// and contexts which must not block. This uses a simple pool of threads,
// creating a new one whenever starvation would occur. Since it is explicitly
// meant for offloading blocking operations, we can make no further assumptions
// about it being legal to run with a more limited number of threads.
class SHORTFIN_API BlockingExecutor {
 public:
  using Task = std::function<void()>;
  BlockingExecutor(iree_allocator_t allocator);
  BlockingExecutor() : BlockingExecutor(iree_allocator_system()) {}
  ~BlockingExecutor();

  // Send a kill signal, optionally waiting indefinitely for shutdown.
  void Kill(bool wait = true,
            iree_timeout_t warn_timeout = iree_make_timeout_ms(5000));

  // Schedule task to run at some point in the future (which may be before
  // this function returns). The task must not throw any exceptions (or the
  // program will terminate).
  void Schedule(Task task);

  // Total number of threads created over the lifetime of this instance.
  // This may not be the same as the count of those currently alive.
  int created_thread_count() const { return created_thread_count_; }

  // Returns whether there are any free threads. This is generally only
  // used for testing.
  bool has_free_threads();

 private:
  // Free thread instances are managed in a simple linked list.
  struct ThreadInstance {
    ThreadInstance(BlockingExecutor *executor);
    BlockingExecutor *executor;
    iree::thread_ptr thread;
    iree::event signal_transact;
    Task current_task;
    ThreadInstance *next = nullptr;

    int RunOnThread() noexcept;
  };
  // Returns whether the thread should keep running. If false, the caller
  // must immediately deallocate and exit. It must not assume that
  // inst->executor is valid.
  bool LockAndMoveToFreeList(ThreadInstance *inst);
  iree_allocator_t allocator_;
  iree::slim_mutex control_mu_;
  ThreadInstance *free_threads_ = nullptr;
  int created_thread_count_ = 0;
  int live_thread_count_ = 0;
  bool kill_ = false;
  bool inhibit_ = false;
};

}  // namespace shortfin

#endif  // SHORTFIN_SUPPORT_BLOCKING_EXECUTOR_H
