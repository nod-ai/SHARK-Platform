// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/support/blocking_executor.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "shortfin/support/logging.h"

namespace shortfin {

class BlockingExecutorTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override { iree::detail::LogLiveRefs(); }
};

TEST_F(BlockingExecutorTest, concurrent_tasks) {
  {
    std::atomic<int> tasks_run{0};

    BlockingExecutor executor;
    for (int i = 0; i < 10; ++i) {
      executor.Schedule([&tasks_run, i]() {
        iree_wait_until(
            iree_timeout_as_deadline_ns(iree_make_timeout_ms(1500)));
        logging::info("TASK {} COMPLETE", i);
        tasks_run.fetch_add(1);
      });
    }

    executor.Kill(/*wait=*/true);
    EXPECT_EQ(tasks_run.load(), 10);
    logging::info("KILL RETURNED");
  }
}

TEST_F(BlockingExecutorTest, inhibit_when_shutdown) {
  {
    std::atomic<int> tasks_run{0};

    BlockingExecutor executor;
    for (int i = 0; i < 10; ++i) {
      executor.Schedule([&tasks_run, i]() {
        iree_wait_until(iree_timeout_as_deadline_ns(iree_make_timeout_ms(150)));
        tasks_run.fetch_add(1);
      });
    }

    executor.Kill(/*wait=*/true);
    logging::info("Killed");

    // New work should be inhibited.
    try {
      executor.Schedule([&tasks_run]() { tasks_run.fetch_add(1); });
      FAIL();
    } catch (std::logic_error &) {
    }
    EXPECT_EQ(tasks_run.load(), 10);
  }
}

TEST_F(BlockingExecutorTest, warn_deadline) {
  {
    std::atomic<int> tasks_run{0};

    BlockingExecutor executor;
    for (int i = 0; i < 10; ++i) {
      executor.Schedule([&tasks_run, i]() {
        iree_wait_until(
            iree_timeout_as_deadline_ns(iree_make_timeout_ms(1000)));
        tasks_run.fetch_add(1);
      });
    }

    executor.Kill(/*wait=*/true, /*warn_timeout=*/iree_make_timeout_ms(200));
    EXPECT_EQ(tasks_run.load(), 10);
  }
}

TEST_F(BlockingExecutorTest, threads_recycle) {
  {
    std::atomic<int> tasks_run{0};

    BlockingExecutor executor;
    for (int i = 0; i < 10; ++i) {
      executor.Schedule([&tasks_run, i]() {
        iree_wait_until(iree_timeout_as_deadline_ns(iree_make_timeout_ms(10)));
        tasks_run.fetch_add(1);
        logging::info("Task {} done", i);
      });
      // Make sure the task is done before scheduling another.
      while (!executor.has_free_threads()) {
        iree_wait_until(iree_timeout_as_deadline_ns(iree_make_timeout_ms(50)));
      }
    }

    executor.Kill(/*wait=*/true);
    EXPECT_EQ(tasks_run.load(), 10);
    // If we only submitted work serially, then there must have only been
    // one thread created to service it.
    EXPECT_EQ(executor.created_thread_count(), 1);
  }
}

}  // namespace shortfin
