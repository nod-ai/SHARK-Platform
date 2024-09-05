// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/support/iree_helpers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "shortfin/support/iree_concurrency.h"

namespace shortfin {

TEST(iree_string_view, to_string_view) {
  iree_string_view_t isv = iree_make_cstring_view("Hello");
  std::string_view sv = to_string_view(isv);
  EXPECT_EQ(sv, "Hello");
}

namespace {

struct iree_dummy_t {
  ~iree_dummy_t() { dtor_count++; }
  int retain_count = 0;
  int release_count = 0;
  int dtor_count = 0;
};

struct dummy_ptr_helper {
  static void steal(iree_dummy_t *obj) {}
  static void retain(iree_dummy_t *obj) { obj->retain_count++; }
  static void release(iree_dummy_t *obj) { obj->release_count++; }
};

using dummy_ptr = iree::object_ptr<iree_dummy_t, dummy_ptr_helper>;

}  // namespace

TEST(iree_object_ptr, steal) {
  auto raw = std::make_unique<iree_dummy_t>();
  dummy_ptr p1 = dummy_ptr::steal_reference(raw.get());
  EXPECT_EQ(raw->retain_count, 0);
  EXPECT_EQ(raw->release_count, 0);
  p1.reset();
  EXPECT_EQ(raw->release_count, 1);
}

TEST(iree_object_ptr, borrow) {
  auto raw = std::make_unique<iree_dummy_t>();
  dummy_ptr p1 = dummy_ptr::borrow_reference(raw.get());
  EXPECT_EQ(raw->retain_count, 1);
  EXPECT_EQ(raw->release_count, 0);
  p1.reset();
  EXPECT_EQ(raw->release_count, 1);
}

TEST(iree_object_ptr, copy) {
  auto raw = std::make_unique<iree_dummy_t>();
  dummy_ptr p1 = dummy_ptr::steal_reference(raw.get());
  dummy_ptr p2 = p1;
  EXPECT_EQ(raw->retain_count, 1);
  EXPECT_EQ(raw->release_count, 0);
  p1.reset();
  EXPECT_EQ(raw->release_count, 1);
  p2.reset();
  EXPECT_EQ(raw->release_count, 2);
}

TEST(iree_object_ptr, move) {
  auto raw = std::make_unique<iree_dummy_t>();
  dummy_ptr p1 = dummy_ptr::steal_reference(raw.get());
  dummy_ptr p2 = std::move(p1);
  EXPECT_EQ(raw->retain_count, 0);
  EXPECT_EQ(raw->release_count, 0);
  p1.reset();
  EXPECT_EQ(raw->release_count, 0);
  p2.reset();
  EXPECT_EQ(raw->release_count, 1);
}

TEST(iree_object_ptr, for_output) {
  auto raw = std::make_unique<iree_dummy_t>();
  dummy_ptr p1 = dummy_ptr::steal_reference(raw.get());
  p1.for_output();
  EXPECT_EQ(raw->retain_count, 0);
  EXPECT_EQ(raw->release_count, 1);
  EXPECT_EQ(p1.get(), nullptr);
  p1.reset();
  EXPECT_EQ(raw->release_count, 1);
}

TEST(iree_error, user_message) {
  try {
    throw iree::error(
        "Something went wrong",
        iree_make_status(IREE_STATUS_CANCELLED, "because I said so"));
  } catch (iree::error &e) {
    EXPECT_THAT(
        std::string(e.what()),
        testing::MatchesRegex(
            "^Something went wrong: .*: CANCELLED; because I said so$"));
  }
}

TEST(iree_error, no_user_message) {
  try {
    throw iree::error(
        iree_make_status(IREE_STATUS_CANCELLED, "because I said so"));
  } catch (iree::error &e) {
    EXPECT_THAT(std::string(e.what()),
                testing::MatchesRegex("^.*: CANCELLED; because I said so$"));
  }
}

TEST(iree_error, throw_if_error_ok) {
  try {
    SHORTFIN_THROW_IF_ERROR(iree_ok_status());
  } catch (iree::error &e) {
    FAIL();
  }
}

TEST(iree_error, throw_if_error) {
  try {
    SHORTFIN_THROW_IF_ERROR(
        iree_make_status(IREE_STATUS_CANCELLED, "because I said so"));
    FAIL();
  } catch (iree::error &e) {
    EXPECT_THAT(std::string(e.what()),
                testing::MatchesRegex("^.*: CANCELLED; because I said so$"));
  }
}

TEST(iree_error, throw_if_error_addl_message) {
  try {
    SHORTFIN_THROW_IF_ERROR(
        iree_make_status(IREE_STATUS_CANCELLED, "because I said so"),
        "oops: %d", 1);
    FAIL();
  } catch (iree::error &e) {
    EXPECT_THAT(
        std::string(e.what()),
        testing::MatchesRegex("^.*: CANCELLED; because I said so; oops: 1$"));
  }
}

}  // namespace shortfin
