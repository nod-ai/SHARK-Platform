// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_SUPPORT_IREE_HELPERS_H
#define SHORTFIN_SUPPORT_IREE_HELPERS_H

#include <memory>
#include <stdexcept>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "shortfin/support/api.h"

namespace shortfin {

// -------------------------------------------------------------------------- //
// Type conversion
// -------------------------------------------------------------------------- //

inline std::string_view to_string_view(iree_string_view_t isv) {
  return std::string_view(isv.data, isv.size);
}

// -------------------------------------------------------------------------- //
// RAII wrappers
// -------------------------------------------------------------------------- //

namespace iree_type_detail {
inline void retain(iree_hal_device_t *obj) { iree_hal_device_retain(obj); }
inline void release(iree_hal_device_t *obj) { iree_hal_device_release(obj); }

inline void retain(iree_hal_driver_t *obj) { iree_hal_driver_retain(obj); }
inline void release(iree_hal_driver_t *obj) { iree_hal_driver_release(obj); }
};  // namespace iree_type_detail

// Wraps an IREE retain/release style object pointer in a smart-pointer
// like wrapper.
template <typename T>
class iree_object_ptr {
 public:
  iree_object_ptr() : ptr(nullptr) {}
  iree_object_ptr(const iree_object_ptr &other) : ptr(other.ptr) {
    if (ptr) {
      iree_type_detail::retain(ptr);
    }
  }
  iree_object_ptr(iree_object_ptr &&other) : ptr(other.ptr) {
    other.ptr = nullptr;
  }
  ~iree_object_ptr() {
    if (ptr) {
      iree_type_detail::release(ptr);
    }
  }

  // Constructs a new iree_object_ptr by transferring ownership of a raw
  // pointer.
  static iree_object_ptr steal_reference(T *owned) {
    return iree_object_ptr(owned);
  }

  operator T *() noexcept { return ptr; }

  // Releases any current reference held by this instance and returns a
  // pointer to the raw backing pointer. This is typically used for passing
  // to out parameters which are expected to store a new owned pointer directly.
  T **for_output() {
    reset();
    return &ptr;
  }

  operator bool() const { return ptr != nullptr; }
  T *get() const { return ptr; }
  void reset(T *other = nullptr) {
    if (ptr) {
      iree_type_detail::release(ptr);
    }
    ptr = other;
  }
  T *release() {
    T *ret = ptr;
    ptr = nullptr;
    return ret;
  }

 private:
  // Assumes the reference count for owned_ptr.
  iree_object_ptr(T *owned_ptr) : ptr(owned_ptr) {}
  T *ptr = nullptr;
};

using iree_hal_driver_ptr = iree_object_ptr<iree_hal_driver_t>;
using iree_hal_device_ptr = iree_object_ptr<iree_hal_device_t>;

// Holds a pointer allocated by some allocator, deleting it if still owned
// at destruction time.
template <typename T>
struct allocated_ptr {
  iree_allocator_t allocator;

  allocated_ptr(iree_allocator_t allocator) : allocator(allocator) {}
  ~allocated_ptr() { reset(); }

  void reset(T *other = nullptr) {
    iree_allocator_free(allocator, ptr);
    ptr = other;
  }

  T *release() {
    T *ret = ptr;
    ptr = nullptr;
    return ret;
  }

  T *operator->() noexcept { return ptr; }
  T **operator&() noexcept { return &ptr; }
  operator T *() noexcept { return ptr; }
  T *get() noexcept { return ptr; }

  // Releases any current reference held by this instance and returns a
  // pointer to the raw backing pointer. This is typically used for passing
  // to out parameters which are expected to store a new owned pointer directly.
  T **for_output() {
    reset();
    return &ptr;
  }

 private:
  T *ptr = nullptr;
};

// -------------------------------------------------------------------------- //
// iree_error and status handling
// -------------------------------------------------------------------------- //

// Captures an iree_status_t as an exception. The intent is that this is
// only used for failing statuses and the iree_error instance will be
// immediately thrown.
class SHORTFIN_API iree_error : public std::exception {
 public:
  iree_error(iree_status_t failing_status);
  iree_error(const iree_error &) = delete;
  iree_error &operator=(const iree_error &) = delete;
  ~iree_error() { iree_status_ignore(failing_status_); }
  const char *what() const noexcept override {
    if (!status_appended_) {
      AppendStatus();
    }
    return message_.c_str();
  };

 private:
  void AppendStatus() const noexcept;
  mutable std::string message_;
  mutable iree_status_t failing_status_;
  mutable bool status_appended_ = false;
};

#define SHORTFIN_IMPL_HANDLE_IF_API_ERROR(var, ...)                          \
  iree_status_t var = (IREE_STATUS_IMPL_IDENTITY_(                           \
      IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_GET_EXPR_)(__VA_ARGS__))); \
  if (IREE_UNLIKELY(var)) {                                                  \
    throw iree_error(IREE_STATUS_IMPL_ANNOTATE_SWITCH_(var, __VA_ARGS__));   \
  }

// Similar to IREE_RETURN_IF_ERROR but throws an iree_error exception instead.
#define SHORTFIN_THROW_IF_ERROR(...)                    \
  SHORTFIN_IMPL_HANDLE_IF_API_ERROR(                    \
      IREE_STATUS_IMPL_CONCAT_(__status_, __COUNTER__), \
      IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_IDENTITY_(__VA_ARGS__)))

}  // namespace shortfin

#endif  // SHORTFIN_SUPPORT_IREE_HELPERS_H
