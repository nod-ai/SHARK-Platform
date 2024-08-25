// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_ARRAY_STORAGE_H
#define SHORTFIN_ARRAY_STORAGE_H

#include <string_view>

#include "shortfin/local/scope.h"
#include "shortfin/support/api.h"

namespace shortfin::array {

// Access to mapped memory.
// Mappings are moveable but not copyable. When default constructed or moved
// from, they will not be valid and have nullptr semantics.
class SHORTFIN_API mapping {
 public:
  mapping();
  mapping(const mapping &) = delete;
  mapping &operator=(const mapping &) = delete;
  mapping &operator=(mapping &&other) {
    access_ = other.access_;
    mapping_ = other.mapping_;
    hal_device_ownership_baton_ = std::move(other.hal_device_ownership_baton_);
    other.access_ = IREE_HAL_MEMORY_ACCESS_NONE;
    std::memset(&other.mapping_, 0, sizeof(other.mapping_));
    return *this;
  }
  mapping(mapping &&other)
      : access_(other.access_),
        mapping_(other.mapping_),
        hal_device_ownership_baton_(
            std::move(other.hal_device_ownership_baton_)) {
    other.access_ = IREE_HAL_MEMORY_ACCESS_NONE;
    std::memset(&other.mapping_, 0, sizeof(other.mapping_));
  }
  ~mapping() noexcept;

  // Whether the mapping is valid.
  operator bool() const { return access_ != IREE_HAL_MEMORY_ACCESS_NONE; }

  // Resets the mapping, making it invalid (if not already so);
  void reset() noexcept;

  // Access the mapped data. The mapping must be valid or else it is UB.
  const uint8_t *data() const {
    assert(*this && "mapping is not valid");
    return mapping_.contents.data;
  }
  uint8_t *data() {
    assert(*this && "mapping is not valid");
    return mapping_.contents.data;
  }

  // The size of the mapped data. Will return 0 if the mapping is not valid.
  iree_device_size_t size() const { return mapping_.contents.data_length; }

  bool readable() const { return access_ & IREE_HAL_MEMORY_ACCESS_READ; }
  bool writable() const { return access_ & IREE_HAL_MEMORY_ACCESS_WRITE; }

 private:
  iree_hal_memory_access_t access_ = IREE_HAL_MEMORY_ACCESS_NONE;
  iree_hal_buffer_mapping_t mapping_;
  iree::hal_device_ptr hal_device_ownership_baton_;
  friend class storage;
};

// Array storage backed by an IREE buffer of some form.
class SHORTFIN_API storage {
 public:
  local::ScopedDevice &device() { return device_; }
  local::Scope &scope() { return device_.scope(); }
  const local::ScopedDevice &device() const { return device_; }
  local::Scope &scope() const { return device_.scope(); }

  // Allocates device storage, compatible with the given device affinity.
  // By default, this will be IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE.
  static storage AllocateDevice(local::ScopedDevice &device,
                                iree_device_size_t allocation_size);

  // Allocates host storage, compatible with the given device affinity.
  // By default, if there are any affinity bits set in the device, then
  // the storage will be device visible and have permitted usage for
  // transfers. This default policy can be overriden based on device defaults
  // or explicit options.
  static storage AllocateHost(local::ScopedDevice &device,
                              iree_device_size_t allocation_size);

  // Creates a subspan view of the current storage given a byte offset and
  // length. The returned storage shares the underlying allocation and
  // scheduling control block.
  storage Subspan(iree_device_size_t byte_offset,
                  iree_device_size_t byte_length);

  // Enqueues a fill of the storage with an arbitrary pattern of the given
  // size. The pattern size must be 1, 2, or 4.
  void Fill(const void *pattern, iree_host_size_t pattern_length);

  // Performs either a d2h, h2d or d2d transfer from a source storage to this
  // storage.
  void CopyFrom(storage &source_storage);

  iree_device_size_t byte_length() const {
    return iree_hal_buffer_byte_length(buffer_.get());
  }

  // Memory type and access.
  iree_hal_memory_type_t memory_type() const;
  iree_hal_memory_access_t memory_access() const;
  iree_hal_buffer_usage_t buffer_usage() const;

  // Formatted type and access.
  std::string formatted_memory_type() const;
  std::string formatted_memory_access() const;
  std::string formatted_buffer_usage() const;

  // Whether the buffer supports host mappable memory.
  bool is_mappable_for_read() const;
  bool is_mappable_for_read_write() const;

  // Maps the memory for access from a host pointer using a scoped mapping.
  void MapExplicit(mapping &mapping, iree_hal_memory_access_t access);

  // Maps the memory for read/write access, preserving any contents.
  mapping MapReadWrite() {
    mapping m;
    MapExplicit(m, IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE);
    return m;
  }

  // Maps the memory for discard write. This is used if populating an initial
  // buffer.
  mapping MapWriteDiscard() {
    mapping m;
    MapExplicit(m, IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE);
    return m;
  }

  // Maps the memory for read-only access.
  mapping MapRead() {
    mapping m;
    MapExplicit(m, IREE_HAL_MEMORY_ACCESS_READ);
    return m;
  }

  const mapping MapRead() const {
    mapping m;
    const_cast<storage *>(this)->MapExplicit(m, IREE_HAL_MEMORY_ACCESS_READ);
    return m;
  }

  std::string to_s() const;

  // Access raw buffer. This must not be retained apart from the storage for
  // any length of time that may extend its lifetime (as the storage keeps
  // underlying device references alive as needed).
  operator iree_hal_buffer_t *() { return buffer_; }

 private:
  storage(local::ScopedDevice device, iree::hal_buffer_ptr buffer,
          local::detail::TimelineResource::Ref timeline_resource)
      : hal_device_ownership_baton_(iree::hal_device_ptr::borrow_reference(
            device.raw_device()->hal_device())),
        buffer_(std::move(buffer)),
        device_(device),
        timeline_resource_(std::move(timeline_resource)) {}
  // TODO(ownership): Since storage is a free-standing object in the system,
  // it needs an ownership baton that keeps the device/driver alive.
  // Otherwise, it can outlive the backing device and then then crashes on
  // buffer deallocation. For now, we stash an RAII hal_device_ptr, which
  // keeps everything alive. This isn't quite what we want but keeps us going
  // for now. When fixing, add a test that creates an array, destroys the
  // System, and then frees the array.
  iree::hal_device_ptr hal_device_ownership_baton_;
  iree::hal_buffer_ptr buffer_;
  local::ScopedDevice device_;
  local::detail::TimelineResource::Ref timeline_resource_;
};

// Wraps an untyped mapping, providing typed access.
template <typename EltTy>
class typed_mapping {
 public:
  using span_type = std::span<EltTy>;
  using const_span_type = std::span<const EltTy>;

  typed_mapping(mapping untyped_mapping)
      : untyped_mapping_(std::move(untyped_mapping)) {}
  typed_mapping(const typed_mapping &) = delete;
  typed_mapping &operator=(const typed_mapping &) = delete;

  iree_device_size_t size() const noexcept {
    return untyped_mapping_.size() / sizeof(EltTy);
  }
  bool empty() const noexcept { return size() == 0; }
  EltTy *data() noexcept {
    return reinterpret_cast<EltTy *>(untyped_mapping_.data());
  }
  EltTy *data() const noexcept {
    return reinterpret_cast<const EltTy *>(untyped_mapping_.data());
  }

  span_type span() { return span_type(data(), size()); }
  const_span_type span() const { return const_span_type(data(), size()); }

  span_type::iterator begin() { return span().begin(); }
  span_type::iterator end() { return span().end(); }

  const_span_type::iterator begin() const { return span().begin(); }
  const_span_type::iterator end() const { return span().end(); }

  const_span_type::iterator cbegin() const { return span().begin(); }
  const_span_type::iterator cend() const { return span().end(); }

 private:
  mapping untyped_mapping_;
};

}  // namespace shortfin::array

#endif  // SHORTFIN_ARRAY_STORAGE_H
