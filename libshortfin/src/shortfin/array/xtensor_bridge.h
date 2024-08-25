// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_ARRAY_XTENSOR_BRIDGE_H
#define SHORTFIN_ARRAY_XTENSOR_BRIDGE_H

#include <fmt/core.h>

#include <span>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

#include "shortfin/array/dims.h"
#include "shortfin/array/dtype.h"
#include "shortfin/array/storage.h"

namespace shortfin::array {

// Polymorphic trampoline methods to a backing typed, xarray adaptor. This
// allows xtensor facilities to be used in a dtype agnostic fashion.
class poly_xt_methods {
 public:
  // Prints the contents of the array.
  virtual std::string contents_to_s() = 0;

 protected:
  // Since we adapt from a pointer-based container with Dims, just pick one
  // as a generic version so that we can reserve space in the class for it.
  using xt_generic_t =
      decltype(xt::adapt(static_cast<double *>(nullptr), Dims()));

  // Placement new an appropriate subclass into the provided storage area,
  // which must be sized to hold the base class (subclasses are statically
  // asserted to be the same size). The appropriate subclass will also placement
  // new an appropriate xtensor adaptor into the adaptor_storage field. It is
  // statically asserted that the type specific adaptor will fit into the
  // storage area reserved.
  // Returns true if an appropriate instance is instantiated. False if no
  // implementation for the dtype exists.
  static bool inplace_new(char *inst_storage, DType dtype, void *array_memory,
                          size_t array_memory_size, Dims &dims);

  // When instantiated via inplace_new, destorys the instance, calling both
  // the type specific adaptor destructor and the subclass destructor.
  virtual void inplace_destruct_this() = 0;

  char adaptor_storage[sizeof(xt_generic_t)];

  template <typename DerivedArrayTy, typename MemoryTy>
  friend class poly_xt_mixin;
};

// Polymorphic xtensor array mixin. Since xt::array is static on element type,
// this class provides a bridge that will polymorphically manage a specialized
// xarray adaptor for a base_array derived class.
//
// This is designed to use via CRTP on a subclass of base_array.
//
// Access is indirected through a heap allocated poly_xt_methods subclass that
// is initialized on-demand by mapping the device memory and constructing an
// appropriate typed subclass. This is done through two layers of generic
// storage (one contained here for the poly_xt_methods subclass and one
// on that class for the concrete xtensor adaptor it contains). The overhead
// on the base_array instance if the xtensor bridge is not used is one pointer.
// On first use, it is a heap allocation and a switch on dtype.
template <typename DerivedArrayTy, typename MemoryTy>
class poly_xt_mixin {
 public:
  poly_xt_mixin() = default;
  // Don't copy the poly instance: if it is needed on the copy, it will be
  // re-allocated.
  poly_xt_mixin(const poly_xt_mixin &other) {}

  std::optional<std::string> contents_to_s() {
    auto *m = optional_xt_methods();
    if (!m) return {};
    return m->contents_to_s();
  }

  std::optional<std::string> contents_to_s() const {
    return const_cast<poly_xt_mixin *>(this)->contents_to_s();
  }

  // Access (potentially instantiating) the polymorphic xt methods trampoline
  // for this array. If no xtensor adaptor can be created or if the memory
  // is not accessible to the host, returns nullptr. The returned pointer
  // must not outlive the creating array.
  poly_xt_methods *optional_xt_methods() {
    if (poly_) {
      return poly_->methods();
    }
    DType dtype = derived_this()->dtype();
    auto inst = std::make_unique<PolyInstance>();
    // CRTP derived class must provide a memory mapping via its
    // map_memory_for_xtensor() method.
    // This must be typed as MemoryTy and have data() and size() accessors.
    std::optional<MemoryTy> mapping = derived_this()->map_memory_for_xtensor();
    if (!mapping) {
      return nullptr;
    }
    inst->memory = std::move(*mapping);
    void *data = static_cast<void *>(inst->memory.data());
    size_t data_size = inst->memory.size();
    if (!poly_xt_methods::inplace_new(inst->methods_storage, dtype, data,
                                      data_size,
                                      derived_this()->shape_container())) {
      return nullptr;
    }
    poly_ = std::move(inst);
    return poly_.get()->methods();
  }

  // Accesses (potentially instantiating) the polymorphic xt methods trampoline.
  // If it cannot be created, throws a std::logic_error. The returned reference
  // must not outlive the creating array.
  poly_xt_methods &xt_methods() {
    auto m = optional_xt_methods();
    if (!m) {
      throw std::logic_error(fmt::format(
          "No xtensor specialization registered for dtype {} or storage type",
          derived_this()->dtype().name()));
    }
    return *m;
  }

 protected:
  ~poly_xt_mixin() {
    if (poly_) {
      // Need to in-place destruct the adaptor and then the methods itself.
      poly_->methods()->inplace_destruct_this();
    }
  }

 private:
  struct PolyInstance {
    MemoryTy memory;
    char methods_storage[sizeof(poly_xt_methods)];
    poly_xt_methods *methods() {
      return reinterpret_cast<poly_xt_methods *>(methods_storage);
    }
  };

  const DerivedArrayTy *derived_this() const {
    return static_cast<const DerivedArrayTy *>(this);
  }
  DerivedArrayTy *derived_this() { return static_cast<DerivedArrayTy *>(this); }

  // If the polymorphic accessor has been instantiated, it will be constructed
  // here.
  std::unique_ptr<PolyInstance> poly_;
};

}  // namespace shortfin::array

#endif  // SHORTFIN_ARRAY_XTENSOR_BRIDGE_H
