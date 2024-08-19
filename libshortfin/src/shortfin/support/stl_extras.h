// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_SUPPORT_STL_EXTRAS_H
#define SHORTFIN_SUPPORT_STL_EXTRAS_H

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace shortfin {

// String interning map, allowing us to mostly operate on string_views without
// tracking ownership.
class string_interner {
 public:
  // Given an arbitrary string_view, return a string_view that is owned by
  // this instance.
  std::string_view intern(std::string_view unowned) {
    auto& it = intern_map_[unowned];
    if (it) return *it;
    it = std::make_unique<std::string>(unowned);
    return *it;
  }

 private:
  std::unordered_map<std::string_view, std::unique_ptr<std::string>>
      intern_map_;
};

}  // namespace shortfin

#endif  // SHORTFIN_SUPPORT_STL_EXTRAS_H
