// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_SUPPORT_CONFIG_H
#define SHORTFIN_SUPPORT_CONFIG_H

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "shortfin/support/api.h"
#include "shortfin/support/stl_extras.h"

namespace shortfin {

// Utility class for querying free-form config options from a string list,
// environment variables, etc.
//
// Config options are looked up in a dict-like map. If not found explicitly,
// they can be optionally found in the environment if |env_lookup_prefix| is
// provided. In this case, the requested upper-cased key is concatted to the
// prefix and lookup up via std::getenv().
class SHORTFIN_API ConfigOptions {
 public:
  // Level of validation to do on configuration options when calling `Create`.
  enum class ValidationLevel {
    UNDEF_WARN,
    UNDEF_DEBUG,
    UNDEF_ERROR,
  };

  ConfigOptions(std::optional<std::string> env_lookup_prefix = {},
                ValidationLevel validation = ValidationLevel::UNDEF_WARN)
      : env_lookup_prefix_(std::move(env_lookup_prefix)),
        validation_(validation) {}
  ConfigOptions(const ConfigOptions &) = delete;
  ConfigOptions(ConfigOptions &&) = default;

  void SetOption(std::string_view key, std::string value);

  const std::optional<std::string_view> GetOption(std::string_view key) const;

  // Gets an option as an integer, optionall enforcing that each is
  // non-negative.
  std::optional<int64_t> GetInt(std::string_view key,
                                bool non_negative = false) const;

  // Gets an option as a bool, returning |default_value| if not found.
  // Bools are interpreted strictly from the string value. The following
  // evaluate to true (case insensitive):
  //   1, TRUE, ON
  // The following evaluate to false:
  //   0, FALSE, OFF
  bool GetBool(std::string_view key, bool default_value = false) const;

  // Gets an option as a list of integers, optionally enforcing that each
  // is non-negative.
  std::optional<std::vector<int64_t>> GetIntList(
      std::string_view key, bool non_negative = false) const;

  // Gets a raw environment variable without looking up in the options or
  // translating the name.
  std::optional<std::string_view> GetRawEnv(const char *key) const;

  // Helper to split on a delimitter.
  static std::vector<std::string_view> Split(std::string_view value,
                                             char delim);

  // After all configuration options have been consumed, perform validation
  // that all options were recognized.
  void ValidateUndef() const;

 private:
  mutable string_interner intern_;
  // Optional environment variable lookup prefix for resolving options not
  // explicitly set.
  std::optional<std::string> env_lookup_prefix_;

  // Level of validation to perform.
  ValidationLevel validation_;

  // Explicit keyword options.
  std::unordered_map<std::string_view, std::string> options_;

  // Keep track of which keys were consumed. Used for error checking.
  mutable std::unordered_set<std::string_view> consumed_keys_;
};

}  // namespace shortfin

#endif  // SHORTFIN_SUPPORT_CONFIG_H
