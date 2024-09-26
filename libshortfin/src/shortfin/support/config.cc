// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/support/config.h"

#include <charconv>
#include <cstdlib>

#include "fmt/format.h"
#include "shortfin/support/logging.h"

namespace shortfin {

void ConfigOptions::SetOption(std::string_view key, std::string value) {
  options_[intern_.intern(key)] = std::move(value);
}

const std::optional<std::string_view> ConfigOptions::GetOption(
    std::string_view key) const {
  // Get explicit option.
  auto found_it = options_.find(key);
  if (found_it != options_.end()) {
    consumed_keys_.insert(key);
    return found_it->second;
  }

  // Consult environment.
  if (env_lookup_prefix_) {
    std::string env_key;
    env_key.reserve(env_lookup_prefix_->size() + key.size());
    env_key.append(*env_lookup_prefix_);
    for (char c : key) {
      env_key.push_back(std::toupper(c));
    }
    char *env_value = std::getenv(env_key.c_str());
    if (env_value && std::strlen(env_value) > 0) {
      return intern_.intern(env_value);
    }
  }

  return {};
}

std::optional<int64_t> ConfigOptions::GetInt(std::string_view key,
                                             bool non_negative) const {
  auto value = GetOption(key);
  if (!value) return {};
  int64_t result;
  auto last = value->data() + value->size();
  auto err = std::from_chars(value->data(), last, result);
  if (err.ec != std::errc{} || err.ptr != last) {
    throw std::invalid_argument(
        fmt::format("Could not parse '{}' as an integer from config option "
                    "{}",
                    *value, key));
  }
  if (non_negative && result < 0) {
    throw std::invalid_argument(fmt::format(
        "Could not parse '{}' as a non-negative integer from config option "
        "{}",
        *value, key));
  }
  return result;
}

std::optional<std::vector<int64_t>> ConfigOptions::GetIntList(
    std::string_view key, bool non_negative) const {
  auto value = GetOption(key);
  if (!value) return {};

  std::vector<int64_t> results;
  auto Consume = [&](std::string_view atom) {
    int64_t result;
    auto last = atom.data() + atom.size();
    auto err = std::from_chars(atom.data(), last, result);
    if (err.ec != std::errc{} || err.ptr != last) {
      throw std::invalid_argument(
          fmt::format("Could not parse '{}' as an integer from config option "
                      "{} (full value: {})",
                      atom, key, *value));
    }
    if (non_negative && result < 0) {
      throw std::invalid_argument(fmt::format(
          "Could not parse '{}' as a non-negative integer from config option "
          "{} (full value: {})",
          atom, key, *value));
    }
    results.push_back(result);
  };
  std::string_view sv_value = *value;
  for (;;) {
    auto found_it = sv_value.find(',');
    if (found_it == std::string_view::npos) {
      Consume(sv_value);
      break;
    }

    Consume(sv_value.substr(0, found_it));
    sv_value.remove_prefix(found_it + 1);
  }

  return results;
}

void ConfigOptions::CheckAllOptionsConsumed() const {
  std::vector<std::string_view> unused_options;
  for (auto it : options_) {
    const auto &key = it.first;
    if (!consumed_keys_.contains(key)) {
      unused_options.push_back(key);
    }
  }
  if (!unused_options.empty()) {
    throw std::invalid_argument(
        fmt::format("Specified options were not used: {}",
                    fmt::join(unused_options, ", ")));
  }
}

}  // namespace shortfin
