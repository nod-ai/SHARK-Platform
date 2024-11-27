// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_COMPONENTS_TOKENIZERS_TOKENIZERS_H
#define SHORTFIN_COMPONENTS_TOKENIZERS_TOKENIZERS_H

#include <string>
#include <vector>

#include "shortfin/support/api.h"

namespace shortfin::tokenizers {

// A vendored Tokenizer class that does not export the details of the backing
// implementation. While a little bit gross, this keeps us from needing to
// re-export a vendor'ed API as part of our public API.
// The current vendor tokenizer is based on mlc-ai/tokenizers-cpp. The API
// is fairly close to that implementation.
// See: https://github.com/mlc-ai/tokenizers-cpp
class SHORTFIN_API Tokenizer {
 public:
  Tokenizer(const Tokenizer &) = delete;
  Tokenizer &operator=(const Tokenizer &) = delete;
  Tokenizer(Tokenizer &&other) : vendor_tokenizer_(other.vendor_tokenizer_) {
    vendor_tokenizer_ = nullptr;
  }
  ~Tokenizer();

  // Factory functions.
  static Tokenizer FromBlobJSON(const std::string &json_blob);

  std::vector<int32_t> Encode(const std::string &text);
  std::vector<std::vector<int32_t>> EncodeBatch(
      const std::vector<std::string> &texts);
  std::string Decode(const std::vector<int32_t> &ids);
  size_t GetVocabSize();
  std::string IdToToken(int32_t token_id);
  int32_t TokenToId(const std::string &token);

 private:
  Tokenizer(void *vendor_tokenizer) : vendor_tokenizer_(vendor_tokenizer) {}

 protected:
  void *vendor_tokenizer_;
};

}  // namespace shortfin::tokenizers

#endif  // SHORTFIN_COMPONENTS_TOKENIZERS_TOKENIZERS_H
