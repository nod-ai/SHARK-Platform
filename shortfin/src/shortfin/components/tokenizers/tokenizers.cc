// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/components/tokenizers/tokenizers.h"

#include <exception>

#include "shortfin/support/logging.h"
#include "tokenizers_cpp.h"

namespace shortfin::tokenizers {

namespace {

class AccessibleTokenizer : public Tokenizer {
 public:
  using Tokenizer::vendor_tokenizer_;
};

::tokenizers::Tokenizer *Get(Tokenizer *self) {
  void *ptr = static_cast<AccessibleTokenizer *>(self)->vendor_tokenizer_;
  if (!ptr) {
    throw std::logic_error("Tokenizer is null");
  }
  return static_cast<::tokenizers::Tokenizer *>(ptr);
}

}  // namespace

Tokenizer::~Tokenizer() { delete Get(this); }

Tokenizer Tokenizer::FromBlobJSON(const std::string &json_blob) {
  SHORTFIN_TRACE_SCOPE_NAMED("Tokenizer::FromBlobJSON");
  return Tokenizer(::tokenizers::Tokenizer::FromBlobJSON(json_blob).release());
}

std::vector<int32_t> Tokenizer::Encode(const std::string &text) {
  SHORTFIN_TRACE_SCOPE_NAMED("Tokenizer::Encode");
  return Get(this)->Encode(text);
}

std::vector<std::vector<int32_t>> Tokenizer::EncodeBatch(
    const std::vector<std::string> &texts) {
  SHORTFIN_TRACE_SCOPE_NAMED("Tokenizer::EncodeBatch");
  return Get(this)->EncodeBatch(texts);
}

std::string Tokenizer::Decode(const std::vector<int32_t> &ids) {
  SHORTFIN_TRACE_SCOPE_NAMED("Tokenizer::Decode");
  return Get(this)->Decode(ids);
}
size_t Tokenizer::GetVocabSize() { return Get(this)->GetVocabSize(); }
std::string Tokenizer::IdToToken(int32_t token_id) {
  return Get(this)->IdToToken(token_id);
}
int32_t Tokenizer::TokenToId(const std::string &token) {
  return Get(this)->TokenToId(token);
}

}  // namespace shortfin::tokenizers
