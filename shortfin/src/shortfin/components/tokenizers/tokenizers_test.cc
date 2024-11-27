// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/components/tokenizers/tokenizers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

using namespace shortfin::tokenizers;

namespace {

std::string ReadFile(std::filesystem::path path) {
  std::ifstream in(path);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

}  // namespace

// TODO: Enable once upstream changes with error handling have landed.
// Currently aborts.
// See: https://github.com/mlc-ai/tokenizers-cpp/issues/50
// TEST(TokenizersTest, FromIllegalBlobJson) {
//   auto tok = Tokenizer::FromBlobJSON("foobar");
// }

TEST(TokenizersTest, BasicTokenizerJson) {
  std::filesystem::path tokenizer_path(
      "src/shortfin/components/tokenizers/tokenizer.json");
  auto tokenizer_json = ReadFile(tokenizer_path);
  ASSERT_GT(tokenizer_json.size(), 0)
      << "reading " << tokenizer_path
      << " (cwd: " << std::filesystem::current_path() << ")";
  auto tok = Tokenizer::FromBlobJSON(tokenizer_json);
  EXPECT_GT(tok.GetVocabSize(), 100);  // Sanity check
  auto encoded = tok.Encode("hello world");
  EXPECT_THAT(encoded,
              ::testing::ContainerEq(std::vector<int32_t>{19082, 1362}));
  auto batch_encoded = tok.EncodeBatch({"hello", "world"});
  ASSERT_EQ(batch_encoded.size(), 2);
  EXPECT_THAT(batch_encoded[0],
              ::testing::ContainerEq(std::vector<int32_t>{19082}));
  EXPECT_THAT(batch_encoded[1],
              ::testing::ContainerEq(std::vector<int32_t>{1362}));
  EXPECT_EQ(tok.TokenToId("hello"), 19082);
  auto decoded = tok.Decode(encoded);
  EXPECT_EQ(decoded, "hello world");
}
