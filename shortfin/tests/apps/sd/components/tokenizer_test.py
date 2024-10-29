# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


@pytest.fixture
def clip_tokenizer():
    from shortfin_apps.sd.components.tokenizer import Tokenizer

    return Tokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", "tokenizer"
    )


def test_transformers_tokenizer(clip_tokenizer):
    enc0 = clip_tokenizer.encode(["This is sequence 1", "Sequence 2"])
    e0 = enc0.input_ids[0, :10]
    e1 = enc0.input_ids[1, :10]
    assert e0.tolist() == [
        49406,
        589,
        533,
        18833,
        272,
        49407,
        49407,
        49407,
        49407,
        49407,
    ]
    assert e1.tolist() == [
        49406,
        18833,
        273,
        49407,
        49407,
        49407,
        49407,
        49407,
        49407,
        49407,
    ]


def test_tokenizer_to_array(cpu_fiber, clip_tokenizer):
    batch_seq_len = 64
    encs = clip_tokenizer.encode(["This is sequence 1", "Sequence 2"])
    ary = clip_tokenizer.encodings_to_array(cpu_fiber.device(0), encs, batch_seq_len)
    print(ary)
    assert ary.view(0).items.tolist()[:5] == [49406, 589, 533, 18833, 272]
    assert ary.view(1).items.tolist()[:5] == [49406, 18833, 273, 49407, 49407]
