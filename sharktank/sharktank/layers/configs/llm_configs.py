# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structured configuration objects for various LLMs.

This draws heavily from the work that ggml has done to systematize the state
of the world for GGUF files:
  https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

When in question, we draw from the vocabulary and normalization they have done
(and indeed, can bootstrap these off of GGUF files).
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

__all__ = ["LlamaHParams", "GrokHParams"]


@dataclass
class GrokHParams:
    """Corresponds 1:1 with the 'LLM' section of the GGUF docs.
    Comments are only provided if they differ from this source.
    """

    context_length: int
    embedding_length: int
    block_count: int
    feed_forward_length: int
    rope_dimension_count: int
    rope_freq_base: float
    attention_head_count: int
    attn_head_dim: int
    attention_layer_norm_rms_epsilon: float
    attention_head_count_kv: int
    expert_count: int
    expert_used_count: int

    @staticmethod
    def from_gguf_props(p: dict[str, Any]):
        default_expert_count = 0
        default_expert_used_count = 0
        default_rope_freq_base = 10000.0
        attention_head_count = _int_prop(p, "grok.attention.head_count")

        return LlamaHParams(
            context_length=_int_prop(p, "grok.context_length"),
            embedding_length=_int_prop(p, "grok.embedding_length"),
            block_count=_int_prop(p, "grok.block_count"),
            feed_forward_length=_int_prop(p, "grok.feed_forward_length"),
            attn_head_dim=128,  # _int_prop(p, "grok.rope.dimension_count"),
            rope_dimension_count=128,  # _int_prop(p, "grok.rope.dimension_count"),
            attention_head_count=attention_head_count,
            attention_layer_norm_rms_epsilon=_float_prop(
                p, "grok.attention.layer_norm_rms_epsilon"
            ),
            attention_head_count_kv=_optional_int_prop(
                p, "grok.attention.head_count_kv", attention_head_count
            ),
            rope_freq_base=_optional_float_prop(
                p, "grok.rope.freq_base", default_rope_freq_base
            ),
            expert_count=_optional_int_prop(
                p, "grok.expert_count", default_expert_count
            ),
            expert_used_count=_optional_int_prop(
                p, "grok.expert_used_count", default_expert_used_count
            ),
        )


@dataclass
class LlamaHParams:
    """Corresponds 1:1 with the 'LLM' section of the GGUF docs.

    Comments are only provided if they differ from this source.
    """

    context_length: int
    embedding_length: int
    block_count: int
    feed_forward_length: int
    rope_dimension_count: int
    rope_freq_base: float
    attention_head_count: int
    attn_head_dim: int
    attention_layer_norm_rms_epsilon: float
    attention_head_count_kv: int
    expert_count: int
    expert_used_count: int

    @staticmethod
    def from_gguf_props(p: dict[str, Any]):
        name_prefix = "llama"
        if "grok.attention.head_count" in p:
            name_prefix = "grok"
        default_expert_count = 0
        default_expert_used_count = 0
        default_rope_freq_base = 10000.0
        attention_head_count = _int_prop(p, f"{name_prefix}.attention.head_count")

        return LlamaHParams(
            context_length=_int_prop(p, f"{name_prefix}.context_length"),
            embedding_length=_int_prop(p, f"{name_prefix}.embedding_length"),
            block_count=_int_prop(p, f"{name_prefix}.block_count"),
            feed_forward_length=_int_prop(p, f"{name_prefix}.feed_forward_length"),
            attn_head_dim=_int_prop(p, f"{name_prefix}.rope.dimension_count"),
            rope_dimension_count=_int_prop(p, f"{name_prefix}.rope.dimension_count"),
            attention_head_count=attention_head_count,
            attention_layer_norm_rms_epsilon=_float_prop(
                p, f"{name_prefix}.attention.layer_norm_rms_epsilon"
            ),
            attention_head_count_kv=_optional_int_prop(
                p, f"{name_prefix}.attention.head_count_kv", attention_head_count
            ),
            rope_freq_base=_optional_float_prop(
                p, f"{name_prefix}.rope.freq_base", default_rope_freq_base
            ),
            expert_count=_optional_int_prop(
                p, f"{name_prefix}.expert_count", default_expert_count
            ),
            expert_used_count=_optional_int_prop(
                p, f"{name_prefix}.expert_used_count", default_expert_used_count
            ),
        )


def _float_prop(p: dict[str, Any], name: str) -> float:
    try:
        return float(p[name])
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be a float and was not") from e
    except KeyError:
        raise KeyError(f"Property '{name}' not found (among keys {p.keys()})")


def _int_prop(p: dict[str, Any], name: str) -> int:
    try:
        return int(p[name])
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be an int and was not") from e
    except KeyError:
        raise KeyError(f"Property '{name}' not found (among keys {p.keys()})")


def _optional_float_prop(p: dict[str, Any], name: str, default_value: float) -> float:
    value = p.get(name, default_value)
    try:
        return float(value)
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be a float and was not") from e


def _optional_int_prop(p: dict[str, Any], name: str, default_value: int) -> int:
    value = p.get(name, default_value)
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be an int and was not") from e
