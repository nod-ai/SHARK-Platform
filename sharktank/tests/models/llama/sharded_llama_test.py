# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from typing import Any, Dict, List, Tuple
from sharktank.models.llama.llama import LlamaModelConfig, PagedLlamaModelV1
import sharktank.ops as ops
from sharktank.types import Dataset
from sharktank.models.llama.testing import make_random_llama_theta
from sharktank.models.llama.sharding import shard_theta
from sharktank.layers.configs import LlamaHParams
from sharktank.utils.math import round_up_to_multiple_of
from sharktank.utils import iterables_equal
import tempfile
import torch
from copy import deepcopy
from shark_turbine.aot import FxProgramsBuilder, export


class ShardedLlamaTest(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(123456)
        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)
        self.batch_size = 3
        self.attention_head_count_kv = 4
        self.attention_head_count = self.attention_head_count_kv * 5
        self.vocabulary_size = 19
        self.rope_dimension_count = 7 * 2
        self.attn_head_dim = self.rope_dimension_count
        self.block_seq_stride = 13
        self.cache_page_count = 11
        self.config = LlamaModelConfig(
            hp=LlamaHParams(
                context_length=self.block_seq_stride * 2,
                embedding_length=self.attention_head_count * self.attn_head_dim,
                block_count=3,
                feed_forward_length=23,
                rope_dimension_count=self.rope_dimension_count,
                rope_freq_base=500000.0,
                attention_head_count=self.attention_head_count,
                attn_head_dim=self.attn_head_dim,
                attention_layer_norm_rms_epsilon=0.01,
                attention_head_count_kv=self.attention_head_count_kv,
                expert_count=0,
                expert_used_count=0,
                model_arch="llama",
            ),
            block_seq_stride=self.block_seq_stride,
            activation_dtype=self.dtype,
            attention_dtype=self.dtype,
        )
        self.theta = make_random_llama_theta(
            config=self.config,
            vocab_size=self.vocabulary_size,
        )
        self.prefill_seq_lens = torch.tensor(
            [14, 9, self.block_seq_stride - 1], dtype=torch.int32
        )

    def make_prefill_args(self, model: PagedLlamaModelV1) -> Dict[str, Any]:
        batch_seq_len = round_up_to_multiple_of(
            int(torch.max(self.prefill_seq_lens)), model.cache.pad_sequence_stride
        )
        token_ids = torch.randint(
            low=0,
            high=self.vocabulary_size,
            size=[self.batch_size, batch_seq_len],
            dtype=torch.int32,
        )
        attention_mask = model.attention_mask(
            model.input_mask(self.prefill_seq_lens, batch_seq_len)
        )
        seq_block_ids = torch.arange(
            self.batch_size * batch_seq_len // self.config.block_seq_stride
        ).view(self.batch_size, -1)
        cache_state = model.cache.paged.allocate(page_count=self.cache_page_count)
        cache_state = [torch.rand_like(cache_state[0])]
        return {
            "tokens": token_ids,
            "attention_mask": attention_mask,
            "seq_block_ids": seq_block_ids,
            "cache_state": cache_state,
        }

    def make_equal_unsharded_and_sharded_prefill_args(
        self, model: PagedLlamaModelV1, sharded_model: PagedLlamaModelV1
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prefill_args = self.make_prefill_args(model)
        sharded_cache_state = sharded_model.cache.paged.allocate(
            page_count=self.cache_page_count
        )
        assert iterables_equal(
            prefill_args["cache_state"][0].shape, sharded_cache_state[0].shape
        )
        sharded_prefill_args = deepcopy(prefill_args)
        sharded_cache_state = sharded_model.cache.paged.shard_state(
            sharded_prefill_args["cache_state"]
        )
        sharded_prefill_args["cache_state"] = sharded_cache_state
        return prefill_args, sharded_prefill_args

    def make_decode_args(self, model: PagedLlamaModelV1) -> Dict[str, Any]:
        start_positions = self.prefill_seq_lens.clone()
        seq_lens = self.prefill_seq_lens + 1
        batch_seq_len = round_up_to_multiple_of(
            int(torch.max(seq_lens)), model.cache.pad_sequence_stride
        )
        decode_token_ids = torch.randint(
            low=0,
            high=self.vocabulary_size,
            size=[self.batch_size, 1],
            dtype=torch.int32,
        )
        attention_mask = model.decode_attention_mask(
            model.input_mask(seq_lens, batch_seq_len)
        )
        seq_block_ids = torch.arange(
            self.batch_size * batch_seq_len // self.config.block_seq_stride
        ).view(self.batch_size, -1)
        cache_state = model.cache.paged.allocate(page_count=self.cache_page_count)
        cache_state = [torch.rand_like(cache_state[0])]
        return {
            "tokens": decode_token_ids,
            "attention_mask": attention_mask,
            "start_positions": start_positions,
            "seq_block_ids": seq_block_ids,
            "cache_state": cache_state,
        }

    def make_equal_unsharded_and_sharded_decode_args(
        self, model: PagedLlamaModelV1, sharded_model: PagedLlamaModelV1
    ):
        decode_args = self.make_decode_args(model)
        sharded_decode_args = deepcopy(decode_args)
        sharded_decode_args["cache_state"] = sharded_model.cache.paged.shard_state(
            sharded_decode_args["cache_state"]
        )
        return decode_args, sharded_decode_args

    def testCompareToySizedModelToUnsharded(self):
        """Run a sharded variant of a toy model size and compare it against the
        unsharded variant."""
        model = PagedLlamaModelV1(self.theta, self.config)
        sharded_config = deepcopy(self.config)
        sharded_config.tensor_parallelism_size = 2
        sharded_theta = shard_theta(self.theta, sharded_config)
        sharded_model = PagedLlamaModelV1(sharded_theta, sharded_config)

        # Verify prefill step.
        (
            prefill_args,
            sharded_prefill_args,
        ) = self.make_equal_unsharded_and_sharded_prefill_args(model, sharded_model)

        expected_prefill_result = model.prefill(**prefill_args)
        sharded_prefill_result = sharded_model.prefill(**sharded_prefill_args)
        # The errors are quite high, but for float64 both errors drop to < 1e-12.
        # The numerics are probably correct.
        torch.testing.assert_close(
            sharded_prefill_result, expected_prefill_result, atol=1e-3, rtol=1e-2
        )
        expected_cache_state = prefill_args["cache_state"][0]
        actual_cache_state = ops.unshard(
            sharded_model.cache.paged.unflatten_page_table(
                sharded_prefill_args["cache_state"]
            )
        ).flatten(start_dim=1)
        torch.testing.assert_close(
            actual_cache_state, expected_cache_state, atol=1e-4, rtol=1e-1
        )

        # Verify decode step.
        (
            decode_args,
            sharded_decode_args,
        ) = self.make_equal_unsharded_and_sharded_decode_args(model, sharded_model)
        expected_decode_result = model.decode(**decode_args)
        sharded_decode_result = sharded_model.decode(**sharded_decode_args)
        torch.testing.assert_close(sharded_decode_result, expected_decode_result)
        expected_decode_cache_state = decode_args["cache_state"][0]
        actual_decode_cache_state = ops.unshard(
            sharded_model.cache.paged.unflatten_page_table(
                sharded_decode_args["cache_state"]
            )
        ).flatten(start_dim=1)
        # TODO: investigate why the Windows machine CI is producing a larger numerical
        # error.
        # The Ubuntu CI runs fine with default tolerances.
        torch.testing.assert_close(
            actual_decode_cache_state, expected_decode_cache_state, atol=1e-4, rtol=1e-4
        )

    def testExportToySizedModelToMlir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sharded_config = deepcopy(self.config)
            sharded_config.tensor_parallelism_size = 2
            sharded_theta = shard_theta(self.theta, sharded_config)
            sharded_theta.rename_tensors_to_paths()
            sharded_dataset = Dataset({}, sharded_theta)
            parameters_path = f"{temp_dir}/parameters.irpa"
            sharded_dataset.save(f"{temp_dir}/parameters.irpa")
            sharded_dataset = Dataset.load(parameters_path, mmap=False)

            model = PagedLlamaModelV1(self.theta, self.config)
            sharded_model = PagedLlamaModelV1(
                sharded_dataset.root_theta, sharded_config
            )
            sharded_fxb = FxProgramsBuilder(sharded_model)

            (
                _,
                sharded_prefill_args,
            ) = self.make_equal_unsharded_and_sharded_prefill_args(model, sharded_model)

            @sharded_fxb.export_program(
                name="prefill", args=tuple(), kwargs=sharded_prefill_args
            )
            def _(model, *args, **kwargs) -> torch.Tensor:
                return model.prefill(*args, **kwargs)

            _, sharded_decode_args = self.make_equal_unsharded_and_sharded_decode_args(
                model, sharded_model
            )
            # TODO: remove strict=False when
            # https://github.com/pytorch/pytorch/issues/136757
            # is resolved.
            @sharded_fxb.export_program(
                name="decode",
                args=tuple(),
                kwargs=sharded_decode_args,
                strict=False,
            )
            def _(model, *args, **kwargs) -> torch.Tensor:
                return model.decode(*args, **kwargs)

            output = export(sharded_fxb)
            output.save_mlir(f"{temp_dir}/program.mlir")
