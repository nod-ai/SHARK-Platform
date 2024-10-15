# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import pytest
from typing import Any, List, Tuple, OrderedDict
from sharktank.models.llama.llama import LlamaModelConfig, PagedLlamaModelV1
import sharktank.ops as ops
from sharktank.types import (
    unbox_tensor,
    Dataset,
)
from sharktank.models.llama.testing import make_random_llama_theta
from sharktank.models.llama.sharding import shard_theta
from sharktank.layers.configs import LlamaHParams
from sharktank.utils.math import round_up_to_multiple_of
from sharktank.utils import iterables_equal
from sharktank.utils.iree import (
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    call_torch_module_function,
)
import tempfile
import torch
from copy import deepcopy
from iree.turbine.aot import FxProgramsBuilder, export
import iree.runtime
import numpy as np
import os


def iree_to_torch(*tensors: iree.runtime.DeviceArray) -> List[torch.Tensor]:
    return [torch.tensor(tensor.to_host()) for tensor in tensors]


@pytest.mark.usefixtures("caching", "path_prefix")
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
        self.sharded_config = deepcopy(self.config)
        self.sharded_config.tensor_parallelism_size = 2
        self.theta = make_random_llama_theta(
            config=self.config,
            vocab_size=self.vocabulary_size,
        )
        self.prefill_seq_lens = torch.tensor(
            [14, 9, self.block_seq_stride - 1], dtype=torch.int64
        )

    def make_prefill_args(self, model: PagedLlamaModelV1) -> OrderedDict[str, Any]:
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
        return OrderedDict(
            [
                ("tokens", token_ids),
                ("attention_mask", attention_mask),
                ("seq_block_ids", seq_block_ids),
                ("cache_state", cache_state),
            ]
        )

    def make_equal_unsharded_and_sharded_prefill_args(
        self, model: PagedLlamaModelV1, sharded_model: PagedLlamaModelV1
    ) -> Tuple[OrderedDict[str, Any], OrderedDict[str, Any]]:
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

    def make_decode_args(self, model: PagedLlamaModelV1) -> OrderedDict[str, Any]:
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
        return OrderedDict(
            [
                ("tokens", decode_token_ids),
                ("attention_mask", attention_mask),
                ("start_positions", start_positions),
                ("seq_block_ids", seq_block_ids),
                ("cache_state", cache_state),
            ]
        )

    def make_equal_unsharded_and_sharded_decode_args(
        self, model: PagedLlamaModelV1, sharded_model: PagedLlamaModelV1
    ) -> Tuple[OrderedDict[str, Any], OrderedDict[str, Any]]:
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
        sharded_theta = shard_theta(self.theta, self.sharded_config)
        sharded_model = PagedLlamaModelV1(sharded_theta, self.sharded_config)

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
        torch.testing.assert_close(
            sharded_decode_result, expected_decode_result, atol=1e-4, rtol=1e-5
        )
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

    @unittest.skip(
        (
            "Before this does not crash at all we need "
            "https://github.com/iree-org/iree/pull/18663 merged."
        )
    )
    def testExportAndRunToySizedModelWithIree(self):
        """Test exporting to MLIR and compiling with IREE the sharded Llama model.
        Test numerical accuracy of the IREE module against PyTorch."""

        if self.path_prefix is not None:
            self.runTestExportAndRunToySizedModelWithIree(
                path_prefix=self.path_prefix, dump_enabled=True
            )
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                self.runTestExportAndRunToySizedModelWithIree(
                    path_prefix=f"{temp_dir}/", dump_enabled=False
                )

    def runTestExportAndRunToySizedModelWithIree(
        self, path_prefix: str, dump_enabled: bool
    ):
        sharded_theta = shard_theta(self.theta, self.sharded_config)
        sharded_theta.rename_tensors_to_paths()
        sharded_dataset = Dataset({}, sharded_theta)
        sharded_parameters_path = f"{path_prefix}parameters.irpa"
        sharded_dataset.save(sharded_parameters_path)
        sharded_dataset = Dataset.load(sharded_parameters_path, mmap=False)
        iree_driver = "local-task"

        model = PagedLlamaModelV1(self.theta, self.config)
        sharded_model = PagedLlamaModelV1(
            sharded_dataset.root_theta, self.sharded_config
        )
        (
            _,
            sharded_prefill_args,
        ) = self.make_equal_unsharded_and_sharded_prefill_args(model, sharded_model)
        (
            _,
            sharded_decode_args,
        ) = self.make_equal_unsharded_and_sharded_decode_args(model, sharded_model)

        iree_module_path = f"{path_prefix}program.vmfb"
        if not self.caching or not os.path.exists(iree_module_path):
            # Export and compile the IREE module.
            sharded_fxb = FxProgramsBuilder(sharded_model)

            @sharded_fxb.export_program(
                name="prefill", args=tuple(), kwargs=sharded_prefill_args
            )
            def _(model, *args, **kwargs) -> torch.Tensor:
                return model.prefill(*args, **kwargs)

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
            if dump_enabled:
                output.save_mlir(f"{path_prefix}program.mlir")
            output.session.set_flags(
                *[
                    f"--iree-hal-target-device=llvm-cpu[{i}]"
                    for i in range(self.sharded_config.tensor_parallelism_size)
                ]
            )
            output.compile(
                save_to=iree_module_path,
                target_backends=None,
            )

        iree_devices = get_iree_devices(
            driver=iree_driver,
            device_count=self.sharded_config.tensor_parallelism_size,
        )
        iree_module, vm_context, vm_instance = load_iree_module(
            module_path=iree_module_path,
            devices=iree_devices,
            parameters_path=sharded_parameters_path,
        )

        # Run prefill step.
        prefill_iree_args = prepare_iree_module_function_args(
            args=deepcopy(sharded_prefill_args).values(), devices=iree_devices
        )
        for i, arg in enumerate(prefill_iree_args):
            np.save(f"{path_prefix}prefill_arg{i}.npy", arg.to_host())
        prefill_iree_result = run_iree_module_function(
            args=prefill_iree_args,
            function_name="prefill",
            module=iree_module,
            vm_context=vm_context,
            driver=iree_driver,
            trace_path_prefix=path_prefix if dump_enabled else None,
        )
        prefill_iree_result = iree_to_torch(*prefill_iree_result)
        assert len(prefill_iree_result) == 1
        expected_prefill_result = call_torch_module_function(
            module=sharded_model,
            function_name="prefill",
            kwargs=sharded_prefill_args,
            trace_path_prefix=f"{path_prefix}expected_" if dump_enabled else None,
        )
        prefill_iree_cache_state_shards = prefill_iree_args[
            -self.config.tensor_parallelism_size - 1 :
        ]
        prefill_iree_cache_state_shards = iree_to_torch(
            *prefill_iree_cache_state_shards
        )

        # Run decode step.
        decode_iree_args = prepare_iree_module_function_args(
            args=deepcopy(sharded_decode_args).values(), devices=iree_devices
        )
        decode_iree_result = run_iree_module_function(
            args=decode_iree_args,
            function_name="decode",
            module=iree_module,
            vm_context=vm_context,
            driver=iree_driver,
            trace_path_prefix=path_prefix if dump_enabled else None,
        )
        decode_iree_result = iree_to_torch(*decode_iree_result)
        expected_decode_result = call_torch_module_function(
            module=sharded_model,
            function_name="decode",
            kwargs=sharded_decode_args,
            trace_path_prefix=f"{path_prefix}expected_" if dump_enabled else None,
        )
        decode_iree_cache_state_shards = decode_iree_args[
            -self.config.tensor_parallelism_size - 1 :
        ]
        decode_iree_cache_state_shards = iree_to_torch(*decode_iree_cache_state_shards)

        # Check IREE's numerical correctness against PyTorch.
        # TODO: Although, not entirely wrong, investigate why this accuracy is that
        # low for fp32 (atol=0.0011, rtol=0.013).
        torch.testing.assert_close(
            prefill_iree_result[0],
            expected_prefill_result,
        )
        for actual_cache_state_shard, expected_cache_state_shard in zip(
            prefill_iree_cache_state_shards,
            sharded_prefill_args["cache_state"][0].shards,
        ):
            # TODO: debug inaccuracy.
            torch.testing.assert_close(
                actual_cache_state_shard, unbox_tensor(expected_cache_state_shard)
            )
        # TODO: debug inaccuracy.
        torch.testing.assert_close(decode_iree_result[0], expected_decode_result)
        for actual_cache_state_shard, expected_cache_state_shard in zip(
            decode_iree_cache_state_shards,
            sharded_decode_args["cache_state"][0].shards,
        ):
            # TODO: debug inaccuracy.
            torch.testing.assert_close(
                actual_cache_state_shard, unbox_tensor(expected_cache_state_shard)
            )
