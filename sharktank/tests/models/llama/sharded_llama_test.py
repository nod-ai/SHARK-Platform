# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from copy import deepcopy
from iree.compiler import compile_file, InputType
from typing import Any
import functools
import os
import pytest
import torch

from sharktank.examples import export_paged_llm_v1
from sharktank.examples.sharding import shard_llm_dataset
from sharktank.examples.paged_llm_v1 import TorchGenerator
from sharktank.models.llama.llama import LlamaModelConfig, PagedLlamaModelV1
from sharktank.layers import CausalLMModelABC
from sharktank.layers.configs import LlamaHParams
from sharktank.layers.testing import CausalLMIreeModel
from sharktank.models.llama.sharding import shard_theta
from sharktank.models.llama.testing import make_random_llama_theta
from sharktank.types import (
    AnyTensor,
    InferenceTensor,
    DefaultPrimitiveTensor,
    Dataset,
    dtype_to_serialized_name,
)
from sharktank.utils.math import round_up_to_multiple_of
from sharktank.utils.iree import (
    get_iree_devices,
    load_iree_module,
)
from sharktank.utils.testing import PathPrefixTestBase, ModuloTokenizer, longrun
from sharktank.utils.tokenizer import load_tokenizer, InferenceTokenizer
import sharktank.ops as ops


AnyTokenizer = Any


def set_float_dtype(tensor: InferenceTensor, dtype: torch.dtype) -> InferenceTensor:
    if isinstance(tensor, DefaultPrimitiveTensor) and tensor.dtype.is_floating_point:
        return DefaultPrimitiveTensor(
            name=tensor.name, data=ops.to(tensor, dtype=dtype)
        )
    assert False, "Unsupported tensor type"


def shard_dataset(
    path: str,
    output_path: str,
    tensor_parallelism_size: int,
    intermediates_caching: bool,
):
    if not intermediates_caching or not os.path.exists(output_path):
        if path.endswith(".gguf"):
            dataset_arg = f"--gguf-file={path}"
        elif path.endswith(".irpa"):
            dataset_arg = f"--irpa-file={path}"
        else:
            raise ValueError(f'Invalid dataset filename "{dataset_arg}"')
        shard_llm_dataset.main(
            [
                f"--tensor-parallelism-size={tensor_parallelism_size}",
                dataset_arg,
                f"--output-irpa-file={output_path}",
            ]
        )


def compile_iree_module(
    intermediates_caching: bool,
    config: LlamaModelConfig,
    dataset_path: str,
    batch_size: int,
    target_device: str,
    output_mlir_path: str,
    output_module_path: str,
    output_config_path: str,
):
    if not intermediates_caching or not os.path.exists(output_module_path):
        export_paged_llm_v1.main(
            [
                f"--output-mlir={output_mlir_path}",
                f"--irpa-file={dataset_path}",
                f"--output-config={output_config_path}",
                f"--bs={batch_size}",
                f"--block-seq-stride={config.block_seq_stride}",
                f"--attention-dtype={dtype_to_serialized_name(config.attention_dtype)}",
                f"--activation-dtype={dtype_to_serialized_name(config.activation_dtype)}",
            ]
        )
        compiler_extra_args = [
            f"--iree-hal-target-device={target_device}[{i}]"
            for i in range(config.tensor_parallelism_size)
        ]

        compile_file(
            output_mlir_path,
            input_type=InputType.TORCH,
            output_file=output_module_path,
            extra_args=compiler_extra_args,
        )


def assert_close_cache_state(
    actual: list[torch.Tensor],
    expected: list[torch.Tensor],
):
    torch.testing.assert_close(
        actual[0].to(dtype=expected[0].dtype), expected[0], atol=1e-3, rtol=0
    )


def assert_close_logits(
    actual: torch.Tensor,
    expected: torch.Tensor,
):
    actual_probabilities = torch.softmax(actual, dim=1)
    expected_probabilities = torch.softmax(expected, dim=1)
    torch.testing.assert_close(
        actual_probabilities.to(dtype=expected_probabilities.dtype),
        expected_probabilities,
        atol=1e-3,
        rtol=0,
    )


def raise_multiple(errors):
    if not errors:  # list emptied, recursion ends
        return
    try:
        raise errors.pop()  # pop removes list entries
    finally:
        raise_multiple(errors)  # recursion


def assert_close_post_call(
    actual_logits: torch.Tensor,
    expected_logits: torch.Tensor,
    actual_cache_state: list[AnyTensor],
    expected_cache_state: list[AnyTensor],
):
    errors = []
    try:
        assert_close_logits(actual_logits, expected_logits)
    except Exception as ex:
        errors.append(ex)
    try:
        assert_close_cache_state(actual_cache_state, expected_cache_state)
    except Exception as ex:
        errors.append(ex)
    raise_multiple(errors)


def compare_models(
    target_model: CausalLMModelABC,
    reference_model: CausalLMModelABC,
    tokenizer: InferenceTokenizer,
    cache_page_count: int,
    prompts: list[str],
):
    generator = TorchGenerator(
        target_model, tokenizer, page_cache_size=cache_page_count
    )
    reference_generator = TorchGenerator(
        reference_model, tokenizer, page_cache_size=cache_page_count
    )
    batch = generator.begin_batch(prompts)
    reference_batch = reference_generator.begin_batch(prompts)

    # Init the cache and copy it to both the target and the reference.
    unsharded_reference_cache_state = reference_model.cache.paged.unshard_state(
        reference_batch.cache_state
    )
    torch.full(
        size=unsharded_reference_cache_state[0].shape,
        fill_value=0,
        out=unsharded_reference_cache_state[0],
    )
    reference_batch.cache_state[0][...] = reference_model.cache.paged.shard_state(
        unsharded_reference_cache_state
    )[0]
    batch.cache_state[0][...] = target_model.cache.paged.shard_state(
        unsharded_reference_cache_state
    )[0]

    batch.prefill()
    reference_batch.prefill()
    assert_close_post_call(
        actual_logits=batch.logits,
        expected_logits=reference_batch.logits,
        actual_cache_state=target_model.cache.paged.unshard_state(batch.cache_state),
        expected_cache_state=reference_batch.cache_state,
    )

    batch.decode()
    reference_batch.decode()
    assert_close_post_call(
        actual_logits=batch.logits,
        expected_logits=reference_batch.logits,
        actual_cache_state=target_model.cache.paged.unshard_state(batch.cache_state),
        expected_cache_state=reference_batch.cache_state,
    )


def run_test_compare_iree_against_torch(
    path_prefix: str,
    intermediates_caching: bool,
    torch_dataset_path: str,
    torch_config: LlamaModelConfig,
    iree_dataset_path: str,
    iree_config: LlamaModelConfig,
    iree_target_device: str,
    iree_driver: str,
    tokenizer: InferenceTokenizer,
    prompts: list[str],
    cache_page_count: int,
):
    iree_module_path = f"{path_prefix}program.vmfb"
    compile_iree_module(
        intermediates_caching=intermediates_caching,
        config=iree_config,
        dataset_path=iree_dataset_path,
        batch_size=len(prompts),
        target_device=iree_target_device,
        output_mlir_path=f"{path_prefix}program.mlir",
        output_module_path=iree_module_path,
        output_config_path=f"{path_prefix}program_config.json",
    )
    iree_devices = get_iree_devices(
        driver=iree_driver,
        device_count=iree_config.tensor_parallelism_size,
    )
    iree_module, vm_context, vm_instance = load_iree_module(
        module_path=iree_module_path,
        devices=iree_devices,
        parameters_path=iree_dataset_path,
    )
    iree_model = CausalLMIreeModel(
        batch_size=len(prompts),
        config=iree_config,
        vm_context=vm_context,
        iree_driver=iree_driver,
        iree_module=iree_module,
        iree_devices=iree_devices,
    )

    torch_dataset = Dataset.load(torch_dataset_path, mmap=False)
    torch_model = PagedLlamaModelV1(theta=torch_dataset.root_theta, config=torch_config)

    compare_models(
        target_model=iree_model,
        reference_model=torch_model,
        tokenizer=tokenizer,
        cache_page_count=cache_page_count,
        prompts=prompts,
    )


@pytest.mark.usefixtures("caching")
class ShardedLlamaTestBase(PathPrefixTestBase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(123456)
        self.intermediates_caching = self.caching
        self.prompts = [
            "The sky is blue",
            "The night is dark",
            "Linguistics is the study of",
        ]


class ShardedLlamaToySizedTest(ShardedLlamaTestBase):
    def setUp(self):
        super().setUp()
        self.reference_dtype = torch.float64
        self.target_dtype = torch.float32
        torch.set_default_dtype(self.reference_dtype)
        self.batch_size = 3
        self.attention_head_count_kv = 4
        self.attention_head_count = self.attention_head_count_kv * 5
        self.vocabulary_size = 19
        self.rope_dimension_count = 7 * 2
        self.attn_head_dim = self.rope_dimension_count
        self.block_seq_stride = 13
        self.context_length = round_up_to_multiple_of(
            functools.reduce(max, [len(prompt) for prompt in self.prompts]),
            self.block_seq_stride,
        )
        # Make this large enough to make torch.export.Dim happy.
        self.context_length = max(self.context_length, 4 * self.block_seq_stride)
        self.cache_page_count = 11
        self.config = LlamaModelConfig(
            hp=LlamaHParams(
                context_length=self.context_length,
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
            activation_dtype=self.reference_dtype,
            attention_dtype=self.reference_dtype,
            static_tables=False,
        )
        self.sharded_config = deepcopy(self.config)
        self.sharded_config.tensor_parallelism_size = 2
        self.sharded_config.activation_dtype = self.target_dtype
        self.sharded_config.attention_dtype = self.target_dtype

        self.theta = make_random_llama_theta(
            config=self.config,
            vocab_size=self.vocabulary_size,
        )
        self.theta.rename_tensors_to_paths()

        self.tokenizer = ModuloTokenizer(self.vocabulary_size)

    def testCompareTensorParallelToUnsharded(self):
        """Run a sharded variant of a toy model size and compare it against the
        unsharded variant."""
        sharded_theta = self.theta.transform(
            functools.partial(set_float_dtype, dtype=self.target_dtype)
        )
        sharded_theta = shard_theta(sharded_theta, self.sharded_config)
        sharded_model = PagedLlamaModelV1(sharded_theta, self.sharded_config)
        reference_model = PagedLlamaModelV1(self.theta, self.config)
        compare_models(
            target_model=sharded_model,
            reference_model=reference_model,
            tokenizer=self.tokenizer,
            prompts=self.prompts,
            cache_page_count=self.cache_page_count,
        )

    def testCompareTensorParallelWithIreeToUnsharded(self):
        """Test exporting to MLIR and compiling with IREE the sharded Llama model.
        Test numerical accuracy of the IREE module against PyTorch."""

        dataset = Dataset(
            properties=self.config.hp.to_gguf_props(), root_theta=self.theta
        )
        torch_dataset_path = f"{self.path_prefix}torch-reference-dataset.irpa"
        if not self.intermediates_caching or not os.path.exists(torch_dataset_path):
            dataset.save(torch_dataset_path)

        iree_unsharded_theta = self.theta.transform(
            functools.partial(set_float_dtype, dtype=self.target_dtype)
        )
        iree_unsharded_dataset = Dataset(
            properties=self.sharded_config.hp.to_gguf_props(),
            root_theta=iree_unsharded_theta,
        )
        iree_usharded_dataset_path = f"{self.path_prefix}iree-dataset-unsharded.irpa"
        if not self.intermediates_caching or not os.path.exists(
            iree_usharded_dataset_path
        ):
            iree_unsharded_dataset.save(iree_usharded_dataset_path)

        iree_dataset_path = f"{self.path_prefix}iree-dataset.irpa"

        shard_dataset(
            path=iree_usharded_dataset_path,
            output_path=iree_dataset_path,
            tensor_parallelism_size=self.sharded_config.tensor_parallelism_size,
            intermediates_caching=self.intermediates_caching,
        )

        run_test_compare_iree_against_torch(
            path_prefix=self.path_prefix,
            intermediates_caching=self.intermediates_caching,
            torch_dataset_path=torch_dataset_path,
            torch_config=self.config,
            iree_dataset_path=iree_dataset_path,
            iree_config=self.sharded_config,
            iree_target_device="llvm-cpu",
            iree_driver="local-task",
            tokenizer=self.tokenizer,
            prompts=self.prompts,
            cache_page_count=self.cache_page_count,
        )


@pytest.mark.usefixtures("get_model_path")
class Llama38BFp16Tp8Test(ShardedLlamaTestBase):
    def setUp(self):
        super().setUp()
        tokenizer_path = self.llama3_8b_tokenizer
        self.tokenizer = load_tokenizer(tokenizer_path.parent)

        self.reference_dtype = torch.float64
        self.dataset_path = str(self.llama3_8b_f16_model)
        self.batch_size = 4
        self.cache_page_count = 8192
        tensor_parallelism_size = 8

        dataset = Dataset.load(self.dataset_path)
        self.theta = dataset.root_theta

        self.config = LlamaModelConfig(
            hp=LlamaHParams.from_gguf_props(dataset.properties),
            activation_dtype=self.reference_dtype,
            attention_dtype=self.reference_dtype,
            static_tables=False,
        )
        self.sharded_config = LlamaModelConfig(
            hp=LlamaHParams.from_gguf_props(dataset.properties),
            tensor_parallelism_size=tensor_parallelism_size,
            static_tables=False,  # Rely on the compiler for hoisting tables.
        )

    def tearDown(self):
        # make sure we don't reference the memory mapped file.
        del self.theta
        super().tearDown()

    @longrun
    @pytest.mark.xfail(
        reason="Numerics are not close.", raises=AssertionError, strict=True
    )
    def testCompareTensorParallelWithIreeToUnsharded(self):
        """Test exporting to MLIR and compiling with IREE the sharded Llama model.
        Test numerical accuracy of the IREE module against PyTorch."""

        reference_theta = self.theta.transform(
            functools.partial(set_float_dtype, dtype=self.reference_dtype)
        )
        reference_dataset = Dataset(
            properties=self.config.hp.to_gguf_props(), root_theta=reference_theta
        )
        reference_dataset_path = f"{self.path_prefix}torch-reference-dataset.irpa"
        if not self.intermediates_caching or not os.path.exists(reference_dataset_path):
            reference_dataset.save(reference_dataset_path)
        target_dataset_path = f"{self.path_prefix}iree-dataset.irpa"

        shard_dataset(
            path=self.dataset_path,
            output_path=target_dataset_path,
            tensor_parallelism_size=self.sharded_config.tensor_parallelism_size,
            intermediates_caching=self.intermediates_caching,
        )

        run_test_compare_iree_against_torch(
            path_prefix=self.path_prefix,
            intermediates_caching=self.intermediates_caching,
            torch_dataset_path=self.dataset_path,
            torch_config=self.config,
            iree_dataset_path=target_dataset_path,
            iree_config=self.sharded_config,
            iree_target_device="llvm-cpu",
            iree_driver="local-task",
            tokenizer=self.tokenizer,
            prompts=self.prompts,
            cache_page_count=self.cache_page_count,
        )
