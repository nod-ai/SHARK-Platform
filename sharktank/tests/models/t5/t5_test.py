# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from transformers.models.t5.modeling_t5 import (
    T5Attention as ReferenceT5Attention,
    T5LayerSelfAttention as ReferenceT5LayerSelfAttention,
    T5LayerFF as ReferenceT5LayerFF,
)
from transformers import (
    AutoTokenizer,
    T5EncoderModel as ReferenceT5EncoderModel,
    T5Config as ReferenceT5Config,
)
from typing import Optional
import os
from collections import OrderedDict
import pytest
import torch
from torch.utils._pytree import tree_map, tree_unflatten, tree_flatten_with_path
from unittest import TestCase
from parameterized import parameterized
from sharktank.types import (
    Theta,
    DefaultPrimitiveTensor,
    unbox_tensor,
    Dataset,
    dtype_to_serialized_short_name,
)
from sharktank.models.t5 import (
    T5Attention,
    T5SelfAttention,
    T5Config,
    T5Encoder,
    T5LayerFF,
    export_encoder_mlir,
    export_encoder_iree_parameters,
)
from sharktank.utils.testing import (
    make_rand_torch,
    make_random_mask,
    TempDirTestBase,
    test_prompts,
)
from sharktank.utils.hf_datasets import get_dataset
from sharktank.utils.iree import (
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    call_torch_module_function,
    flatten_for_iree_signature,
    iree_to_torch,
)
from sharktank.transforms.dataset import set_float_dtype
from sharktank import ops
import iree.compiler

with_t5_data = pytest.mark.skipif("not config.getoption('with_t5_data')")


@pytest.mark.usefixtures("get_model_artifacts")
class T5EncoderEagerTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        torch.no_grad()

    @with_t5_data
    def testXxlBf16AgainstFluxGolden(self):
        """The ground-truth values were acquired from the Flux pipeline."""
        target_model_name = (
            f"{'google/t5-v1_1-xxl'.replace('/', '__').replace('-', '_')}_f32_model"
        )
        target_model_path = getattr(self, target_model_name)
        dataset = Dataset.load(target_model_path)
        dataset.root_theta = dataset.root_theta.transform(
            functools.partial(set_float_dtype, dtype=torch.bfloat16)
        )
        config = T5Config.from_gguf_properties(
            dataset.properties,
            feed_forward_proj="gated-gelu",
        )
        model = T5Encoder(theta=dataset.root_theta, config=config)
        model.eval()

        with open(
            "/data/t5/xxl/flux_schnell_t5_v1_1_xxl_encoder_bf16_input_ids.pt", "rb"
        ) as f:
            reference_input_ids = torch.load(f)

        outputs = model(
            input_ids=reference_input_ids,
            attention_mask=None,
            output_hidden_states=False,
        )

        with open(
            "/data/t5/xxl/flux_schnell_t5_v1_1_xxl_encoder_bf16_output_last_hidden_state.pt",
            "rb",
        ) as f:
            reference_last_hidden_state = torch.load(f)

        torch.testing.assert_close(
            outputs["last_hidden_state"], reference_last_hidden_state
        )

    def runTestV1_1CompareTorchEagerHuggingFace(
        self,
        huggingface_repo_id: str,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        get_dataset(
            huggingface_repo_id,
        ).download()
        tokenizer = AutoTokenizer.from_pretrained(huggingface_repo_id)
        reference_model = ReferenceT5EncoderModel.from_pretrained(
            huggingface_repo_id, torch_dtype=reference_dtype
        )
        reference_model.eval()

        model = ReferenceT5EncoderModel.from_pretrained(
            huggingface_repo_id, torch_dtype=target_dtype
        )
        model.eval()

        input_ids = tokenizer(
            test_prompts,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=16,
        ).input_ids

        expected_outputs = dict(reference_model(input_ids=input_ids))
        actual_outputs = dict(model(input_ids=input_ids))
        actual_outputs = tree_map(
            lambda t: ops.to(t, dtype=reference_dtype), actual_outputs
        )

        torch.testing.assert_close(
            actual_outputs, expected_outputs, atol=atol, rtol=rtol
        )

    def runTestV1_1CompareTorchEagerAgainstHuggingFace(
        self,
        huggingface_repo_id: str,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        get_dataset(
            huggingface_repo_id,
        ).download()
        tokenizer = AutoTokenizer.from_pretrained(huggingface_repo_id)
        reference_model = ReferenceT5EncoderModel.from_pretrained(
            huggingface_repo_id, torch_dtype=reference_dtype
        )
        reference_model.eval()

        target_model_name = (
            f"{huggingface_repo_id.replace('/', '__').replace('-', '_')}_f32_model"
        )
        target_model_path = getattr(self, target_model_name)
        dataset = Dataset.load(target_model_path)
        dataset.root_theta = dataset.root_theta.transform(
            functools.partial(set_float_dtype, dtype=target_dtype)
        )
        config = T5Config.from_gguf_properties(
            dataset.properties,
            feed_forward_proj="gated-gelu",
        )

        input_ids = tokenizer(
            test_prompts,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=config.context_length_padding_block_size,
        ).input_ids

        model = T5Encoder(theta=dataset.root_theta, config=config)
        model.eval()

        expected_outputs = reference_model(input_ids=input_ids)
        actual_outputs = model(input_ids=input_ids)
        actual_outputs = tree_map(
            lambda t: ops.to(t, dtype=reference_dtype), actual_outputs
        )

        torch.testing.assert_close(
            actual_outputs, expected_outputs, atol=atol, rtol=rtol
        )

    @pytest.mark.xfail(
        raises=AssertionError,
        reason=(
            "The accuracy is bad, "
            "but for XXL we get the same result as the Flux pipeline. "
            "This need further investigation how Flux works at all like that."
        ),
    )
    @with_t5_data
    def testV1_1SmallCompareTorchEagerHuggingFaceBf16AgainstF32(self):
        self.runTestV1_1CompareTorchEagerHuggingFace(
            "google/t5-v1_1-small",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @with_t5_data
    def testV1_1SmallF32CompareTorchEagerAgainstHuggingFace(self):
        self.runTestV1_1CompareTorchEagerAgainstHuggingFace(
            "google/t5-v1_1-small",
            reference_dtype=torch.float32,
            target_dtype=torch.float32,
        )

    @pytest.mark.xfail(
        raises=AssertionError,
        reason=(
            "The accuracy is bad, "
            "but for XXL we get the same result as the Flux pipeline. "
            "This need further investigation how Flux works at all like that."
        ),
    )
    @with_t5_data
    def testV1_1SmallBf16CompareTorchEagerAgainstHuggingFaceF32(self):
        self.runTestV1_1CompareTorchEagerAgainstHuggingFace(
            "google/t5-v1_1-small",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @with_t5_data
    def testV1_1SmallBf16CompareTorchEagerAgainstHuggingFace(self):
        self.runTestV1_1CompareTorchEagerAgainstHuggingFace(
            "google/t5-v1_1-small",
            reference_dtype=torch.bfloat16,
            target_dtype=torch.bfloat16,
        )

    @with_t5_data
    def testV1_1XxlF32CompareTorchEagerAgainstHuggingFace(self):
        self.runTestV1_1CompareTorchEagerAgainstHuggingFace(
            "google/t5-v1_1-xxl",
            reference_dtype=torch.float32,
            target_dtype=torch.float32,
        )

    @pytest.mark.xfail(
        raises=AssertionError,
        reason=(
            "The accuracy is bad, but we get the same result as the Flux pipeline. "
            "This need further investigation how Flux works at all like that."
        ),
    )
    @with_t5_data
    def testV1_1XxlBf16CompareTorchEagerAgainstHuggingFaceF32(self):
        self.runTestV1_1CompareTorchEagerAgainstHuggingFace(
            "google/t5-v1_1-xxl",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
            atol=1e-2,
            rtol=1.6e-2,
        )


@pytest.mark.usefixtures("caching", "get_model_artifacts", "path_prefix")
class T5EncoderIreeTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        if self.path_prefix is None:
            self.path_prefix = f"{self._temp_dir}/"

    def runTestV1_1CompareIreeAgainstTorchEager(
        self,
        huggingface_repo_id: str,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        get_dataset(
            huggingface_repo_id,
        ).download()
        tokenizer = AutoTokenizer.from_pretrained(huggingface_repo_id)

        huggingface_repo_id_as_path = (
            f"{huggingface_repo_id.replace('/', '__').replace('-', '_')}"
        )
        source_model_name = f"{huggingface_repo_id_as_path}_f32_model"
        source_model_path = getattr(self, source_model_name)

        reference_dataset = Dataset.load(source_model_path)
        reference_dataset.root_theta = reference_dataset.root_theta.transform(
            functools.partial(set_float_dtype, dtype=reference_dtype)
        )
        config = T5Config.from_gguf_properties(
            reference_dataset.properties,
            feed_forward_proj="gated-gelu",
        )

        input_ids = tokenizer(
            test_prompts,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=config.context_length_padding_block_size,
        ).input_ids
        input_args = OrderedDict([("input_ids", input_ids)])
        batch_size = input_ids.shape[0]

        reference_dtype_name = dtype_to_serialized_short_name(reference_dtype)
        target_dtype_name = dtype_to_serialized_short_name(target_dtype)
        target_model_path_prefix = f"{self.path_prefix}{huggingface_repo_id_as_path}_encoder_{target_dtype_name}"

        reference_model = T5Encoder(theta=reference_dataset.root_theta, config=config)
        reference_result_dict = call_torch_module_function(
            module=reference_model,
            function_name="forward",
            kwargs=input_args,
            trace_path_prefix=f"{self.path_prefix}{huggingface_repo_id_as_path}_encoder_{reference_dtype_name}_torch_",
        )
        reference_result = flatten_for_iree_signature(reference_result_dict)

        parameters_path = f"{target_model_path_prefix}.irpa"
        if not self.caching or not os.path.exists(parameters_path):
            export_encoder_iree_parameters(
                source_model_path, parameters_path, dtype=target_dtype
            )

        mlir_path = f"{target_model_path_prefix}.mlir"
        if not self.caching or not os.path.exists(mlir_path):
            export_encoder_mlir(
                parameters_path, batch_sizes=[batch_size], mlir_output_path=mlir_path
            )
        iree_module_path = f"{target_model_path_prefix}.vmfb"
        if not self.caching or not os.path.exists(iree_module_path):
            iree.compiler.compile_file(
                mlir_path,
                output_file=iree_module_path,
                extra_args=["--iree-hal-target-device=hip", "--iree-hip-target=gfx942"],
            )

        iree_devices = get_iree_devices(driver="hip", device_count=1)
        iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
            module_path=iree_module_path,
            devices=iree_devices,
            parameters_path=parameters_path,
        )
        iree_args = prepare_iree_module_function_args(
            args=flatten_for_iree_signature(input_args), devices=iree_devices
        )
        iree_result = iree_to_torch(
            *run_iree_module_function(
                module=iree_module,
                vm_context=iree_vm_context,
                args=iree_args,
                driver="hip",
                function_name=f"forward_bs{batch_size}",
                trace_path_prefix=f"{target_model_path_prefix}_iree_",
            )
        )
        iree_result = [
            ops.to(iree_result[i], dtype=reference_result[i].dtype)
            for i in range(len(reference_result))
        ]

        torch.testing.assert_close(reference_result, iree_result, atol=atol, rtol=rtol)

    @with_t5_data
    def testV1_1CompareSmallIreeF32AgainstTorchEagerF32(self):
        self.runTestV1_1CompareIreeAgainstTorchEager(
            "google/t5-v1_1-small",
            reference_dtype=torch.float32,
            target_dtype=torch.float32,
            atol=1e-4,
            rtol=2.0e-3,
        )

    @pytest.mark.xfail(
        raises=AssertionError,
        reason=(
            "The accuracy is bad, "
            "but but it is no worse than the accuracy for of eager bfloat16. "
            "This need further investigation how Flux works at all like that."
        ),
    )
    @with_t5_data
    def testV1_1CompareSmallIreeBf16AgainstTorchEagerF32(self):
        self.runTestV1_1CompareIreeAgainstTorchEager(
            "google/t5-v1_1-small",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @with_t5_data
    def testV1_1CompareXxlIreeF32AgainstTorchEagerF32(self):
        self.runTestV1_1CompareIreeAgainstTorchEager(
            "google/t5-v1_1-xxl",
            reference_dtype=torch.float32,
            target_dtype=torch.float32,
            atol=1e-4,
            rtol=2.0e-3,
        )

    @pytest.mark.xfail(
        raises=AssertionError,
        reason=(
            "The accuracy is bad, "
            "but but it is no worse than the accuracy for of eager bfloat16. "
            "This need further investigation how Flux works at all like that."
        ),
    )
    @with_t5_data
    def testV1_1CompareXxlIreeBf16AgainstTorchEagerF32(self):
        self.runTestV1_1CompareIreeAgainstTorchEager(
            "google/t5-v1_1-xxl",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
            atol=1e-2,
            rtol=1.6e-2,
        )


class T5AttentionTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        torch.no_grad()

    @parameterized.expand(
        [
            [torch.float32, torch.float32],
            [torch.bfloat16, torch.bfloat16],
            [torch.float32, torch.bfloat16, 1e-2, 1.6e-2],
        ]
    )
    def testCompareAgainstTransformers(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        torch.set_default_dtype(reference_dtype)
        batch_size = 19
        batch_seq_len = 23
        reference_config = ReferenceT5Config(
            vocab_size=11,
            d_model=13,
            d_kv=7,
            d_ff=3,
            num_heads=2,
            relative_attention_num_buckets=5,
            relative_attention_max_distance=17,
            dropout_rate=0.0,
        )
        reference_model = ReferenceT5Attention(
            reference_config, has_relative_attention_bias=True
        )
        reference_model.eval()

        theta = Theta(
            {
                "attn_q.weight": DefaultPrimitiveTensor(
                    data=reference_model.q.weight.to(dtype=target_dtype)
                ),
                "attn_k.weight": DefaultPrimitiveTensor(
                    data=reference_model.k.weight.to(dtype=target_dtype)
                ),
                "attn_v.weight": DefaultPrimitiveTensor(
                    data=reference_model.v.weight.to(dtype=target_dtype)
                ),
                "attn_o.weight": DefaultPrimitiveTensor(
                    data=reference_model.o.weight.to(dtype=target_dtype)
                ),
                "attn_rel_b.weight": DefaultPrimitiveTensor(
                    data=reference_model.relative_attention_bias.weight.to(
                        dtype=target_dtype
                    )
                ),
            }
        )
        model = T5Attention(
            theta=theta,
            is_decoder=reference_config.is_decoder,
            relative_attention_num_buckets=reference_config.relative_attention_num_buckets,
            relative_attention_max_distance=reference_config.relative_attention_max_distance,
            d_model=reference_config.d_model,
            d_kv=reference_config.d_kv,
            num_heads=reference_config.num_heads,
            activation_dtype=target_dtype,
            has_relative_attention_bias=True,
        )
        model.eval()

        reference_hidden_states = make_rand_torch(
            shape=[batch_size, batch_seq_len, reference_config.d_model],
            dtype=reference_dtype,
        )
        reference_mask = make_random_mask(
            shape=[batch_size, 1, 1, batch_seq_len], dtype=reference_dtype
        )
        expected_outputs = reference_model(
            hidden_states=reference_hidden_states, mask=reference_mask
        )

        hidden_states = ops.to(reference_hidden_states, dtype=target_dtype)
        mask = ops.to(reference_mask, dtype=target_dtype)
        actual_outputs = model(
            hidden_states=DefaultPrimitiveTensor(data=hidden_states),
            mask=DefaultPrimitiveTensor(data=mask),
        )
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        torch.testing.assert_close(
            actual_outputs, expected_outputs, atol=atol, rtol=rtol
        )

    @parameterized.expand(
        [
            [torch.float32, torch.float32],
            [torch.bfloat16, torch.bfloat16],
            [torch.float32, torch.bfloat16, 1e-2, 1.6e-2],
        ]
    )
    def testCompareSelfAttentionAgainstTransformers(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        torch.set_default_dtype(reference_dtype)
        batch_size = 19
        batch_seq_len = 23
        reference_config = ReferenceT5Config(
            vocab_size=11,
            d_model=13,
            d_kv=7,
            d_ff=3,
            num_heads=2,
            relative_attention_num_buckets=5,
            relative_attention_max_distance=17,
            dropout_rate=0.0,
            layer_norm_epsilon=1e-6,
        )
        reference_model = ReferenceT5LayerSelfAttention(
            reference_config, has_relative_attention_bias=True
        )
        reference_model.eval()

        theta = Theta(
            {
                "attn_q.weight": DefaultPrimitiveTensor(
                    data=reference_model.SelfAttention.q.weight.to(dtype=target_dtype)
                ),
                "attn_k.weight": DefaultPrimitiveTensor(
                    data=reference_model.SelfAttention.k.weight.to(dtype=target_dtype)
                ),
                "attn_v.weight": DefaultPrimitiveTensor(
                    data=reference_model.SelfAttention.v.weight.to(dtype=target_dtype)
                ),
                "attn_o.weight": DefaultPrimitiveTensor(
                    data=reference_model.SelfAttention.o.weight.to(dtype=target_dtype)
                ),
                "attn_rel_b.weight": DefaultPrimitiveTensor(
                    data=reference_model.SelfAttention.relative_attention_bias.weight.to(
                        dtype=target_dtype
                    )
                ),
                "attn_norm.weight": DefaultPrimitiveTensor(
                    data=reference_model.layer_norm.weight.to(dtype=target_dtype)
                ),
            }
        )
        model = T5SelfAttention(
            theta=theta,
            is_decoder=reference_config.is_decoder,
            relative_attention_num_buckets=reference_config.relative_attention_num_buckets,
            relative_attention_max_distance=reference_config.relative_attention_max_distance,
            d_model=reference_config.d_model,
            d_kv=reference_config.d_kv,
            num_heads=reference_config.num_heads,
            activation_dtype=torch.float32,
            layer_norm_epsilon=reference_config.layer_norm_epsilon,
            has_relative_attention_bias=True,
        )
        model.eval()

        reference_hidden_states = make_rand_torch(
            shape=[batch_size, batch_seq_len, reference_config.d_model],
            dtype=reference_dtype,
        )
        reference_mask = make_random_mask(
            shape=[batch_size, 1, 1, batch_seq_len], dtype=reference_dtype
        )
        reference_position_bias = make_rand_torch(
            shape=[
                batch_size,
                reference_config.num_heads,
                batch_seq_len,
                batch_seq_len,
            ],
            dtype=reference_dtype,
        )
        expected_outputs = reference_model(
            hidden_states=reference_hidden_states,
            attention_mask=reference_mask,
            position_bias=reference_position_bias,
        )

        hidden_states = ops.to(reference_hidden_states, dtype=target_dtype)
        mask = ops.to(reference_mask, dtype=target_dtype)
        position_bias = ops.to(reference_position_bias, dtype=target_dtype)
        actual_outputs = model(
            hidden_states=DefaultPrimitiveTensor(data=hidden_states),
            attention_mask=DefaultPrimitiveTensor(data=mask),
            position_bias=DefaultPrimitiveTensor(data=position_bias),
        )
        actual_outputs = [
            unbox_tensor(t) if t is not None else t for t in actual_outputs
        ]
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        torch.testing.assert_close(
            actual_outputs, expected_outputs, atol=atol, rtol=rtol
        )


class T5LayerFFTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        torch.no_grad()

    @parameterized.expand(
        [
            [torch.float32, torch.float32],
            [torch.bfloat16, torch.bfloat16],
            [torch.float32, torch.bfloat16, 1e-2, 1.6e-2],
        ]
    )
    def testCompareAgainstTransformers(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        torch.set_default_dtype(reference_dtype)
        batch_size = 19
        batch_seq_len = 23
        reference_config = ReferenceT5Config(
            d_model=13,
            d_ff=3,
            dropout_rate=0.0,
            layer_norm_epsilon=1e-6,
            feed_forward_proj="gated-gelu",
        )

        reference_model = ReferenceT5LayerFF(reference_config)
        reference_model.eval()

        theta = Theta(
            {
                "ffn_gate.weight": DefaultPrimitiveTensor(
                    data=reference_model.DenseReluDense.wi_0.weight.to(
                        dtype=target_dtype
                    )
                ),
                "ffn_up.weight": DefaultPrimitiveTensor(
                    data=reference_model.DenseReluDense.wi_1.weight.to(
                        dtype=target_dtype
                    )
                ),
                "ffn_down.weight": DefaultPrimitiveTensor(
                    data=reference_model.DenseReluDense.wo.weight.to(dtype=target_dtype)
                ),
                "ffn_norm.weight": DefaultPrimitiveTensor(
                    data=reference_model.layer_norm.weight.to(dtype=target_dtype)
                ),
            }
        )
        model = T5LayerFF(
            theta=theta,
            is_gated_act=reference_config.is_gated_act,
            dense_act_fn=reference_config.dense_act_fn,
            layer_norm_epsilon=reference_config.layer_norm_epsilon,
            activation_dtype=torch.float32,
        )

        reference_hidden_states = make_rand_torch(
            shape=[batch_size, batch_seq_len, reference_config.d_model],
            dtype=reference_dtype,
        )
        expected_output = reference_model(
            hidden_states=reference_hidden_states,
        )

        hidden_states = ops.to(reference_hidden_states, dtype=target_dtype)
        actual_output = model(
            hidden_states=DefaultPrimitiveTensor(data=hidden_states),
        )
        actual_output = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_output,
        )

        torch.testing.assert_close(actual_output, expected_output, atol=atol, rtol=rtol)
