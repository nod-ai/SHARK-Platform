# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
import os
from collections import OrderedDict
import pytest
import torch
from unittest import TestCase
from parameterized import parameterized
from sharktank.types import Theta, DefaultPrimitiveTensor, unbox_tensor, Dataset
from sharktank.models.t5 import (
    T5Attention,
    T5SelfAttention,
    T5Config,
    T5Encoder,
    T5LayerFF,
    export_encoder_mlir,
    export_encoder_iree_parameters,
)
from sharktank.utils.testing import make_rand_torch, TempDirTestBase
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
import iree.compiler

with_t5_data = pytest.mark.skipif("not config.getoption('with_t5_data')")


def make_random_mask(shape: tuple[int], dtype: torch.dtype):
    mask = make_rand_torch(shape=shape, dtype=dtype)
    mask = (mask >= 0).to(dtype=dtype)
    return mask


test_prompts = [
    "Studies have been shown that owning a dog is good for you",
    "The horse went into the river",
    "We need at least one sentence long enough so that it spans more than one padding block which by default is of size 16.",
    "Make the batch size 4",
]


@pytest.mark.usefixtures("get_model_artifacts")
class T5EncoderEagerTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        torch.no_grad()

    def runTestV1_1Fp32CompareTorchEagerAgainstHuggingFace(
        self, huggingface_repo_id: str
    ):
        get_dataset(
            huggingface_repo_id,
        ).download()
        tokenizer = AutoTokenizer.from_pretrained(huggingface_repo_id)
        reference_model = ReferenceT5EncoderModel.from_pretrained(huggingface_repo_id)
        reference_model.eval()

        input_ids = tokenizer(
            test_prompts,
            return_tensors="pt",
            padding=True,
        ).input_ids

        target_model_name = (
            f"{huggingface_repo_id.replace('/', '__').replace('-', '_')}_fp32_model"
        )
        target_model_path = getattr(self, target_model_name)
        dataset = Dataset.load(target_model_path)
        config = T5Config.from_gguf_properties(
            dataset.properties,
            feed_forward_proj="gated-gelu",
        )
        model = T5Encoder(theta=dataset.root_theta, config=config)
        model.eval()

        expected_outputs = reference_model(input_ids=input_ids)
        actual_outputs = model(input_ids=input_ids)
        torch.testing.assert_close(actual_outputs, expected_outputs, atol=1e-5, rtol=0)

    @with_t5_data
    def testV1_1SmallFp32CompareTorchEagerAgainstHuggingFace(self):
        self.runTestV1_1Fp32CompareTorchEagerAgainstHuggingFace("google/t5-v1_1-small")

    @with_t5_data
    def testV1_1XxlFp32CompareTorchEagerAgainstHuggingFace(self):
        self.runTestV1_1Fp32CompareTorchEagerAgainstHuggingFace("google/t5-v1_1-xxl")


@pytest.mark.usefixtures("caching", "get_model_artifacts", "path_prefix")
class T5EncoderIreeTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        if self.path_prefix is None:
            self.path_prefix = f"{self._temp_dir}/"

    @parameterized.expand(
        [
            "google/t5-v1_1-small",
            "google/t5-v1_1-xxl",
        ]
    )
    @with_t5_data
    def testV1_1Fp32CompareIreeAgainstTorchEager(self, huggingface_repo_id: str):
        get_dataset(
            huggingface_repo_id,
        ).download()
        tokenizer = AutoTokenizer.from_pretrained(huggingface_repo_id)

        huggingface_repo_id_as_path = (
            f"{huggingface_repo_id.replace('/', '__').replace('-', '_')}"
        )
        source_model_name = f"{huggingface_repo_id_as_path}_fp32_model"
        source_model_path = getattr(self, source_model_name)

        dataset = Dataset.load(source_model_path)
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
        input_args = OrderedDict([("input_ids", input_ids)])
        batch_size = input_ids.shape[0]

        reference_model = T5Encoder(theta=dataset.root_theta, config=config)
        reference_result = flatten_for_iree_signature(
            call_torch_module_function(
                module=reference_model,
                function_name="forward",
                kwargs=input_args,
                trace_path_prefix=f"{self.path_prefix}{huggingface_repo_id_as_path}_torch_",
            )
        )

        mlir_path = f"{self.path_prefix}{huggingface_repo_id_as_path}_encoder_fp32.mlir"
        if not self.caching or not os.path.exists(mlir_path):
            export_encoder_mlir(
                source_model_path, batch_sizes=[batch_size], mlir_output_path=mlir_path
            )
        iree_module_path = (
            f"{self.path_prefix}{huggingface_repo_id_as_path}_encoder_fp32.vmfb"
        )
        if not self.caching or not os.path.exists(iree_module_path):
            iree.compiler.compile_file(
                mlir_path,
                output_file=iree_module_path,
                extra_args=["--iree-hal-target-device=hip", "--iree-hip-target=gfx942"],
            )

        parameters_path = (
            f"{self.path_prefix}{huggingface_repo_id_as_path}_encoder_fp32.irpa"
        )
        if not self.caching or not os.path.exists(parameters_path):
            export_encoder_iree_parameters(source_model_path, parameters_path)

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
                trace_path_prefix=f"{self.path_prefix}{huggingface_repo_id_as_path}_iree_",
            )
        )

        torch.testing.assert_close(
            reference_result, iree_result, atol=1e-4, rtol=2.0e-3
        )


class T5AttentionTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        torch.no_grad()

    def testCompareAgainstTransformersFp32(self):
        dtype = torch.float32
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
                    data=reference_model.q.weight.data
                ),
                "attn_k.weight": DefaultPrimitiveTensor(
                    data=reference_model.k.weight.data
                ),
                "attn_v.weight": DefaultPrimitiveTensor(
                    data=reference_model.v.weight.data
                ),
                "attn_o.weight": DefaultPrimitiveTensor(
                    data=reference_model.o.weight.data
                ),
                "attn_rel_b.weight": DefaultPrimitiveTensor(
                    data=reference_model.relative_attention_bias.weight.data
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
            activation_dtype=dtype,
            has_relative_attention_bias=True,
        )
        model.eval()

        hidden_states = make_rand_torch(
            shape=[batch_size, batch_seq_len, reference_config.d_model], dtype=dtype
        )
        mask = make_random_mask(shape=[batch_size, 1, 1, batch_seq_len], dtype=dtype)
        expected_outputs = reference_model(hidden_states=hidden_states, mask=mask)
        actual_outputs = model(
            hidden_states=DefaultPrimitiveTensor(data=hidden_states),
            mask=DefaultPrimitiveTensor(data=mask),
        )
        torch.testing.assert_close(actual_outputs, expected_outputs, atol=1e-5, rtol=0)

    def testCompareSelfAttentionAgainstTransformersFp32(self):
        dtype = torch.float32
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
                    data=reference_model.SelfAttention.q.weight.data
                ),
                "attn_k.weight": DefaultPrimitiveTensor(
                    data=reference_model.SelfAttention.k.weight.data
                ),
                "attn_v.weight": DefaultPrimitiveTensor(
                    data=reference_model.SelfAttention.v.weight.data
                ),
                "attn_o.weight": DefaultPrimitiveTensor(
                    data=reference_model.SelfAttention.o.weight.data
                ),
                "attn_rel_b.weight": DefaultPrimitiveTensor(
                    data=reference_model.SelfAttention.relative_attention_bias.weight.data
                ),
                "attn_norm.weight": DefaultPrimitiveTensor(
                    data=reference_model.layer_norm.weight.data
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
            activation_dtype=dtype,
            layer_norm_epsilon=reference_config.layer_norm_epsilon,
            has_relative_attention_bias=True,
        )
        model.eval()

        hidden_states = make_rand_torch(
            shape=[batch_size, batch_seq_len, reference_config.d_model], dtype=dtype
        )
        mask = make_random_mask(shape=[batch_size, 1, 1, batch_seq_len], dtype=dtype)
        position_bias = make_rand_torch(
            shape=[batch_size, reference_config.num_heads, batch_seq_len, batch_seq_len]
        )
        expected_outputs = reference_model(
            hidden_states=hidden_states,
            attention_mask=mask,
            position_bias=position_bias,
        )
        actual_outputs = model(
            hidden_states=DefaultPrimitiveTensor(data=hidden_states),
            attention_mask=DefaultPrimitiveTensor(data=mask),
            position_bias=DefaultPrimitiveTensor(data=position_bias),
        )
        actual_outputs = [
            unbox_tensor(t) if t is not None else t for t in actual_outputs
        ]
        torch.testing.assert_close(actual_outputs, expected_outputs, atol=1e-5, rtol=0)


class T5LayerFFTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        torch.no_grad()

    def testCompareAgainstTransformersFp32(self):
        dtype = torch.float32
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
                    data=reference_model.DenseReluDense.wi_0.weight
                ),
                "ffn_up.weight": DefaultPrimitiveTensor(
                    data=reference_model.DenseReluDense.wi_1.weight
                ),
                "ffn_down.weight": DefaultPrimitiveTensor(
                    data=reference_model.DenseReluDense.wo.weight
                ),
                "ffn_norm.weight": DefaultPrimitiveTensor(
                    data=reference_model.layer_norm.weight
                ),
            }
        )
        model = T5LayerFF(
            theta=theta,
            is_gated_act=reference_config.is_gated_act,
            dense_act_fn=reference_config.dense_act_fn,
            layer_norm_epsilon=reference_config.layer_norm_epsilon,
            activation_dtype=dtype,
        )

        hidden_states = make_rand_torch(
            shape=[batch_size, batch_seq_len, reference_config.d_model], dtype=dtype
        )

        expected_output = reference_model(
            hidden_states=hidden_states,
        )
        actual_output = model(
            hidden_states=DefaultPrimitiveTensor(data=hidden_states),
        )
        torch.testing.assert_close(actual_output, expected_output, atol=1e-5, rtol=0)
