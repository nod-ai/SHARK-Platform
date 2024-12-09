# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from parameterized import parameterized
import pytest
import torch
from torch.utils._pytree import tree_map
from typing import Optional
from unittest import TestCase
import transformers
from transformers import CLIPTextModel as TransformersCLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import (
    CLIPAttention as TransformersCLIPAttention,
    CLIPEncoderLayer as TransformersCLIPEncoderLayer,
    CLIPEncoder as TransformersCLIPEncoder,
)

from sharktank.types import DefaultPrimitiveTensor
from sharktank.transforms.dataset import set_float_dtype
from sharktank.utils.hf_datasets import get_dataset
from sharktank.utils.math import cosine_similarity
from sharktank.utils.testing import (
    make_rand_torch,
    make_random_mask,
    TempDirTestBase,
    test_prompts,
)
from sharktank.models.clip.export import (
    export_clip_text_model_dataset_from_hugging_face,
    transformers_clip_attention_to_theta,
    transformers_clip_encoder_layer_to_theta,
    transformers_clip_encoder_to_theta,
    transformers_clip_text_model_to_theta,
)
from sharktank.models.clip import (
    ClipAttention,
    ClipEncoderLayer,
    ClipEncoder,
    ClipTextModel,
)
from sharktank.layers.configs.llm_configs import ClipTextConfig
from sharktank import ops

with_clip_data = pytest.mark.skipif("not config.getoption('with_clip_data')")


@pytest.mark.usefixtures("path_prefix")
class ClipExportTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        if self.path_prefix is None:
            self.path_prefix = f"{self._temp_dir}/"

    @with_clip_data
    def testSmokeExportLargeF32FromHuggingFace(self):
        repo_id = "openai/clip-vit-large-patch14"
        get_dataset(
            repo_id,
        ).download()
        output_path = f"{self.path_prefix}{repo_id.replace('/', '--')}.irpa"
        export_clip_text_model_dataset_from_hugging_face(repo_id, output_path)


@pytest.mark.usefixtures("get_model_artifacts")
class ClipTextEagerTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        torch.no_grad()

    def runTestCompareTorchEagerAgainstHuggingFace(
        self,
        huggingface_repo_id: str,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: float,
    ):
        """Compares the last hidden states with the cosine similarity metric.
        This metric is sensible as the outputs are the result of layer normalization.
        The angle between the vectors would indicate how close they are."""
        get_dataset(
            huggingface_repo_id,
        ).download()

        reference_model: TransformersCLIPTextModel = (
            TransformersCLIPTextModel.from_pretrained(
                huggingface_repo_id, torch_dtype=reference_dtype
            )
        )

        theta = transformers_clip_text_model_to_theta(reference_model)
        theta.rename_tensors_to_paths()
        theta = theta.transform(functools.partial(set_float_dtype, dtype=target_dtype))
        config = ClipTextConfig.from_transformers_clip_text_config(
            reference_model.config
        )
        model = ClipTextModel(theta, config)

        tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            huggingface_repo_id,
            max_length=reference_model.config.max_position_embeddings,
        )
        input_ids = tokenizer(
            test_prompts,
            truncation=True,
            max_length=reference_model.config.max_position_embeddings,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]

        expected_outputs = reference_model(input_ids=input_ids)
        actual_outputs = model(input_ids=DefaultPrimitiveTensor(data=input_ids))
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        cosine_similarity_per_token = cosine_similarity(
            actual_outputs["last_hidden_state"],
            expected_outputs["last_hidden_state"],
            dim=-1,
        )
        torch.testing.assert_close(
            cosine_similarity_per_token,
            torch.ones_like(cosine_similarity_per_token),
            atol=atol,
            rtol=0,
        )

    @with_clip_data
    def testLargeCompareTorchEagerF32AgainstHuggingFaceF32(self):
        self.runTestCompareTorchEagerAgainstHuggingFace(
            "openai/clip-vit-large-patch14",
            reference_dtype=torch.float32,
            target_dtype=torch.float32,
            atol=1e-5,
        )

    @with_clip_data
    def testLargeCompareTorchEagerBf16AgainstHuggingFaceF32(self):
        self.runTestCompareTorchEagerAgainstHuggingFace(
            "openai/clip-vit-large-patch14",
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
            atol=1e-3,
        )

    @parameterized.expand(
        [
            [torch.float32, torch.float32],
            [torch.bfloat16, torch.bfloat16, 4e-2, 1.6e-2],
            [torch.float32, torch.bfloat16, 4e-2, 1.6e-2],
        ]
    )
    def testCompareEagerToySizedModelAgainstTransformers(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        torch.set_default_dtype(reference_dtype)
        batch_size = 19
        tgt_len = 23
        num_attention_heads = 5
        vocab_size = 11
        reference_config = transformers.CLIPTextConfig(
            vocab_size=vocab_size,
            hidden_size=13 * num_attention_heads,
            intermediate_size=7,
            projection_dim=3,
            num_attention_heads=num_attention_heads,
            layer_norm_eps=1e-4,
            num_hidden_layers=2,
            final_layer_norm=1e-3,
            bos_token_id=vocab_size - 2,
            eos_token_id=vocab_size - 1,
        )
        reference_model = TransformersCLIPTextModel(
            reference_config,
        )
        reference_model.eval()

        theta = transformers_clip_text_model_to_theta(reference_model)
        theta.rename_tensors_to_paths()
        theta = theta.transform(functools.partial(set_float_dtype, dtype=target_dtype))
        config = ClipTextConfig.from_transformers_clip_text_config(reference_config)
        model = ClipTextModel(theta, config)

        input_ids = torch.randint(low=0, high=vocab_size, size=[batch_size, tgt_len])

        expected_outputs = reference_model(input_ids=input_ids)

        actual_outputs = model(input_ids=DefaultPrimitiveTensor(data=input_ids))
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        torch.testing.assert_close(
            actual_outputs, expected_outputs, atol=atol, rtol=rtol
        )


class ClipAttentionTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        torch.no_grad()

    @parameterized.expand(
        [
            [torch.float32, torch.float32],
            # Default values are not enough because torch.nn.Linear does fused
            # multiply-add, while our implementation is decomposed.
            # There may be other source of discrepancy.
            [torch.bfloat16, torch.bfloat16, 0.5e-2, 1.6e-2],
            [torch.float32, torch.bfloat16, 1e-2, 1.6e-2],
        ]
    )
    def testCompareEagerToySizedModelAgainstTransformers(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        torch.set_default_dtype(reference_dtype)
        batch_size = 19
        tgt_len = 23
        src_len = tgt_len
        num_attention_heads = 2
        reference_config = transformers.CLIPTextConfig(
            vocab_size=11,
            hidden_size=13 * num_attention_heads,
            intermediate_size=7,
            projection_dim=3,
            num_attention_heads=num_attention_heads,
        )
        reference_model = TransformersCLIPAttention(
            reference_config,
        )
        reference_model.eval()

        theta = transformers_clip_attention_to_theta(reference_model)
        theta.rename_tensors_to_paths()
        theta = theta.transform(functools.partial(set_float_dtype, dtype=target_dtype))
        config = ClipTextConfig.from_transformers_clip_text_config(reference_config)
        model = ClipAttention(theta, config)

        reference_hidden_states = make_rand_torch(
            shape=[batch_size, tgt_len, reference_config.hidden_size],
            dtype=reference_dtype,
        )
        reference_attention_mask = make_random_mask(
            shape=[batch_size, 1, tgt_len, src_len], dtype=reference_dtype
        )
        reference_causal_attention_mask = make_random_mask(
            shape=[batch_size, 1, tgt_len, src_len], dtype=reference_dtype
        )
        expected_outputs = reference_model(
            hidden_states=reference_hidden_states,
            attention_mask=reference_attention_mask,
            causal_attention_mask=reference_causal_attention_mask,
        )

        hidden_states = ops.to(reference_hidden_states, dtype=target_dtype)
        attention_mask = ops.to(reference_attention_mask, dtype=target_dtype)
        causal_attention_mask = ops.to(
            reference_causal_attention_mask, dtype=target_dtype
        )
        actual_outputs = model(
            hidden_states=DefaultPrimitiveTensor(data=hidden_states),
            attention_mask=DefaultPrimitiveTensor(data=attention_mask),
            causal_attention_mask=DefaultPrimitiveTensor(data=causal_attention_mask),
        )
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        torch.testing.assert_close(
            actual_outputs, expected_outputs, atol=atol, rtol=rtol
        )


class ClipEncoderLayerTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        torch.no_grad()

    @parameterized.expand(
        [
            [torch.float32, torch.float32],
            [torch.bfloat16, torch.bfloat16, 1e-2, 1.6e-2],
            [torch.float32, torch.bfloat16, 1e-2, 1.6e-2],
        ]
    )
    def testCompareEagerToySizedModelAgainstTransformers(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        torch.set_default_dtype(reference_dtype)
        batch_size = 19
        tgt_len = 23
        src_len = tgt_len
        num_attention_heads = 2
        reference_config = transformers.CLIPTextConfig(
            vocab_size=11,
            hidden_size=13 * num_attention_heads,
            intermediate_size=7,
            projection_dim=3,
            num_attention_heads=num_attention_heads,
            layer_norm_eps=1e-4,
        )
        reference_model = TransformersCLIPEncoderLayer(
            reference_config,
        )
        reference_model.eval()

        theta = transformers_clip_encoder_layer_to_theta(reference_model)
        theta.rename_tensors_to_paths()
        theta = theta.transform(functools.partial(set_float_dtype, dtype=target_dtype))
        config = ClipTextConfig.from_transformers_clip_text_config(reference_config)
        model = ClipEncoderLayer(theta, config)

        reference_hidden_states = make_rand_torch(
            shape=[batch_size, tgt_len, reference_config.hidden_size],
            dtype=reference_dtype,
        )
        reference_attention_mask = make_random_mask(
            shape=[batch_size, 1, tgt_len, src_len], dtype=reference_dtype
        )
        reference_causal_attention_mask = make_random_mask(
            shape=[batch_size, 1, tgt_len, src_len], dtype=reference_dtype
        )
        expected_outputs = reference_model(
            hidden_states=reference_hidden_states,
            attention_mask=reference_attention_mask,
            causal_attention_mask=reference_causal_attention_mask,
        )

        hidden_states = ops.to(reference_hidden_states, dtype=target_dtype)
        attention_mask = ops.to(reference_attention_mask, dtype=target_dtype)
        causal_attention_mask = ops.to(
            reference_causal_attention_mask, dtype=target_dtype
        )
        actual_outputs = model(
            hidden_states=DefaultPrimitiveTensor(data=hidden_states),
            attention_mask=DefaultPrimitiveTensor(data=attention_mask),
            causal_attention_mask=DefaultPrimitiveTensor(data=causal_attention_mask),
        )
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        torch.testing.assert_close(
            actual_outputs, expected_outputs, atol=atol, rtol=rtol
        )


class ClipEncoderTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        torch.no_grad()

    @parameterized.expand(
        [
            [torch.float32, torch.float32],
            [torch.bfloat16, torch.bfloat16, 2e-2, 1.6e-2],
            [torch.float32, torch.bfloat16, 2e-2, 1.6e-2],
        ]
    )
    def testCompareEagerToySizedModelAgainstTransformers(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        torch.set_default_dtype(reference_dtype)
        batch_size = 19
        tgt_len = 23
        src_len = tgt_len
        num_attention_heads = 5
        reference_config = transformers.CLIPTextConfig(
            vocab_size=11,
            hidden_size=13 * num_attention_heads,
            intermediate_size=7,
            projection_dim=3,
            num_attention_heads=num_attention_heads,
            layer_norm_eps=1e-4,
            num_hidden_layers=2,
        )
        reference_model = TransformersCLIPEncoder(
            reference_config,
        )
        reference_model.eval()

        theta = transformers_clip_encoder_to_theta(reference_model)
        theta.rename_tensors_to_paths()
        theta = theta.transform(functools.partial(set_float_dtype, dtype=target_dtype))
        config = ClipTextConfig.from_transformers_clip_text_config(reference_config)
        model = ClipEncoder(theta, config)

        reference_inputs_embeds = make_rand_torch(
            shape=[batch_size, tgt_len, reference_config.hidden_size],
            dtype=reference_dtype,
        )
        reference_attention_mask = make_random_mask(
            shape=[batch_size, 1, tgt_len, src_len], dtype=reference_dtype
        )
        reference_causal_attention_mask = make_random_mask(
            shape=[batch_size, 1, tgt_len, src_len], dtype=reference_dtype
        )
        expected_outputs = reference_model(
            inputs_embeds=reference_inputs_embeds,
            attention_mask=reference_attention_mask,
            causal_attention_mask=reference_causal_attention_mask,
        )

        inputs_embeds = ops.to(reference_inputs_embeds, dtype=target_dtype)
        attention_mask = ops.to(reference_attention_mask, dtype=target_dtype)
        causal_attention_mask = ops.to(
            reference_causal_attention_mask, dtype=target_dtype
        )
        actual_outputs = model(
            inputs_embeds=DefaultPrimitiveTensor(data=inputs_embeds),
            attention_mask=DefaultPrimitiveTensor(data=attention_mask),
            causal_attention_mask=DefaultPrimitiveTensor(data=causal_attention_mask),
        )
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=reference_dtype),
            actual_outputs,
        )

        torch.testing.assert_close(
            actual_outputs, expected_outputs, atol=atol, rtol=rtol
        )
