# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest

import torch

from diffusers.models.transformers.transformer_flux import (
    FluxTransformerBlock,
    FluxSingleTransformerBlock,
)

from iree.turbine import aot
from sharktank.layers import (
    MMDITDoubleBlock,
    MMDITSingleBlock,
)
import sharktank.ops as ops
from sharktank.layers.testing import (
    make_mmdit_double_block_random_theta,
    make_mmdit_single_block_random_theta,
)
from sharktank.types.tensors import DefaultPrimitiveTensor
from sharktank.types.theta import torch_module_to_theta
from sharktank.utils.math import cosine_similarity


def assert_last_hidden_states_close(
    actual: torch.Tensor, expected: torch.Tensor, atol: float
):
    """The cosine similarity has been suggested to compare encoder states.
    Dehua Peng, Zhipeng Gui, Huayi Wu -
    Interpreting the Curse of Dimensionality from Distance Concentration and Manifold
    Effect (2023)
    shows that cosine and all Minkowski distances suffer from the curse of
    dimensionality.
    The cosine similarity ignores the vector magnitudes. We can probably come up with a
    better metric, but this is maybe good enough.
    """
    cosine_similarity_per_token = cosine_similarity(
        actual,
        expected,
        dim=-1,
    )
    torch.testing.assert_close(
        cosine_similarity_per_token,
        torch.ones_like(cosine_similarity_per_token),
        atol=atol,
        rtol=0,
    )


class MMDITTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)
        self.hidden_size = 3072
        self.num_heads = 24
        self.batch_size = 3
        self.dim = 8
        self.attention_head_dim = 16
        self.dtype = torch.float32

    def testFluxDoubleBlockEagerAgainstHuggingface(self):
        reference_model: FluxTransformerBlock = FluxTransformerBlock(
            self.dim, self.num_heads, self.attention_head_dim
        )

        theta = torch_module_to_theta(reference_model)
        theta.rename_tensors_to_paths()
        theta = theta.transform(functools.partial(set_float_dtype, dtype=self.dtype))
        model = MMDITDoubleBlock(theta, config)

        img = torch.rand([self.batch_size, 1024, self.hidden_size])
        txt = torch.rand([self.batch_size, 512, self.hidden_size])
        vec = torch.rand([self.batch_size, self.hidden_size])
        rot = torch.rand([self.batch_size, 1, 1536, 64, 2, 2])

        expected_outputs = reference_model(img, txt, vec, rot)
        actual_outputs = model(
            DefaultPrimitiveTensor(data=img),
            DefaultPrimitiveTensor(data=txt),
            DefaultPrimitiveTensor(data=vec),
            DefaultPrimitiveTensor(data=rot),
        )
        actual_outputs = tree_map(
            lambda t: None if t is None else ops.to(t, dtype=self.dtype),
            actual_outputs,
        )

        assert_last_hidden_states_close(
            actual_outputs["last_hidden_state"],
            expected_outputs["last_hidden_state"],
            atol=atol,
        )

    def testFluxDoubleBlockExport(self):

        theta = make_mmdit_double_block_random_theta()
        mmdit = MMDITDoubleBlock(
            theta=theta,
            num_heads=self.num_heads,
        )

        img = torch.rand([self.batch_size, 1024, self.hidden_size])
        txt = torch.rand([self.batch_size, 512, self.hidden_size])
        vec = torch.rand([self.batch_size, self.hidden_size])
        rot = torch.rand([self.batch_size, 1, 1536, 64, 2, 2])
        mmdit.forward(img, txt, vec, rot)
        fxb = aot.FxProgramsBuilder(mmdit)

        @fxb.export_program(name="mmdit", args=(img, txt, vec, rot), strict=False)
        def _(model, img, txt, vec, rot) -> torch.Tensor:
            return model.forward(img, txt, vec, rot)

        output = aot.export(fxb)
        output.verify()
        asm = str(output.mlir_module)

    def testFluxSingleBlockExport(self):

        theta = make_mmdit_single_block_random_theta()
        mmdit = MMDITSingleBlock(
            theta=theta,
            num_heads=self.num_heads,
        )

        inp = torch.rand([self.batch_size, 1024, self.hidden_size])
        vec = torch.rand([self.batch_size, self.hidden_size])
        rot = torch.rand([self.batch_size, 1, 1024, 64, 2, 2])
        mmdit.forward(inp, vec, rot)
        fxb = aot.FxProgramsBuilder(mmdit)

        @fxb.export_program(name="mmdit", args=(inp, vec, rot), strict=False)
        def _(model, inp, vec, rot) -> torch.Tensor:
            return model.forward(inp, vec, rot)

        output = aot.export(fxb)
        output.verify()
        asm = str(output.mlir_module)


if __name__ == "__main__":
    unittest.main()
