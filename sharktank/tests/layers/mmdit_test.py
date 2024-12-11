# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest
import functools

import torch

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
from sharktank.transforms.dataset import set_float_dtype
from sharktank.types.tensors import DefaultPrimitiveTensor
from sharktank.types.theta import Dataset, Theta
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
        self.dtype = torch.float32

    def testFluxDoubleBlockGolden(self):
        theta = Dataset.load(
            "/home/kherndon/goldens/flux.irpa", file_type="irpa"
        ).root_theta
        theta.rename_tensors_to_paths()
        dbl_blks = theta.tensor("double_blocks")
        blk = dbl_blks["0"]
        model = MMDITDoubleBlock(Theta(blk), self.num_heads)

        img = torch.load("/home/kherndon/goldens/img.pt", weights_only=True)
        txt = torch.load("/home/kherndon/goldens/txt.pt", weights_only=True)
        vec = torch.load("/home/kherndon/goldens/vec.pt", weights_only=True)
        rot = torch.load("/home/kherndon/goldens/rot.pt", weights_only=True)

        expected_out1, expected_out2 = torch.load(
            "/home/kherndon/goldens/out1.pt", weights_only=True
        ), torch.load("/home/kherndon/goldens/out2.pt", weights_only=True)
        img_qkv, img_mod = torch.load(
            "/home/kherndon/goldens/img_qkv.pt", weights_only=True
        ), torch.load("/home/kherndon/goldens/img_modulated.pt", weights_only=True)
        img_mod2 = torch.load(
            "/home/kherndon/goldens/img_modulated2.pt", weights_only=True
        )
        eim1s, eim1c = torch.load("/home/kherndon/goldens/im1s.pt"), torch.load(
            "/home/kherndon/goldens/im1c.pt"
        )
        eq, ek, ev = (
            torch.load("/home/kherndon/goldens/q.pt", weights_only=True),
            torch.load("/home/kherndon/goldens/k.pt", weights_only=True),
            torch.load("/home/kherndon/goldens/v.pt", weights_only=True),
        )
        out1, out2, mod, mod2 = model(
            DefaultPrimitiveTensor(data=img),
            DefaultPrimitiveTensor(data=txt),
            DefaultPrimitiveTensor(data=vec),
            DefaultPrimitiveTensor(data=rot),
        )

        atol = 1e-2
        torch.testing.assert_close(mod2, img_mod2, atol=0.1, rtol=0.1)
        torch.testing.assert_close(qkv, img_qkv, atol=0.1, rtol=0.1)
        torch.testing.assert_close(expected_out1, out1, atol=0.1, rtol=0.1)
        assert_last_hidden_states_close(
            expected_out1,
            out1,
            atol=atol,
        )

        assert_last_hidden_states_close(
            expected_out2,
            out2,
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
