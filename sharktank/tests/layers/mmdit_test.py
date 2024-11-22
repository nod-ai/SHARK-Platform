# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest

import torch

from iree.turbine import aot
from sharktank.layers import (
    MMDITDoubleBlock,
)
import sharktank.ops as ops
from sharktank.layers.testing import (
    make_mmdit_block_theta,
)
from sharktank.types.tensors import DefaultPrimitiveTensor


class MMDITTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)
        self.hidden_size = 3072
        self.num_heads = 24
        self.block_seqlen = 7
        self.block_seq_stride = 17
        self.max_seqlen = self.block_seq_stride * self.block_seqlen
        self.batch_size = 3

    def testExport(self):
        dtype = torch.float32

        txt_ids = torch.rand([self.batch_size, 3, self.max_seqlen, 3])
        img_ids = torch.rand([self.batch_size, 3, self.max_seqlen, 3])
        pe_dim = self.hidden_size // self.num_heads
        axes_dim = [16, 56, 56]
        theta = 10000

        theta = make_mmdit_block_theta()
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


if __name__ == "__main__":
    unittest.main()
