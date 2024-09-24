# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from typing import List

import torch
from iree.turbine.aot import *
from sharktank.types import *
from sharktank.models.llama.testing import make_moe_block_theta, make_rand_torch
from sharktank.layers.linear import LinearLayer
from sharktank import ops


class NativeFP8Test(unittest.TestCase):
    def test(self):
        scale = torch.tensor(0.0118, dtype=torch.float64)
        orig = torch.tensor(
            [
                -58,
                -48,
                -70,
                53,
                -53,
                76,
                -71,
                -90,
                50,
                77,
                62,
                -98,
                66,
                -54,
                55,
                -80,
                -66,
                -62,
                -61,
                -56,
                56,
                -67,
                79,
                -60,
                -71,
                42,
                72,
                -73,
                91,
                63,
                124,
                -128,
                -128,
                -128,
                -128,
            ],
            dtype=torch.int8,
        )
        # mirrors dequant logic  in quark and our importer
        orig = orig.view(torch.float8_e4m3fn)
        orig = (orig.to(torch.float64) * scale).to(torch.float16)
        orig = orig.reshape(1, 35)
        orig = torch.cat([orig.clone() for x in range(100)], dim=0)
        # Note that for fnuz  we have to do scale*2 to account for the difference between types
        # We specify the reciprocal scale explicitly to avoid adding more floating point error noise
        fnuz = StaticScaledQuantizer(
            name="dopoo",
            scale=1.0 / (scale * 2),
            reciprocal_scale=scale * 2,
            offset=None,
            dtype=torch.float8_e4m3fnuz,
        )
        quant_weight = fnuz.quantize(orig)
        theta = {
            "weight": quant_weight,
            #   "q_input":fnuz,
        }
        theta = Theta(theta)
        ds = Dataset({"hi": "what?"}, theta)
        ds.save("test.irpa", io_report_callback=print)
        model = LinearLayer(
            theta,
            weight_name="weight",
            dequantize_output=False,
        )
        fxb = FxProgramsBuilder(model)
        input = quant_weight.unpack().qs  # make_rand_torch((1, 35))

        @fxb.export_program(name="linearF8Test", args=(input,), strict=False)
        def _(model, input: torch.Tensor) -> torch.Tensor:
            return model(input)

        output = export(fxb)
        out.save("test.mlir")


if __name__ == "__main__":
    unittest.main()
