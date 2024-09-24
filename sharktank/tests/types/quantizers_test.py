# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch

from sharktank.types import *
from sharktank.types.layout_utils import saturate_cast
from sharktank.utils.testing import TempDirTestBase


class StaticScaledQuantizerTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def _roundtrip(self, it, suffix=""):
        dataset_path = self._temp_dir / f"poodoo{suffix}.irpa"
        theta = Theta([it])
        Dataset({}, theta).save(dataset_path)
        ds = Dataset.load(dataset_path)
        return ds.root_theta.tensor(it.name)

    def testPerTensorRoundtrip(self):
        ssq = StaticScaledQuantizer(
            name="poodoo",
            scale=torch.tensor(0.2, dtype=torch.float32),
            dtype=torch.float16,
        )
        ssq = self._roundtrip(ssq, "_ssq")
        self.assertIs(ssq.axis, None)
        self.assertEqual(ssq.scale, 0.2)
        self.assertEqual(ssq.reciprocal_scale, 5.0)
        self.assertIs(ssq.dtype, torch.float16)

    def testPerTensorQuantDequant(self):
        ssq = StaticScaledQuantizer(
            scale=torch.tensor(2.0, dtype=torch.float32), dtype=torch.uint8
        )
        ssq = self._roundtrip(ssq, "_ssq")
        orig_value = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
        qt_value = ssq.quantize(orig_value)
        qt_value = self._roundtrip(qt_value, "_qt_value")
        layout = qt_value.unpack()
        dequant_value = layout.dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-3, rtol=1e-3)

    def testPerTensorOffsetQuantDequant(self):
        ssq = StaticScaledQuantizer(
            scale=torch.tensor(2.0, dtype=torch.float32),
            offset=torch.tensor(8, dtype=torch.int8),
            dtype=torch.int8,
        )
        ssq = self._roundtrip(ssq, "_ssq")
        orig_value = torch.tensor([9.0, 10.0, 11.0, 12.0], dtype=torch.float16)
        qt_value = ssq.quantize(orig_value)
        qt_value = self._roundtrip(qt_value, "_qt_value")
        layout = qt_value.unpack()
        dequant_value = layout.dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-3, rtol=1e-3)

    def testPerAxisRoundtrip(self):
        ssq = StaticScaledQuantizer(
            name="poodoo",
            axis=1,
            scale=torch.tensor([0.2, 0.4, 0.8], dtype=torch.float32),
            dtype=torch.float16,
        )
        ssq = self._roundtrip(ssq, "_ssq")
        self.assertEqual(ssq.axis, 1)
        torch.testing.assert_close(ssq.scale, torch.tensor([0.2, 0.4, 0.8]))
        torch.testing.assert_close(ssq.reciprocal_scale, torch.tensor([5.0, 2.5, 1.25]))
        self.assertIs(ssq.dtype, torch.float16)

    def testPerAxisQuantDequant(self):
        ssq = StaticScaledQuantizer(
            name="poodoo",
            axis=1,
            # Note that the range of the third channel requires a smaller scale
            # to pass the test (otherwise, will saturate at ~30 for scale >= 4
            # or so).
            scale=torch.tensor([8.0, 4.0, 2.0], dtype=torch.float32),
            dtype=torch.int8,
        )
        ssq = self._roundtrip(ssq, "_ssq")
        orig_value = torch.tensor(
            [[1.0, -2.0, 3.0], [10.0, -20.0, 60.0]], dtype=torch.float16
        )
        qt_value = ssq.quantize(orig_value)
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
        torch.testing.assert_close(dequant_value, orig_value, atol=1e-3, rtol=1e-3)

    def testPerAxisOffsetQuantDequant(self):
        ssq = StaticScaledQuantizer(
            name="poodoo",
            axis=1,
            # Carefully chosen scale and offset channels that are big enough
            # to handle outliers below.
            scale=torch.tensor([8.0, 4.0, 2.0], dtype=torch.float32),
            offset=torch.tensor([16, 127, 136], dtype=torch.uint8),
            dtype=torch.uint8,
        )
        ssq = self._roundtrip(ssq, "_ssq")
        orig_value = torch.tensor(
            [[9.0, -11.0, 13.0], [18.0, -29.0, 40.0]], dtype=torch.float16
        )
        qt_value = ssq.quantize(orig_value)
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
        torch.testing.assert_close(dequant_value, orig_value, atol=1e-3, rtol=1e-3)


class DynamicScaledQuantizerTest(TempDirTestBase):
    def _roundtrip(self, it, suffix=""):
        dataset_path = self._temp_dir / f"poodoo{suffix}.irpa"
        theta = Theta([it])
        Dataset({}, theta).save(dataset_path)
        ds = Dataset.load(dataset_path)
        return ds.root_theta.tensor(it.name)

    def testQuantDequantInt(self):
        qr = DynamicScaledQuantizer(dtype=torch.int8)
        qr = self._roundtrip(qr, "_qr")
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-3)

    def testQuantDequantf16(self):
        qr = DynamicScaledQuantizer(dtype=torch.float16)
        qr = self._roundtrip(qr, "_qr")
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-3)

    def testQuantDequantf8fn(self):
        qr = DynamicScaledQuantizer(dtype=torch.float8_e4m3fn)
        qr = self._roundtrip(qr, "_qr")
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-1)

    def testQuantDequantf8fnuz(self):
        qr = DynamicScaledQuantizer(dtype=torch.float8_e4m3fnuz)
        qr = self._roundtrip(qr, "_qr")
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-1)

    def testQuarkF8Hell(self):
        # we use hardcoded values here because they're representative of actual values from a quark model
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
            ],
            dtype=torch.int8,
        )
        # mirrors dequant logic  in quark and our importer
        orig = orig.view(torch.float8_e4m3fn)
        orig = (orig.to(torch.float64) * scale).to(torch.float16)
        # Note that for fnuz  we have to do scale*2 to account for the difference between types
        # We specify the reciprocal scale explicitly to avoid adding more floating point error noise
        fnuz = StaticScaledQuantizer(
            name="dopoo",
            scale=1.0 / (scale * 2),
            reciprocal_scale=scale * 2,
            offset=None,
            dtype=torch.float8_e4m3fnuz,
        )
        fn = StaticScaledQuantizer(
            name="poodoo",
            scale=1.0 / scale,
            reciprocal_scale=scale,
            offset=None,
            dtype=torch.float8_e4m3fn,
        )
        fnuz_quant = fnuz.quantize(orig)
        fn_quant = fn.quantize(orig)

        dequant_fnuz = fnuz_quant.unpack().dequant()
        dequant_fn = fn_quant.unpack().dequant()

        # redundant asserts for sanity
        torch.testing.assert_close(
            orig.to(torch.float16), dequant_fnuz, atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            orig.to(torch.float16), dequant_fn, atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(dequant_fnuz, dequant_fn, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
