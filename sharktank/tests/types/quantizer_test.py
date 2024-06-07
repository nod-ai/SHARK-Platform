# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch

from sharktank.types import *
from sharktank.utils.testing import TempDirTestBase


class StaticScaledQuantizerTest(TempDirTestBase):
    def _roundtrip(self, it):
        dataset_path = self._temp_dir / "poodoo.irpa"
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
        ssq = self._roundtrip(ssq)
        self.assertIs(ssq.axis, None)
        self.assertEqual(ssq.scale, 0.2)
        self.assertEqual(ssq.reciprocal_scale, 5.0)
        self.assertIs(ssq.dtype, torch.float16)

    def testPerTensorQuantDequant(self):
        ssq = StaticScaledQuantizer(
            scale=torch.tensor(0.2, dtype=torch.float32), dtype=torch.float16
        )
        ssq = self._roundtrip(ssq)

        orig_value = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        qt_value = ssq.quantize(orig_value)
        layout = qt_value.unpack()
        expected_quant_value = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float16)
        torch.testing.assert_close(layout.planes["qs"], expected_quant_value)
        dequant_value = layout.dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-3, rtol=1e-3)

    def testPerTensorOffsetQuantDequant(self):
        ssq = StaticScaledQuantizer(
            scale=torch.tensor(0.2, dtype=torch.float32),
            offset=torch.tensor(8.0, dtype=torch.float32),
            dtype=torch.float16,
        )
        ssq = self._roundtrip(ssq)
        orig_value = torch.tensor([9.0, 10.0, 11.0, 12.0], dtype=torch.float32)
        qt_value = ssq.quantize(orig_value)
        layout = qt_value.unpack()
        expected_quant_value = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float16)
        qs = layout.planes["qs"]
        torch.testing.assert_close(qs, expected_quant_value)
        dequant_value = layout.dequant()
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-3, rtol=1e-3)

    def testPerAxisRoundtrip(self):
        ssq = StaticScaledQuantizer(
            name="poodoo",
            axis=1,
            scale=torch.tensor([0.2, 0.4, 0.8], dtype=torch.float32),
            dtype=torch.float16,
        )
        ssq = self._roundtrip(ssq)
        self.assertEqual(ssq.axis, 1)
        torch.testing.assert_close(ssq.scale, torch.tensor([0.2, 0.4, 0.8]))
        torch.testing.assert_close(ssq.reciprocal_scale, torch.tensor([5.0, 2.5, 1.25]))
        self.assertIs(ssq.dtype, torch.float16)

    def testPerAxisQuantDequant(self):
        ssq = StaticScaledQuantizer(
            name="poodoo",
            axis=1,
            scale=torch.tensor([0.2, 0.4, 0.8], dtype=torch.float32),
            dtype=torch.float16,
        )
        ssq = self._roundtrip(ssq)
        orig_value = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        qt_value = ssq.quantize(orig_value)
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        torch.testing.assert_close(
            qs,
            torch.tensor(
                [[0.2000, 0.7998, 2.4004], [2.0000, 8.0000, 24.0000]],
                dtype=torch.float16,
            ),
        )
        dequant_value = layout.dequant()
        torch.testing.assert_close(dequant_value, orig_value, atol=1e-3, rtol=1e-3)

    def testPerAxisOffsetQuantDequant(self):
        ssq = StaticScaledQuantizer(
            name="poodoo",
            axis=1,
            scale=torch.tensor([0.2, 0.4, 0.8], dtype=torch.float32),
            offset=torch.tensor([8.0, 9.0, 10.0], dtype=torch.float32),
            dtype=torch.float16,
        )
        ssq = self._roundtrip(ssq)
        orig_value = torch.tensor([[9.0, 11.0, 13.0], [18.0, 29.0, 40.0]])
        qt_value = ssq.quantize(orig_value)
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        torch.testing.assert_close(
            qs,
            torch.tensor(
                [[0.2000, 0.7998, 2.4004], [2.0000, 8.0000, 24.0000]],
                dtype=torch.float16,
            ),
        )
        dequant_value = layout.dequant()
        torch.testing.assert_close(dequant_value, orig_value, atol=1e-3, rtol=1e-3)


class DynamicScaledQuantizerTest(TempDirTestBase):
    def _roundtrip(self, it):
        dataset_path = self._temp_dir / "poodoo.irpa"
        theta = Theta([it])
        Dataset({}, theta).save(dataset_path)
        ds = Dataset.load(dataset_path)
        return ds.root_theta.tensor(it.name)

    def testQuantDequantInt(self):
        qr = DynamicScaledQuantizer(dtype=torch.int8)
        qr = self._roundtrip(qr)
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        layout = qt_value.unpack()
        expected_quant_value = torch.tensor([-127, -50, 76, 114], dtype=torch.int8)
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
        print("i8 QS:", qs)
        print("i8 DQ:", dequant_value)
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-3)
        torch.testing.assert_close(qs, expected_quant_value)

    def testQuantDequantf16(self):
        qr = DynamicScaledQuantizer(dtype=torch.float16)
        qr = self._roundtrip(qr)
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        layout = qt_value.unpack()
        expected_quant_value = torch.tensor(
            [-65504.0, -26208.0, 39296.0, 58944.0], dtype=torch.float16
        )
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
        print("f16 QS:", qs)
        print("f16 DQ:", dequant_value)
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-3)
        torch.testing.assert_close(qs, expected_quant_value)

    def testQuantDequantf8fn(self):
        qr = DynamicScaledQuantizer(dtype=torch.float8_e4m3fn)
        qr = self._roundtrip(qr)
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
        print("f8fn QS:", qs)
        print("f8fn DQ:", dequant_value)
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-1)

    def testQuantDequantf8fnuz(self):
        qr = DynamicScaledQuantizer(dtype=torch.float8_e4m3fnuz)
        qr = self._roundtrip(qr)
        orig_value = torch.tensor([-5.0, -2.0, 3.0, 4.5], dtype=torch.float32)
        qt_value = qr.quantize(orig_value)
        layout = qt_value.unpack()
        qs = layout.planes["qs"]
        dequant_value = layout.dequant()
        print("f8fnuz QS:", qs)
        print("f8fnuz DQ:", dequant_value)
        torch.testing.assert_close(orig_value, dequant_value, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    unittest.main()
