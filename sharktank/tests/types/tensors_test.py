# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch

from sharktank.types import *


def _createTestLayout():
    n = 128
    k = 1024
    bs = 32

    return BlockScaledLayout(
        [n, k],
        d=torch.empty(n, k // bs, 1, dtype=torch.float32),
        qs=torch.empty(n, k // bs, bs, dtype=torch.int8),
        m=torch.empty(n, k // bs, bs, dtype=torch.float32),
    )


class PlanarQuantizedTensorTest(unittest.TestCase):
    def testTransform(self):
        pqt1 = PlanarQuantizedTensor("t1", [128, 1024], _createTestLayout())

        def transform1(d):
            new_d = {}
            for k, t in d.items():
                if k.endswith(":qs"):
                    t = t.to(torch.int16)
                new_d[k] = t
            return new_d

        def transform2(d):
            new_d = {}
            for k, t in d.items():
                if k.endswith(":d") or k.endswith(":m"):
                    t = t.to(torch.float16)
                new_d[k] = t
            return new_d

        pqt2 = pqt1.transform_globals(transform1, transform2)
        self.assertIsNot(pqt1, pqt2)
        print(pqt2)
        self.assertEqual(pqt2.name, pqt1.name)
        self.assertEqual(pqt2.shape, pqt1.shape)
        new_planes = pqt2.layout.planes
        self.assertEqual(new_planes["qs"].dtype, torch.int16)
        self.assertEqual(new_planes["m"].dtype, torch.float16)
        self.assertEqual(new_planes["d"].dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
