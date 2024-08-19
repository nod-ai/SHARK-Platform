# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch
import tempfile
import os

from sharktank.types import *
from sharktank import ops


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
        pqt1 = PlanarQuantizedTensor(
            name="t1", shape=[128, 1024], layout=_createTestLayout()
        )

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


class ShardedTensorTest(unittest.TestCase):
    def testReplicatedTensorSaveLoad(self):
        tensor = torch.rand([2, 3, 4], dtype=torch.float32)
        replicated_tensor = ReplicatedTensor(
            ts=tensor, shard_count=3, name="the_tensor"
        )
        theta = Theta([replicated_tensor])
        dataset = Dataset({}, theta)
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "dataset.irpa")
            dataset.save(file_path)
            # TODO: figure out why when memory mapping (mmap=True) even when deleting
            # the Python objects the underlying files are still open causing
            # TemporaryDirectory cleanup to fail under Windows.
            loaded_dataset = Dataset.load(file_path, mmap=False)
            loaded_replicated_tensor = loaded_dataset.root_theta.tensor("the_tensor")
            assert replicated_tensor.is_deep_equal(loaded_replicated_tensor)

    def testShardedPrimitiveTensorSaveLoad(self):
        tensor = torch.rand([2, 6, 4], dtype=torch.float32)
        sharded_tensor = SplitPrimitiveTensor(
            ts=tensor, shard_count=3, name="the_tensor", shard_dim=1
        )
        theta = Theta([sharded_tensor])
        dataset = Dataset({}, theta)
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "dataset.irpa")
            dataset.save(file_path)
            loaded_dataset = Dataset.load(file_path, mmap=False)
            loaded_sharded_tensor = loaded_dataset.root_theta.tensor("the_tensor")
            assert sharded_tensor.is_deep_equal(loaded_sharded_tensor)

    def testUnreducedTensorSaveLoad(self):
        tensor = torch.rand([2, 6, 4], dtype=torch.float32)
        sharded_tensor = UnreducedTensor(
            ts=torch.split(tensor, 1, dim=1), name="the_tensor"
        )
        theta = Theta([sharded_tensor])
        dataset = Dataset({}, theta)
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "dataset.irpa")
            dataset.save(file_path)
            loaded_dataset = Dataset.load(file_path, mmap=False)
            loaded_sharded_tensor = loaded_dataset.root_theta.tensor("the_tensor")
            assert sharded_tensor.is_deep_equal(loaded_sharded_tensor)

    def testReplicatedTensorExtractSlice(self):
        tensor = torch.rand([2, 3, 4], dtype=torch.float32)
        replicated_tensor = ReplicatedTensor(ts=tensor, shard_count=3)
        s = [slice(1, 2), slice(0, 3, 2), None]
        expected_result = tensor[s]
        replicated_sliced_tensor = replicated_tensor[s]
        assert isinstance(replicated_sliced_tensor, ReplicatedTensor)
        actual_result = ops.reshard_like(replicated_sliced_tensor, expected_result)
        assert ops.equal(expected_result, actual_result)

    def testReplicatedTensorExtractElement(self):
        tensor = torch.rand([2, 3, 4], dtype=torch.float32)
        replicated_tensor = ReplicatedTensor(ts=tensor, shard_count=3)
        idx = (
            1,
            2,
            3,
        )
        expected_result = tensor[idx]
        replicated_result = replicated_tensor[idx]
        assert isinstance(replicated_result, ReplicatedTensor)
        actual_result = ops.reshard_like(replicated_result, expected_result)
        assert ops.equal(expected_result, actual_result)

    def testSplitTensorExtractSliceOfNonSplitDim(self):
        tensor = torch.rand([5, 6], dtype=torch.float32)
        sharded_tensor = SplitPrimitiveTensor(ts=tensor, shard_count=3, shard_dim=1)
        s = [slice(0, 2), slice(None), None, None]
        expected_result = tensor[s]
        sharded_slice = sharded_tensor[s]
        assert isinstance(sharded_slice, SplitPrimitiveTensor)
        actual_result = ops.reshard_like(sharded_slice, expected_result)
        assert ops.equal(expected_result, actual_result)


if __name__ == "__main__":
    unittest.main()
