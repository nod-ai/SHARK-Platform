# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# For testing dataset transforms, we like to use an example CLI tool, invoked
# programmatically to test the entire interaction. See examples under the
# examples/ directory.

from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from sharktank.types import *


class TransformTestBase(unittest.TestCase):
    def setUp(self):
        self._temp_dir = Path(tempfile.mkdtemp("transform.test"))

    def tearDown(self):
        shutil.rmtree(self._temp_dir)

    def get_irpa_path(self, name: str) -> Path:
        return self._temp_dir / f"{name}.irpa"

    def save_dataset(self, ds: Dataset, name: str) -> Path:
        p = self.get_irpa_path(name)
        ds.save(p)
        return p

    def run_main(self, main_func, *args):
        new_args = [str(arg) for arg in args]
        main_func(new_args)


class MmtRHSShardingTransformTest(TransformTestBase):
    def testPrimitive(self):
        orig_pts = [
            DefaultPrimitiveTensor("blk.1.attn_k.weight", torch.randn([32, 128])),
            DefaultPrimitiveTensor("blk.2.attn_q.weight", torch.randn([48, 64])),
            DefaultPrimitiveTensor("other", torch.randn([2, 2])),
        ]
        ds_orig = Dataset({}, Theta(orig_pts))
        input_path = self.save_dataset(ds_orig, "input")
        output_path = self.get_irpa_path("output")
        from sharktank.examples.sharding import shard_llm_dataset

        self.run_main(
            shard_llm_dataset.main,
            "--irpa-file",
            input_path,
            "--output-irpa-file",
            output_path,
            "--num-shards",
            8,
        )
        ds_tran = Dataset.load(output_path, mmap=False)

        # Verify.
        flat_sts = ds_tran.root_theta.flatten()
        self.assertEqual(3, len(flat_sts))
        st_1 = flat_sts["blk.1.attn_k.weight"]
        st_2 = flat_sts["blk.2.attn_q.weight"]
        pt_3 = flat_sts["other"]
        self.assertIsInstance(st_1, ShardedPrimitiveTensor)
        self.assertIsInstance(st_2, ShardedPrimitiveTensor)
        self.assertIsInstance(pt_3, DefaultPrimitiveTensor)
        self.assertListEqual(st_1.shape, [32, 128])
        self.assertListEqual(st_2.shape, [48, 64])

        # Verify component shapes for st_1.
        self.assertEqual(8, len(st_1.shards))
        self.assertTrue(all(pt.shape == [32, 16] for pt in st_1.shards))
        self.assertTrue(
            all(list(pt.as_torch().shape) == [32, 16] for pt in st_1.shards)
        )

        # Verify component shapes for st_2.
        self.assertEqual(8, len(st_2.shards))
        self.assertTrue(all(pt.shape == [48, 8] for pt in st_2.shards))
        self.assertTrue(all(list(pt.as_torch().shape) == [48, 8] for pt in st_2.shards))

        # Verify contents for one shard for sanity.
        new_t = st_1.shards[0].as_torch()
        torch.testing.assert_close(new_t, orig_pts[0].as_torch().split(16, dim=1)[0])


if __name__ == "__main__":
    unittest.main()
