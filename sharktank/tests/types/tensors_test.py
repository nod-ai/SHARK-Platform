# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch
import tempfile
import os
import pytest
from collections import OrderedDict

from sharktank.types import *
from sharktank.utils.iree import (
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    call_torch_module_function,
    iree_to_torch,
)
from sharktank import ops
from copy import deepcopy
from iree.turbine.aot import FxProgramsBuilder, export


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


@pytest.mark.usefixtures("path_prefix")
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

    def testSplitTensorExtractSliceWithEllipsis(self):
        tensor = torch.rand([2, 3, 4, 5])
        sharded_tensor = ops.reshard_split(tensor, dim=2, count=2)
        expected_result = tensor[0, ..., 1:3]
        expected_sharded_result = ops.reshard_split(expected_result, dim=1, count=2)
        actual_sharded_result = sharded_tensor[0, ..., 1:3]
        assert ops.equal(actual_sharded_result, expected_sharded_result)

    def testSplitTensorInsertSliceOfAllDimsWithEllipsis(self):
        dst = torch.rand([2, 3, 4])
        src = torch.rand([2, 3, 4])
        sharded_dst = ops.reshard_split(dst.clone(), dim=1, count=3)
        sharded_src = ops.reshard_like(src, like=sharded_dst)
        dst[...] = src
        sharded_dst[...] = sharded_src
        actual_result = ops.unshard(sharded_dst)
        assert ops.equal(actual_result, dst)

    def testSplitTensorInsertSliceWithEllipsis(self):
        dst = torch.rand([2, 3, 4, 5])
        src = torch.rand([3, 4, 2])
        sharded_dst = ops.reshard_split(dst.clone(), dim=2, count=2)
        sharded_src = ops.reshard_split(src, dim=1, count=2)
        dst[0, ..., 1:3] = src
        sharded_dst[0, ..., 1:3] = sharded_src
        actual_result = ops.unshard(sharded_dst)
        assert ops.equal(actual_result, dst)

    def testInPlaceUpdate(self):
        if self.path_prefix is not None:
            self.runTestInPlaceUpdate(path_prefix=self.path_prefix, dump_enabled=True)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                self.runTestInPlaceUpdate(
                    path_prefix=f"{temp_dir}/", dump_enabled=False
                )

    def runTestInPlaceUpdate(self, path_prefix: str, dump_enabled: bool):
        shard_dim = 2
        shard_count = 2

        class Module(torch.nn.Module):
            def main(self, tensor: AnyTensor):
                tensor += 1
                # TODO: figure out why when not returning anything fails the export
                # fails.
                return torch.empty([1])

        shape = [2, 3, 4]
        tensor = torch.rand(shape)
        sharded_tensor = SplitPrimitiveTensor(
            ts=tensor,
            shard_dim=shard_dim,
            shard_count=shard_count,
            insert_device_assignment=False,
        )

        # Avoid aliasing with tensor.
        # Torch exporting complains about mutating an aliased input.
        # Doing
        # sharded_tensor = deepcopy(sharded_tensor)
        # is not enough.
        shards = [
            torch.empty_like(unbox_tensor(shard)) for shard in sharded_tensor.shards
        ]
        for src_shard, dst_shard in zip(sharded_tensor.shards, shards):
            dst_shard[...] = unbox_tensor(src_shard)
        sharded_tensor = SplitPrimitiveTensor(
            ts=shards, shard_dim=shard_dim, insert_device_assignment=False
        )

        sharded_tensor_snapshot = deepcopy(sharded_tensor)
        module = Module()
        module.main(sharded_tensor)
        actual_result = ops.unshard(sharded_tensor)
        expected_result = tensor + 1
        assert ops.equal(expected_result, actual_result)

        fxb = FxProgramsBuilder(module)

        @fxb.export_program(
            args=(deepcopy(sharded_tensor),),
            name="main",
            strict=False,
        )
        def _(model, *args, **kwargs) -> AnyTensor:
            return model.main(*args, **kwargs)

        if dump_enabled:
            for program_name, ep in fxb.programs.items():
                with open(
                    f"{path_prefix}{program_name}.torch.fx.txt",
                    "w",
                ) as f:
                    print(str(ep), file=f)

        output = export(fxb)
        if dump_enabled:
            output.save_mlir(f"{path_prefix}program.mlir")

        iree_module_path = f"{path_prefix}program.vmfb"
        output.session.set_flags(
            *[f"--iree-hal-target-device=llvm-cpu[{i}]" for i in range(shard_count)]
        )
        output.compile(
            save_to=iree_module_path,
            target_backends=None,
        )

        iree_driver = "local-task"
        iree_devices = get_iree_devices(
            driver=iree_driver,
            device_count=shard_count,
        )
        iree_module, vm_context, vm_instance = load_iree_module(
            module_path=iree_module_path,
            devices=iree_devices,
        )
        iree_args = prepare_iree_module_function_args(
            args=[deepcopy(sharded_tensor_snapshot)], devices=iree_devices
        )
        run_iree_module_function(
            args=iree_args,
            function_name="main",
            module=iree_module,
            vm_context=vm_context,
            driver=iree_driver,
            trace_path_prefix=path_prefix if dump_enabled else None,
        )
        iree_args_as_torch = iree_to_torch(*iree_args)
        iree_args_sharded_tensor = SplitPrimitiveTensor(
            ts=iree_args_as_torch, shard_dim=shard_dim, insert_device_assignment=False
        )
        actual_iree_result = ops.unshard(iree_args_sharded_tensor)
        if dump_enabled:
            call_torch_module_function(
                module=module,
                function_name="main",
                kwargs=OrderedDict(
                    [
                        (
                            "tensor",
                            deepcopy(sharded_tensor_snapshot),
                        )
                    ]
                ),
                trace_path_prefix=f"{path_prefix}expected_",
            )
        torch.testing.assert_close(actual_iree_result, expected_result)


if __name__ == "__main__":
    unittest.main()
