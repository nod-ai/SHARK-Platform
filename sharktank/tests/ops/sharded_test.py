# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from parameterized import parameterized
import tempfile
import pytest
from copy import deepcopy
import torch

from sharktank import ops
from sharktank.types import *
from sharktank.types import sharding
from sharktank.layers import Conv2DLayer
from sharktank.utils.iree import (
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    iree_to_torch,
)
from iree.turbine.aot import FxProgramsBuilder, export


class AllGatherTest(unittest.TestCase):
    def testAllGather(self):
        shard_count = 3
        shard_shape = [3, 4]
        shard_dim = 1
        shards = [
            torch.rand(shard_shape, dtype=torch.float32) for i in range(shard_count)
        ]
        expected_result = torch.cat(shards, dim=shard_dim)

        sharded = SplitPrimitiveTensor(shard_dim=shard_dim, ts=shards)
        actual_result = ops.all_gather(sharded)

        for shard in actual_result.shards:
            torch.testing.assert_close(shard.as_torch(), expected_result)


class AllReduceTest(unittest.TestCase):
    def testAllReduce(self):
        shard_count = 3
        shard_shape = [3, 4]
        shard_dim = 1
        shards = [
            torch.rand(shard_shape, dtype=torch.float32) for i in range(shard_count)
        ]
        expected_result = torch.add(torch.add(shards[0], shards[1]), shards[2])

        sharded = SplitPrimitiveTensor(shard_dim=shard_dim, ts=shards)
        actual_result = ops.all_reduce(sharded)

        for shard in actual_result.shards:
            torch.testing.assert_close(shard.as_torch(), expected_result)


class CatTest(unittest.TestCase):
    def testCatSplitDim(self):
        """Concatenation along the sharded split dimension."""
        shard_dim = 1
        shard_count = 2
        cat_dim = 1
        a = torch.rand(3, 6, dtype=torch.float32)
        b = torch.rand(3, 4, dtype=torch.float32)
        unsharded_result = torch.cat([a, b], dim=cat_dim)
        expected_result = ops.reshard_split(
            unsharded_result, count=shard_count, dim=shard_dim
        )
        sharded_a = ops.reshard_split(a, count=shard_count, dim=shard_dim)
        sharded_b = ops.reshard_split(b, count=shard_count, dim=shard_dim)
        actual_result = ops.cat([sharded_a, sharded_b], dim=cat_dim)
        assert ops.equal(expected_result, actual_result)

    def testCatNonSplitDim(self):
        """Concatenation along a non-split dimension."""
        shard_dim = 1
        shard_count = 2
        cat_dim = 0
        a = torch.rand(5, 4, dtype=torch.float32)
        b = torch.rand(3, 4, dtype=torch.float32)
        unsharded_result = torch.cat([a, b], dim=cat_dim)
        expected_result = ops.reshard_split(
            unsharded_result, count=shard_count, dim=shard_dim
        )
        sharded_a = ops.reshard_split(a, count=shard_count, dim=shard_dim)
        sharded_b = ops.reshard_split(b, count=shard_count, dim=shard_dim)
        actual_result = ops.cat([sharded_a, sharded_b], dim=cat_dim)
        assert ops.equal(expected_result, actual_result)


class ConvTest(unittest.TestCase):
    def testConv2dShardedInputAndOutputChannelsOneGroup(self):
        batches = 2
        in_channels = 6
        out_channels = 12
        groups = 1
        height = 17
        width = 19
        stride = 2
        padding = 3
        dilation = 2
        kernel_height = 3
        kernel_width = 4
        x = torch.rand(batches, in_channels, height, width, dtype=torch.float32)
        weight = torch.rand(
            out_channels,
            in_channels // groups,
            kernel_height,
            kernel_width,
            dtype=torch.float32,
        )
        bias = torch.rand(out_channels, dtype=torch.float32)

        expected_result = ops.conv2d(
            x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        shard_count = 2
        x_sharded = SplitPrimitiveTensor(shard_dim=1, ts=x, shard_count=shard_count)
        weight_sharded = SplitPrimitiveTensor(
            shard_dim=0, ts=weight, shard_count=shard_count
        )
        bias_sharded = SplitPrimitiveTensor(
            shard_dim=0, ts=bias, shard_count=shard_count
        )
        sharded_result = ops.conv2d(
            x_sharded,
            weight=weight_sharded,
            bias=bias_sharded,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        actual_result = ops.sharded_cat(sharded_result)

        torch.testing.assert_close(actual_result, expected_result)

    def testCov2dShardedOutputChannelsOneGroup(self):
        batches = 2
        in_channels = 6
        out_channels = 12
        groups = 1
        height = 17
        width = 19
        stride = 2
        padding = 3
        dilation = 2
        kernel_height = 3
        kernel_width = 4
        x = torch.rand(batches, in_channels, height, width, dtype=torch.float32)
        weight = torch.rand(
            out_channels,
            in_channels // groups,
            kernel_height,
            kernel_width,
            dtype=torch.float32,
        )
        bias = torch.rand(out_channels, dtype=torch.float32)

        expected_result = ops.conv2d(
            x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        shard_count = 2
        weight_sharded = SplitPrimitiveTensor(
            shard_dim=0, ts=weight, shard_count=shard_count
        )
        bias_sharded = SplitPrimitiveTensor(
            shard_dim=0, ts=bias, shard_count=shard_count
        )
        sharded_result = ops.conv2d(
            x,
            weight=weight_sharded,
            bias=bias_sharded,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        actual_result = ops.sharded_cat(sharded_result)

        torch.testing.assert_close(actual_result, expected_result)

    def testConv2DSplitOutputChannelShardingSpec(self):
        batches = 2
        in_channels = 6
        out_channels = 12
        groups = 1
        height = 17
        width = 19
        stride = 2
        padding = [2, 3]
        kernel_height = 3
        kernel_width = 4
        input = torch.rand(batches, in_channels, height, width, dtype=torch.float32)
        weight = torch.rand(
            out_channels,
            in_channels // groups,
            kernel_height,
            kernel_width,
            dtype=torch.float32,
        )
        bias = torch.rand(out_channels, dtype=torch.float32)
        theta = Theta(
            {
                "weight": DefaultPrimitiveTensor(data=weight),
                "bias": DefaultPrimitiveTensor(data=bias),
            }
        )
        conv2d_layer = Conv2DLayer(theta, padding=padding, stride=stride)

        shard_count = 3
        sharded_input = ops.reshard_split(input, dim=1, count=shard_count)
        conv2d_sharding = sharding.Conv2DSplitOutputChannelSharding(
            shard_count=shard_count
        )
        sharded_theta = ops.reshard(theta, conv2d_sharding)
        sharded_conv2d_layer = Conv2DLayer(
            sharded_theta, padding=padding, stride=stride
        )

        expected_result = conv2d_layer.forward(input)
        sharded_result = sharded_conv2d_layer.forward(sharded_input)
        actual_result = ops.reshard_like(sharded_result, expected_result)
        assert ops.equal(expected_result, actual_result)


@pytest.mark.usefixtures("path_prefix")
class ElementwiseTest(unittest.TestCase):
    def testRhsAndLhsShardedAdd(self):
        a = torch.rand(4, 5, 6, dtype=torch.float32)
        b = torch.rand(4, 5, 6, dtype=torch.float32)

        expected_result = a + b

        shard_dim = 2
        shard_count = 3
        sharded_a = ops.reshard_split(a, dim=shard_dim, count=shard_count)
        sharded_b = ops.reshard_split(b, dim=shard_dim, count=shard_count)
        sharded_result = sharded_a + sharded_b
        actual_result = ops.reshard_like(sharded_result, expected_result)

        torch.testing.assert_close(actual_result, expected_result)

    def testRhsAndLhsShardedAddWithBroadcasting(self):
        a = torch.rand(1, 4, 5, 6, dtype=torch.float32)
        b = torch.rand(3, 4, 1, 6, dtype=torch.float32)

        expected_result = a + b

        shard_dim = 3
        shard_count = 3
        sharded_a = ops.reshard_split(a, dim=shard_dim, count=shard_count)
        sharded_b = ops.reshard_split(b, dim=shard_dim, count=shard_count)
        sharded_result = sharded_a + sharded_b
        actual_result = ops.reshard_like(sharded_result, expected_result)

        torch.testing.assert_close(actual_result, expected_result)

    @parameterized.expand(
        [
            (torch.add,),
            (torch.div,),
            (torch.fmin,),
            (torch.fmax,),
            (torch.sub),
        ]
    )
    def testBinaryOperators(self, operator):
        a = torch.rand(4, 5, 6, dtype=torch.float32)
        b = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 2
        shard_count = 3
        expected_result = operator(a, b)

        # Sharded LHS and RHS
        sharded_a = ops.reshard_split(a, dim=shard_dim, count=shard_count)
        sharded_b = ops.reshard_split(b, dim=shard_dim, count=shard_count)
        sharded_result = ops.elementwise(operator, sharded_a, sharded_b)
        assert isinstance(sharded_result, ShardedTensor)
        assert not sharded_result.is_replicated
        assert sharded_result.shard_count == sharded_a.shard_count
        assert sharded_result.shard_dim == sharded_a.shard_dim
        actual_result = ops.reshard_like(sharded_result, expected_result)
        torch.testing.assert_close(actual_result, expected_result)

        # Replicated LHS and Sharded RHS
        sharded_a = ops.replicate(a, count=shard_count)
        sharded_b = ops.reshard_split(b, dim=shard_dim, count=shard_count)
        sharded_result = ops.elementwise(operator, sharded_a, sharded_b)
        assert isinstance(sharded_result, ShardedTensor)
        assert not sharded_result.is_replicated
        assert sharded_result.shard_count == sharded_b.shard_count
        assert sharded_result.shard_dim == sharded_b.shard_dim
        actual_result = ops.reshard_like(sharded_result, expected_result)
        torch.testing.assert_close(actual_result, expected_result)

        # Sharded LHS and Replicated RHS
        sharded_a = ops.reshard_split(a, dim=shard_dim, count=shard_count)
        sharded_b = ops.replicate(b, count=shard_count)
        sharded_result = ops.elementwise(operator, sharded_a, sharded_b)
        assert isinstance(sharded_result, ShardedTensor)
        assert not sharded_result.is_replicated
        assert sharded_result.shard_count == sharded_a.shard_count
        assert sharded_result.shard_dim == sharded_a.shard_dim
        actual_result = ops.reshard_like(sharded_result, expected_result)
        torch.testing.assert_close(actual_result, expected_result)

    def testInPlaceAddWithIree(self):
        if self.path_prefix is not None:
            self.runTestInPlaceAddWithIree(
                path_prefix=self.path_prefix, dump_enabled=True
            )
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                self.runTestInPlaceAddWithIree(
                    path_prefix=f"{temp_dir}/", dump_enabled=False
                )

    def runTestInPlaceAddWithIree(self, path_prefix: str, dump_enabled: bool):
        dtype = torch.float32

        class Module(torch.nn.Module):
            def main(self, tensor: torch.Tensor) -> torch.Tensor:
                tensor += 1
                # TODO: figure out why when not returning anything the export fails.
                return torch.empty([1], dtype=dtype)

        tensor = torch.tensor([1, 2, 3], dtype=dtype)
        expected_result = torch.tensor([2, 3, 4], dtype=dtype)

        module = Module()
        fxb = FxProgramsBuilder(module)

        @fxb.export_program(
            args=(tensor,),
            name="main",
            strict=False,
        )
        def _(model, *args, **kwargs):
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
        output.session.set_flags("--iree-hal-target-device=llvm-cpu")
        output.compile(
            save_to=iree_module_path,
            target_backends=None,
        )

        iree_driver = "local-task"
        iree_devices = get_iree_devices(
            driver=iree_driver,
            device_count=1,
        )
        iree_module, vm_context, vm_instance = load_iree_module(
            module_path=iree_module_path,
            devices=iree_devices,
        )
        iree_args = prepare_iree_module_function_args(
            args=[deepcopy(tensor)], devices=iree_devices
        )
        run_iree_module_function(
            args=iree_args,
            function_name="main",
            module=iree_module,
            vm_context=vm_context,
            driver=iree_driver,
            trace_path_prefix=path_prefix if dump_enabled else None,
            # trace_path_prefix=None
        )
        actual_result = iree_to_torch(*iree_args)[0]
        torch.testing.assert_close(actual_result, expected_result)

    @pytest.mark.xfail(
        reason=(
            "We are inserting device transfers that introduce a copy a screw up the"
            " in-place semantics."
        ),
        strict=True,
        raises=AssertionError,
    )
    def testInPlaceAddWithIreeOfShardedSplitTensor(self):
        if self.path_prefix is not None:
            self.runTestInPlaceAddWithIreeOfShardedSplitTensor(
                path_prefix=self.path_prefix, dump_enabled=True
            )
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                self.runTestInPlaceAddWithIreeOfShardedSplitTensor(
                    path_prefix=f"{temp_dir}/", dump_enabled=False
                )

    def runTestInPlaceAddWithIreeOfShardedSplitTensor(
        self, path_prefix: str, dump_enabled: bool
    ):
        shard_dim = 0
        shard_count = 2
        dtype = torch.float32

        class Module(torch.nn.Module):
            def main(self, tensor: AnyTensor):
                tensor += 1
                # TODO: figure out why when not returning anything the export fails.
                return torch.empty([1], dtype=dtype)

        tensor = torch.tensor([1, 2, 3, 4], dtype=dtype)
        expected_result = torch.tensor([2, 3, 4, 5], dtype=dtype)
        sharded_tensor = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)

        module = Module()

        fxb = FxProgramsBuilder(module)

        @fxb.export_program(
            args=(sharded_tensor,),
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
            args=[sharded_tensor],
            devices=iree_devices,
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
            ts=iree_args_as_torch,
            shard_dim=shard_dim,
        )
        actual_result = ops.unshard(iree_args_sharded_tensor)
        torch.testing.assert_close(actual_result, expected_result)


class EqualTest(unittest.TestCase):
    def testNotEqualReplicated(self):
        a = torch.rand(3, 4, 5, dtype=torch.float32)
        b = torch.clone(a)
        shard_count = 2
        a_sharded = ops.replicate(a, count=shard_count)
        b_sharded = ops.reshard_like(b, a_sharded)
        assert ops.equal(a_sharded, b_sharded)
        assert ops.equal(b_sharded, a_sharded)

    def testNotEqualReplicated(self):
        a = torch.rand(3, 4, 5, dtype=torch.float32)
        b = torch.clone(a)
        b[0, 0, 0] += 1
        shard_count = 2
        a_sharded = ops.replicate(a, count=shard_count)
        b_sharded = ops.reshard_like(b, a_sharded)
        assert not ops.equal(a_sharded, b_sharded)
        assert not ops.equal(b_sharded, a_sharded)

    def testEqualSharded(self):
        a = torch.rand(3, 4, 5, dtype=torch.float32)
        b = torch.clone(a)
        shard_dim = 1
        shard_count = 2
        a_sharded = ops.reshard_split(a, dim=shard_dim, count=shard_count)
        b_sharded = ops.reshard_like(b, a_sharded)
        assert ops.equal(a_sharded, b_sharded)
        assert ops.equal(b_sharded, a_sharded)

    def testNotEqualSharded(self):
        a = torch.rand(3, 4, 5, dtype=torch.float32)
        b = torch.clone(a)
        b[0, 0, 0] += 1
        shard_dim = 1
        shard_count = 2
        a_sharded = ops.reshard_split(a, dim=shard_dim, count=shard_count)
        b_sharded = ops.reshard_like(b, a_sharded)
        assert not ops.equal(a_sharded, b_sharded)
        assert not ops.equal(b_sharded, a_sharded)


class FlattenTest(unittest.TestCase):
    def testReplicated(self):
        tensor = torch.rand(2, 3, 4, 5)
        unsharded_expected_result = torch.flatten(tensor, start_dim=1, end_dim=2)
        expected_result = ops.replicate(unsharded_expected_result, count=2)
        sharded_tensor = ops.replicate(tensor, count=2)
        actual_result = ops.flatten(sharded_tensor, start_dim=1, end_dim=2)
        assert expected_result.is_deep_equal(actual_result)

    def testSplitTensorFlattenNonSplitDim(self):
        tensor = torch.rand(2, 3, 4, 5)
        unsharded_expected_result = torch.flatten(tensor, start_dim=1, end_dim=2)
        expected_result = ops.reshard_split(unsharded_expected_result, dim=2, count=2)
        sharded_tensor = ops.reshard_split(tensor, dim=3, count=2)
        actual_result = ops.flatten(sharded_tensor, start_dim=1, end_dim=2)
        assert expected_result.is_deep_equal(actual_result)

    def testSplitTensorSplitDimIsLeadingFlattenDim(self):
        tensor = torch.rand(3, 4, 5, 6)
        unsharded_expected_result = torch.flatten(tensor, start_dim=1, end_dim=2)
        expected_result = ops.reshard_split(unsharded_expected_result, dim=1, count=2)
        sharded_tensor = ops.reshard_split(tensor, dim=1, count=2)
        actual_result = ops.flatten(sharded_tensor, start_dim=1, end_dim=2)
        assert expected_result.is_deep_equal(actual_result)


class GemmTest(unittest.TestCase):
    def testShardedParallelDim(self):
        a = torch.rand(4, 3)
        b = torch.rand(5, 3)
        c = torch.rand(4, 5)
        alpha = 2
        beta = 3
        shard_count = 2
        expected = ops.gemm(a, b, c, alpha, beta, False, True)
        sharded_a = ops.reshard_split(a, dim=0, count=shard_count)
        sharded_c = ops.reshard_split(c, dim=0, count=shard_count)
        sharded_result = ops.gemm(sharded_a, b, sharded_c, alpha, beta, False, True)
        assert isinstance(sharded_result, SplitPrimitiveTensor)
        assert sharded_result.shard_count == 2
        assert sharded_result.shard_dim == 0
        actual = ops.unshard(sharded_result)
        torch.testing.assert_close(actual, expected)


class IndexCopyTest(unittest.TestCase):
    def testSplitInPlace(self):
        torch.set_default_dtype(torch.float32)
        tensor = torch.rand(3, 4, 5, 6)
        dim = 2
        source = torch.rand(3, 4, 2, 6)
        index = torch.tensor([1, 3])
        expected_result = torch.index_copy(tensor, dim, index, source)

        split_dim = 1
        shard_count = 2
        sharded_tensor = ops.reshard_split(tensor, dim=split_dim, count=shard_count)
        sharded_index = ops.replicate(index, count=shard_count)
        sharded_source = ops.reshard_split(source, dim=split_dim, count=shard_count)
        sharded_result = ops.index_copy_(
            sharded_tensor, dim, sharded_index, sharded_source
        )
        assert sharded_tensor is sharded_result
        actual_result = ops.unshard(sharded_tensor)
        assert ops.equal(actual_result, expected_result)


class IndexPutTest(unittest.TestCase):
    def testSplitNonIndexDimInPlace(self):
        torch.set_default_dtype(torch.float32)
        tensor = torch.rand(3, 4, 5, 6)
        indices = (
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([2, 3], dtype=torch.long),
        )
        values = torch.rand(2, 5, 6)
        expected_result = tensor.clone().index_put_(indices, values)
        shard_count = 2
        sharded_tensor = ops.reshard_split(tensor.clone(), dim=3, count=shard_count)
        sharded_values = ops.reshard_split(values, dim=2, count=shard_count)
        sharded_result = ops.index_put_(sharded_tensor, indices, sharded_values)
        assert sharded_tensor is sharded_result
        actual_result = ops.unshard(sharded_tensor)
        assert ops.equal(actual_result, expected_result)


class InterpolateTest(unittest.TestCase):
    def testInterpolateSplitChannelDim(self):
        batches = 2
        channels = 6
        height = 5
        width = 4
        scale_factor = 2.0
        mode = "bilinear"
        align_corners = True
        recompute_scale_factor = True
        antialias = True
        input = torch.rand(batches, channels, height, width, dtype=torch.float32)
        expected_result = torch.nn.functional.interpolate(
            input=input,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )
        shard_count = 3
        sharded_input = ops.reshard_split(input, dim=1, count=shard_count)
        sharded_result = ops.interpolate(
            input=sharded_input,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )
        assert isinstance(sharded_result, SplitPrimitiveTensor)
        assert sharded_result.shard_count == shard_count
        assert sharded_result.shard_dim == 1
        actual_result = ops.unbox_tensor(ops.unshard(sharded_result))
        torch.testing.assert_close(actual_result, expected_result)

    def testInterpolateReplicated(self):
        batches = 2
        channels = 6
        height = 5
        width = 4
        scale_factor = 2.0
        mode = "bilinear"
        align_corners = True
        recompute_scale_factor = True
        antialias = True
        input = torch.rand(batches, channels, height, width, dtype=torch.float32)
        expected_result = torch.nn.functional.interpolate(
            input=input,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )
        shard_count = 3
        sharded_input = ops.replicate(input, count=shard_count)
        sharded_result = ops.interpolate(
            input=sharded_input,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )
        assert isinstance(sharded_result, ReplicatedTensor)
        assert sharded_result.shard_count == shard_count
        actual_result = ops.unbox_tensor(ops.unshard(sharded_result))
        torch.testing.assert_close(actual_result, expected_result)


class NormalizationTest(unittest.TestCase):
    def testGroupNormShardedGroups(self):
        """Shard the channel dimension such that the group count is multiple of the
        shard count."""
        batches = 3
        groups = 6
        height = 17
        width = 19
        channels = 12
        eps = 0.01
        x = torch.rand(batches, channels, height, width, dtype=torch.float32)
        weight = torch.rand(channels, dtype=torch.float32)
        bias = torch.rand(channels, dtype=torch.float32)

        expected_result = ops.group_norm_affine(
            x, weight=weight, bias=bias, num_groups=groups, eps=eps
        )

        shard_count = 3
        x_sharded = SplitPrimitiveTensor(shard_dim=1, ts=x, shard_count=shard_count)
        weight_sharded = SplitPrimitiveTensor(
            shard_dim=0, ts=weight, shard_count=shard_count
        )
        bias_sharded = SplitPrimitiveTensor(
            shard_dim=0, ts=bias, shard_count=shard_count
        )
        sharded_result = ops.group_norm_affine(
            x_sharded,
            weight=weight_sharded,
            bias=bias_sharded,
            num_groups=groups,
            eps=eps,
        )
        actual_result = ops.sharded_cat(sharded_result)

        torch.testing.assert_close(actual_result, expected_result)

    def testLayerNorm(self):
        """Shard an input dimension other than the trailing normalization dimensions."""
        batches = 3
        eps = 0.01
        weight = torch.rand(3, 4, dtype=torch.float32)
        bias = torch.rand_like(weight)
        input_shape = [batches, 11, 12] + list(weight.shape)
        x = torch.rand(input_shape, dtype=torch.float32)

        expected_result = ops.layer_norm(x, weight=weight, bias=bias, eps=eps)

        x_sharded = SplitPrimitiveTensor(shard_dim=2, ts=x, shard_count=3)
        sharded_result = ops.layer_norm(
            x_sharded,
            weight=weight,
            bias=bias,
            eps=eps,
        )
        actual_result = ops.sharded_cat(sharded_result)

        torch.testing.assert_close(actual_result, expected_result)


class PermuteTest(unittest.TestCase):
    def testShardedPrimitiveTensorPermute(self):
        torch_tensor = torch.rand(3, 8, 5, dtype=torch.float32)
        permutation = [1, 0, 2]
        sharded_tensor = SplitPrimitiveTensor(
            ts=torch_tensor, shard_dim=1, shard_count=4
        )
        expected_result = torch.permute(torch_tensor, permutation)

        permuted_sharded_tensor = ops.permute(sharded_tensor, permutation)
        result = ops.sharded_cat(permuted_sharded_tensor)

        assert ops.equal(expected_result, result)


class MatmulTest(unittest.TestCase):
    def testTorchRHSColumnShardedTransposed(self):
        t1 = torch.rand(4, 32, 16, dtype=torch.float32)
        t2 = torch.rand(48, 16, dtype=torch.float16)
        # RHS is transposed, so dim0 is the "column". Shard into 12.
        t2_sharded = SplitPrimitiveTensor(shard_dim=0, ts=t2.split(4, dim=0))
        sharded_result = ops.matmul(t1, t2_sharded.T)
        expected_result = ops.matmul(t1, t2.T)
        unsharded_result = ops.sharded_cat(sharded_result)
        torch.testing.assert_close(unsharded_result, expected_result)

    def testTorchRHSColumnSharded(self):
        t1 = torch.rand(4, 32, 16, dtype=torch.float32)
        t2 = torch.rand(16, 48, dtype=torch.float16)
        t2_sharded = SplitPrimitiveTensor(shard_dim=1, ts=t2.split(4, dim=1))
        sharded_result = ops.matmul(t1, t2_sharded)
        expected_result = ops.matmul(t1, t2)
        unsharded_result = ops.sharded_cat(sharded_result)
        torch.testing.assert_close(unsharded_result, expected_result)

    def testReplicatedLhsShardedParallelDimRhs(self):
        a = torch.rand(2, 5, 3, dtype=torch.float32)
        b = torch.rand(3, 6, dtype=torch.float32)
        shard_count = 3
        unsharded_result = torch.matmul(a, b)
        expected_result = ops.reshard_split(unsharded_result, dim=2, count=shard_count)
        b_sharded = ops.reshard_split(b, dim=1, count=shard_count)
        a_sharded = ops.replicate(a, count=shard_count)
        actual_result = ops.matmul(a_sharded, b_sharded)
        assert ops.equal(expected_result, actual_result)

    def testShardedChainMatmulX2Transposed(self):
        # Computes Z = (XA)B (sharded by 8).
        X = torch.rand(4, 32, 16, dtype=torch.float32)
        A = torch.rand(48, 16, dtype=torch.float16)
        B = torch.rand(16, 48, dtype=torch.float16)
        XA = ops.matmul(X, A.T)
        Z = ops.matmul(XA, B.T)

        # Columnwise sharding of A matrix (transposed).
        A_sharded = SplitPrimitiveTensor(shard_dim=0, ts=A.split(6, dim=0))
        assert A_sharded.shard_count == 8
        # Rowwise sharding of B matrix (transposed).
        B_sharded = SplitPrimitiveTensor(shard_dim=1, ts=B.split(6, dim=1))
        assert B_sharded.shard_count == 8

        XA_sharded = ops.matmul(X, A_sharded.T)
        Z_sharded = ops.matmul(XA_sharded, B_sharded.T)
        Z_unsharded = ops.sharded_sum(Z_sharded)
        torch.testing.assert_close(Z_unsharded, Z)

    def testShardedParallelAxisInLhs(self):
        a = torch.rand(2, 12, 5, dtype=torch.float32)
        b = torch.rand(5, 9, dtype=torch.float32)
        expected_result = torch.matmul(a, b)
        shard_count = 3
        a_sharded = SplitPrimitiveTensor(ts=a, shard_dim=1, shard_count=shard_count)
        res_sharded = ops.matmul(a_sharded, b)
        assert isinstance(res_sharded, SplitPrimitiveTensor)
        assert res_sharded.shard_dim == 1
        assert res_sharded.shard_count == shard_count
        actual_result = ops.sharded_cat(res_sharded)
        torch.testing.assert_close(actual_result, expected_result)

    def testShardedParallelAxesInLhsAndRhs(self):
        a = torch.rand(2, 12, 5, dtype=torch.float32)
        b = torch.rand(5, 9, dtype=torch.float32)
        expected_result = torch.matmul(a, b)
        shard_count = 3
        a_sharded = SplitPrimitiveTensor(ts=a, shard_dim=1, shard_count=shard_count)
        b_sharded = SplitPrimitiveTensor(ts=b, shard_dim=1, shard_count=shard_count)
        res_sharded = ops.matmul(a_sharded, b_sharded)
        assert isinstance(res_sharded, SplitPrimitiveTensor)
        assert res_sharded.shard_dim == 1
        assert res_sharded.shard_count == shard_count
        actual_result = ops.sharded_cat(res_sharded)
        torch.testing.assert_close(actual_result, expected_result)

    def testShardedParallelAxesInLhsAndTransposedRhs(self):
        a = torch.rand(2, 12, 5, dtype=torch.float32)
        b = torch.rand(9, 5, dtype=torch.float32)
        expected_result = torch.matmul(a, b.T)
        shard_count = 3
        a_sharded = SplitPrimitiveTensor(ts=a, shard_dim=1, shard_count=shard_count)
        b_sharded = SplitPrimitiveTensor(ts=b, shard_dim=0, shard_count=shard_count)
        res_sharded = ops.matmul(a_sharded, b_sharded, transpose_rhs=True)
        assert isinstance(res_sharded, SplitPrimitiveTensor)
        assert res_sharded.shard_dim == 1
        assert res_sharded.shard_count == shard_count
        actual_result = ops.sharded_cat(res_sharded)
        torch.testing.assert_close(actual_result, expected_result)

    def testShardedLhsBatchDimAndRhsParallelDim(self):
        a = torch.rand(12, 2, 5, dtype=torch.float32)
        b = torch.rand(5, 9, dtype=torch.float32)
        expected_result = torch.matmul(a, b)
        shard_count = 3
        a_sharded = SplitPrimitiveTensor(ts=a, shard_dim=0, shard_count=shard_count)
        b_sharded = SplitPrimitiveTensor(ts=b, shard_dim=1, shard_count=shard_count)
        res_sharded = ops.matmul(a_sharded, b_sharded)
        assert isinstance(res_sharded, SplitPrimitiveTensor)
        assert res_sharded.shard_dim == 0
        assert res_sharded.shard_count == shard_count
        actual_result = ops.sharded_cat(res_sharded)
        torch.testing.assert_close(actual_result, expected_result)

    def testShardedLhsReplcatedRhs(self):
        a = torch.rand(12, 3, 5, dtype=torch.float32)
        b = torch.rand(5, 9, dtype=torch.float32)
        expected_result = torch.matmul(a, b)
        shard_count = 3
        a_sharded = SplitPrimitiveTensor(ts=a, shard_dim=1, shard_count=shard_count)
        b_sharded = ReplicatedTensor(ts=b, shard_count=shard_count)
        res_sharded = ops.matmul(a_sharded, b_sharded)
        assert isinstance(res_sharded, SplitPrimitiveTensor)
        assert res_sharded.shard_dim == 1
        assert res_sharded.shard_count == shard_count
        actual_result = ops.sharded_cat(res_sharded)
        torch.testing.assert_close(actual_result, expected_result)

    def testShardedFFNTransposed(self):
        input = torch.rand(4, 32, 64, dtype=torch.float32)
        unsharded_ffn_gate_weight = torch.rand(128, 64, dtype=torch.float16)
        unsharded_ffn_down_weight = torch.rand(64, 128, dtype=torch.float16)
        unsharded_ffn_up_weight = torch.rand(128, 64, dtype=torch.float16)

        def compute(input, ffn_gate_weight, ffn_down_weight, ffn_up_weight):
            ffn_gate = ops.elementwise(
                torch.nn.functional.silu, ops.linear(input, ffn_gate_weight)
            )
            ffn_up = ops.linear(input, ffn_up_weight)
            ffn_down = ops.linear(
                ops.elementwise(torch.mul, ffn_gate, ffn_up), ffn_down_weight
            )
            summed = ops.sharded_sum(ffn_down)
            return summed

        Z_ref = compute(
            input,
            unsharded_ffn_gate_weight,
            unsharded_ffn_down_weight,
            unsharded_ffn_up_weight,
        )

        # Columnwise sharding of gate and up weight (transposed).
        sharded_ffn_gate_weight = SplitPrimitiveTensor(
            shard_dim=0, ts=unsharded_ffn_gate_weight.split(16, dim=0)
        )
        sharded_ffn_up_weight = SplitPrimitiveTensor(
            shard_dim=0, ts=unsharded_ffn_up_weight.split(16, dim=0)
        )
        assert sharded_ffn_gate_weight.shard_count == 8
        assert sharded_ffn_up_weight.shard_count == 8

        # Rowwise sharding of down weight (transposed).
        sharded_ffn_down_weight = SplitPrimitiveTensor(
            shard_dim=1, ts=unsharded_ffn_down_weight.split(16, dim=1)
        )
        assert sharded_ffn_down_weight.shard_count == 8
        Z_sharded = compute(
            input,
            sharded_ffn_gate_weight,
            sharded_ffn_down_weight,
            sharded_ffn_up_weight,
        )
        torch.testing.assert_close(Z_sharded, Z_ref)

    def testSameSplitLhsAndRhsBatchDim(self):
        a = torch.rand(3, 4, 5, 6)
        b = torch.rand(3, 4, 6, 7)
        shard_count = 2
        shard_dim = 1
        expected_result = torch.matmul(a, b)
        sharded_a = ops.reshard_split(a, dim=shard_dim, count=shard_count)
        sharded_b = ops.reshard_split(b, dim=shard_dim, count=shard_count)
        sharded_result = ops.matmul(sharded_a, sharded_b)
        assert isinstance(sharded_result, SplitPrimitiveTensor)
        assert sharded_result.shard_count == shard_count
        assert sharded_result.shard_dim == shard_dim
        actual_result = unbox_tensor(ops.unshard(sharded_result))
        torch.testing.assert_close(actual_result, expected_result)


class ReplicateTest(unittest.TestCase):
    def testReplicateReplicated(self):
        tensor = torch.rand(4, 5, dtype=torch.float32)
        shard_count = 3
        expected_result = ops.replicate(tensor, count=shard_count)
        actual_result = ops.replicate(expected_result, count=shard_count)
        assert expected_result.is_deep_equal(actual_result)

    def testReplicateUnsharded(self):
        tensor = torch.rand(4, 5, dtype=torch.float32)
        shard_count = 3
        actual_result = ops.replicate(tensor, count=shard_count)
        expected_result = ReplicatedTensor(ts=tensor, shard_count=shard_count)
        assert expected_result.is_deep_equal(actual_result)

        # Test not a copy.
        tensor[...] = torch.rand_like(tensor)
        assert all(ops.equal(tensor, shard) for shard in actual_result.shards)


class ReshapeTest(unittest.TestCase):
    def testSplitTensorFlattenNonSplitDim(self):
        tensor = torch.rand(2, 3, 4, 5)
        new_shape = [2, 12, 5]
        unsharded_expected_result = torch.reshape(tensor, new_shape)
        expected_result = ops.reshard_split(unsharded_expected_result, dim=2, count=2)
        sharded_tensor = ops.reshard_split(tensor, dim=3, count=2)
        actual_result = ops.reshape(sharded_tensor, new_shape)
        assert expected_result.is_deep_equal(actual_result)

    def testSplitTensorSplitDimIsLeadingFlattenDim(self):
        tensor = torch.rand(3, 4, 5, 6)
        new_shape = [3, 20, 6]
        unsharded_expected_result = torch.reshape(tensor, new_shape)
        expected_result = ops.reshard_split(unsharded_expected_result, dim=1, count=2)
        sharded_tensor = ops.reshard_split(tensor, dim=1, count=2)
        actual_result = ops.reshape(sharded_tensor, new_shape)
        assert expected_result.is_deep_equal(actual_result)

    def testSplitTensorInsertSize1DimBeforeSplitDim(self):
        tensor = torch.rand(4, 5, 6, 7)
        new_shape = [4, 1, 5, 6, 7]
        unsharded_expected_result = torch.reshape(tensor, new_shape)
        shard_dim = 2
        expected_result = ops.reshard_split(
            unsharded_expected_result, dim=shard_dim + 1, count=2
        )
        sharded_tensor = ops.reshard_split(tensor, dim=shard_dim, count=2)
        actual_result = ops.reshape(sharded_tensor, new_shape)
        assert expected_result.is_deep_equal(actual_result)

    def testSplitTensorInsertMultipleSize1DimsBeforeSplitDim(self):
        tensor = torch.rand(4, 5, 6, 7)
        new_shape = [4, 1, 1, 5, 6, 7]
        unsharded_expected_result = torch.reshape(tensor, new_shape)
        shard_dim = 2
        expected_result = ops.reshard_split(
            unsharded_expected_result, dim=shard_dim + 2, count=2
        )
        sharded_tensor = ops.reshard_split(tensor, dim=shard_dim, count=2)
        actual_result = ops.reshape(sharded_tensor, new_shape)
        assert expected_result.is_deep_equal(actual_result)

    def testSplitTensorInsertMultipleSize1TrailingDimsNotRightAfterSplitDim(self):
        tensor = torch.rand(4, 5, 6, 7)
        new_shape = [4, 5, 6, 7, 1, 1]
        unsharded_expected_result = torch.reshape(tensor, new_shape)
        shard_dim = 2
        expected_result = ops.reshard_split(
            unsharded_expected_result, dim=shard_dim, count=2
        )
        sharded_tensor = ops.reshard_split(tensor, dim=shard_dim, count=2)
        actual_result = ops.reshape(sharded_tensor, new_shape)
        assert expected_result.is_deep_equal(actual_result)

    def testSplitTensorUnflattenNonSplitDim(self):
        tensor = torch.rand(3, 20, 6)
        new_shape = [3, 4, 5, 6]
        unsharded_expected_result = torch.reshape(tensor, new_shape)
        expected_result = ops.reshard_split(unsharded_expected_result, dim=3, count=2)
        sharded_tensor = ops.reshard_split(tensor, dim=2, count=2)
        actual_result = ops.reshape(sharded_tensor, new_shape)
        assert expected_result.is_deep_equal(actual_result)

    def testSplitTensorUnflattenTrailingNonSplitDim(self):
        tensor = torch.rand(3, 4, 30)
        new_shape = [3, 4, 5, 6]
        unsharded_expected_result = torch.reshape(tensor, new_shape)
        expected_result = ops.reshard_split(unsharded_expected_result, dim=1, count=2)
        sharded_tensor = ops.reshard_split(tensor, dim=1, count=2)
        actual_result = ops.reshape(sharded_tensor, new_shape)
        assert expected_result.is_deep_equal(actual_result)

    def testSplitTensorUnflattenSplitDim(self):
        tensor = torch.rand(3, 20, 6)
        new_shape = [3, 4, 5, 6]
        unsharded_expected_result = torch.reshape(tensor, new_shape)
        expected_result = ops.reshard_split(unsharded_expected_result, dim=1, count=2)
        sharded_tensor = ops.reshard_split(tensor, dim=1, count=2)
        actual_result = ops.reshape(sharded_tensor, new_shape)
        assert expected_result.is_deep_equal(actual_result)

    def testSplitTensorUnflattenTrailingSplitDim(self):
        tensor = torch.rand(2, 3, 20)
        new_shape = [2, 3, 4, 5]
        unsharded_expected_result = torch.reshape(tensor, new_shape)
        expected_result = ops.reshard_split(unsharded_expected_result, dim=2, count=2)
        sharded_tensor = ops.reshard_split(tensor, dim=2, count=2)
        actual_result = ops.reshape(sharded_tensor, new_shape)
        assert expected_result.is_deep_equal(actual_result)


class ReshardSplitTest(unittest.TestCase):
    def testReshardReplicated(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 2
        shard_count = 3
        replicated_tensor = ops.replicate(tensor, count=shard_count)
        actual_result = ops.reshard_split(
            replicated_tensor, dim=shard_dim, count=shard_count
        )
        expected_result = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)
        assert expected_result.is_deep_equal(actual_result)

    def testReshardUnsharded(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 2
        shard_count = 3
        actual_result = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)
        expected_result = SplitPrimitiveTensor(
            ts=tensor, shard_count=shard_count, shard_dim=shard_dim
        )
        assert expected_result.is_deep_equal(actual_result)

        # Test not a copy.
        tensor[...] = torch.rand_like(tensor)
        result_split2 = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)
        assert ops.equal(actual_result, result_split2)

    def testReshardSharded(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 2
        shard_count = 3
        expected_result = SplitPrimitiveTensor(
            ts=tensor, shard_count=shard_count, shard_dim=shard_dim
        )
        actual_result = ops.reshard_split(
            expected_result, dim=shard_dim, count=shard_count
        )
        assert expected_result.is_deep_equal(actual_result)


class ReshardTest(unittest.TestCase):
    def testTensorSplit(self):
        tensor = torch.rand(5, 6, dtype=torch.float32)
        shard_count = 3
        shard_dim = 1
        expected_result = ops.reshard_split(tensor, count=shard_count, dim=shard_dim)
        split_sharding = sharding.Split(shard_count=shard_count, shard_dim=shard_dim)
        actual_result = ops.reshard(tensor, split_sharding)
        assert ops.equal(expected_result, actual_result)

    def testGroupNormSplitChannelSharding(self):
        channels = 12
        weight = torch.rand(channels, dtype=torch.float32)
        bias = torch.rand(channels, dtype=torch.float32)
        theta = Theta(
            {
                "weight": DefaultPrimitiveTensor(data=weight),
                "bias": DefaultPrimitiveTensor(data=bias),
            }
        )
        shard_count = 3
        sharding_spec = sharding.GroupNormSplitChannelSharding(shard_count=shard_count)
        sharded_theta = ops.reshard(theta, sharding_spec)
        expected_weight = ops.reshard_split(weight, dim=0, count=shard_count)
        expected_bias = ops.reshard_split(bias, dim=0, count=shard_count)
        assert ops.equal(expected_weight, sharded_theta("weight"))
        assert ops.equal(expected_bias, sharded_theta("bias"))


class ShardLikeTest(unittest.TestCase):
    def testReshardLikeReplicatedToReplicated(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_count = 2
        expected_result = ops.replicate(tensor, count=shard_count)
        actual_result = ops.reshard_like(expected_result, expected_result)
        assert expected_result.is_deep_equal(actual_result)

    def testReshardLikeReplicatedToSharded(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 2
        shard_count = 3
        expected_result = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)
        replicated_tensor = ops.replicate(tensor, count=shard_count)
        actual_result = ops.reshard_like(replicated_tensor, expected_result)
        assert expected_result.is_deep_equal(actual_result)

    def testReshardLikeReplicatedToUnsharded(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_count = 2
        replicated = ops.replicate(tensor, count=shard_count)
        actual_result = ops.reshard_like(replicated, tensor)
        expected_result = tensor
        assert ops.equal(expected_result, actual_result)

    def testReshardLikeShardedToUnsharded(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 0
        shard_count = 2
        sharded = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)
        actual_result = ops.reshard_like(sharded, tensor)
        expected_result = tensor
        assert ops.equal(expected_result, actual_result)

    def testReshardLikeUnshardedToReplicated(self):
        tensor = torch.rand(4, 5, dtype=torch.float32)
        shard_count = 3
        expected_result = ops.replicate(tensor, count=shard_count)
        actual_result = ops.reshard_like(tensor, expected_result)
        assert expected_result.is_deep_equal(actual_result)

    def testReshardLikeUnshardedToSharded(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 2
        shard_count = 3
        expected_result = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)
        actual_result = ops.reshard_like(tensor, expected_result)
        assert expected_result.is_deep_equal(actual_result)

    def testReshardLikeShardedToShared(self):
        tensor = torch.rand(5, 6, dtype=torch.float32)
        shard_dim = 1
        shard_count = 3
        expected_result = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)
        actual_result = ops.reshard_like(expected_result, expected_result)
        assert expected_result.is_deep_equal(actual_result)


class UnshardTest(unittest.TestCase):
    def testUnshardSplitTensor(self):
        tensor = torch.rand(4, 5, 6, dtype=torch.float32)
        shard_dim = 0
        shard_count = 2
        sharded = ops.reshard_split(tensor, dim=shard_dim, count=shard_count)
        actual_result = ops.unshard(sharded)
        expected_result = tensor
        assert ops.equal(expected_result, actual_result)


if __name__ == "__main__":
    unittest.main()
