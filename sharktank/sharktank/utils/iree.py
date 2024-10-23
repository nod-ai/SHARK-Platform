# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.runtime
from typing import List, Tuple, Optional, Union
from pathlib import Path
import torch
import numpy as np
import collections.abc
from collections import OrderedDict
from ..types.tensors import (
    AnyTensor,
    InferenceTensor,
    ShardedTensor,
    DefaultPrimitiveTensor,
    unbox_tensor,
    torch_tree_flatten,
)
from .tree import Tree


def get_iree_devices(driver: str, device_count: int) -> List[iree.runtime.HalDevice]:
    hal_driver = iree.runtime.get_driver(driver)
    available_devices = hal_driver.query_available_devices()
    if driver in ["local-task", "local-sync"]:
        # Use the same actual device for all devices.
        return [
            hal_driver.create_device(available_devices[0]) for _ in range(device_count)
        ]
    else:
        return [
            hal_driver.create_device(available_devices[i]) for i in range(device_count)
        ]


def load_iree_module(
    module_path: str,
    devices: List[iree.runtime.HalDevice],
    parameters_path: Optional[str] = None,
) -> Tuple[iree.runtime.VmModule, iree.runtime.VmContext, iree.runtime.VmInstance]:
    """The VmContext and VmInstance need to outlive the VmModule and any device
    buffers."""
    vm_instance = iree.runtime.VmInstance()
    hal_module = iree.runtime.create_hal_module(instance=vm_instance, devices=devices)
    modules = [hal_module]
    if parameters_path is not None:
        params_path = Path(parameters_path)
        parameter_index = iree.runtime.ParameterIndex()
        if len(devices) > 1:
            # TODO: make IREE able to load the parameters from the top parameter file
            # without having to specify the parameter file for each shard separately.
            for i in range(len(devices)):
                parameter_index.load(
                    file_path=str(
                        Path(params_path).with_suffix(f".rank{i}{params_path.suffix}")
                    )
                )
        else:
            parameter_index.load(file_path=str(params_path))
        parameter_provider = parameter_index.create_provider(scope="model")
        parameters_module = iree.runtime.create_io_parameters_module(
            vm_instance, parameter_provider
        )
        modules.append(parameters_module)
    vm_module = iree.runtime.VmModule.mmap(vm_instance, str(module_path))
    modules.append(vm_module)
    vm_context = iree.runtime.VmContext(instance=vm_instance, modules=modules)
    return vm_module, vm_context, vm_instance


def run_iree_module_function(
    module: iree.runtime.VmModule,
    vm_context: iree.runtime.VmContext,
    args: List[iree.runtime.DeviceArray],
    driver: str,
    function_name: str = "main",
    trace_path_prefix: Optional[str] = None,
) -> List[iree.runtime.DeviceArray]:
    """Run IREE module function with optional tracing of arguments/results."""
    vm_function = module.lookup_function(function_name)
    invoker = iree.runtime.FunctionInvoker(
        vm_context=vm_context,
        # TODO: rework iree.runtime.FunctionInvoker interface for multiple devices.
        # This works, but does not look right.
        device=iree.runtime.get_device(driver, cache=False),
        vm_function=vm_function,
    )
    if trace_path_prefix is not None:
        for i, arg in enumerate(args):
            np.save(f"{trace_path_prefix}{function_name}_arg{i}.npy", arg.to_host())
    results = invoker(*args)
    if isinstance(results, iree.runtime.DeviceArray):
        results = (results,)

    if trace_path_prefix is not None:
        for i, arg in enumerate(args):
            np.save(
                f"{trace_path_prefix}{function_name}_arg{i}_post_call.npy",
                arg.to_host(),
            )
        for i, arg in enumerate(results):
            np.save(f"{trace_path_prefix}{function_name}_result{i}.npy", arg.to_host())
    return results


def prepare_iree_module_function_args(
    args: List[Union[AnyTensor, List[AnyTensor]]], devices: List[iree.runtime.HalDevice]
) -> List[iree.runtime.DeviceArray]:
    """Flatten composite tensors into their parts and place them on devices.
    Sharded tensors become a list of their shards while placing them onto their
    corresponding device.
    All unsharded tensors go on device 0.
    """
    res = []
    for arg in args:
        if isinstance(arg, ShardedTensor):
            assert len(devices) == len(arg.shards)
            res.extend(
                [
                    prepare_iree_module_function_args([shard], [device])[0]
                    for shard, device in zip(arg.shards, devices)
                ]
            )
        elif isinstance(arg, (DefaultPrimitiveTensor, torch.Tensor)):
            res.append(
                iree.runtime.asdevicearray(
                    devices[0], unbox_tensor(arg).to("cpu").numpy()
                )
            )
        else:
            assert isinstance(arg, collections.abc.Sequence)
            res.extend(prepare_iree_module_function_args(arg, devices))
    return res


def flatten_for_iree_signature(tree: Tree) -> List[torch.Tensor]:
    """Flatten a tree of arguments or results for an IREE call.
    E.g. sharded tensors gets flattened into their shards."""

    return torch_tree_flatten(tree)[0]


def call_torch_module_function(
    module: torch.nn.Module,
    function_name: str,
    kwargs: OrderedDict,
    trace_path_prefix: Optional[str] = None,
):
    """Call a torch module function with optional tracing.
    For tracing the arguments/results are flattened to match IREE's signature."""
    assert isinstance(
        kwargs, OrderedDict
    ), "Make sure when flattening the order is preserved"
    if trace_path_prefix is not None:
        flat_args = flatten_for_iree_signature(kwargs)
        for i, arg in enumerate(flat_args):
            np.save(
                f"{trace_path_prefix}{function_name}_arg{i}.npy",
                arg.to("cpu").numpy(),
            )
    res = getattr(module, function_name)(**kwargs)
    if trace_path_prefix is not None:
        flat_args = flatten_for_iree_signature(kwargs)
        for i, arg in enumerate(flat_args):
            np.save(
                f"{trace_path_prefix}{function_name}_arg{i}.npy",
                arg.to("cpu").numpy(),
            )
        results = (
            (res,)
            if isinstance(
                res,
                (
                    torch.Tensor,
                    InferenceTensor,
                ),
            )
            else res
        )
        flat_results = flatten_for_iree_signature(results)
        for i, result in enumerate(flat_results):
            np.save(
                f"{trace_path_prefix}{function_name}_result{i}.npy",
                result.to("cpu").numpy(),
            )
    return res


def iree_to_torch(*tensors: iree.runtime.DeviceArray) -> List[torch.Tensor]:
    return [torch.tensor(tensor.to_host()) for tensor in tensors]
