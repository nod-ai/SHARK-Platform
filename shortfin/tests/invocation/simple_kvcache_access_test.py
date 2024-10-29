# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
This test file closely mimics the kvcache indexing, reading, and writing for open-llama-3b-v2-f16.gguf.

It's a simple test to verify that shortfin can:
- Create device arrays
- Map and fill them
- Invoke a VMFB and pass arguments to it
- Properly retain changes made by the VMFB to device arrays provided as arguments
"""

import urllib.request
import tempfile
from pathlib import Path
import functools
import pytest
import shortfin as sf
import shortfin.array as sfnp
import array


@pytest.fixture(scope="session")
def kvcache_compiled_cpu_path():
    try:
        import iree.compiler.tools as tools
    except ModuleNotFoundError:
        raise pytest.skip("iree.compiler packages not available")

    print("Compiling kvcache module")

    KVCACHE_MODULE_CONTENTS = """
    module @kvcache {
      func.func @write_kvcache(%kvcache: !torch.tensor<[?,2662400],f16>, %new_data: !torch.vtensor<[16,32,100],f16>, %page_index: !torch.vtensor<[1],si64>, %layer_index: !torch.vtensor<[1],si64>) {
        %int0 = torch.constant.int 0
        %int1 = torch.constant.int 1
        %int2 = torch.constant.int 2
        %int16 = torch.constant.int 16
        %int26 = torch.constant.int 26
        %int32 = torch.constant.int 32
        %int100 = torch.constant.int 100
        %int2662400 = torch.constant.int 2662400
        %false = torch.constant.bool false
        %none = torch.constant.none

        %0 = torch.copy.to_vtensor %kvcache : !torch.vtensor<[?,2662400],f16>

        // Get the number of pages
        %num_pages = torch.aten.size.int %0, %int0 : !torch.vtensor<[?,2662400],f16>, !torch.int -> !torch.int

        // Reshape kvcache to [?,26,2,16,32,100]
        %1 = torch.prim.ListConstruct %num_pages, %int26, %int2, %int16, %int32, %int100 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
        %2 = torch.aten.view %0, %1 : !torch.vtensor<[?,2662400],f16>, !torch.list<int> -> !torch.vtensor<[?,26,2,16,32,100],f16>

        // Create index list with the provided tensors
        %3 = torch.prim.ListConstruct %page_index, %layer_index : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.list<optional<vtensor>>

        // Update the kvcache
        %4 = torch.aten.index_put %2, %3, %new_data, %false : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[16,32,100],f16>, !torch.bool -> !torch.vtensor<[?,26,2,16,32,100],f16>

        // Reshape back to original shape
        %5 = torch.prim.ListConstruct %num_pages, %int2662400 : (!torch.int, !torch.int) -> !torch.list<int>
        %6 = torch.aten.view %4, %5 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.list<int> -> !torch.vtensor<[?,2662400],f16>

        // Overwrite the original tensor
        torch.overwrite.tensor.contents %6 overwrites %kvcache : !torch.vtensor<[?,2662400],f16>, !torch.tensor<[?,2662400],f16>

        return
      }

      func.func @read_kvcache(%kvcache: !torch.tensor<[?,2662400],f16>, %page_index: !torch.vtensor<[1],si64>, %layer_index: !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,16,32,100],f16> {
        %int0 = torch.constant.int 0
        %int1 = torch.constant.int 1
        %int2 = torch.constant.int 2
        %int16 = torch.constant.int 16
        %int26 = torch.constant.int 26
        %int32 = torch.constant.int 32
        %int100 = torch.constant.int 100
        %int2662400 = torch.constant.int 2662400
        %none = torch.constant.none

        %0 = torch.copy.to_vtensor %kvcache : !torch.vtensor<[?,2662400],f16>

        // Get the number of pages
        %num_pages = torch.aten.size.int %0, %int0 : !torch.vtensor<[?,2662400],f16>, !torch.int -> !torch.int

        // Reshape kvcache to [?,26,2,16,32,100]
        %1 = torch.prim.ListConstruct %num_pages, %int26, %int2, %int16, %int32, %int100 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
        %2 = torch.aten.view %0, %1 : !torch.vtensor<[?,2662400],f16>, !torch.list<int> -> !torch.vtensor<[?,26,2,16,32,100],f16>

        // Create index list with the provided tensors
        %3 = torch.prim.ListConstruct %page_index, %layer_index : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.list<optional<vtensor>>

        // Read from the kvcache and squeeze the result
        %4 = torch.aten.index.Tensor %2, %3 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.list<optional<vtensor>> -> !torch.vtensor<[1,2,16,32,100],f16>
        %5 = torch.aten.squeeze.dim %4, %int0 : !torch.vtensor<[1,2,16,32,100],f16>, !torch.int -> !torch.vtensor<[2,16,32,100],f16>

        return %5 : !torch.vtensor<[2,16,32,100],f16>
      }
    }
    """

    # Create temporary directory for our files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        # Write MLIR to temporary file
        mlir_path = tmp_dir_path / "kvcache.mlir"
        mlir_path.write_text(KVCACHE_MODULE_CONTENTS)

        # Define output path for compiled binary
        vmfb_path = tmp_dir_path / "kvcache_cpu.vmfb"

        # Compile the MLIR to VMFB
        tools.compile_file(
            str(mlir_path),
            output_file=str(vmfb_path),
            target_backends=["llvm-cpu"],
            input_type="AUTO",
        )

        # Read the compiled binary
        compiled_binary = vmfb_path.read_bytes()

        # Create a new temporary file for the final vmfb
        final_vmfb = tmp_dir_path / "final_kvcache_cpu.vmfb"
        final_vmfb.write_bytes(compiled_binary)

        yield final_vmfb


@pytest.fixture
def lsys():
    sc = sf.host.CPUSystemBuilder()
    lsys = sc.create_system()
    yield lsys
    lsys.shutdown()


@pytest.fixture
def fiber(lsys):
    return lsys.create_fiber()


@pytest.fixture
def device(fiber):
    return fiber.device(0)


def create_scalar_device_array(device, value, dtype=sfnp.int64):
    """Helper function to create a scalar device array."""
    arr = sfnp.device_array.for_device(device, [1], dtype)
    staging = arr.for_transfer()
    with staging.map(discard=True) as m:
        m.fill(value)
    arr.copy_from(staging)
    return arr


@pytest.mark.parametrize(
    "await_before_invoke",
    [
        True,
        False,  # if we don't await GPU it might fail.
    ],
)
def test_kvcache_noreturn(lsys, fiber, kvcache_compiled_cpu_path, await_before_invoke):
    try:
        import numpy as np
    except ImportError:
        raise pytest.skip("numpy not available")
    device = fiber.device(0)
    program_module = lsys.load_module(kvcache_compiled_cpu_path)
    program = sf.Program([program_module], devices=fiber.raw_devices)

    write_function = program["kvcache.write_kvcache"]
    read_function = program["kvcache.read_kvcache"]

    # Test parameters
    num_pages = 4
    num_layers = 26  # Number of transformer layers
    num_kv = 2  # K and V states
    batch_size = 16
    num_heads = 32
    head_dim = 100

    # Initialize test data - note we're only writing one layer at a time
    test_data = np.random.uniform(-1, 1, (batch_size, num_heads, head_dim)).astype(
        np.float16
    )

    # The kvcache shape should match the MLIR module's expected shape
    # [num_pages, num_layers * num_kv * batch_size * num_heads * head_dim]
    total_dim = num_layers * num_kv * batch_size * num_heads * head_dim
    assert total_dim == 2662400
    kvcache_shape = [num_pages, total_dim]
    kvcache_data = np.zeros(kvcache_shape, dtype=np.float16)

    async def main():
        # Create device arrays
        device_kvcache = sfnp.device_array(device, kvcache_shape, sfnp.float16)
        device_new_data = sfnp.device_array(
            device, [batch_size, num_heads, head_dim], sfnp.float16
        )

        # Initialize kvcache on device
        staging_kvcache = device_kvcache.for_transfer()
        with staging_kvcache.map(discard=True) as m:
            m.fill(array.array("H", kvcache_data.tobytes()))
        device_kvcache.copy_from(staging_kvcache)

        # Initialize new data on device
        staging_new_data = device_new_data.for_transfer()
        with staging_new_data.map(discard=True) as m:
            m.fill(array.array("H", test_data.tobytes()))
        device_new_data.copy_from(staging_new_data)

        # Test writing and reading for both K and V states
        for layer_idx in range(2):  # Test first two layers
            for kv_idx in range(num_kv):  # Test both K and V
                page_index = create_scalar_device_array(device, 1)  # Write to page 1
                layer_index = create_scalar_device_array(device, layer_idx)

                # Write to kvcache
                if await_before_invoke:
                    await device
                ret = await write_function(
                    device_kvcache,
                    device_new_data,
                    page_index,
                    layer_index,
                    fiber=fiber,
                )

                # Read from kvcache
                if await_before_invoke:
                    await device
                (read_result,) = await read_function(
                    device_kvcache, page_index, layer_index, fiber=fiber
                )

                # Transfer results back to host
                host_result = read_result.for_transfer()
                host_result.copy_from(read_result)
                await device

                # Convert result to numpy array for comparison
                result_array = np.frombuffer(
                    host_result.items.tobytes(), dtype=np.float16
                ).reshape(num_kv, batch_size, num_heads, head_dim)

                # Verify numerical correctness for the specific K/V state
                np.testing.assert_allclose(
                    result_array[kv_idx],
                    test_data,
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"KV cache read/write mismatch for layer {layer_idx}, {'key' if kv_idx == 0 else 'value'} state",
                )

                # Additional statistical checks
                result_mean = np.mean(np.abs(result_array[kv_idx]))
                test_mean = np.mean(np.abs(test_data))
                np.testing.assert_allclose(
                    result_mean,
                    test_mean,
                    rtol=1e-3,
                    err_msg=f"Mean absolute values don't match for layer {layer_idx}, {'key' if kv_idx == 0 else 'value'} state: got {result_mean}, expected {test_mean}",
                )

    lsys.run(main())
