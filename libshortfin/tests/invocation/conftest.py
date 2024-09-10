# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import urllib.request


def upgrade_onnx(original_path, converted_path):
    import onnx

    original_model = onnx.load_model(original_path)
    converted_model = onnx.version_converter.convert_version(original_model, 17)
    onnx.save(converted_model, converted_path)


@pytest.fixture(scope="session")
def mobilenet_onnx_path(tmp_path_factory):
    try:
        import onnx
    except ModuleNotFoundError:
        raise pytest.skip("onnx python package not available")
    print("Downloading mobilenet.onnx")
    parent_dir = tmp_path_factory.mktemp("mobilenet_onnx")
    orig_onnx_path = parent_dir / "mobilenet_orig.onnx"
    urllib.request.urlretrieve(
        "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
        orig_onnx_path,
    )
    upgraded_onnx_path = parent_dir / "mobilenet.onnx"
    upgrade_onnx(orig_onnx_path, upgraded_onnx_path)
    return upgraded_onnx_path


@pytest.fixture(scope="session")
def mobilenet_compiled_cpu_path(mobilenet_onnx_path):
    try:
        import iree.compiler.tools as tools
        import iree.compiler.tools.import_onnx.__main__ as import_onnx
    except ModuleNotFoundError:
        raise pytest.skip("iree.compiler packages not available")
    print("Compiling mobilenet")
    mlir_path = mobilenet_onnx_path.parent / "mobilenet.mlir"
    vmfb_path = mobilenet_onnx_path.parent / "mobilenet_cpu.vmfb"
    args = import_onnx.parse_arguments(["-o", str(mlir_path), str(mobilenet_onnx_path)])
    import_onnx.main(args)
    tools.compile_file(
        str(mlir_path),
        output_file=str(vmfb_path),
        target_backends=["llvm-cpu"],
        input_type="onnx",
    )
    return vmfb_path
