# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from shortfin.build import *


@entrypoint
def mobilenet():
    fetch_http(
        name="mobilenet.onnx",
        url="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
    )
    onnx_import(
        name="mobilenet.mlir",
        source="mobilenet.onnx",
    )
    return "mobilenet.mlir"
