# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler import ir

# substitute replace=True so that colliding registration don't error
def register_attribute_builder(kind, replace=True):
    def decorator_builder(func):
        AttrBuilder.insert(kind, func, replace=replace)
        return func

    return decorator_builder

ir.register_attribute_builder = register_attribute_builder
