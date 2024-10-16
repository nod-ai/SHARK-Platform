# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from pathlib import Path

class BaseCompileTest(unittest.TestCase):
    def setUp(self):
        raise NotImplementedError("Subclasses should implement this method.")

    