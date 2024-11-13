# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence

import logging
import math
import threading

import shortfin as sf

from .config_struct import ModelParams, human_size

logger = logging.getLogger(__name__)
