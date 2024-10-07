# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from iree.turbine.aot import (
    ParameterArchiveBuilder,
)


class ShardedArchiveBuilder(ParameterArchiveBuilder):
    """A ParameterArchiveBuilder that can contain subordinate builders for each rank.

    TODO: This currently collects all data in memory and commits at once. This
    can be made much more memory efficient for computed datasets by exposing
    a Python streaming save helper upstream and using that.
    """

    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = save_path
        self._rank_builders: dict[int, ParameterArchiveBuilder] = {}

    def for_rank(self, rank: int) -> ParameterArchiveBuilder:
        """Returns a ParameterArchiveBuilder for tensors specific to the given rank."""
        if rank in self._rank_builders:
            return self._rank_builders[rank]
        b = ParameterArchiveBuilder()
        self._rank_builders[rank] = b
        return b

    def commit(self):
        """Performs final commit of all builders to disk."""
        self.save(self.save_path)
        for i, rank_builder in self._rank_builders.items():
            rank_builder.save(ShardedArchiveBuilder.path_for_rank(self.save_path, i))

    @staticmethod
    def path_for_rank(path: Path, rank: int):
        """Returns a path to a file modified with a rank.

        Example input:
          /tmp/foobar.irpa

        Example output:
          /tmp/foobar.rank0.irpa
        """
        return path.with_suffix(f".rank{rank}{path.suffix}")
