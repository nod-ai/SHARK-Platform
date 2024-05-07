# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shards an LLM dataset.

This is an ad-hoc transformation which operates on the layer structure of
weights of an LLM by converting the RHS of all eligible layers to a sharded
form.
"""
from typing import Optional, Union

import re

from ...types import *


# TODO: Remove this in favor of real logging once we have the setup right for CLI work.
def _log(message, *args):
    formatted = message % args
    print(formatted)


class MmtRHSShardingTransform:
    def __init__(
        self,
        *patterns: Union[str, re.Pattern],
        num_shards: int,
        skip_on_unsupported: bool = True,
    ):
        self.patterns = patterns
        self.num_shards = num_shards
        self.skip_on_unsupported = skip_on_unsupported

    def __call__(self, it: InferenceTensor):
        name = it.name
        if not any(re.match(p, name) for p in self.patterns):
            return it
        if isinstance(it, PrimitiveTensor):
            sharded = self._shard_primitive_tensor(it)
            if sharded is not None:
                return sharded

        if self.skip_on_unsupported:
            _log("Skipping unsupported tensor: %r", it)
            return it
        else:
            raise ValueError(f"Unsupporting sharding for tensor: {it}")

    def _shard_primitive_tensor(
        self, pt: PrimitiveTensor
    ) -> Optional[list[PrimitiveTensor]]:
        t = pt.as_torch()
        shape = list(t.shape)
        if len(shape) < 2:
            return None
        shard_dim = 1
        shard_dim_size = shape[shard_dim]
        if (shard_dim_size % self.num_shards) != 0:
            return None
        shard_split_size = shard_dim_size // self.num_shards
        shard_ts = t.split(shard_split_size, dim=shard_dim)
        st = ShardedPrimitiveTensor(pt.name, pt.shape, shard_dim, shard_ts)
        _log("Sharding tensor %r -> %r", pt, st)
        return st

    def __repr__(self):
        return f"ShardingTransform()"


def main():
    from ...utils import cli

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
    cli.add_output_dataset_options(parser)
    args = cli.parse(parser)
    dataset = cli.get_input_dataset(args)

    tr = MmtRHSShardingTransform(
        r"^blk\.[0-9]+\.(attn_k|attn_q|attn_v|ffn_gate|ffn_up|ffn_down)\.weight$",
        num_shards=8,
    )
    dataset.transform(tr)
    dataset.save(args.output_irpa_file, io_report_callback=print)


if __name__ == "__main__":
    main()
