# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Specifications describing how a tensor, ops, layers and blocks are
sharded."""

from abc import ABC, abstractmethod
from ..utils import tree
from ..types.theta import flat_to_nested_dict


class Sharding(ABC):
    def __init__(self):
        pass


class TensorSharding(Sharding):
    def __init__(self, *, shard_count: int):
        super().__init__()
        self.shard_count = shard_count


class Unsharded(TensorSharding):
    def __init__(self):
        super().__init__(shard_count=1)


class Replicated(TensorSharding):
    def __init__(self, *, shard_count: int):
        super().__init__(shard_count=shard_count)


class Split(TensorSharding):
    def __init__(self, *, shard_count: int, shard_dim: int):
        super().__init__(shard_count=shard_count)
        self.shard_dim = shard_dim


class ThetaSharding(dict):
    """Sharding for each tensor in a theta.
    It is of type dict[str, "ThetaSharding" | TensorSharding].
    """

    def __init__(self, *args, **kwargs):
        d = flat_to_nested_dict(dict(*args, **kwargs))
        for k, v in d.items():
            d[k] = tree.map_nodes(
                tree=v,
                f=lambda x: x if isinstance(x, TensorSharding) else ThetaSharding(x),
            )
        super().__init__(d)


class ThetaLayerSharding(Sharding):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def theta_sharding(self) -> ThetaSharding:
        """Returns the leaf tensor shardings.
        The nested structure would match the one of a corresponding theta for this
        layer.

        ```python
        from sharktank.ops import reshard
        theta = ...
        theta_layer_sharding = ...
        theta_sharding = theta_layer_sharding.theta_sharding()
        sharded_theta = reshard(theta, theta_sharding)
        ```
        """
        ...


class Conv2DSplitOutputChannelSharding(ThetaLayerSharding):
    def __init__(self, shard_count: int):
        super().__init__()
        self.shard_count = shard_count

    def theta_sharding(self) -> ThetaSharding:
        return ThetaSharding(
            {
                "weight": Split(shard_count=self.shard_count, shard_dim=0),
                "bias": Split(shard_count=self.shard_count, shard_dim=0),
            }
        )


class GroupNormSplitChannelSharding(ThetaLayerSharding):
    def __init__(self, shard_count: int):
        super().__init__()
        self.shard_count = shard_count

    def theta_sharding(self) -> ThetaSharding:
        return ThetaSharding(
            {
                "weight": Split(shard_count=self.shard_count, shard_dim=0),
                "bias": Split(shard_count=self.shard_count, shard_dim=0),
            }
        )


class LinearReplicatedInputSplitWeightAndBiasSharding(ThetaLayerSharding):
    def __init__(self, shard_count: int, weight_and_bias_spit_dim: int = 0):
        super().__init__()
        self.shard_count = shard_count
        self.weight_and_bias_spit_dim = weight_and_bias_spit_dim

    def theta_sharding(self) -> ThetaSharding:
        return ThetaSharding(
            {
                "premul_input": Replicated(shard_count=self.shard_count),
                "weight": Split(
                    shard_count=self.shard_count,
                    shard_dim=self.weight_and_bias_spit_dim,
                ),
                "bias": Split(
                    shard_count=self.shard_count,
                    shard_dim=self.weight_and_bias_spit_dim,
                ),
            }
        )
