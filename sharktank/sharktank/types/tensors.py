# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Optional, Union, TypeVar, Generic, Type

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from ..utils.math import ceildiv
from shark_turbine.aot import (
    ExternalTensorTrait,
)

from ..utils.io import ShardedArchiveBuilder

__all__ = [
    "register_quantized_layout",
    "InferenceTensor",
    "MetaDataValueType",
    "DefaultPrimitiveTensor",
    "PlanarQuantizedTensor",
    "PrimitiveTensor",
    "QuantizedTensor",
    "QuantizedLayout",
    "ReplicatedTensor",
    "ShardedTensor",
    "SplitPrimitiveTensor",
    "UnreducedTensor",
]

# JSON encodable value types.
MetaDataValueType = Union[int, bool, float, str]
UnnamedTensorName = "<unnamed>"


class QuantizedLayout(ABC):
    @abstractmethod
    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        ...

    @classmethod
    @abstractmethod
    def serialized_name(cls) -> str:
        """Returns the globally unique serialization name for this layout."""
        ...

    @classmethod
    @abstractmethod
    def create(
        cls,
        shape: list[int],
        metadata: Optional[dict[str, MetaDataValueType]],
        planes: dict[str, torch.Tensor],
    ) -> "QuantizedLayout":
        ...

    @property
    @abstractmethod
    def planes(self) -> dict[str, torch.Tensor]:
        """Returns the planes of this layout as concrete, named tensors.

        When transforming, the name will form a local suffix (i.e. ":name")
        for stored values by combining the global name with the ":" separator.
        """
        ...

    @property
    def metadata(self) -> Optional[dict[str, MetaDataValueType]]:
        """Additional metadata needed to reconstruct a layout."""
        return None


QuantizedLayoutT = TypeVar("QuantizedLayoutT", bound=QuantizedLayout)


REGISTERED_LAYOUT_CLASSES: dict[str, Type[QuantizedLayout]] = {}


def register_quantized_layout(ty: Type[QuantizedLayoutT]) -> Type[QuantizedLayoutT]:
    """Class decorator which registers a serializable layout class."""
    name = ty.serialized_name()
    existing = REGISTERED_LAYOUT_CLASSES.get(name)
    assert (
        existing is None
    ), f"Duplicate QuantizedLayoutRegistration '{name}' ({ty} vs {existing})"
    REGISTERED_LAYOUT_CLASSES[name] = ty
    return ty


@dataclass
class InferenceTensorMetadata:
    # Registered name of an InferenceTensor subclass.
    type_name: str
    # Mapping of constituent local names to parameter archive global names
    # of individual tensors that make up this InferenceTensor.
    raw_tensors: dict[str, str]
    # Additional properties needed to restore the instance. Must be JSON
    # legal types. Will be added to the root JSON dictionary.
    extra_properties: Optional[dict[str, Any]] = None

    def create_instance(self) -> "InferenceTensor":
        try:
            clazz = REGISTERED_INFERENCE_TENSOR_CLASSES[self.type_name]
        except KeyError as e:
            raise IOError(
                f"Unable to create instance of unregistered type {self.type_name}"
            ) from e
        assert issubclass(clazz, InferenceTensor)

    def to_json(self) -> dict:
        d = {
            "type_name": self.type_name,
            "raw_tensors": self.raw_tensors,
        }
        if self.extra_properties is not None:
            d.update(self.extra_properties)
        return d

    def from_json(obj: dict) -> "InferenceTensorMetadata":
        extra_properties = dict(obj)
        try:
            type_name = extra_properties["type_name"]
            assert isinstance(type_name, str)
            del extra_properties["type_name"]
            raw_tensors = extra_properties["raw_tensors"]
            assert isinstance(raw_tensors, dict)
            del extra_properties["raw_tensors"]
        except Exception as e:
            raise IOError(f"Error decoding InferenceTensorMetadata object") from e

        # Validate.
        for k, v in raw_tensors.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise IOError(
                    f"Bad format for InferenceTensorMetadata.raw_tensors ({type(k)}, {type(v)})"
                )

        return InferenceTensorMetadata(
            type_name=type_name,
            raw_tensors=raw_tensors,
            extra_properties=extra_properties,
        )


class InferenceTensor(ABC):
    """Provides access to a tensor in the model used for inference.

    InferenceTensors have a richer structure than "normal" training tensors
    since they often involve a degree of layout on top of the raw data tensor.
    """

    def __init__(self, *, shape: list[int], name: str = UnnamedTensorName):
        self.name = name
        self.shape = shape

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ) -> "InferenceTensor":
        raise NotImplementedError(
            f"InferenceTensor {cls} cannot be deserialized "
            f"because it does not implement create()"
        )

    @classmethod
    def serialized_name(cls) -> str:
        """Returns the globally unique serialization name for this type."""
        raise NotImplementedError(
            f"InferenceTensor {cls} cannot be directly "
            f"serialized (does not implement serialized_name())"
        )

    @property
    @abstractmethod
    def globals(self) -> dict[str, torch.Tensor]:
        """Returns a mapping of global name to root tensor.

        The primary accessors on an InferenceTensor access the root tensors in
        the global set, all of which in a root Theta must have unique names.
        """
        ...

    @abstractmethod
    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        ...

    def is_deep_equal(self, other: Any) -> bool:
        """Deep equality including metadata and exact equality of tensor elements.
        It is a representational equality."""
        raise NotImplementedError()

    def transform_globals(
        self, *transforms: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    ) -> "InferenceTensor":
        """Appplies transformation functions to the InferenceTensors backing
        globals.

        Each transformation must produce a new dict of a form that the subclass
        can handle. Practically, this means that placement and layout related
        changes are always allowed, while more invasive changes (like dtype)
        are more case by case.

        Returns a new InferenceTensor, mutated.
        """
        prev_globals = self.globals
        for transform in transforms:
            next_globals = transform(prev_globals)
            # Copy any metadata from prior to next.
            for k, prev_t in prev_globals.items():
                new_t = next_globals.get(k)
                if new_t is None:
                    continue
                if new_t is not prev_t:
                    ext_trait = ExternalTensorTrait.get(prev_t)
                    if ext_trait is not None:
                        ext_trait.set(new_t)
            prev_globals = next_globals
        return self._clone_with_globals(prev_globals)

    def to(
        self, *, device: Optional[Union[str, torch.device]] = None
    ) -> "InferenceTensor":
        return self.transform_globals(
            lambda d: {k: t.to(device=device) for k, t in d.items()}
        )

    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        raise NotImplementedError(
            f"InferenceTensor {type(self)} does not implement _clone_with_globals"
        )

    @property
    def T(self) -> "InferenceTensor":
        from ..ops import permute

        # Reverse the dimension range.
        rank = len(self.shape)
        assert rank == 2, "T will be deprecated in torch for non-2D tensors"
        dims = [rank - 1 - i for i in range(rank)]

        return permute(self, dims=dims)

    def __add__(self, rhs):
        from ..ops import elementwise

        return elementwise(torch.add, self, rhs)


REGISTERED_INFERENCE_TENSOR_CLASSES: dict[str, Type[InferenceTensor]] = {}


def register_inference_tensor(ty: Type[QuantizedLayoutT]) -> Type[QuantizedLayoutT]:
    """Class decorator which registers a serializable InferenceTensor class.

    This should only be used to decorate concrete implementations that need to
    be loaded by name.
    """
    name = ty.serialized_name()
    existing = REGISTERED_INFERENCE_TENSOR_CLASSES.get(name)
    assert (
        existing is None
    ), f"Duplicate InferenceTensor registration '{name}' ({ty} vs {existing})"
    REGISTERED_INFERENCE_TENSOR_CLASSES[name] = ty
    return ty


########################################################################################
# Primitive tensors
########################################################################################


class PrimitiveTensor(InferenceTensor):
    """An InferenceTensor without any kind of special layout.

    These can be directly operated on as a torch.Tensor.
    """

    @abstractmethod
    def as_torch(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Accesses the raw data as a torch tensor.

        If the tensor is packed in some way, this may bare no resemblance to
        the logical arrangement of the data.
        """
        ...


@register_inference_tensor
class DefaultPrimitiveTensor(PrimitiveTensor):
    """Concrete implementation of a PrimitiveTensor based on a single tensor."""

    def __init__(
        self,
        *,
        data: torch.Tensor,
        name: str = UnnamedTensorName,
    ):
        super().__init__(name=name, shape=list(data.shape))
        self._data = data

    @classmethod
    def serialized_name(cls) -> str:
        return "PrimitiveTensor"

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ) -> "InferenceTensor":
        try:
            data = raw_tensors[""]
        except KeyError as e:
            raise IOError(f"Missing component tensor") from e
        return cls(name=name, data=data)

    def as_torch(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is not None:
            return self._data.to(dtype)
        return self._data

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        return {
            self.name: self._data,
        }

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        builder.add_tensor(self.name, self._data)
        return InferenceTensorMetadata(self.serialized_name(), {"": self.name})

    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        return DefaultPrimitiveTensor(name=self.name, data=new_globals[self.name])

    def __repr__(self):
        return f"PrimitiveTensor({self.name}, {self.shape}, {self._data.dtype})"

    def is_deep_equal(self, other: Any) -> bool:
        if not isinstance(other, DefaultPrimitiveTensor):
            return False
        if self.shape != other.shape or self.name != other.name:
            return False
        return torch.equal(self.as_torch(), other.as_torch())


########################################################################################
# Quantized tensors
########################################################################################


class QuantizedTensor(InferenceTensor, Generic[QuantizedLayoutT]):
    """An inference tensor that is quantized/packed."""

    def __init__(
        self,
        *,
        shape: list[int],
        layout_type: Type[QuantizedLayout],
        name: str = UnnamedTensorName,
    ):
        super().__init__(name=name, shape=shape)
        self.layout_type = layout_type

    @abstractmethod
    def unpack(self) -> QuantizedLayoutT:
        ...

    def to_planar(self) -> "PlanarQuantizedTensor":
        """Converts this QuantizedTensor to a generic planar form.

        This is done for serialization and to materialize unpacking.
        If a subclass cannot be converted to planar form generically like this,
        it should override this method to implement properly or raise
        NotImplementedError.
        """
        return PlanarQuantizedTensor(self.name, self.shape, self.unpack())

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """By default all QuantizedTensors serialize as a generic PlanarQuantizedTensor.

        If this is not desirable, subclasses should override.
        """
        return self.to_planar().add_to_archive(builder)


@register_inference_tensor
class PlanarQuantizedTensor(QuantizedTensor):
    """Generic planar tensor backed by an instantiated QuantizedLayout.

    This is used for materialized, unpacked layouts (i.e. no unpacking
    will be done).
    """

    def __init__(
        self,
        *,
        shape: list[int],
        layout: QuantizedLayout,
        name: str = UnnamedTensorName,
    ):
        super().__init__(name=name, shape=shape, layout_type=type(layout))
        self.layout = layout

    def to_planar(self) -> "PlanarQuantizedTensor":
        # Already planar.
        return self

    @classmethod
    def serialized_name(cls) -> str:
        return "PlanarQuantizedTensor"

    def unpack(self) -> QuantizedLayout:
        return self.layout

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        global_name = self.name
        planes = self.layout.planes
        return {f"{global_name}:{k}": v for k, v in planes.items()}

    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        # Clone it via layout serialization.
        serialized_name = self.layout.serialized_name()
        global_prefix = f"{self.name}:"
        orig_planes = self.layout.planes
        new_planes = {}
        for plane_name in orig_planes.keys():
            # Planes are stored in the globals dict with the inference
            # tensor's name and colon prepended, so look up that way.
            new_planes[plane_name] = new_globals[f"{global_prefix}{plane_name}"]

        # Create a new layout via the serialization adaptor.
        try:
            layout_clazz = REGISTERED_LAYOUT_CLASSES[serialized_name]
        except KeyError:
            raise IOError(
                f"Cannot deserialize PlanarQuantizedTensor because of unregistered layout "
                f"{serialized_name}"
            )
        new_layout = layout_clazz.create(self.shape, self.layout.metadata, new_planes)

        return PlanarQuantizedTensor(
            name=self.name,
            shape=self.shape,
            layout=new_layout,
        )

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ) -> "InferenceTensor":
        try:
            shape = extra_properties["shape"]
            layout_type_name = extra_properties["layout_type"]
            layout_metadata = extra_properties.get("layout_metadata")
        except KeyError as e:
            raise IOError(f"Missing PlanarQuantizedTensor deserialization prop") from e

        shape = [int(d) for d in shape]
        try:
            layout_clazz = REGISTERED_LAYOUT_CLASSES[layout_type_name]
        except KeyError:
            raise IOError(
                f"Cannot deserialize PlanarQuantizedTensor because of unregistered layout "
                f"{layout_type_name}"
            )

        layout = layout_clazz.create(shape, layout_metadata, raw_tensors)
        return PlanarQuantizedTensor(name=name, shape=shape, layout=layout)

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        root_name = self.name
        layout = self.unpack()
        name_map: dict[str, str] = {}
        for suffix, plane in layout.planes.items():
            irpa_name = f"{root_name}:{suffix}"
            builder.add_tensor(irpa_name, plane)
            name_map[suffix] = irpa_name
        extra_properties = {
            "shape": [int(d) for d in self.shape],
            "layout_type": self.layout.serialized_name(),
        }
        layout_metadata = self.layout.metadata
        if layout_metadata is not None:
            extra_properties["layout_metadata"] = layout_metadata
        return InferenceTensorMetadata(
            PlanarQuantizedTensor.serialized_name(),
            name_map,
            extra_properties=extra_properties,
        )

    def __repr__(self):
        return f"PlanarQuantized({self.name}, {self.shape}, layout={self.layout})"


########################################################################################
# Sharded tensors
########################################################################################


class ShardedTensor(InferenceTensor):
    """A sharded tensor contains a list of tensor-parallel shards, one for each rank.

    The shape of the overall sharded tensor is the un-sharded shape.
    """

    def __init__(
        self, *, shape: list[int], shard_dim: int | None, name: str = UnnamedTensorName
    ):
        super().__init__(name=name, shape=shape)
        self.shard_dim = shard_dim

    @property
    @abstractmethod
    def shard_count(self) -> int:
        ...

    @property
    @abstractmethod
    def shards(self) -> tuple[InferenceTensor]:
        """Accesses the underlying shards"""
        ...

    @property
    @abstractmethod
    def is_replicated(self) -> bool:
        """Returns whether the original tensor is replicated.
        If replicated, `shard_dim` does not make sense and is None."""
        ...


@register_inference_tensor
class ShardedTensorBase(ShardedTensor):
    """Sharded tensor which contains tensors.

    The sharded tensors have names with this tensor's name as the stem and
    a suffix of f".shard.{i}" where i is from 0..shard_count-1.
    """

    def __init__(
        self,
        *,
        shard_dim: int | None,
        ts: list[torch.Tensor],
        name: str = UnnamedTensorName,
        shape: Optional[list[int]],
    ):
        assert len(ts) > 0
        assert shard_dim is None or len(ts[0].shape) > shard_dim
        super().__init__(name=name, shape=shape, shard_dim=shard_dim)
        self._shards: tuple[DefaultPrimitiveTensor] = tuple(
            DefaultPrimitiveTensor(name=f"{name}.shard.{i}", data=t)
            for i, t in enumerate(ts)
        )

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    @property
    def shards(self) -> tuple[InferenceTensor]:
        return self._shards

    @property
    def is_replicated(self) -> bool:
        return False

    @classmethod
    def serialized_name(cls) -> str:
        return cls.__name__

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        return {pt.name: pt._data for pt in self._shards}

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        for i, pt in enumerate(self._shards):
            builder.for_rank(i).add_tensor(pt.name, pt._data)
        extra_properties = {
            "shard_count": len(self._shards),
            "shape": list(self.shape),
        }
        if self.shard_dim is not None:
            extra_properties.update({"shard_dim": self.shard_dim})
        return InferenceTensorMetadata(
            self.serialized_name(),
            {str(i): pt.name for i, pt in enumerate(self._shards)},
            extra_properties=extra_properties,
        )

    @classmethod
    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        ts = []
        for k in self.globals.keys():
            ts.append(new_globals[ts[k]])
        return self.__class__(
            name=self.name, shape=self.shape, shard_dim=self.shard_dim, ts=ts
        )

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ) -> "InferenceTensor":
        shard_count = int(extra_properties["shard_count"])
        shape = list(extra_properties["shape"])
        shard_dim = (
            int(extra_properties["shard_dim"])
            if "shard_dim" in extra_properties
            else None
        )
        ts = []
        for i in range(shard_count):
            t_name = str(i)
            try:
                t = raw_tensors[t_name]
                ts.append(t)
            except KeyError as e:
                raise IOError(
                    f"Missing component tensor '{t_name}' in {raw_tensors.keys()}"
                ) from e
        if shard_dim is None:
            return cls(name=name, shape=shape, ts=ts)
        else:
            return cls(name=name, shape=shape, ts=ts, shard_dim=shard_dim)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.name}, {self.shape}, "
            + ("" if self.shard_dim is None else f"shard_dim={self.shard_dim}, ")
            + f"shard_count={len(self._shards)} "
            f"of {self.shards[0].shape})"
        )

    def is_deep_equal(self, other: Any) -> bool:
        if type(self) != type(other):
            return False
        if (
            self.shard_count != other.shard_count
            or self.shard_dim != other.shard_dim
            or self.name != other.name
            or self.shape != other.shape
        ):
            return False
        return all(a.is_deep_equal(b) for a, b in zip(self.shards, other.shards))


@register_inference_tensor
class SplitPrimitiveTensor(ShardedTensorBase):
    """Sharded tensor split along a dimension into primitive tensors.

    The sharded tensors have names with this tensor's name as the stem and
    a suffix of f".shard.{i}" where i is from 0..shard_count-1.
    """

    def __init__(
        self,
        *,
        shard_dim: int,
        ts: list[torch.Tensor] | torch.Tensor,
        shard_count: None | int = None,
        name: str = UnnamedTensorName,
        shape: Optional[list[int]] = None,
    ):
        """
        If `ts` is a list of tensors, it is interpreted as the shards.
        Then `shard_count` must be None.
        If `ts` is a tensor then `shard_count` must be provided and it,
        will be split along dimension `shard_dim` into `shard_count`
        number of pieces.
        """
        if isinstance(ts, torch.Tensor):
            assert shard_count is not None
            ts = ts.split(ceildiv(ts.shape[shard_dim], shard_count), dim=shard_dim)
            assert len(ts) == shard_count
            shard_count = None

        assert shard_count is None
        assert len(ts) > 0
        first_shape = ts[0].shape
        assert len(first_shape) > shard_dim
        if shape is None:
            # Compute the shape.
            shape = list(first_shape)
            shape[shard_dim] *= len(ts)

        # Assert the shape.
        shard_dim_size = first_shape[shard_dim]
        for t in ts[1:]:
            assert (
                t.shape == first_shape
            ), f"Shape mismatch for split tensors: {t.shape} vs {first_shape}"
            shard_dim_size += t.shape[shard_dim]
        assert (
            shard_dim_size == shape[shard_dim]
        ), f"Sharding mismatch: Sharded dims do not cover the whole volume {shard_dim_size} vs {shape[shard_dim]}"

        super().__init__(name=name, ts=ts, shape=shape, shard_dim=shard_dim)


@register_inference_tensor
class ReplicatedTensor(ShardedTensor):
    """A tensor that is replicated across all shards."""

    def __init__(
        self,
        *,
        ts: list[torch.Tensor] | torch.Tensor,
        shard_count: None | int = None,
        name: str = UnnamedTensorName,
    ):
        """
        If `ts` is a list of tensors, it is interpreted as the shards.
        Then `shard_count` must be None.
        If `ts` is a tensor then `shard_count` must be provided and it,
        will be replicated that many times.
        """

        if isinstance(ts, torch.Tensor):
            assert shard_count is not None
            ts = [ts] * shard_count
            shard_count = None

        assert shard_count is None
        assert len(ts) > 0
        first_shape = ts[0].shape
        shape = list(first_shape)

        super().__init__(name=name, shape=shape, shard_dim=None)
        for shard in ts:
            assert list(shape) == list(shard.shape)

        self._shards: tuple[DefaultPrimitiveTensor] = tuple(
            DefaultPrimitiveTensor(name=f"{name}.shard.{i}", data=t)
            for i, t in enumerate(ts)
        )

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    @property
    def shards(self) -> tuple[InferenceTensor]:
        return self._shards

    @property
    def is_replicated(self) -> bool:
        return True

    @classmethod
    def serialized_name(cls) -> str:
        return "ReplicatedTensor"

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        return {pt.name: pt._data for pt in self._shards}

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        builder.for_rank(0).add_tensor(self.name, self._shards[0]._data)
        return InferenceTensorMetadata(
            self.serialized_name(),
            {"": self.name},
            extra_properties={
                "shard_count": len(self._shards),
            },
        )

    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        ts = []
        for k in self.globals.keys():
            ts.append(new_globals[ts[k]])
        return ReplicatedTensor(name=self.name, shape=self.shape, ts=ts)

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ) -> "InferenceTensor":
        shard_count = int(extra_properties["shard_count"])
        try:
            ts = raw_tensors[""]
        except KeyError as e:
            raise IOError(f"Missing component tensor '' in {raw_tensors.keys()}") from e
        return cls(name=name, ts=ts, shard_count=shard_count)

    def __repr__(self):
        return (
            f"ReplicatedTensor({self.name}, {self.shape}, "
            f"shard_count={len(self._shards)} "
            f"of {self.shards[0].shape})"
        )

    def is_deep_equal(self, other: Any) -> bool:
        if not isinstance(other, ReplicatedTensor):
            return False
        if (
            self.shard_count != other.shard_count
            or self.name != other.name
            or self.shape != other.shape
        ):
            return False
        if self.shard_count == 0:
            return True
        return self.shards[0].is_deep_equal(other.shards[0])


@register_inference_tensor
class UnreducedTensor(ShardedTensorBase):
    """Sharded tensor which contains primitive tensors.
    To obtain the actual tensor a sum-reduction over the shards must be performed.
    """

    def __init__(
        self,
        *,
        ts: list[torch.Tensor],
        name: str = UnnamedTensorName,
        shape: Optional[list[int]] = None,
    ):
        assert len(ts) > 0
        shape = list(ts[0].shape if shape is None else shape)
        assert all(shape == list(t.shape) for t in ts)
        super().__init__(name=name, ts=ts, shape=shape, shard_dim=None)


########################################################################################
# Serialization helpers
########################################################################################


def _dtype_to_serialized_name(dtype: torch.dtype) -> str:
    try:
        return _DTYPE_TO_NAME[dtype]
    except KeyError as e:
        raise KeyError(
            f"Missing mapping for dtype {dtype}. Please add to the _NAME_TO_DTYPE dict"
        ) from e


def _serialized_name_to_dtype(dtype_name: str) -> torch.dtype:
    try:
        return _NAME_TO_DTYPE[dtype_name]
    except KeyError as e:
        raise KeyError(
            f"Missing mapping for dtype '{dtype_name}'. Please add to the _NAME_TO_DTYPE dict"
        ) from e


_NAME_TO_DTYPE: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def _maybe_dtype(*names: str):
    for name in names:
        try:
            cls = getattr(torch, name)
        except AttributeError:
            pass
        else:
            _NAME_TO_DTYPE[name] = cls


_maybe_dtype(
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "uint1",
    "uint2",
    "uint3",
    "uint4",
    "uint5",
    "uint6",
    "uint7",
)

_DTYPE_TO_NAME: dict[torch.dtype, str] = {v: k for k, v in _NAME_TO_DTYPE.items()}
