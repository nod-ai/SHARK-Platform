# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Quantizer Tensors
These tensors contain quantization parameters that can be used to quantize some other
tensor. These are typically stored in a dataset to signal a transformation into
a quantized representation for some layer (typically for activations or other dynamic
value) for which the underlying parameters themselves are fixed.

Note that there is no need for a "DequantizerTensor" or a "dequantize" method on
this class, since any `QuantizedTensor` already knows how to dequantize itself.
"""

from typing import Any, Optional

from abc import abstractmethod

import torch

from ..utils.io import ShardedArchiveBuilder

from .layouts import (
    TensorScaledLayout,
)

from .tensors import (
    InferenceTensor,
    InferenceTensorMetadata,
    PlanarQuantizedTensor,
    PrimitiveTensor,
    QuantizedTensor,
    UnnamedTensorName,
    register_inference_tensor,
    _serialized_name_to_dtype,
    _dtype_to_serialized_name,
)

__all__ = [
    "DynamicScaledQuantizer",
    "QuantizerTensor",
    "StaticScaledQuantizer",
]


class QuantizerTensor(InferenceTensor):
    """A tensor that knows how to quantize some other tensor."""

    @abstractmethod
    def quantize(self, t: PrimitiveTensor) -> QuantizedTensor:
        """Performs a quantizing transformation on t, returning a QuantizeTensor."""
        ...


@register_inference_tensor
class StaticScaledQuantizer(QuantizerTensor):
    """Quantizes to a `TensorScaledLayout` (per-tensor) or (TBD) for per-axis.

    If `scale` is a scalar, it produces a PlanarQuantizedTensor of a
    TensorScaledLayout where the `d` (scale) is the reciprocal of the scale
    specified here.

    An optional `offset` can be provided. If provided, when quantizing, this
    will be subtracted from the value. Upon dequantizing, it will be added.
    """

    def __init__(
        self,
        *,
        scale: torch.Tensor,
        dtype: torch.dtype,
        axis: Optional[int] = None,
        reciprocal_scale: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        name: str = UnnamedTensorName,
    ):
        super().__init__(shape=scale.shape, name=name)
        assert axis is None or axis >= 0
        self._scale = scale
        if reciprocal_scale is None:
            reciprocal_scale = 1.0 / scale
        self._reciprocal_scale = reciprocal_scale
        self._offset = offset
        self._dtype = dtype
        self._axis = axis
        assert self._scale.shape == self._reciprocal_scale.shape
        assert self._scale.dtype == self._reciprocal_scale.dtype
        if self._offset is not None:
            assert self._offset.shape == self._scale.shape
            assert self._offset.dtype == self._scale.dtype
        if self._axis is not None:
            assert len(self._scale.shape) == 1, "Expected per-axis scale to be 1D"
        else:
            assert len(self._scale.shape) == 0, "Expected per-tensor scale to be 0D"

    def quantize(self, t: PrimitiveTensor) -> QuantizedTensor:
        """Performs a quantizing transformation on t, returning a QuantizeTensor."""
        shape = list(t.shape)
        axis = self._axis
        offset = self._offset
        if axis is None:
            # Per tensor.
            if offset is None:
                qs = (t * self._scale).to(dtype=self.dtype)
            else:
                qs = ((t - offset) * self._scale).to(dtype=self.dtype)
            return PlanarQuantizedTensor(
                shape=shape,
                layout=TensorScaledLayout(
                    shape=shape,
                    d=self._reciprocal_scale,
                    qs=qs,
                    m=self._offset,
                ),
            )
        else:
            # Expand the scale/reciprocal to correspond to the broadcast axis.
            scale = self._scale
            reciprocal_scale = self._reciprocal_scale
            offset = self._offset
            assert axis >= 0 and axis < len(
                shape
            ), f"Per-axis scale {axis} out of bounds of shape {shape}"
            scale_shape = [1] * len(shape)
            scale_shape[axis] = scale.shape[0]
            broadcast_scale = scale.reshape(scale_shape)
            broadcast_reciprocal_scale = reciprocal_scale.reshape(scale_shape)
            if offset is None:
                broadcast_offset = None
                qs = (t * broadcast_scale).to(dtype=self.dtype)
            else:
                broadcast_offset = offset.reshape(scale_shape)
                qs = ((t - broadcast_offset) * broadcast_scale).to(dtype=self.dtype)
            return PlanarQuantizedTensor(
                shape=shape,
                layout=TensorScaledLayout(
                    shape=shape,
                    d=broadcast_reciprocal_scale,
                    qs=qs,
                    m=broadcast_offset,
                ),
            )

    @property
    def axis(self) -> Optional[int]:
        """Returns the axis that is scaled or None for whole tensor."""
        return self._axis

    @property
    def offset(self) -> Optional[torch.Tensor]:
        return self._offset

    @property
    def scale(self) -> torch.Tensor:
        return self._scale

    @property
    def reciprocal_scale(self) -> torch.Tensor:
        return self._reciprocal_scale

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @classmethod
    def serialized_name(cls) -> str:
        return "StaticScaledQuantizer"

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ):
        offset = None
        try:
            scale = raw_tensors["scale"]
            reciprocal_scale = raw_tensors["rscale"]
            if "offset" in raw_tensors:
                offset = raw_tensors["offset"]
        except KeyError as e:
            raise IOError("Missing component tensor") from e
        try:
            dtype_name = extra_properties["dtype"]
        except KeyError as e:
            raise IOError("Missing property") from e
        axis = int(extra_properties["axis"]) if "axis" in extra_properties else None
        dtype = _serialized_name_to_dtype(dtype_name)
        return cls(
            name=name,
            scale=scale,
            offset=offset,
            reciprocal_scale=reciprocal_scale,
            dtype=dtype,
            axis=axis,
        )

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        d = {
            f"{self.name}:scale": self._scale,
            f"{self.name}:rscale": self._reciprocal_scale,
        }
        if self._offset is not None:
            d[f"{self.name}:offset"] = self._offset
        return d

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        scale_name = f"{self.name}:scale"
        rscale_name = f"{self.name}:rscale"
        offset_name = f"{self.name}:offset"
        extra_properties = {"dtype": _dtype_to_serialized_name(self._dtype)}
        if self._axis is not None:
            extra_properties["axis"] = self._axis
        raw_tensors = {
            "scale": scale_name,
            "rscale": rscale_name,
        }
        builder.add_tensor(scale_name, self._scale)
        builder.add_tensor(rscale_name, self._reciprocal_scale)
        if self._offset is not None:
            raw_tensors["offset"] = offset_name
            builder.add_tensor(offset_name, self._offset)

        return InferenceTensorMetadata(
            self.serialized_name(),
            raw_tensors=raw_tensors,
            extra_properties=extra_properties,
        )

    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        offset_name = f"{self.name}:offset"
        return StaticScaledQuantizer(
            name=self.name,
            dtype=self.dtype,
            scale=new_globals[f"{self.name}:scale"],
            reciprocal_scale=new_globals[f"{self.name}:rscale"],
            offset=new_globals.get(offset_name),
        )

    def __repr__(self):
        return (
            f"StaticScaledQuantizer({self.name}, {self.shape}, "
            f"scale=({self._scale.shape}, {self._scale.dtype}) along {self._axis}) "
            f"offset={self._offset} "
            f"-> dtype={self._dtype})"
        )


class DynamicScaledQuantizer(QuantizedTensor):
    """Quantizer that produced a `TensorScaledLayout` (per-tensor) based on
    computing the dynamic scale of the source tensor.

    This is done via a computation like:

    ```
    finfo = torch.finfo(output_dtype)
    amax = abs(max(x))
    scale = finfo.max / amax.clamp(eps)
    ```
    """

    ...
