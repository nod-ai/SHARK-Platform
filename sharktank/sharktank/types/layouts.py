# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Abstract layout structs describing various physical arrangements.

These are typically logical, planar layouts over some fundamental data types.
Concrete sub-classes implement any necessary physical to logical mapping.

While many of these layouts will have one or more vendor specific, custom
packed realizations as a QuantizedTensor subtype, each also has a generic
planar QuantizedTensor which carries its tensors unpacked.
"""

from typing import Optional

import torch

from .tensors import (
    register_quantized_layout,
    MetaDataValueType,
    QuantizedLayout,
)

from .layout_utils import (
    promote_linear_i4_block_to_i8,
    promote_linear_i6_block_to_i8,
)

__all__ = [
    "BlockScaledI4Layout",
    "BlockScaledLayout",
    "SuperBlockOffsetScaled_4_6_Layout",
    "TensorScaledLayout",
]


@register_quantized_layout
class TensorScaledLayout(QuantizedLayout):
    """Quantized layout which combines some scalar scale (`d`) tensor with a
    quantized sample (`qs`) tensor. An optional offset (`m`) tensor
    can be provided.

    The dequantization formula:

    ```
    dtype = d.dtype
    result = d.to(dtype) * (qs - m)
    ```

    If provided, `m` must be of the same dtype as `d`. `qs` must be cast
    compatible to `d.dtype`. Generally, `qs` will be a lower precision
    floating point format or an integer dtype.

    If d/m are scalar tensors, then this implements whole tensor quantization.
    Otherwise, they must be broadcast to the axis along which scaling is
    performed.
    """

    def __init__(
        self,
        *,
        shape: list[int],
        d: torch.Tensor,
        qs: torch.Tensor,
        m: Optional[torch.Tensor] = None,
    ):
        self._shape = shape
        self._d = d
        self._qs = qs
        self._m = m

    @classmethod
    def serialized_name(cls) -> str:
        return "TensorScaledLayout"

    @classmethod
    def create(
        cls,
        shape: list[int],
        metadata: dict[str, MetaDataValueType],
        planes: dict[str, torch.Tensor],
    ):
        m = planes.get("m")
        return cls(shape=shape, d=planes["d"], qs=planes["qs"], m=m)

    @property
    def planes(self) -> dict[str, torch.Tensor]:
        p = {
            "d": self._d,
            "qs": self._qs,
        }
        if self._m is not None:
            p["m"] = self._m
        return p

    @property
    def shape(self) -> list[int]:
        """The flattened shape of the logical (unblocked) result."""
        return self._shape

    @property
    def d(self) -> torch.Tensor:
        """Per tensor scale."""
        return self._d

    @property
    def m(self) -> Optional[torch.Tensor]:
        """Per tensor offset."""
        return self._m

    @property
    def qs(self) -> torch.Tensor:
        """Per sample quantized values."""
        return self._qs

    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.dequant_blocked(dtype)

    def dequant_blocked(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        d = self.d
        m = self.m
        qs = self.qs
        if dtype is not None:
            d = d.to(dtype)
        else:
            dtype = d.dtype
        qs = qs.to(dtype=dtype)
        if m is not None:
            m = m.to(dtype)
            return (qs - m) * d
        else:
            return qs * d

    def __repr__(self):
        r = (
            f"{type(self).__name__}(d({list(self.d.shape)}, dtype={self.d.dtype}), "
            f"qs({list(self.qs.shape)}, dtype={self.qs.dtype}))"
        )
        if self.m is not None:
            r += f", m({list(self.m.shape)}, dtype={self.m.dtype})"
        return r


@register_quantized_layout
class BlockScaledLayout(QuantizedLayout):
    """Block-quantized representation which consists of a scale (`d`)
    and offset (`m`) per block in a higher precision type. The offset, if
    present, is pre-scaled.

    The dequantization formula:

    ```
    result = d.to(dtype) * qs.to(dtype) + m.to(dtype)
    ```

    The inner-most dims will retain block structure. For example, if the
    block size is 32 and the original shape was NxK, then the component
    shapes would be:

    * `d`: `[N, K // 32, 1]`
    * `m`: `[N, K // 32, 1]`
    * `qs`: `[N, K // 32, 32]`

    Note that the offset (`m`) is optional.
    """

    def __init__(
        self,
        shape: list[int],
        d: torch.Tensor,
        qs: torch.Tensor,
        *,
        m: Optional[torch.Tensor] = None,
    ):
        self._shape = shape
        self._d = d
        self._m = m
        self._qs = qs

    @classmethod
    def serialized_name(self) -> str:
        return "BlockScaledLayout"

    @classmethod
    def create(
        cls,
        shape: list[int],
        metadata: dict[str, MetaDataValueType],
        planes: dict[str, torch.Tensor],
    ):
        m = planes.get("m")
        return cls(shape, planes["d"], planes["qs"], m=m)

    @property
    def planes(self) -> dict[str, torch.Tensor]:
        p = {
            "d": self._d,
            "qs": self._qs,
        }
        if self._m is not None:
            p["m"] = self._m
        return p

    @property
    def shape(self) -> list[int]:
        """The flattened shape of the logical (unblocked) result."""
        return self._shape

    @property
    def d(self) -> torch.Tensor:
        """Per block scales."""
        return self._d

    @property
    def m(self) -> torch.Tensor:
        """Per block offsets."""
        return self._m

    @property
    def qs(self) -> torch.Tensor:
        """Per sample quantized values."""
        return self._qs

    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.dequant_blocked(dtype).reshape(self.shape)

    def dequant_blocked(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        d = self.d
        m = self.m
        qs = self.qs
        if dtype:
            d = d.to(dtype)
            if m is not None:
                m = m.to(dtype)
        else:
            dtype = d.dtype
            assert m is None or m.dtype == d.dtype
        scaled = d * qs.to(dtype)
        shifted = scaled if m is None else scaled + m
        return shifted

    def __repr__(self):
        r = (
            f"{type(self).__name__}(d({list(self.d.shape)}, dtype={self.d.dtype}), "
            f"qs({list(self.qs.shape)}, dtype={self.qs.dtype}))"
        )
        if self.m is not None:
            r += f", m({list(self.m.shape)}, dtype={self.m.dtype})"
        return r


@register_quantized_layout
class BlockScaledI4Layout(BlockScaledLayout):
    """A BlockScaledLayout where the `qs` are internally packed 2 values per byte.

    Per convention, the `qs` property returns a tensor as either uint8 or
    int8 (depending on `signed=`) that can be used directly for arithmetic.
    The underlying bit-packed tensor can be accessed via `qs_bit_packed` and
    it is laid out in little endian bit order, linearly across the block
    dimension. There are an arbitrary ways to organize such things, and
    if more specificity is needed, a dedicated layout class should be used. In
    general, for these "generic" layouts, we choose defaults that mate well
    with how the compiler infra and prevailing targets are built and trust that
    optimizations that care will choose a specific packing.
    """

    def __init__(
        self,
        shape: list[int],
        d: torch.Tensor,
        qs: torch.Tensor,
        *,
        m: Optional[torch.Tensor] = None,
        signed: bool = False,
    ):
        super().__init__(shape, d, qs, m=m)
        self.signed = signed

    @classmethod
    def serialized_name(self) -> str:
        return "BlockScaledI4Layout"

    @classmethod
    def create(
        cls,
        shape: list[int],
        metadata: dict[str, MetaDataValueType],
        planes: dict[str, torch.Tensor],
    ):
        m = planes.get("m")
        return cls(shape, planes["d"], planes["qs"], m=m, signed=metadata["signed"])

    @property
    def metadata(self) -> dict[str, MetaDataValueType]:
        return {"signed": self.signed}

    @property
    def qs(self) -> torch.Tensor:
        # `qs` is defined as something that we can do integer arithmetic on
        # for cases where we only have non-packed kernels available. Therefore,
        # we promote it to i8. The `qs_packed` is available for the sub-byte
        # bit pattern.
        return promote_linear_i4_block_to_i8(self._qs, signed=self.signed)

    @property
    def qs_bit_packed(self) -> torch.Tensor:
        return self._qs


@register_quantized_layout
class SuperBlockOffsetScaled_4_6_Layout(QuantizedLayout):
    """Effectively a planarized version of the ggml Q4_K layout."""

    def __init__(
        self,
        shape: list[int],
        *,
        d: torch.Tensor,
        dmin: torch.Tensor,
        sb_scales_high: torch.Tensor,
        sb_scales_low: torch.Tensor,
        sb_mins_high: torch.Tensor,
        sb_mins_low: torch.Tensor,
        qs: torch.Tensor,
    ):
        self._shape = shape
        self._d = d
        self._dmin = dmin
        self._sb_scales_high = sb_scales_high
        self._sb_scales_low = sb_scales_low
        self._sb_mins_high = sb_mins_high
        self._sb_mins_low = sb_mins_low
        self._qs = qs

    @classmethod
    def serialized_name(self) -> str:
        return "SuperBlockOffsetScaled_4_6_Layout"

    @classmethod
    def create(
        cls,
        shape: list[int],
        metadata: dict[str, MetaDataValueType],
        planes: dict[str, torch.Tensor],
    ):
        return cls(
            shape,
            d=planes["d"],
            dmin=planes["dmin"],
            sb_scales_high=planes["sb_scales_high"],
            sb_scales_low=planes["sb_scales_low"],
            sb_mins_high=planes["sb_mins_high"],
            sb_mins_low=planes["sb_mins_low"],
            qs=planes["qs"],
        )

    @property
    def planes(self) -> dict[str, torch.Tensor]:
        return {
            "d": self._d,
            "dmin": self._dmin,
            "sb_scales_high": self._sb_scales_high,
            "sb_scales_low": self._sb_scales_low,
            "sb_mins_high": self._sb_mins_high,
            "sb_mins_low": self._sb_mins_low,
            "qs": self._qs,
        }

    @property
    def shape(self) -> list[int]:
        """The flattened shape of the logical (unblocked) result.

        Shape: [N, SUPER_COUNT * SUB_COUNT * BLOCK_SIZE]
        """
        return self._shape

    @property
    def d(self) -> torch.Tensor:
        """Super-block scales.

        Shape: [N, SUPER_COUNT, 1]
        """
        return self._d

    @property
    def dmin(self) -> torch.Tensor:
        """Super-block mins.

        Shape: [N, SUPER_COUNT, 1]
        """
        return self._dmin

    @property
    def sb_scales(self) -> torch.Tensor:
        """Returns sub-block scales combined and cast to a uint8 tensor.

        Shape: [N, SUPER_COUNT, SUB_COUNT]
        """
        return promote_linear_i6_block_to_i8(self._sb_scales_high, self._sb_scales_low)

    @property
    def sb_mins(self) -> torch.Tensor:
        """Returns sub-block mins combined and cast to a uint8 tensor.

        Shape: [N, SUPER_COUNT, SUB_COUNT]
        """
        return promote_linear_i6_block_to_i8(self._sb_mins_high, self._sb_mins_low)

    @property
    def sb_scales_bit_packed(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Bit packed sub-block scales.

        Shape:
            high = [N, SUPER_COUNT, SUB_COUNT // 4]
            low  = [N, SUPER_COUNT, SUB_COUNT // 2]

        The 'high' tensor contains upper 2 bits of each. The 'low' tensor
        contains the lower nibble.
        """
        return self._sb_scales_high, self._sb_scales_low

    @property
    def sb_mins_bit_packed(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Bit packed sub-block mins.

        Shape:
            high = [N, SUPER_COUNT, SUB_COUNT // 4]
            low  = [N, SUPER_COUNT, SUB_COUNT // 2]

        The 'high' tensor contains upper 2 bits of each. The 'low' tensor
        contains the lower nibble.
        """
        return self._sb_mins_high, self._sb_mins_low

    @property
    def qs_bit_packed(self) -> torch.Tensor:
        """Gets the qs as a bit-packed i4 tensor (as uint8).

        Shape: [N, SUPER_COUNT, SUB_COUNT, BLOCK_SIZE // 2]
        """
        return self._qs

    @property
    def qs(self) -> torch.Tensor:
        """Per sample quantized values.

        Shape: [N, SUPER_COUNT, SUB_COUNT, BLOCK_SIZE]
        """
        # `qs` is defined as something that we can do integer arithmetic on
        # for cases where we only have non-packed kernels available. Therefore,
        # we promote it to i8. The `qs_packed` is available for the sub-byte
        # bit pattern.
        return promote_linear_i4_block_to_i8(self._qs, signed=False)

    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.dequant_blocked(dtype).reshape(self.shape)

    def dequant_blocked(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        d = self.d
        dmin = self.dmin
        qs = self.qs
        sb_scales = self.sb_scales
        sb_mins = self.sb_mins

        d_scaled = (d * sb_scales).unsqueeze(-1)
        dmin_scaled = (dmin * sb_mins).unsqueeze(-1)
        return d_scaled * qs - dmin_scaled

    def __repr__(self):
        r = (
            f"{type(self).__name__}(d({list(self.d.shape)}, dtype={self.d.dtype}), "
            f"d({list(self.d.shape)}, dtype={self.d.dtype}), "
            f"dmin({list(self.dmin.shape)}, dtype={self.dmin.dtype}), "
            f"sb_scales_high({list(self._sb_scales_high.shape)}, dtype={self._sb_scales_high.dtype}), "
            f"sb_scales_low({list(self._sb_scales_low.shape)}, dtype={self._sb_scales_low.dtype}), "
            f"sb_mins_high({list(self._sb_mins_high.shape)}, dtype={self._sb_mins_high.dtype}), "
            f"sb_mins_low({list(self._sb_mins_low.shape)}, dtype={self._sb_mins_low.dtype}), "
            f"qs({list(self._qs.shape)}, dtype={self._qs.dtype}))"
        )
        return r
