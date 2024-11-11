# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import namedtuple
from typing import Optional, Union

import torch

from .base import BaseLayer
from .. import ops
from ..types import SplitPrimitiveTensor, ReplicatedTensor, unbox_tensor


class RotaryEmbeddingLayer(BaseLayer):
    """Computes a rotary embedding in the style popularized by llama (RoPE)."""

    def __init__(
        self,
        *,
        rope_dimension_count: int,
        max_seqlen: int,
        rope_freq_base: Optional[float],
        device: Optional[torch.device] = None,
        use_hf: bool = False,
        static_tables: bool = False,
        use_table: bool = True,
        tensor_parallelism_size: int = 1,
    ):
        super().__init__()
        self.device = device
        self.rope_dimension_count = rope_dimension_count
        self.max_seqlen = max_seqlen
        self.use_hf = use_hf
        self.static_tables = static_tables
        self.use_table = use_table

        self.rope_freq_base = rope_freq_base if rope_freq_base is not None else 10000.0
        self.tensor_parallelism_size = tensor_parallelism_size
        if static_tables:
            ops.module_register_buffer(
                self, "static_rotary_embed_table", self._create_rotary_embed_table()
            )
        else:
            self.static_rotary_embed_table = None

    @property
    def rotary_embed_table(self):
        if self.use_table:
            if self.static_tables:
                return self.static_rotary_embed_table
            return self._create_rotary_embed_table()

        return None

    def forward(
        self,
        *,
        xt: Union[torch.Tensor, SplitPrimitiveTensor],
        start_index: int,
    ):
        if isinstance(xt, SplitPrimitiveTensor):
            rotary_shards = [None] * xt.shard_count
            if self.rotary_embed_table is not None:
                assert (
                    isinstance(self.rotary_embed_table, ReplicatedTensor)
                    and xt.shard_count == self.rotary_embed_table.shard_count
                )
                rotary_shards = [
                    unbox_tensor(shard) for shard in self.rotary_embed_table.shards
                ]

            xt_shards = [
                self.forward_unsharded(
                    xt=unbox_tensor(xt_shard),
                    start_index=start_index,
                    rotary_embed_table=rotary_shard,
                )
                for xt_shard, rotary_shard in zip(xt.shards, rotary_shards)
            ]
            xt = SplitPrimitiveTensor(ts=xt_shards, shard_dim=xt.shard_dim)
            return xt
        else:
            return self.forward_unsharded(
                xt=xt,
                start_index=start_index,
                rotary_embed_table=self.rotary_embed_table,
            )

    def forward_unsharded(
        self,
        *,
        xt: torch.Tensor,
        start_index: int,
        rotary_embed_table: Optional[torch.Tensor],
    ):
        # xq_, xk_ shape: bs, sl, _, dim
        # freqs_cis shape: max_sl, dim

        def create_interleaved_tensor(dim):
            """Creates a tensor which indexes an tensor such that
            it alternates between elements of its first and second
            half. Intended for use for HuggingFace's rotation
            implementation.

            Args:
              dim: Size of tensor

            Returns:
              Interleaved indexing tensor
            """
            first_half = torch.arange(dim // 2)
            second_half = torch.arange(dim // 2, dim)

            interleaved_tensor = torch.empty(dim, dtype=torch.long)
            interleaved_tensor[0::2] = first_half
            interleaved_tensor[1::2] = second_half

            return interleaved_tensor

        def create_ordering_tensor(dim):
            """Creates a tensor which indexes an tensor such that
            it reverses the alternation induced by create_interleaved_tesnor.
            Intended for use for HuggingFace's rotation implementation.

            Args:
              dim: Size of tensor

            Returns:
              Ordering indexing tensor
            """
            order_tensor = torch.empty(dim, dtype=torch.long)
            order_tensor[: dim // 2] = torch.arange(0, dim, 2)
            order_tensor[dim // 2 :] = torch.arange(1, dim, 2)
            return order_tensor

        if self.use_hf:
            xt = xt[..., create_interleaved_tensor(xt.shape[-1])]
        xt_ = xt.unflatten(-1, (-1, 2))
        _, sl, _, dim, _ = xt_.shape

        # Offset the table based on starting position.
        if self.use_table:
            freqs_cis = rotary_embed_table[:, start_index : start_index + sl, :]
        else:
            freqs_cis = torch.arange(start_index, start_index + sl, device=xt.device)
            freqs_cis = self._compute_rotary_embed_table(freqs_cis)

        assert freqs_cis.shape[-1] == dim
        assert (
            freqs_cis.shape[1] >= sl
        ), f"Sequence length longer than embedding table ({sl} vs {freqs_cis.shape[0]})"

        broadcast_freqs_cis = freqs_cis[:, None, 0:sl, None, :]

        cos = broadcast_freqs_cis[0]
        sin = broadcast_freqs_cis[1]
        xt_r = xt_[..., 0]
        xt_i = xt_[..., 1]

        xt_out_r = xt_r * cos - xt_i * sin
        xt_out_i = xt_i * cos + xt_r * sin

        xt_out = torch.concatenate((xt_out_r, xt_out_i), dim=-1)

        if self.use_hf:
            xt_out = xt_out[..., create_ordering_tensor(xt_out.shape[-1])]
            return xt_out.type_as(xt)

        return xt_out.type_as(xt)

    def complex_multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Function for elementwise-multiplication of two complex torch tensors.
        Functionally similar to a*b, but numerically accurate for HuggingFace
        LLaMa implementation.

        Args:
          a: First torch tensor operand
          b: Second torch tensor operand
        Returns:
          Tensor of same size to a, b whose elements is product of corresponding
          elements in a, b
        """
        return torch.complex(
            a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real
        )

    def compute_batch_mask(
        self, start_positions: Union[torch.Tensor, ReplicatedTensor], batch_seq_len: int
    ) -> torch.Tensor:
        """Computes a mask for a batch that can be repeatedly applied.

        Args:
          start_positions: Tensor of [bs] with start positions for every sequence
            in the batch.
          batch_seq_len: The sequence length dimension of the batch.
        Returns:
          Tensor of [bs, sl, 1, d] that will be later passed to apply_batch_mask.
        """
        self.trace_tensor("rope.start_positions", start_positions)
        positions_seq = torch.arange(0, batch_seq_len, device=self.device).unsqueeze(
            0
        ) + start_positions.unsqueeze(1)
        # Broadcast lookup to [b, ...].
        self.trace_tensor("rope.positions_seq", positions_seq)

        if self.use_table:
            freqs_cis = self.rotary_embed_table[:, positions_seq]
        else:
            shape = positions_seq.shape
            freqs_cis = self._compute_rotary_embed_table(positions_seq.flatten())
            freqs_cis = freqs_cis.unflatten(1, shape)

        # Unsqueeze a unit dim for attention heads.
        broadcast_freqs_cis = freqs_cis.unsqueeze(2)
        return broadcast_freqs_cis

    def apply_batched_mask(
        self,
        *,
        xt: Union[torch.Tensor, SplitPrimitiveTensor],
        mask: Union[torch.Tensor, ReplicatedTensor],
    ):
        if not isinstance(xt, SplitPrimitiveTensor):
            return self.apply_batched_mask_unsharded(xt=xt, mask=mask)

        assert isinstance(mask, ReplicatedTensor) and mask.shard_count == xt.shard_count
        xt_shards = [
            self.apply_batched_mask_unsharded(
                xt=unbox_tensor(xt_shard),
                mask=unbox_tensor(mask_shard),
            )
            for xt_shard, mask_shard in zip(xt.shards, mask.shards)
        ]
        xt = SplitPrimitiveTensor(ts=xt_shards, shard_dim=xt.shard_dim)
        return xt

    def apply_batched_mask_unsharded(self, *, xt: torch.Tensor, mask: torch.Tensor):
        """Applies the embedding to a ragged batch of queries and keys.

        This does a more complicated indexing operation for cases when the each
        sequence in the batch has a potentially different start position.

        positions should be of [bs, sl] and enumerate positions of all tokens.
        """
        # xq_, xk_ shape: bs, sl, _, dim
        # freqs_cis shape: max_sl, dim
        cos = mask[0]
        sin = mask[1]

        xt_ = xt.unflatten(-1, (-1, 2))
        xt_r = xt_[..., 0]
        xt_i = xt_[..., 1]

        xt_out_r = xt_r * cos - xt_i * sin
        xt_out_i = xt_r * sin + xt_i * cos
        xt_out = torch.concatenate((xt_out_r, xt_out_i), dim=-1)
        return xt_out.type_as(xt)

    def _compute_rotary_embed_table(self, t):
        dim = self.rope_dimension_count
        freqs = 1.0 / (
            self.rope_freq_base
            ** (torch.arange(0, dim, 2, device=t.device)[: (dim // 2)].float() / dim)
        )
        freqs = torch.outer(t, freqs).float()

        cos = torch.cos(freqs).unsqueeze(0)
        sin = torch.sin(freqs).unsqueeze(0)

        return torch.concatenate((cos, sin), dim=0)

    def _create_rotary_embed_table(self):
        t = torch.arange(self.max_seqlen, device=self.device)
        freqs_cis = self._compute_rotary_embed_table(t)
        return self._replicate(freqs_cis)

    def _replicate(self, t):
        if self.tensor_parallelism_size > 1:
            # Replicate across all devices, the data is not a lot and the computation is cheap.
            t = ops.replicate(t, self.tensor_parallelism_size)

        return t
