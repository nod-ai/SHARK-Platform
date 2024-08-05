# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from .base import BaseLayer


class RotaryEmbeddingLayer(BaseLayer):
    """Computes a rotary embedding in the style popularized by llama (RoPE)."""

    def __init__(
        self,
        *,
        rope_dimension_count: int,
        max_seqlen: int,
        device: Optional[torch.device] = None,
        hf: bool = False,
    ):
        super().__init__()
        self.device = device
        self._table = self._create_rotary_embed_table(
            max_seqlen=max_seqlen,
            dim=rope_dimension_count,
            hf=hf,
        )
        self.hf = hf

    def forward(self, *, xq: torch.Tensor, xk: torch.Tensor, start_index: int):
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
            first_half = torch.arange(dim//2)
            second_half = torch.arange(dim//2, dim)
            
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
            order_tensor[:dim // 2] = torch.arange(0, dim, 2)
            order_tensor[dim // 2:] = torch.arange(1, dim, 2)
            return order_tensor
        
        if self.hf:
            xq = xq[..., create_interleaved_tensor(xq.shape[-1])]
            xk = xk[..., create_interleaved_tensor(xq.shape[-1])]

        xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
        _, sl, _, dim = xq_.shape

        # Offset the table based on starting position.
        freqs_cis = self._table[start_index : start_index + sl, :]
        assert freqs_cis.shape[-1] == dim
        assert (
            freqs_cis.shape[0] >= sl
        ), f"Sequence length longer than embedding table ({sl} vs {freqs_cis.shape[0]})"

        broadcast_freqs_cis = freqs_cis[None, 0:sl, None, :]
        
        if self.hf:
            xq_out = torch.view_as_real(self.complex_multiply(xq_, broadcast_freqs_cis)).flatten(3)
            xk_out = torch.view_as_real(self.complex_multiply(xk_, broadcast_freqs_cis)).flatten(3)

            xq_out = xq_out[...,create_ordering_tensor(xq_out.shape[-1])]
            xk_out = xk_out[...,create_ordering_tensor(xq_out.shape[-1])]
        
            return xq_out.type_as(xq), xk_out.type_as(xk)
        
        xq_out = torch.view_as_real(xq_ * broadcast_freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * broadcast_freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def complex_multiply(
            self, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
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
        return torch.complex(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real)

    def compute_batch_mask(
        self, start_positions: torch.Tensor, batch_seq_len: int
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
        freqs_cis = self._table[positions_seq]

        # Unsqueeze a unit dim for attention heads.
        broadcast_freqs_cis = freqs_cis.unsqueeze(2)
        return broadcast_freqs_cis

    def apply_batched_mask(
        self, *, xq: torch.Tensor, xk: torch.Tensor, mask: torch.Tensor
    ):
        """Applies the embedding to a ragged batch of queries and keys.

        This does a more complicated indexing operation for cases when the each
        sequence in the batch has a potentially different start position.

        positions should be of [bs, sl] and enumerate positions of all tokens.
        """
        # xq_, xk_ shape: bs, sl, _, dim
        # freqs_cis shape: max_sl, dim
        xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
        _, sl, _, dim = xq_.shape

        xq_out = torch.view_as_real(xq_ * mask).flatten(3)
        xk_out = torch.view_as_real(xk_ * mask).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def _create_rotary_embed_table(
        self,
        max_seqlen: int,
        dim: int,
        theta_value: float = 10000.0,
        hf: bool = False,
    ):
        freqs = 1.0 / (
            theta_value
            ** (torch.arange(0, dim, 2, device=self.device)[: (dim // 2)].float() / dim)
        )
        t = torch.arange(max_seqlen, device=freqs.device)
        freqs = torch.outer(t, freqs).float()

        freqs_cis = torch.complex(torch.cos(freqs), torch.sin(freqs)) if hf else torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis
