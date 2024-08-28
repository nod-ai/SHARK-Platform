# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Export support for the PagedLLMV1 protocol of models."""

import json
import torch
from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from shark_turbine.aot import *

from sharktank.layers import *
from sharktank.types import *

# TODO: Should be using a base class with the protocol supported.
# from ..models.llama.llama import LlamaModelConfig, PagedLlamaModelV1


################################################################################
# Config
################################################################################


@dataclass
class LlamaModelConfig:
    context_length=4096
    embedding_length=4096
    block_count=32
    feed_forward_length=11008
    rope_dimension_count=128
    attention_head_count=32
    attn_head_dim=128
    attention_layer_norm_rms_epsilon=9.999999747378752e-06
    attention_head_count_kv=32

    # Block sequence stride for a paged KV cache. This must divide evenly
    # into the context length.
    block_seq_stride: int = 16

    # Either "paged" or "direct".
    kv_cache_type: str = "paged"

    # The device on which to place intermediate state.
    device: Optional[torch.device] = None

    # Dtype to use for general FP activations not otherwise configured.
    activation_dtype: torch.dtype = torch.float16

    # Dtype to use for attention.
    attention_dtype: torch.dtype = torch.float16

    # Indicates if running with HuggingFace implementation and ensures
    # numerical equivalency to HuggingFace's LLaMa if true (by modifying
    # rotary embedding).
    use_hf: bool = False

    # local params
    bs = 1
    sl = 1
    start_index = 0
    q_len = 1
    feature_dim = 4096
    kv_seq_len = 1 
    head_dim = 128

    def create_kv_cache(self) -> BaseKVCache:
        if self.kv_cache_type == "direct":
            return DirectKVCache(
                block_seq_stride=self.block_seq_stride,
                transformer_block_count=self.block_count,
                attn_head_count=self.attention_head_count_kv,
                attn_head_dim=self.attn_head_dim,
                seq_length=self.context_length,
                device=self.device,
                dtype=self.attention_dtype,
            )
        elif self.kv_cache_type == "paged":
            return PagedKVCache(
                transformer_block_count=self.block_count,
                attn_head_count=self.attention_head_count_kv,
                attn_head_dim=self.attn_head_dim,
                cache_partition_count=2,  # One for each of K/V.
                block_seq_stride=self.block_seq_stride,
                device=self.device,
                dtype=self.attention_dtype,
            )
        else:
            raise NotImplementedError(f"kv_cache_type = {self.kv_cache_type}")

################################################################################
# Models
################################################################################


class PagedLlamaModelV1(torch.nn.Module):
    """LlamaModel with a paged KV cache and supporting variable sequence
    length batched inference.

    As both the caching and batching setup is complicated, this model variant
    is modular, intending to be instantiated and used in an overall assembly
    vs trying to providing one-stop methods that do everything.

    The inference procedure is typically:

    1. Initialize the PagedKVCache state tensors.
    2. Generate an input mask given a vector of sequence lengths.
    3. Generate an attention mask from the input mask.
    4. Allocate a block mapping table.
    5. Invoke prefill() with a batch of sequences.
    6. Extract tokens from batched logits.
    7. Iteratively invoke decode() for as long as there are sequences needing
       to be serviced.

    Various samplers and schedulers can be interleaved throughout.
    """

    def __init__(self, config: LlamaModelConfig):
        super().__init__()
        self.config = config
        self.context_length=self.config.context_length
        self.device=self.config.device
        self.activation_dtype=self.config.activation_dtype
        self.attention_dtype=self.config.attention_dtype
        self.cache = self.config.create_kv_cache()
        self.use_hf = self.config.use_hf

        self.attn_blocks = nn.ModuleList(
            [
                PagedLlamaAttentionBlock(
                    block_index=n,
                    cache=self.cache,
                    head_count=self.config.attention_head_count,
                    head_dim=self.config.attn_head_dim,
                    head_count_kv=self.config.attention_head_count_kv,
                    use_hf=self.use_hf,
                )
                for n in range(1)
            ]
        )

    def _assert_device(self, *ts: torch.Tensor, dtype: Optional[torch.dtype] = None):
        if self.device is not None:
            for t in ts:
                assert (
                    t.device == self.device
                ), f"Expected tensor to be on device {self.device} but it is on {t.device}"
                if dtype is not None:
                    assert (
                        t.dtype == dtype
                    ), f"Expected tensor to have dtype {dtype} but it is {t.dtype}"

    def _maximally_negative_value(self, dtype):
        """Returns a maximally negative value for the given dtype.

        This can be overriden to decide on a different policy.
        """
        return float("-inf")

    def generate_causal_context_mask(self) -> torch.Tensor:
        context_length = self.context_length
        causal_context_mask = torch.triu(
            torch.ones(
                [context_length, context_length], dtype=torch.bool, device=self.device
            ),
            diagonal=1,
        )[None, None, :, :]
        return causal_context_mask
    
    def input_mask(
        self,
        # [bs] of integers
        seq_lens: torch.Tensor,
        batch_seqlen: int,
    ):
        """Compute a boolean input mask for a batch of sequence lengths.

        The mask will be [bs, batch_seqlen] with True at any position that is
        masked.
        """
        range_vector = torch.arange(0, batch_seqlen, 1)
        matrix = torch.unsqueeze(seq_lens, dim=-1)
        mask = range_vector >= matrix
        return mask
        
    def attention_mask(
        self,
        input_mask: torch.Tensor,
        *,
        causal_context_mask: Optional[torch.Tensor] = None,
    ):
        """Generates a causal attention mask of [1, 1, sl, sl] of activation dtype.

        All masked positions are -inf and unmasked are 0.0.

        The pre-initialized causal context mask can be passed in. If not, then
        it will either be generated or use the initialization time buffer.
        Since this is a bool tensor of context_length^2, different deployment
        scenarios can benefit from managing this in different ways.
        """

        if causal_context_mask is None:
            causal_context_mask = self.generate_causal_context_mask()

        # Combine the causal context mask and input mask.
        dtype = self.attention_dtype
        _, batch_seq_len = input_mask.shape
        causal_mask = causal_context_mask[:, :, :batch_seq_len, :batch_seq_len]
        boolean_mask = causal_mask + input_mask[:, None, None, :]
        numeric_mask = torch.zeros_like(boolean_mask, dtype=dtype)
        numeric_mask.masked_fill_(boolean_mask, self._maximally_negative_value(dtype))
        return numeric_mask.to(self.device)

    def decode_attention_mask(self, boolean_input_mask: torch.Tensor):
        dtype = self.attention_dtype
        numeric_mask = torch.zeros_like(boolean_input_mask, dtype=dtype)
        numeric_mask.masked_fill_(
            boolean_input_mask, self._maximally_negative_value(dtype)
        )
        return numeric_mask.unsqueeze(1).unsqueeze(1).to(self.device)
    
    def prefill(
        self,
        # [bs, batch_seq_len]
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        *,
        # [1, 1, batch_seq_len, batch_seq_len]
        attention_mask: torch.Tensor,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        cache_state: list[torch.Tensor],
    ):
        self._assert_device(attention_mask, dtype=self.activation_dtype)
        self._assert_device(seq_block_ids)
        self._assert_device(*cache_state, dtype=self.activation_dtype)

        # Iterate over attention blocks.
        for block_idx, block in enumerate(self.attn_blocks):
            h = block(
                xq=xq,
                xk=xk,
                xv=xv,
                start_index=0,
                attention_mask=attention_mask,
                cache_state=cache_state,
                seq_block_ids=seq_block_ids,
            )
            # self.trace_tensor(f"llama.attn_block.{block_idx}.output", h)

        return h
    
    def decode(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        *,
        # [bs, 1, 1, batch_seq_len]
        attention_mask: torch.Tensor,
        # [bs] of starting positions
        start_positions: torch.Tensor,
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        cache_state: list[torch.Tensor],
    ):
        self._assert_device(attention_mask, dtype=self.activation_dtype)
        self._assert_device(start_positions)
        self._assert_device(*cache_state, dtype=self.activation_dtype)

        bs, _, _, _ = xq.shape

        # Allocate per-block temporary K/V tensors. These temporaries hold
        # one block's K/V state for the maximum context length.
        xk_temp = torch.empty(
            [
                bs,
                self.context_length,
                self.config.attention_head_count_kv,
                self.config.attn_head_dim,
            ],
            dtype=self.config.activation_dtype,
            device=self.device,
        )
        xv_temp = torch.empty(
            [
                bs,
                self.context_length,
                self.config.attention_head_count_kv,
                self.config.attn_head_dim,
            ],
            dtype=self.config.activation_dtype,
            device=self.device,
        )

        # Iterate over attention blocks.
        for block_idx, block in enumerate(self.attn_blocks):
            h = block(
                xq=xq,
                xk=xk,
                xv=xv,
                start_positions=start_positions,
                attention_mask=attention_mask,
                cache_state=cache_state,
                seq_block_ids=seq_block_ids,
                xk_temp=xk_temp,
                xv_temp=xv_temp,
            )
            # self.trace_tensor(f"llama.attn_block.{block_idx}.output", h)

        return h

################################################################################
# Layers
################################################################################


class PagedLlamaAttentionBlock(ThetaLayer):
    """Implements a self attention layer in the style of Llama using a
    paged cache."""

    def __init__(
        self,
        block_index: int,
        cache: PagedKVCache,
        head_count: int,
        head_dim: int,
        head_count_kv: int,
        use_hf: bool = False,
    ):
        super().__init__(theta)

        self.block_index = block_index
        self.cache = cache
        assert isinstance(head_count, int)
        self.head_count = head_count
        self.head_dim = head_dim
        self.head_count_kv = head_count_kv
        self.use_hf = use_hf

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        seq_block_ids: torch.Tensor,
        start_index: Optional[int] = None,
        start_positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_state: list[torch.Tensor] = None,
        xk_temp: Optional[torch.Tensor] = None,
        xv_temp: Optional[torch.Tensor] = None,
    ):
    
        bs, batch_seq_len, _,_ = xq.shape

        # Full sequence length.
        kv_seq_len = seq_block_ids.shape[1] * self.cache.block_seq_stride

        if self.cache.is_paged:
            xk, xv = self.transact_cache_paged(
                xk_cache_update=xk,
                xv_cache_update=xv,
                seq_block_ids=seq_block_ids,
                kv_seq_len=kv_seq_len,
                start_positions=start_positions,
                cache_state=cache_state,
                xk_temp=xk_temp,
                xv_temp=xv_temp,
            )
        elif self.cache.is_direct:
            xk, xv = self.transact_cache_direct(
                xk_cache_update=xk,
                xv_cache_update=xv,
                start_positions=start_positions,
                kv_seq_len=kv_seq_len,
                cache_state=cache_state,
            )
        else:
            raise NotImplementedError(f"Unsupported KV cache type: {type(self.cache)}")

        # Expand kv heads for GQA.
        gqa_n_rep = self.head_count // self.head_count_kv
        assert gqa_n_rep > 0
        if gqa_n_rep > 1:

            def repeat_kv(x: torch.Tensor) -> torch.Tensor:
                bs, slen, n_kv_heads, head_dim = x.shape
                return (
                    x.unsqueeze(-2)
                    .expand(bs, slen, n_kv_heads, gqa_n_rep, head_dim)
                    .reshape(bs, slen, n_kv_heads * gqa_n_rep, head_dim)
                )

            xk = repeat_kv(xk)
            xv = repeat_kv(xv)

        # Transpose into [bs, heads, sl, dim]
        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(xq, keys, values, attn_mask=None, is_causal=False)
        attn_output = attn_output.transpose(1, 2).reshape(bs, batch_seq_len, -1)
        return attn_output

    def transact_cache_direct(
        self,
        *,
        cache_state: list[torch.Tensor],
        xk_cache_update: torch.Tensor,
        xv_cache_update: torch.Tensor,
        kv_seq_len: int,
        start_positions: Optional[torch.Tensor] = None,
    ):
        bs, batch_seq_len, _, _ = xk_cache_update.shape
        cache_k = cache_state[self.block_index * 2]
        cache_v = cache_state[self.block_index * 2 + 1]

        if start_positions is None:
            # Prefill. Write the entire cache.
            cache_k[:, :batch_seq_len] = xk_cache_update
            cache_v[:, :batch_seq_len] = xv_cache_update
            return xk_cache_update, xv_cache_update
        else:
            # Decode. Write a single timestep.
            # TODO: This needs to be reworked with index ops.
            assert xk_cache_update.shape[1] == 1
            assert xv_cache_update.shape[1] == 1
            max_start_pos = 0
            for row_index in range(bs):
                row_start_pos = start_positions[row_index].item()
                max_start_pos = max(row_start_pos, max_start_pos)
                cache_k[row_index, row_start_pos] = xk_cache_update[row_index, 0]
                cache_v[row_index, row_start_pos] = xv_cache_update[row_index, 0]
            return cache_k[:, :kv_seq_len], cache_v[:, :kv_seq_len]

    def transact_cache_paged(
        self,
        *,
        xk_cache_update: torch.Tensor,
        xv_cache_update: torch.Tensor,
        cache_state: list[torch.Tensor],
        # [bs, batch_seq_len // block_seq_stride]
        seq_block_ids: torch.Tensor,
        kv_seq_len: int,
        start_positions: Optional[torch.Tensor] = None,
        xk_temp: Optional[torch.Tensor] = None,
        xv_temp: Optional[torch.Tensor] = None,
    ):
        cache = self.cache.paged

        # Manage the cache.
        if start_positions is None:
            # Prefill: Write the entire cache.
            cache.write(
                cache_state,
                cache_partitions=[xk_cache_update, xv_cache_update],
                transformer_block_index=self.block_index,
                page_ids=seq_block_ids,
            )
            return xk_cache_update, xv_cache_update
        else:
            # Decode at ragged start positions.
            # We need to initialize/read the K/V from the cache for the whole
            # sequence. Note that at this point, it is possible to fork and
            # use a memory efficient attention kernel that can do indirect
            # reads, skipping this materialization. This path is taken for
            # a decode step.
            assert xk_temp is not None and xv_temp is not None
            assert xk_cache_update.shape[1] == 1
            assert xv_cache_update.shape[1] == 1
            assert kv_seq_len == seq_block_ids.shape[1] * cache.block_seq_stride

            # Write our one updated cache row into the cache.
            cache.write_timestep(
                cache_state,
                cache_partitions=[
                    xk_cache_update,
                    xv_cache_update,
                ],
                transformer_block_index=self.block_index,
                seq_positions=start_positions + 1,
                page_ids=seq_block_ids,
            )

            # Restore from the cache.
            cache.read(
                cache_state,
                read_into_partitions=[
                    xk_temp[:, 0:kv_seq_len, ...],
                    xv_temp[:, 0:kv_seq_len, ...],
                ],
                transformer_block_index=self.block_index,
                page_ids=seq_block_ids,
            )

            # For computation, we create a subview of the xk/xv tensors to have
            # a sequence length covering the blocked size. This must include
            # the newly added row (the caller is responsible for ensuring that
            # every block has at least one row left). We'll compute on this
            # ragged view and use an appropriate mask.
            xk = xk_temp[:, 0:kv_seq_len, ...]
            xv = xv_temp[:, 0:kv_seq_len, ...]
            return xk, xv


def main():
    from ..utils import cli

    parser = cli.create_parser()
    # cli.add_input_dataset_options(parser)
    parser.add_argument(
        "--output-mlir",
        help="Output file path for exported MLIR file",
        default="/tmp/batch_llama_v1.mlir",
    )
    parser.add_argument(
        "--output-config",
        help="Output file path for exported config file",
        default="/tmp/batch_llama_v1.json",
    )
    parser.add_argument(
        "--bs",
        help="Comma-separated batch size(s) to generate, e.g. `4` or `2,4`",
        type=lambda arg: [int(bs) for bs in arg.split(",")],
        default="4",
    )
    parser.add_argument(
        "--verbose",
        help="Include verbose logging",
        action="store_true",
    )

    args = cli.parse(parser)
    # dataset = cli.get_input_dataset(args)

    # hp = configs.LlamaHParams.from_gguf_props(dataset.properties)
    hp = LlamaModelConfig()
    hp.kv_cache_type = "direct" if args.bs == [1] else "paged"
    hp.bs = args.bs
    model = PagedLlamaModelV1(hp)

    def generate_params_json(hp, prefill_bs: list[int], decode_bs: list[int]):
        return {
            "module_name": "module",
            "module_abi_version": 1,
            "max_seq_len": hp.context_length,
            "attn_head_count": hp.attention_head_count,
            "attn_head_dim": hp.attn_head_dim,
            "prefill_batch_sizes": prefill_bs,
            "decode_batch_sizes": decode_bs,
            "transformer_block_count": hp.block_count,
            "block_seq_stride": hp.block_seq_stride,
        }

    # Unrolling cache updates by batch row makes dynamo sad without an
    # override. There may be a better way to do this.
    import torch._dynamo.config as dynamo_config

    # TODO: Seems removed from 2.3+
    # dynamo_config.max_loop_unroll_nodes = 0

    fxb = FxProgramsBuilder(model)

    def generate_batch_prefill(
        bs: int, 
        ):
        
        tokens = torch.empty(bs, 64, dtype=torch.int64)
        seq_lens = torch.empty(bs, dtype=torch.int64)
        seq_block_ids = torch.empty(bs, 4, dtype=torch.int64)
        block_dim = torch.export.Dim(
            "block", max=(hp.context_length - 1) // hp.block_seq_stride
        )
        sl_dim = hp.block_seq_stride * block_dim

        if model.config.kv_cache_type == "paged":
            cache_state = model.cache.allocate(
                page_count=hp.context_length // hp.block_seq_stride
            )
            page_dim = torch.export.Dim("page")
            cache_state_dynamic_shapes = [{0: page_dim}]
        elif model.config.kv_cache_type == "direct":
            cache_state = model.cache.allocate(bs=1)
            # Direct cache dimensions:
            #   2 * transformer_block_count of...
            #   [bs, seq_length, attn_head_count, attn_head_dim]
            cache_state_dynamic_shapes = (2 * hp.block_count) * [{}]
        else:
            raise NotImplementedError(f"Unsupported KV cache type: {type(model.cache)}")

        q = torch.zeros((bs, 64, 32, 128), dtype=torch.float16)
        k = torch.zeros((bs, 64, 32, 128), dtype=torch.float16)
        v = torch.zeros((bs, 64, 32, 128), dtype=torch.float16)

        print(f"Exporting prefill_bs{bs}")
        example_args = (q, k, v, seq_lens, seq_block_ids, cache_state)

        @fxb.export_program(
            name=f"prefill_bs{bs}",
            args=example_args,
        )
        def _(model, q, k, v, seq_lens, seq_block_ids, cache_state):
            sl = tokens.shape[1]
            input_mask = model.input_mask(seq_lens, sl)
            attention_mask = model.attention_mask(input_mask)
            h = model.prefill(
                xq=q,
                xk=k,
                xv=v,
                attention_mask=attention_mask,
                seq_block_ids=seq_block_ids,
                cache_state=cache_state,
            )

            return h

    def generate_batch_decode(
        bs: int, 
        ):

        tokens = torch.ones(bs, 64, dtype=torch.int64)
        seq_lens = torch.ones(bs, dtype=torch.int64)
        start_positions = torch.ones(bs, dtype=torch.int64)
        seq_block_ids = torch.zeros(bs, 4, dtype=torch.int64)
        block_dim = torch.export.Dim(
            "block", max=(hp.context_length - 1) // hp.block_seq_stride
        )

        if model.config.kv_cache_type == "paged":
            cache_state = model.cache.allocate(
                page_count=hp.context_length // hp.block_seq_stride
            )
            page_dim = torch.export.Dim("page")
            cache_state_dynamic_shapes = [{0: page_dim}]
        elif model.config.kv_cache_type == "direct":
            cache_state = model.cache.allocate(bs=1)
            # Direct cache dimensions:
            #   2 * transformer_block_count of...
            #   [bs, seq_length, attn_head_count, attn_head_dim]
            cache_state_dynamic_shapes = (2 * hp.block_count) * [{}]
        else:
            raise NotImplementedError(f"Unsupported KV cache type: {type(model.cache)}")

        # dynamic_shapes = {
        #     "tokens": {},
        #     "seq_lens": {},
        #     "start_positions": {},
        #     "seq_block_ids": {1: block_dim},
        #     "cache_state": cache_state_dynamic_shapes,
        # }

        q = torch.zeros((bs, 1, 32, 128), dtype=torch.float16)
        k = torch.zeros((bs, 1, 32, 128), dtype=torch.float16)
        v = torch.zeros((bs, 1, 32, 128), dtype=torch.float16)

        print(f"Exporting decode_bs{bs}")
        example_args = (q, k, v, seq_lens, start_positions, seq_block_ids, cache_state)

        # @fxb.export_program(
        #     name=f"decode_bs{bs}",
        #     args=(
        #         tokens,
        #         seq_lens,
        #         start_positions,
        #         seq_block_ids,
        #         cache_state,
        #     ),
        #     dynamic_shapes=dynamic_shapes,
        # )

        @fxb.export_program(
            name=f"decode_bs{bs}",
            args=example_args,
        )
        def _(
            model,
            q, 
            k, 
            v,
            seq_lens,
            start_positions,
            seq_block_ids,
            cache_state,
        ):
            input_mask = model.input_mask(
                seq_lens, seq_block_ids.shape[1] * model.cache.block_seq_stride
            )
            attention_mask = model.decode_attention_mask(input_mask)
            h = model.decode(
                xq=q,
                xk=k,
                xv=v,
                attention_mask=attention_mask,
                start_positions=start_positions,
                seq_block_ids=seq_block_ids,
                cache_state=cache_state,
            )

            return h

    bsizes = []
    for bs in args.bs:
        generate_batch_prefill(bs)
        generate_batch_decode(bs)
        bsizes.append(bs)

    config = generate_params_json(hp, bsizes, bsizes)
    print("GENERATED!")

    # if args.verbose:
    #     for name, ep in fxb.programs.items():
    #         print(f"EXPORT {name}:\n{ep}")

    print("Exporting")
    output = export(fxb)
    print(f"Saving to '{args.output_mlir}'")
    output.save_mlir(args.output_mlir)
    json.dump(config, open(args.output_config, "w"))

if __name__ == "__main__":
    main()
