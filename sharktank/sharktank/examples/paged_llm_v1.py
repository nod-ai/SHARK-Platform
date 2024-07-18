# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Inference support for the PagedLLMV1 protocol of models."""

from typing import Optional

import math
import sys

import torch

from sharktank.layers import *
from sharktank.types import *

# TODO: Should be using a base class with the protocol supported.
from sharktank.models.llama.llama import *
from sharktank.utils.debugging import trace_tensor
from sharktank.utils.tokenizer import InferenceTokenizer, load_tokenizer


class TorchGenerator:
    """Generator that runs directly on the Torch model."""

    def __init__(
        self,
        model: PagedLlamaModelV1,
        tokenizer: InferenceTokenizer,
        page_cache_size: int = 128,
        # Need to look at the model more for this.
        end_token: int = 2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if model.cache.is_paged:
            self.shared_cache_state = model.cache.paged.allocate(page_cache_size)
        else:
            self.shared_cache_state = None
        self.free_pages = list(range(1, 128))
        self.end_token = end_token

    @property
    def block_seq_stride(self) -> int:
        return self.model.cache.block_seq_stride

    def begin_batch(self, prompts: list[str]):
        token_ids, seq_lens = self.tokenizer.encode(
            prompts, pad_to_multiple_of=self.model.cache.pad_sequence_stride
        )
        token_ids = torch.tensor(token_ids, device=self.model.device)
        seq_lens = torch.tensor(seq_lens, device=self.model.device)
        if self.shared_cache_state is not None:
            cache_state = self.shared_cache_state
        else:
            cache_state = self.model.cache.direct.allocate(bs=len(prompts))
        return Batch(self, token_ids, seq_lens, cache_state)

    def alloc_page(self) -> int:
        if self.model.cache.is_direct:
            # We don't allocate block ids for the direct cache.
            return 0

        return self.free_pages.pop()

    def release_page(self, index: int):
        if self.model.cache.is_direct:
            return
        self.free_pages.append(index)


class Batch:
    def __init__(
        self,
        parent: TorchGenerator,
        token_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        cache_state: list[torch.Tensor],
    ):
        self.bs = token_ids.shape[0]
        assert seq_lens.shape[0] == self.bs
        self.parent = parent
        self.token_ids = token_ids
        self.seq_lens = seq_lens
        self.cache_state = cache_state
        self.results: list[list[int]] = [[] for _ in range(self.bs)]
        self.done_result_indices: set[int] = set()

        # Assemble the batch.
        seq_stride = self.parent.block_seq_stride
        self.seq_block_ids: list[list[int]] = []
        for seq_len in self.seq_lens:
            blocks_needed = (
                int(math.ceil(seq_len / seq_stride)) if seq_stride > 0 else 0
            )
            row = []
            for _ in range(blocks_needed):
                row.append(self.parent.alloc_page())
            self.seq_block_ids.append(row)

    @property
    def done(self) -> bool:
        return len(self.done_result_indices) == self.bs

    def detokenize(self) -> list[str]:
        return self.parent.tokenizer.decode(self.results)

    def print_current_results(self):
        results = self.detokenize()
        for i, s in enumerate(results):
            seq_len = int(self.seq_lens[i])
            print(f"  {i}({len(self.results[i])}, {seq_len}): {s}")

    def add_result_token(self, tokens: torch.Tensor):
        for i in range(self.bs):
            token = tokens[i][0]
            if token == self.parent.end_token:
                self.done_result_indices.add(i)
            if i in self.done_result_indices:
                continue
            token = int(tokens[i, 0])
            self.results[i].append(token)

    def allocate_seq_block_ids(self):
        for i in range(self.bs):
            sl = int(self.seq_lens[i])
            if (sl % self.parent.block_seq_stride) == 0:
                needed_blocks = sl // self.parent.block_seq_stride + 1
            else:
                needed_blocks = math.ceil(sl / self.parent.block_seq_stride)
            block_ids_row = self.seq_block_ids[i]
            while len(block_ids_row) < needed_blocks:
                block_ids_row.append(self.parent.alloc_page())

    def prefill(self):
        model = self.parent.model
        attention_mask = model.attention_mask(
            model.input_mask(self.seq_lens, self.token_ids.shape[1])
        )
        seq_block_ids_tensor = self.pad_block_ids()
        print(f":: Invoke prefill:")
        trace_tensor("prefill.token_ids", self.token_ids)
        trace_tensor("prefill.seq_block_ids", seq_block_ids_tensor)
        trace_tensor("prefill.attention_mask", attention_mask)
        logits = model.prefill(
            self.token_ids,
            attention_mask=attention_mask,
            seq_block_ids=seq_block_ids_tensor,
            cache_state=self.cache_state,
        )

        # TODO: Generalize the sampling and don't make it swap on/off cpu.
        # TODO: Normalize the output of extract_tokens_from_logits into
        # tensor [bs, 1].
        tokens = torch.tensor(
            model.extract_tokens_from_logits(logits, self.seq_lens)
        ).unsqueeze(1)
        print(f":: Prefill results:\n{tokens.tolist()}")
        self.add_result_token(tokens)
        self.next_tokens = tokens.to(device=model.device)

    def decode(self):
        model = self.parent.model
        start_positions = self.seq_lens.clone()
        self.seq_lens.add_(1)
        self.allocate_seq_block_ids()
        # TODO: Allocate more blocks on overflow.
        seq_block_ids_tensor = self.pad_block_ids()
        decode_attention_mask = model.decode_attention_mask(
            model.input_mask(
                self.seq_lens,
                seq_block_ids_tensor.shape[1] * self.parent.block_seq_stride,
            )
        )
        trace_tensor("decode.token_ids", self.next_tokens)
        trace_tensor("decode.start_positions", start_positions)
        trace_tensor("decode.seq_block_ids", seq_block_ids_tensor)
        trace_tensor("decode.attention_mask", decode_attention_mask)
        logits = model.decode(
            self.next_tokens,
            attention_mask=decode_attention_mask,
            start_positions=start_positions,
            seq_block_ids=seq_block_ids_tensor,
            cache_state=self.cache_state,
        )
        trace_tensor("decode.logits", logits)
        # TODO: Normalize the output of extract_tokens_from_logits into
        # tensor [bs, 1].
        tokens = torch.tensor(
            model.extract_tokens_from_logits(logits, [1] * self.bs),
            device=self.parent.model.device,
        ).unsqueeze(1)
        self.add_result_token(tokens)
        self.next_tokens = tokens

    def pad_block_ids(self) -> torch.Tensor:
        max_length = max(len(r) for r in self.seq_block_ids)
        rows = [r + (max_length - len(r)) * [0] for r in self.seq_block_ids]
        return torch.tensor(rows, device=self.parent.model.device)


def main():
    from ..utils import cli

    parser = cli.create_parser()
    parser.add_argument("prompt", nargs="+", help="Prompt strings")
    parser.add_argument("--kv-cache-type", default="paged", help="KV cache type")
    parser.add_argument("--device", help="Torch device (or default)")
    parser.add_argument("--save_intermediates_path", 
            help="save module forward outputs to safetensors, ex: run_0 will save to run_0_prefill.savetensors")
    parser.add_argument(
        "--activation-dtype",
        help="DType to use for activations in the model",
        default="float32",
    )
    cli.add_input_dataset_options(parser)
    cli.add_tokenizer_options(parser)
    args = cli.parse(parser)

    device = torch.device(args.device) if args.device else None
    activation_dtype = getattr(torch, args.activation_dtype)
    assert isinstance(activation_dtype, torch.dtype)
    dataset = cli.get_input_dataset(args)
    tokenizer = cli.get_tokenizer(args)
    prompts = args.prompt

    config = LlamaModelConfig(
        hp=configs.LlamaHParams.from_hf_props(dataset.properties),
        block_seq_stride=16,
        kv_cache_type=args.kv_cache_type,
        device=device,
        activation_dtype=activation_dtype,
        attention_dtype=activation_dtype,
    )
    model = PagedLlamaModelV1(dataset.root_theta, config)
    if args.save_intermediates_path:
        from ..utils.patching import SaveModuleResultTensorsPatch
        intermediates_saver = SaveModuleResultTensorsPatch()
        intermediates_saver.patch_child_modules(model)
    generator = TorchGenerator(model, tokenizer)

    print(f":: Prompting:")
    for prompt in prompts:
        print(f"    {prompt.encode()}")

    batch = generator.begin_batch(prompts)
    print(f":: Prompt tokens: {batch.token_ids}")
    batch.prefill()
    print(batch.detokenize())

    if args.save_intermediates_path:
        intermediates_saver.save_file(args.save_intermediates_path+"_prefill.safetensors")
    counter = 0
    while not batch.done:
        batch.decode()
        if args.save_intermediates_path:
            intermediates_saver.save_file(args.save_intermediates_path+f"_step_{counter}.safetensors")
        print(f":: Result tokens: {batch.results}")
        batch.print_current_results()


if __name__ == "__main__":
    main()
