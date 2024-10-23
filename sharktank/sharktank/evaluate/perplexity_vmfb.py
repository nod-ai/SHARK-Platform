# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import logging
import json
import time
import random
from datetime import timedelta
from tqdm import tqdm

import numpy as np

from datasets import load_dataset

import torch
from torch.nn import CrossEntropyLoss

from sharktank.layers import *
from sharktank.types import *

from sharktank.utils.vmfb_runner import *
from sharktank.utils import cli
from sharktank.utils.load_llm import *

import iree.runtime as ireert

log_levels = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
logger = logging.getLogger("eval")

logger.setLevel(log_levels["info"])

logger.root.handlers[0].setFormatter(
    logging.Formatter(fmt="\n%(levelname)s:%(name)-8s %(message)s")
)

__all__ = ["Perplexity", "run_perplexity"]


class Perplexity:
    """
    Perplexity (PPL) is one of the most common metrics for evaluating language models.
    It is defined as the exponentiated average negative log-likelihood of a sequence,
    calculated with exponent base `e`.

    For more information, see https://huggingface.co/docs/transformers/perplexity
    """

    def __init__(
        self,
        device,
        tokenizer,
    ):
        self.device = device
        self.tokenizer = tokenizer
        self.pad_sequence_stride = 16
        self.block_seq_stride = 16
        self.free_pages = list(range(1, 8192))
        # TODO: investigate cache
        self.cache_state = model.cache.paged.allocate(page_cache_size)

    def timeit(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            seconds = end - start
            time_taken = abs(timedelta(seconds=round(seconds)))

            if seconds < 1:
                time_taken = f" {seconds * 1000} ms"

            func_name = func.__name__
            if func_name == "get_perplexity":
                func_name = "Total time"
            logger.info(f" {func_name}: {time_taken}")
            return result

        return wrapper

    def print_token_comparison(self, i):
        if i <= self.max_prompt_length:
            batch_predicted_token_id = [[i[-1]] for i in self.batch.results]
            batch_predicted_token = self.tokenizer.decode(batch_predicted_token_id)
            logger.debug(f"Predicted:")
            logger.debug(f"{batch_predicted_token}")
            logger.debug(f"{batch_predicted_token_id}")

            expected_token_id = self.token_ids[:, i + 1 : i + 2].tolist()
            expected_token = self.tokenizer.decode(expected_token_id)
            logger.debug(f"Expected:")
            logger.debug(f"{expected_token}")
            logger.debug(f"{expected_token_id}")

    def alloc_page(self) -> int:
        # Only applies for paged attention
        return self.free_pages.pop()

    def pad_block_ids(self, seq_block_ids) -> torch.Tensor:
        max_length = max(len(r) for r in seq_block_ids)
        rows = [r + (max_length - len(r)) * [0] for r in seq_block_ids]
        return torch.tensor(rows)

    @timeit
    def load_model(self, vmfb_path, gguf_weight_path):
        return vmfbRunner(
            device=self.device,
            vmfb_path=vmfb_path,
            external_weight_path=gguf_weight_path,
        )

    def get_args(self, seq_lens_batch):
        # Assemble the batch.
        seq_stride = self.block_seq_stride
        seq_block_ids: list[list[int]] = []
        for seq_len in seq_lens_batch:
            blocks_needed = (
                int(math.ceil(seq_len / seq_stride)) if seq_stride > 0 else 0
            )
            row = []
            for _ in range(blocks_needed):
                row.append(self.alloc_page())
            seq_block_ids.append(row)

        return seq_block_ids

    @timeit
    def get_prompts(self):
        test_prompts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")[
            "text"
        ]

        num_test_prompts = 219

        random.seed(0)
        test_prompts = random.sample(test_prompts, num_test_prompts)

        # Ignore prompts that are: empty, less than 20 tokens or a title.
        test_prompts = [
            s.replace("\n", "").rstrip()
            for s in test_prompts
            if s != "" and len(s.split()) >= 20 and s.count("=") < 2
        ]

        logger.info(f" num_test_prompts: {len(test_prompts)}")

        return test_prompts

    @timeit
    def get_logits(
        self,
    ):

        token_ids, seq_lens = self.tokenizer.encode(
            self.test_prompts,
            pad_to_multiple_of=self.pad_sequence_stride,
        )

        logger.info(f" Prompts for Evaluation:")
        for idx, prompt in enumerate(self.test_prompts):
            logger.info(
                f" Prompt {idx}: \nTokens: {prompt.encode()}\nToken ids: {token_ids[idx]}\n"
            )

        self.max_prompt_length = max(seq_lens)
        self.token_ids = torch.tensor(token_ids)
        self.attention_mask = (self.token_ids != 0).int().detach().clone()

        self.bs = len(self.test_prompts)

        is_first_token = True
        start = 0
        for i in tqdm(
            range(start, self.max_prompt_length - 1),
            desc="eval: Calculating logits",
        ):
            logger.debug(f"Iteration: {i}")

            if is_first_token:

                token_batch = self.token_ids[:, : i + 1]
                logger.debug(f"Prefill:")

                logger.debug("Input:")
                logger.debug(f"{self.tokenizer.decode(token_batch)}")

                token_batch, seq_lens_batch = self.tokenizer.pad_tokens(
                    token_ids=token_batch.tolist(),
                    pad_to_multiple_of=self.pad_sequence_stride,
                )

                logger.debug(f"{token_batch}")

                token_batch = torch.tensor(token_batch, device=self.device)
                seq_lens_batch = torch.tensor(seq_lens_batch, device=self.device)

                seq_block_ids = self.get_args(seq_lens_batch)
                seq_block_ids = self.pad_block_ids(seq_block_ids)
                prefill_logits = self.runner.ctx.modules.module.prefill_bs4(
                    token_batch, seq_lens_batch, seq_block_ids, self.cache_state
                )

                self.out_logits = prefill_logits[:, -1, :]
                is_first_token = False

                self.print_token_comparison(i)

            else:
                token_batch = self.token_ids[:, i : i + 1]

                logger.debug("Decode:")

                logger.debug("Input:")
                logger.debug(f"{self.tokenizer.decode(token_batch)}")
                logger.debug(f"{token_batch.tolist()}")

                start_positions = seq_lens_batch.clone()
                seq_lens_batch.add_(1)

                seq_block_ids = self.get_args(seq_lens_batch)
                seq_block_ids = self.pad_block_ids(seq_block_ids)
                decode_logits = self.runner.ctx.modules.module.decode_bs4(
                    token_batch, start_positions, seq_block_ids, self.cache_state
                )

                self.out_logits = torch.cat((self.out_logits, decode_logits), 1)

                self.print_token_comparison(i)

        pad_logits_shape = self.token_ids.shape[1] - self.out_logits.shape[1]

        self.pad_logits = torch.zeros(
            self.out_logits.shape[0], pad_logits_shape, self.out_logits.shape[2]
        )

        self.out_logits = torch.cat((self.out_logits, self.pad_logits), 1).to(
            self.device
        )

    @timeit
    def compute_perplexity(self):
        loss_fct = CrossEntropyLoss(reduction="none")

        ## perplexity = e ^ (sum(losses) / num_tokenized_tokens)
        crossentropy_loss = (
            loss_fct(self.out_logits.transpose(1, 2), self.token_ids)
            * self.attention_mask
        ).sum(1)
        crossentropy_loss = torch.tensor(crossentropy_loss.tolist())
        perplexity_batch = torch.exp(
            crossentropy_loss / self.attention_mask.sum(1)
        ).tolist()

        perplexity_batch = [round(ppl, 6) for ppl in perplexity_batch]

        return {
            "perplexities": perplexity_batch,
            "mean_perplexity": round(np.mean(perplexity_batch), 6),
        }

    @timeit
    def get_perplexity(self, test_prompts):

        self.test_prompts = test_prompts
        self.get_logits()

        self.out_logits = self.out_logits[..., :-1, :].contiguous()
        self.token_ids = self.token_ids[..., 1:].contiguous()
        self.attention_mask = self.attention_mask[..., 1:].contiguous()

        logger.debug(f"Final Logits shape: {self.out_logits.shape}")
        logger.debug(f"Token ids: {self.token_ids}, \n{self.token_ids.shape}")
        logger.debug(
            f"Mask shape: {self.attention_mask}, \n{self.attention_mask.shape}"
        )

        assert self.token_ids.shape == self.out_logits.shape[0:2]

        return self.compute_perplexity()


def run_perplexity(
    vmfb_path,
    gguf_weight_path,
    tokenizer,
    device,
):
    perplexity = Perplexity(device=device, tokenizer=tokenizer)
    perplexity.load_model(tokenizer, vmfb_path, gguf_weight_path)
    test_prompts = perplexity.get_prompts()
    ppl = perplexity.get_perplexity(test_prompts=test_prompts)

    return ppl


def main(argv):
    parser = cli.create_parser()
    parser.add_argument("--device", help="Torch device (or default)")

    cli.add_tokenizer_options(parser)
    args = cli.parse(parser, args=argv)

    device = torch.device(args.device) if args.device else None
    tokenizer = cli.get_tokenizer(args)

    # device could be local-sync:// local-task://
    device = "hip://GPU-34346462-3466-6333-3231-353561336563"
    vmfb_path = "/home/aramalin/SHARK-Platform/artifacts/llama70b_q4_1.vmfb"
    gguf_weight_path = "/data/extra/models/llama70b_q4_1.gguf"

    ppl = run_perplexity(
        vmfb_path=vmfb_path,
        gguf_weight_path=gguf_weight_path,
        tokenizer=tokenizer,
        device=device,
    )

    logger.info(f"\n{json.dumps(ppl, indent=2)}")
    return ppl


if __name__ == "__main__":
    main(sys.argv[1:])
