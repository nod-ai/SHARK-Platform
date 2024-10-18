# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import logging
import time
from datetime import timedelta

import json
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss

from sharktank.layers import *
from sharktank.types import *

from sharktank.models.llama.llama import *
from sharktank.models.mixtral.mixtral import *
from sharktank.models.grok.grok import *

from sharktank.utils import cli
from sharktank.utils.load_llm import *

log_levels = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
logger = logging.getLogger("eval")

logger.setLevel(log_levels["debug"])

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
        prompts: list,
        device,
        kv_cache_type,
    ):
        self.prompts = prompts
        self.add_start_token = False
        self.batch_size = 16
        self.bs = len(prompts)
        self.device = device
        self.kv_cache_type = kv_cache_type

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
            batch_predicted_token = self.generator.tokenizer.decode(
                batch_predicted_token_id
            )
            logger.debug(f"Predicted:")
            logger.debug(f"{batch_predicted_token}")
            logger.debug(f"{batch_predicted_token_id}")

            expected_token_id = self.token_ids[:, i + 1 : i + 2].tolist()
            expected_token = self.generator.tokenizer.decode(expected_token_id)
            logger.debug(f"Expected:")
            logger.debug(f"{expected_token}")
            logger.debug(f"{expected_token_id}")

    @timeit
    def load_model(self, dataset, tokenizer):

        theta = dataset.root_theta

        config = LlamaModelConfig(
            hp=configs.LlamaHParams.from_gguf_props(dataset.properties),
            block_seq_stride=16,
            kv_cache_type=self.kv_cache_type,
            device=self.device,
            activation_dtype=torch.float32,
            attention_dtype=torch.float32,
        )

        if config.hp.expert_count:
            if config.hp.model_arch == "grok":
                model = PagedGrokModelV1(theta, config)
            else:
                model = PagedMixtralModelV1(theta, config)
        else:
            model = PagedLlamaModelV1(theta, config)

        self.generator = TorchGenerator(model, tokenizer)

    @timeit
    def get_logits(self):

        token_ids, seq_lens = self.generator.tokenizer.encode(
            self.prompts,
            pad_to_multiple_of=self.generator.model.cache.pad_sequence_stride,
            add_start_token=self.add_start_token,
        )

        logger.info(f" Prompts:")
        for idx, prompt in enumerate(self.prompts):
            logger.info(f" Prompt {idx} - {prompt.encode()}\n{token_ids[idx]}")

        self.max_prompt_length = max(seq_lens)

        self.token_ids = torch.tensor(token_ids, device=self.device)
        self.attention_mask = (
            (self.token_ids != 0).int().detach().clone().to(self.device)
        )

        is_first_token = True
        for i in tqdm(
            range(0, self.max_prompt_length - 1),
            desc="eval: Calculating logits",
        ):
            token_batch = self.token_ids[:, : i + 1]
            logger.debug(f"Prefill:")

            logger.debug("Input:")
            logger.debug(f"{self.generator.tokenizer.decode(token_batch)}")

            token_batch, seq_lens_batch = self.generator.tokenizer.pad_tokens(
                token_ids=token_batch.tolist(),
                pad_to_multiple_of=self.generator.model.cache.pad_sequence_stride,
            )

            token_batch = torch.tensor(token_batch, device=self.device)
            seq_lens_batch = torch.tensor(seq_lens_batch, device=self.device)

            self.batch = self.generator.begin_eval_batch(
                token_batch=token_batch,
                seq_lens_batch=seq_lens_batch,
                bs=self.bs,
            )

            self.cache_state = self.batch.prefill()
            self.print_token_comparison(i)

            if is_first_token:
                self.out_logits = self.batch.prefill_logits[:, 0:1, :]
                is_first_token = False
            else:
                self.out_logits = torch.cat(
                    (self.out_logits, self.batch.prefill_logits[:, 0:1, :]), 1
                )

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

        return {
            "perplexities": perplexity_batch,
            "mean_perplexity": np.mean(perplexity_batch),
        }

    @timeit
    def get_perplexity(self):

        self.get_logits()

        self.out_logits = self.out_logits[..., :-1, :].contiguous()
        self.token_ids = self.token_ids[..., 1:].contiguous()
        self.attention_mask = self.attention_mask[..., 1:].contiguous()

        assert self.token_ids.shape == self.out_logits.shape[0:2]

        logger.debug(f"Logits shape: {self.out_logits.shape}")
        logger.debug(f"Token ids: {self.token_ids}, {self.token_ids.shape}")
        logger.debug(
            f"Logits shape: {self.attention_mask}, {self.attention_mask.shape}"
        )

        return self.compute_perplexity()


def run_perplexity(
    prompts: list[str],
    dataset,
    tokenizer,
    device,
    kv_cache_type,
):
    perplexity = Perplexity(prompts=prompts, device=device, kv_cache_type=kv_cache_type)

    perplexity.load_model(dataset, tokenizer)
    ppl = perplexity.get_perplexity()

    return ppl


def main(argv):
    parser = cli.create_parser()
    parser.add_argument("--kv-cache-type", default="paged", help="KV cache type")
    parser.add_argument("--device", help="Torch device (or default)")

    cli.add_input_dataset_options(parser)
    cli.add_tokenizer_options(parser)
    args = cli.parse(parser, args=argv)

    device = torch.device(args.device) if args.device else None
    kv_cache_type = args.kv_cache_type
    dataset = cli.get_input_dataset(args)
    tokenizer = cli.get_tokenizer(args)

    prompt_path = "sharktank/evaluate/data/eval_prompts.txt"
    with open(prompt_path, "r") as f:
        input_texts = f.read().splitlines()

    ppl = run_perplexity(
        prompts=input_texts[0:1],
        dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        kv_cache_type=kv_cache_type,
    )

    logger.info(f"\n{json.dumps(ppl, indent=2)}")
    return ppl


if __name__ == "__main__":
    main(sys.argv[1:])
