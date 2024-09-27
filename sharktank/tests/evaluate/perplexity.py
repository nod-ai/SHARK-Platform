# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss

from datasets import load_dataset, load_from_disk

from sharktank.layers import *
from sharktank.types import *

# TODO: Should be using a base class with the protocol supported.
from sharktank.models.mixtral.mixtral import *

# from sharktank.models.grok.grok import *
from sharktank.models.llama.llama import *

from sharktank.utils import cli
from sharktank.utils.load_llm import *


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
    ):
        self.prompts = prompts
        self.add_start_token = False
        self.batch_size = 16
        self.bs = len(prompts)

    def load_model(self, dataset, tokenizer):

        theta = dataset.root_theta

        config = LlamaModelConfig(
            hp=configs.LlamaHParams.from_gguf_props(dataset.properties),
            block_seq_stride=16,
            kv_cache_type=args.kv_cache_type,
            device=device,
            activation_dtype=torch.float16,
            attention_dtype=torch.float16,
        )

        if config.hp.expert_count:
            # TODO Add grok after merging grok changes
            # if config.hp.model_arch == "grok":
            # pass
            # model = PagedGrokModelV1(theta, config)
            # else:
            model = PagedMixtralModelV1(theta, config)
        else:
            model = PagedLlamaModelV1(theta, config)

        self.generator = TorchGenerator(model, tokenizer)

    def get_logits(self):

        # print(f":: Prompting:")
        # for prompt in prompts:
        #     print(f"    {prompt.encode()}")

        token_ids, seq_lens = self.generator.tokenizer.encode(
            self.prompts,
            pad_to_multiple_of=self.generator.model.cache.pad_sequence_stride,
            add_start_token=self.add_start_token,
        )

        # print(f":: Prompt tokens: {token_ids}")
        max_prompt_length = max(seq_lens)

        self.token_ids = torch.tensor(token_ids, device=self.generator.model.device)
        self.attention_mask = (self.token_ids != 0).int()

        is_first_token = True
        for i in tqdm(
            range(0, max_prompt_length),
            desc="eval-perplexity: Load models & Fetching logits",
        ):

            token_batch = self.token_ids[:, i : i + 1]
            seq_lens_batch = torch.tensor([i] * self.bs)
            # print('loop', i, token_batch, seq_lens_batch)

            if is_first_token:
                token_batch, seq_lens_batch = self.generator.tokenizer.pad_tokens(
                    token_ids=token_batch.tolist(),
                    pad_to_multiple_of=self.generator.model.cache.pad_sequence_stride,
                )
                token_batch = torch.tensor(
                    token_batch, device=self.generator.model.device
                )
                seq_lens_batch = torch.tensor(
                    seq_lens_batch, device=self.generator.model.device
                )

            self.batch = self.generator.begin_eval_batch(
                token_batch=token_batch,
                seq_lens_batch=seq_lens_batch,
                bs=self.bs,
            )

            if is_first_token:
                is_first_token = False
                self.cache_state = self.batch.prefill()
                # print(self.batch.detokenize())
                self.pad_logits = self.batch.prefill_logits[:, 1:2, :]
                self.out_logits = self.batch.prefill_logits[:, 0:1, :]

                # print('prefill_logits', self.batch.prefill_logits.shape, self.pad_logits.shape, self.out_logits.shape)
                continue

            self.cache_state = self.batch.decode(self.cache_state)

            # print(f":: Result tokens: {self.batch.results}")
            # self.batch.print_current_results()
            # print('decode_logits', self.batch.decode_logits.shape)
            self.out_logits = torch.cat((self.out_logits, self.batch.decode_logits), 1)

            # print('out_logits', self.out_logits.shape)
        shape_diff = self.token_ids.shape[1] - self.out_logits.shape[1]
        tensor_pad = torch.cat([self.pad_logits] * shape_diff, 1)
        # print(shape_diff, tensor_pad.shape)
        self.out_logits = torch.cat((self.out_logits, tensor_pad), 1)
        # print('out_logits', self.out_logits.shape)

    def compute_perplexity(self):

        self.get_logits()

        loss_fct = CrossEntropyLoss(reduction="none")
        # num_prompts = len(self.prompts)

        self.out_logits = self.out_logits.contiguous()
        self.token_ids = self.token_ids.contiguous()
        assert self.token_ids.shape == self.out_logits.shape[0:2]

        # print('shift3', self.out_logits.shape, self.out_logits.transpose(1, 2).shape)
        # print('shift4', self.token_ids, self.token_ids.shape)
        # print('shift6', self.out_logits[1,0,:100], self.out_logits[1,1,:100], self.out_logits[1,14,:100])
        # print('shift7', self.attention_mask, self.attention_mask.shape, self.attention_mask.sum(1))

        # perplexity = e ^ (sum(losses) / num_tokenized_tokens)
        crossentropy_loss = (
            loss_fct(self.out_logits.transpose(1, 2), self.token_ids)
            * self.attention_mask
        ).sum(1)
        crossentropy_loss = torch.tensor(crossentropy_loss.tolist())
        print("crossentropy_loss", crossentropy_loss)
        perplexity_batch = torch.exp(
            crossentropy_loss / self.attention_mask.sum(1)
        ).tolist()

        perplexity = dict(map(lambda i, j: (i, j), self.prompts, perplexity_batch))

        return {
            "perplexities": perplexity,
            "mean_perplexity": np.mean(perplexity_batch),
        }


if __name__ == "__main__":
    parser = cli.create_parser()
    # parser.add_argument("--prompt", nargs="+", help="Prompt strings")
    parser.add_argument("--kv-cache-type", default="paged", help="KV cache type")
    parser.add_argument("--device", help="Torch device (or default)")

    cli.add_input_dataset_options(parser)
    cli.add_tokenizer_options(parser)
    args = cli.parse(parser)

    device = torch.device(args.device) if args.device else None

    dataset = cli.get_input_dataset(args)
    tokenizer = cli.get_tokenizer(args)

    input_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"][
        :10
    ]

    input_texts = [s for s in input_texts if s != ""]

    # input_texts = ["Happy Birthday!", "Write a story about LLamas"]

    perplexity = Perplexity(prompts=input_texts)

    perplexity.load_model(dataset, tokenizer)
    ppl = perplexity.compute_perplexity()

    print(json.dumps(ppl, indent=2))
