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

from sharktank.models.llama.llama import *
from sharktank.models.mixtral.mixtral import *
from sharktank.models.grok.grok import *

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

    def get_logits(self):

        token_ids, seq_lens = self.generator.tokenizer.encode(
            self.prompts,
            pad_to_multiple_of=self.generator.model.cache.pad_sequence_stride,
            add_start_token=self.add_start_token,
        )

        print(f":: Prompts:")
        for prompt in self.prompts:
            print(f"    {prompt.encode()}")

        print(f":: Prompt token ids: {token_ids}")

        max_prompt_length = max(seq_lens)

        self.token_ids = torch.tensor(token_ids, device=self.generator.model.device)
        self.attention_mask = torch.tensor((self.token_ids != 0).int())

        is_first_token = True
        for i in tqdm(
            range(0, max_prompt_length),
            desc="eval-perplexity: Load models & Fetching logits",
        ):
            # for i in range(0, max_prompt_length):
            token_batch = self.token_ids[:, i : i + 1]
            seq_lens_batch = torch.tensor([i] * self.bs)

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
                # torch.save(self.batch.prefill_logits, '/home/aramalin/sharktank/logits_perplexity_prefill.pt')

                self.pad_logits = self.batch.prefill_logits[:, :, :]
                self.out_logits = self.batch.prefill_logits[:, 0:1, :]
                continue

            self.cache_state = self.batch.decode(self.cache_state)

            # print(f":: Result tokens: {self.batch.results}")
            # self.batch.print_current_results()
            # print('decode_logits', self.batch.decode_logits.shape)
            self.out_logits = torch.cat((self.out_logits, self.batch.decode_logits), 1)

        pad_logits_shape = self.out_logits.shape[1]
        self.out_logits = torch.cat(
            (self.out_logits, self.pad_logits[:, pad_logits_shape:, :]), 1
        )
        # print('out_logits', self.out_logits.shape, pad_logits_shape)

    def compute_perplexity(self):

        self.get_logits()

        loss_fct = CrossEntropyLoss(reduction="none")

        self.out_logits = self.out_logits[..., :-1, :].contiguous()
        self.token_ids = self.token_ids[..., 1:].contiguous()
        self.attention_mask = self.attention_mask[..., 1:].contiguous()
        assert self.token_ids.shape == self.out_logits.shape[0:2]

        # print('shift3', self.out_logits[0,0,20], self.out_logits.shape)
        # print('shift4', self.token_ids, self.token_ids.shape)
        # print('shift7', self.attention_mask, self.attention_mask.shape)

        # perplexity = e ^ (sum(losses) / num_tokenized_tokens)
        crossentropy_loss = (
            loss_fct(self.out_logits.transpose(1, 2), self.token_ids)
            * self.attention_mask
        ).sum(1)
        crossentropy_loss = torch.tensor(crossentropy_loss.tolist())
        perplexity_batch = torch.exp(
            crossentropy_loss / self.attention_mask.sum(1)
        ).tolist()

        perplexity = dict(map(lambda i, j: (i, j), self.prompts, perplexity_batch))

        return {
            "perplexities": perplexity,
            "mean_perplexity": np.mean(perplexity_batch),
        }


def run_perplexity(
    prompts: list[str],
    dataset,
    tokenizer,
):
    perplexity = Perplexity(prompts=prompts)

    perplexity.load_model(dataset, tokenizer)
    ppl = perplexity.compute_perplexity()

    return ppl


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

    input_texts = ["Happy Birthday!", "Write a story about LLamas"]

    # input_texts = [' Robert Boulter is an English film , television and theatre actor .', ' He had a guest starring role on the television series The Bill in 2000 .', ' This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre .', ' He had a guest role in the television series Judge John Deed in 2002 . In 2004 Boulter landed a role as " Craig " in the episode " Teddy \'s Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi .', ' He was cast in the 2005 theatre productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the Menier Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall . \n', ' In 2006 , Boulter starred alongside Whishaw in the play Citizenship written by Mark Ravenhill . He appeared on a 2006 episode of the television series , Doctors , followed by a role in the 2007 theatre production of How to Curse directed by Josie Rourke . How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham . Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris Leonti , and Donkey Punch directed by Olly Blackburn .', ' In May 2008 , Boulter made a guest appearance on a two @-@ part episode arc of the television series Waking the Dead , followed by an appearance on the television series Survivors in November 2008 . He had a recurring role in ten episodes of the television series Casualty in 2010 , as " Kieron Fletcher " . Boulter starred in the 2011 film Mercenaries directed by Paris Leonti . \n']

    ppl = run_perplexity(prompts=input_texts, dataset=dataset, tokenizer=tokenizer)

    print(json.dumps(ppl, indent=2))
