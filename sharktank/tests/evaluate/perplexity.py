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
            activation_dtype=torch.float16,
            attention_dtype=torch.float16,
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

        print(f"\n\n:: Prompts:")
        for idx, prompt in enumerate(self.prompts):
            print(f"    {prompt.encode()}")
            print(f"    {token_ids[idx]}\n")

        max_prompt_length = max(seq_lens)

        self.token_ids = torch.tensor(token_ids, device=self.generator.model.device)
        self.attention_mask = torch.tensor((self.token_ids != 0).int())

        is_first_token = True
        for i in tqdm(
            range(0, max_prompt_length),
            desc="eval-perplexity: Load models & Fetching logits",
        ):
            token_batch = self.token_ids[:, : i + 1]
            seq_lens_batch = torch.tensor([i] * self.bs)
            # print('token_batch\n', token_batch, self.generator.tokenizer.decode(token_batch))

            token_batch, seq_lens_batch = self.generator.tokenizer.pad_tokens(
                token_ids=token_batch.tolist(),
                pad_to_multiple_of=self.generator.model.cache.pad_sequence_stride,
            )
            token_batch = torch.tensor(token_batch, device=self.generator.model.device)
            seq_lens_batch = torch.tensor(
                seq_lens_batch, device=self.generator.model.device
            )

            self.batch = self.generator.begin_eval_batch(
                token_batch=token_batch,
                seq_lens_batch=seq_lens_batch,
                bs=self.bs,
            )

            self.cache_state = self.batch.prefill()
            # print(self.batch.detokenize())

            if is_first_token:
                self.out_logits = self.batch.prefill_logits[:, 0:1, :]
                is_first_token = False
            else:
                self.out_logits = torch.cat(
                    (self.out_logits, self.batch.prefill_logits[:, 0:1, :]), 1
                )

            if i == max_prompt_length - 1:
                self.pad_logits = self.batch.prefill_logits[:, i + 1 :, :]

            # self.cache_state = self.batch.decode(self.cache_state)
            # print(f":: Result tokens: {self.batch.results}")
            # self.batch.print_current_results()
            # self.out_logits = torch.cat((self.out_logits, self.batch.decode_logits), 1)

        # print('out_logits', self.out_logits.shape, self.pad_logits.shape, self.token_ids.shape)
        # shape_diff = self.token_ids.shape[1] - self.out_logits.shape[1]
        # self.pad_logits = torch.cat([self.pad_logits] * shape_diff, 1)
        self.out_logits = torch.cat((self.out_logits, self.pad_logits), 1)
        # print('out_logits', self.out_logits.shape, self.pad_logits.shape, self.token_ids.shape)

    def compute_perplexity(self):

        self.get_logits()

        loss_fct = CrossEntropyLoss(reduction="none")

        self.out_logits = self.out_logits[..., :-1, :].contiguous()
        self.token_ids = self.token_ids[..., 1:].contiguous()
        self.attention_mask = self.attention_mask[..., 1:].contiguous()

        # print('shift3', self.out_logits[0,0,:20], self.out_logits.shape)
        # print('shift4', self.token_ids, self.token_ids.shape)
        # print('shift7', self.attention_mask, self.attention_mask.shape)

        assert self.token_ids.shape == self.out_logits.shape[0:2]

        # torch.save([self.out_logits, self.token_ids, self.attention_mask], '/home/aramalin/sharktank/logits_perplexity_sharktank.pt')

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
    # parser.add_argument("-p", "--prompt", nargs='+', help="Prompt strings", required=True)
    parser.add_argument("--kv-cache-type", default="paged", help="KV cache type")
    parser.add_argument("--device", help="Torch device (or default)")

    cli.add_input_dataset_options(parser)
    cli.add_tokenizer_options(parser)
    args = cli.parse(parser)
    # input_texts = args.prompt

    device = torch.device(args.device) if args.device else None

    dataset = cli.get_input_dataset(args)
    tokenizer = cli.get_tokenizer(args)

    input_texts = [
        'Robert Boulter is an English film, television and theatre actor. He had a guest-starring role on the television series "The Bill" in 2000.',
        "Du Fu was a prominent Chinese poet of the Tang dynasty. Along with Li Bai (Li Po), he is frequently called the greatest of the Chinese poets.",
        "The Ise-class battleships were a pair of dreadnought battleships built for the Imperial Japanese Navy (IJN) during World War I. Originally intended to be repeats of the preceding Fusō class, they were redesigned before construction began. Both ships carried supplies for the survivors of the Great Kantō earthquake in 1923. They were modernized in 1934-37 with improvements to their armour and machinery and a rebuilt superstructure in the pagoda mast style. Afterwards they played a minor role in the Second Sino-Japanese War.",
        'Richard Gale "Dick" Rifenburg (August 21, 1926-December 5, 1994) was an American football player and a pioneering television broadcaster for the forerunner to WIVB-TV in Buffalo. He played college football for the University of Michigan Wolverines in 1944 and from 1946 to 1948. He was a consensus selection at end on the 1948 College Football All-America Team. Rifenburg played professionally in the National Football League (NFL) with the Detroit Lions for one season in 1950. After retiring from football he settled in Buffalo and became a sports broadcaster.',
        "An oxaziridine is an organic molecule that features a three-membered heterocycle containing oxygen, nitrogen, and carbon. In their largest application, oxazidines are intermediates in the industrial production of hydrazine. Oxaziridine derivatives are also used as specialized reagents in organic chemistry for a variety of oxidations, including alpha hydroxylation of enolates, epoxidation and aziridination of olefins, and other heteroatom transfer reactions.",
    ]
    input_texts = ["Happy Birthday!", "Write a story about LLamas"]

    ppl = run_perplexity(prompts=input_texts, dataset=dataset, tokenizer=tokenizer)

    print(json.dumps(ppl, indent=2))
