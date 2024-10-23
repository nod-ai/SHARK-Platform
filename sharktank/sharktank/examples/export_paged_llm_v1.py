# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Export support for the PagedLLMV1 protocol of models."""

import json
import torch

from iree.turbine.aot import *

from sharktank.layers import *
from sharktank.types import *

# TODO: Should be using a base class with the protocol supported.
from ..models.llama.llama import LlamaModelConfig, PagedLlamaModelV1
from ..models.llama.sharding import shard_theta
from ..models.mixtral.mixtral import *
from ..models.grok.grok import *
from .. import ops


def main():
    from ..utils import cli

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
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
    parser.add_argument(
        "--strict",
        help="Enables strictness during export",
        action="store_true",
    )
    parser.add_argument(
        "--attention-kernel",
        type=str,
        default="decomposed",
        choices=["decomposed", "torch_sdpa"],
    )
    parser.add_argument(
        "--tensor-parallelism-size",
        type=int,
        default=1,
        help="How many devices are involved for tensor parallel sharding.",
    )

    args = cli.parse(parser)
    dataset_type = cli.get_input_data_files(args)
    dataset_type = "irpa" if "irpa" in dataset_type else "gguf"
    dataset = cli.get_input_dataset(args)

    hp = configs.LlamaHParams.from_gguf_props(dataset.properties)
    llama_config = LlamaModelConfig(hp)
    if args.tensor_parallelism_size > 1:
        dataset.root_theta = shard_theta(dataset.root_theta, llama_config)
    llama_config.use_hf = False
    llama_config.static_tables = False  # Rely on the compiler for hoisting tables.
    llama_config.kv_cache_type = "direct" if args.bs == [1] else "paged"
    llama_config.attention_kernel = args.attention_kernel

    # This is a bit gross and should be changed in the future. Best Idea I had so far.
    attn_q_weight = dataset.root_theta.tensor("blk")["0"]["attn_q"]["weight"]
    if isinstance(attn_q_weight, SplitPrimitiveTensor):
        llama_config.tensor_parallelism_size = attn_q_weight.shard_count

    if llama_config.hp.expert_count:
        if llama_config.hp.model_arch == "grok":
            model = PagedGrokModelV1(dataset.root_theta, llama_config)
        else:
            model = PagedMixtralModelV1(dataset.root_theta, llama_config)
    else:
        model = PagedLlamaModelV1(dataset.root_theta, llama_config)

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
            "block_seq_stride": llama_config.block_seq_stride,
        }

    # Unrolling cache updates by batch row makes dynamo sad without an
    # override. There may be a better way to do this.
    import torch._dynamo.config as dynamo_config

    # TODO: Seems removed from 2.3+
    # dynamo_config.max_loop_unroll_nodes = 0

    fxb = FxProgramsBuilder(model)

    def setup_cache(model, shard_count):
        if model.config.kv_cache_type == "paged":
            cache_state = model.cache.allocate(
                page_count=hp.context_length // llama_config.block_seq_stride
            )
            page_dim = torch.export.Dim("page")
            dynamic_shapes = [{0: page_dim}]
        elif model.config.kv_cache_type == "direct":
            cache_state = model.cache.allocate(bs=1)
            # Direct cache dimensions:
            #   2 * transformer_block_count of...
            #   [bs, seq_length, attn_head_count, attn_head_dim]
            cache_state_dynamic_shapes = (2 * hp.block_count) * [{}]
        else:
            raise NotImplementedError(f"Unsupported KV cache type: {type(model.cache)}")

        unpacked = cache_state
        dynamic_shapes = dynamic_shapes
        arg_affinities = {}
        shard_dim = None

        # Need to unpacke that state when sharded
        if llama_config.tensor_parallelism_size > 1:
            shard_dim = cache_state[0].shard_dim

            unpacked = [[shard._data for shard in cs.shards] for cs in cache_state]
            dynamic_shapes = [
                [ds] * llama_config.tensor_parallelism_size for ds in dynamic_shapes
            ]

            for i in range(llama_config.tensor_parallelism_size):
                arg_affinities[i] = DeviceAffinity(str(i))

        return unpacked, shard_dim, dynamic_shapes, arg_affinities

    def repack_cache(cache, shard_dim):
        return [SplitPrimitiveTensor(ts=c, shard_dim=shard_dim) for c in cache]

    def generate_batch_prefill(bs: int):
        tokens = torch.empty(bs, 64, dtype=torch.int64)
        seq_lens = torch.empty(bs, dtype=torch.int64)
        seq_block_ids = torch.empty(bs, 4, dtype=torch.int64)
        block_dim = torch.export.Dim(
            "block", max=(hp.context_length - 1) // llama_config.block_seq_stride
        )
        sl_dim = llama_config.block_seq_stride * block_dim

        cache, cache_shard_dim, cache_dynamic_shapes, arg_affinities = setup_cache(
            model, llama_config.tensor_parallelism_size
        )

        # We need to offset the indices for the cache
        arg_affinities = {key + 3: arg_affinities[key] for key in arg_affinities}

        dynamic_shapes = {
            "tokens": {1: sl_dim},
            "seq_lens": {},
            "seq_block_ids": {1: block_dim},
            "cs": cache_dynamic_shapes,
        }

        print(f"Exporting prefill_bs{bs}")

        @fxb.export_program(
            name=f"prefill_bs{bs}",
            args=(tokens, seq_lens, seq_block_ids, cache),
            dynamic_shapes=dynamic_shapes,
            strict=args.strict,
            argument_device_affinities=arg_affinities,
        )
        def _(model, tokens, seq_lens, seq_block_ids, cs):
            cache_tensors = cs

            sl = tokens.shape[1]
            input_mask = model.input_mask(seq_lens, sl)
            attention_mask = model.attention_mask(input_mask)

            if llama_config.tensor_parallelism_size != 1:
                shard_count = llama_config.tensor_parallelism_size

                tokens = ops.replicate(tokens, count=shard_count)
                attention_mask = ops.replicate(attention_mask, count=shard_count)
                seq_block_ids = ops.replicate(seq_block_ids, count=shard_count)

                cache_tensors = repack_cache(cs, cache_shard_dim)

            logits = model.prefill(
                tokens,
                attention_mask=attention_mask,
                seq_block_ids=seq_block_ids,
                cache_state=cache_tensors,
            )

            if llama_config.tensor_parallelism_size != 1:
                logits = ops.unshard(logits)

            return logits

    def generate_batch_decode(bs: int):
        tokens = torch.ones(bs, 1, dtype=torch.int64)
        seq_lens = torch.ones(bs, dtype=torch.int64)
        start_positions = torch.ones(bs, dtype=torch.int64)
        seq_block_ids = torch.zeros(bs, 4, dtype=torch.int64)
        block_dim = torch.export.Dim(
            "block", max=(hp.context_length - 1) // llama_config.block_seq_stride
        )

        (
            cache_state,
            cache_shard_dim,
            cache_dynamic_shapes,
            arg_affinities,
        ) = setup_cache(model, llama_config.tensor_parallelism_size)

        # We need to offset the indices for the cache
        arg_affinities = {key + 4: arg_affinities[key] for key in arg_affinities}

        dynamic_shapes = {
            "tokens": {},
            "seq_lens": {},
            "start_positions": {},
            "seq_block_ids": {1: block_dim},
            "cache_state": cache_dynamic_shapes,
        }

        print(f"Exporting decode_bs{bs}")

        @fxb.export_program(
            name=f"decode_bs{bs}",
            args=(
                tokens,
                seq_lens,
                start_positions,
                seq_block_ids,
                cache_state,
            ),
            dynamic_shapes=dynamic_shapes,
            strict=args.strict,
            argument_device_affinities=arg_affinities,
        )
        def _(
            model,
            tokens,
            seq_lens,
            start_positions,
            seq_block_ids,
            cache_state,
        ):
            input_mask = model.input_mask(
                seq_lens, seq_block_ids.shape[1] * model.cache.block_seq_stride
            )
            attention_mask = model.decode_attention_mask(input_mask)

            if llama_config.tensor_parallelism_size != 1:
                shard_count = llama_config.tensor_parallelism_size

                tokens = ops.replicate(tokens, count=shard_count)
                attention_mask = ops.replicate(attention_mask, count=shard_count)
                start_positions = ops.replicate(start_positions, count=shard_count)
                seq_block_ids = ops.replicate(seq_block_ids, count=shard_count)

                cache_state = repack_cache(cache_state, cache_shard_dim)

            logits = model.decode(
                tokens,
                attention_mask=attention_mask,
                start_positions=start_positions,
                seq_block_ids=seq_block_ids,
                cache_state=cache_state,
            )

            if llama_config.tensor_parallelism_size != 1:
                logits = ops.unshard(logits)

            return logits

    bsizes = []
    for bs in args.bs:
        generate_batch_prefill(bs)
        generate_batch_decode(bs)
        bsizes.append(bs)
    config = generate_params_json(hp, bsizes, bsizes)
    print("GENERATED!")

    if args.verbose:
        for name, ep in fxb.programs.items():
            print(f"EXPORT {name}:\n{ep}")

    print("Exporting")
    output = export(fxb)
    print(f"Saving to '{args.output_mlir}'")
    output.save_mlir(args.output_mlir)
    json.dump(config, open(args.output_config, "w"))


if __name__ == "__main__":
    main()
