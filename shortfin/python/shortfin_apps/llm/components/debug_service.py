# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Some code for debugging service.py. Importing this should do nothing and import no additional dependencies if DEBUG_LLM_SERVICE == False
"""


import asyncio
import logging
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import List
from pprint import pformat

logger = logging.getLogger(__name__)

import os

# Get environment variable, default to False if not set
SHORTFIN_DEBUG_LLM_SERVICE = os.getenv('SHORTFIN_DEBUG_LLM_SERVICE', 'False').lower() in ('true', 'yes', '1', 'y')
if SHORTFIN_DEBUG_LLM_SERVICE:
    logger.info("DEBUG_LLM_SERVICE=True")
    dump_id = 0
    boot_timestamp = datetime.now().isoformat()
    DEBUG_DATA_DIR = Path.home() / "sfdebug"
    DUMP_DIR_THIS_SESSION = DEBUG_DATA_DIR / f"llm_service_invocation_dumps_from_{boot_timestamp}"
    DUMP_DIR_THIS_SESSION.mkdir(parents=True, exist_ok=False)
    logger.info(f"[debug_service.py] Please find debug dumps for service.py in {DUMP_DIR_THIS_SESSION}")

async def pre_invocation_debug_dump(
    phase,
    is_decode,
    device0,
    fn,
    req_bs,
    bsl,
    seq_stride,
    block_count,
    req_count,
    exec_requests,
    tokens,
    start_positions,
    seq_lens,
    seq_block_ids,
    model_params,
    args
):
    """Comprehensive debug dump before inference invocation."""
    if not SHORTFIN_DEBUG_LLM_SERVICE:
        return
    
    global dump_id
    dump_path = DUMP_DIR_THIS_SESSION / f"{dump_id}"
    dump_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare debug info dictionary
    debug_info = {
        "metadata": {
            "dump_id": dump_id,
            "dump_timestamp": datetime.now().isoformat(),
            "phase": str(phase),
            "is_decode": is_decode,
            "device": str(device0),
            "function": str(fn)
        },
        "batch_info": {
            "request_batch_size": req_bs,
            "block_sequence_length": int(bsl),
            "sequence_stride": seq_stride,
            "block_count": block_count,
            "actual_request_count": req_count
        },
        "requests": [
            {
                "index": i,
                "start_position": req.start_position,
                "rid": req.rid,
                "input_token_ids": req.input_token_ids.tolist() if hasattr(req.input_token_ids, 'tolist') else list(req.input_token_ids),
                "input_length": len(req.input_token_ids),
                "cache_pages": req.cache_page_indices(block_count)
            }
            for i, req in enumerate(exec_requests)
        ],
        "tensor_shapes": {
            "tokens": tokens.shape,
            **({"start_positions": start_positions.shape} if is_decode else {}),
            "seq_lens": seq_lens.shape,
            "seq_block_ids": seq_block_ids.shape,
        },
        "tensor_values": {
            "tokens": tokens.for_transfer().items.tolist() if hasattr(tokens.for_transfer().items, 'tolist') else list(tokens.for_transfer().items),
            **({"start_positions": start_positions.for_transfer().items.tolist() if hasattr(start_positions.for_transfer().items, 'tolist') else list(start_positions.for_transfer().items)} if is_decode else {}),
            "sequence_lengths": seq_lens.for_transfer().items.tolist() if hasattr(seq_lens.for_transfer().items, 'tolist') else list(seq_lens.for_transfer().items),
            "sequence_block_ids": seq_block_ids.for_transfer().items.tolist() if hasattr(seq_block_ids.for_transfer().items, 'tolist') else list(seq_block_ids.for_transfer().items)
        },
        "model_config": {
            "prefill_batch_sizes": model_params.prefill_batch_sizes,
            "decode_batch_sizes": model_params.decode_batch_sizes,
            "attn_dtype": str(model_params.attn_dtype),
            "paged_kv_cache": {
                "device_block_count": model_params.paged_kv_cache.device_block_count,
                "block_seq_stride": model_params.paged_kv_cache.block_seq_stride,
                "prefix_sharing_algorithm": model_params.paged_kv_cache.prefix_sharing_algorithm
            }
        }
    }

    # Save debug info as JSON
    with open(dump_path / "info.json", "w") as f:
        json.dump(debug_info, f, indent=2)

    # Save program arguments
    path = dump_path
    args_np = []
    for i, a in enumerate(args):
        host_array = a.for_transfer()
        host_array.copy_from(a)
        await a.device
        args_np.append(np.array(host_array))
        
    # Save binary numpy arrays
    for i, arr in enumerate(args_np):
        np.save(path / f"{i}.npy", arr)
    
    # Generate human-readable report
    with open(path / "saved_program_args.txt", "w") as f:
        for i, arr in enumerate(args_np):
            f.write(f"\n{'='*80}\n")
            f.write(f"{i}.npy:\n")
            f.write(f"{'='*80}\n\n")
            
            # Basic info
            f.write(f"Shape: {arr.shape}\n")
            f.write(f"Dtype: {arr.dtype}\n")
            f.write(f"Total elements: {arr.size}\n")
            f.write(f"Dimensions: {arr.ndim}\n\n")
            
            # Stats
            f.write("Statistics:\n")
            nan_count = np.count_nonzero(np.isnan(arr))
            inf_count = np.count_nonzero(np.isinf(arr))
            f.write(f"- NaN count: {nan_count}\n")
            f.write(f"- Inf count: {inf_count}\n")
            
            if nan_count == 0 and inf_count == 0:
                f.write(f"- Min: {np.min(arr)}\n")
                f.write(f"- Max: {np.max(arr)}\n")
                f.write(f"- Mean: {np.mean(arr):.6f}\n")
                f.write(f"- Median: {np.median(arr):.6f}\n")
                f.write(f"- Range: {np.ptp(arr)}\n")
                try:
                    mode = pd.Series(arr.flatten()).mode().iloc[0]
                    f.write(f"- Mode: {mode}\n")
                except:
                    f.write("- Mode: Unable to compute\n")
                
                if np.issubdtype(arr.dtype, np.number):
                    try:
                        hist, bins = np.histogram(arr.flatten(), bins='auto')
                        f.write("\nHistogram:\n")
                        f.write("Bins: " + pformat(bins.tolist(), width=80, compact=True) + "\n")
                        f.write("Counts: " + pformat(hist.tolist(), width=80, compact=True) + "\n")
                    except Exception as e:
                        f.write(f"\nHistogram computation failed: {str(e)}\n")
            else:
                f.write("Skipping additional statistics due to NaN/Inf values\n")
            
            f.write("\nArray contents:\n")
            if arr.size <= 64:
                formatted = pformat(arr.tolist(), width=80, compact=True)
                f.write(formatted + "\n")
            else:
                f.write("\nFirst 5 elements:\n")
                f.write(pformat(arr.flatten()[:5].tolist(), width=80, compact=True) + "\n")
                f.write("\nLast 5 elements:\n")
                f.write(pformat(arr.flatten()[-5:].tolist(), width=80, compact=True) + "\n")
                
    dump_id += 1
