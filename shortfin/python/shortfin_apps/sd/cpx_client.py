# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import requests
import argparse
import base64
import time
import asyncio
import aiohttp
import sys
import os

from datetime import datetime as dt
from PIL import Image

sample_request = {
    "prompt": [
        " a cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo, animal",
    ],
    "neg_prompt": ["Watermark, blurry, oversaturated, low resolution, pollution"],
    "height": [1024],
    "width": [1024],
    "steps": [20],
    "guidance_scale": [7.5],
    "seed": [0],
    "output_type": ["base64"],
    "rid": ["string"],
}


def bytes_to_img(bytes, idx=0, width=1024, height=1024, outputdir="./gen_imgs"):
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    image = Image.frombytes(
        mode="RGB", size=(width, height), data=base64.b64decode(bytes)
    )
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    im_path = os.path.join(outputdir, f"shortfin_sd_output_{timestamp}_{idx}.png")
    image.save(im_path)
    print(f"Saved to {im_path}")


def get_batched(request, arg, idx):
    if isinstance(request[arg], list):
        if len(request[arg]) == 1:
            indexed = request[arg][0]
        else:
            indexed = request[arg][idx]
    else:
        indexed = request[arg]
    return indexed


async def send_request(session, rep, args, data, warmup):
    try:
        print(f"Sending request batch #{rep} at {time.time()}")
        url = f"http://0.0.0.0:{args.port}/generate"
        loop = asyncio.get_running_loop()
        start = loop.time()
        async with session.post(url, json=data) as response:
            end = loop.time()
            # Check if the response was successful
            if response.status == 200:
                response.raise_for_status()  # Raise an error for bad responses
                timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
                res_json = await response.json(content_type=None)
                if args.save and not warmup:
                    for idx, item in enumerate(res_json["images"]):
                        width = get_batched(data, "width", idx)
                        height = get_batched(data, "height", idx)
                        print("Saving response as image...")
                        bytes_to_img(
                            item.encode("utf-8"), idx, width, height, args.outputdir
                        )
                latency = end - start
                print("Responses processed.")
                return latency, len(data["prompt"])
            else:
                print(f"Error: Received {response.status} from server")
                raise Exception
    except Exception as e:
        print(f"Request failed: {e}")
        raise Exception


async def static(args, warmup=False):
    # Create an aiohttp session for sending requests
    async with aiohttp.ClientSession() as session:
        pending = []
        latencies = []
        sample_counts = []
        # Read the JSON file if supplied. Otherwise, get user input.
        try:
            if args.file == "default":
                data = sample_request
            else:
                with open(args.file, "r") as json_file:
                    data = json.load(json_file)
        except Exception as e:
            print(f"Error reading the JSON file: {e}")
            return
        data["prompt"] = (
            [data["prompt"]] if isinstance(data["prompt"], str) else data["prompt"]
        )

        # Schedule concurrent requests without delays
        for i in range(args.concurrent):
            pending.append(asyncio.create_task(send_request(session, i, args, data, warmup)))
        
        # Wait for all tasks to complete
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.ALL_COMPLETED
            )
            for task in done:
                latency, num_samples = await task
                latencies.append(latency)
                sample_counts.append(num_samples)

        if not any([i is None for i in [latencies, sample_counts]]):
            if not warmup:
                print(f"Latencies: {latencies}")
                print(f"Total throughput: {args.concurrent/max(latencies)} samples per second")
        else:
            raise ValueError("Received error response from server.")


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--file",
        type=str,
        default="default",
        help="A non-default request to send to the server.",
    )
    p.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save images. To disable, use --no-save",
    )
    p.add_argument(
        "--outputdir",
        type=str,
        default="gen_imgs",
        help="Directory to which images get saved.",
    )
    p.add_argument("--port", type=str, default="8000", help="Server port")
    p.add_argument(
        "--concurrent",
        type=int,
        default=64,
        help="Number of concurrent requests to send.",
    )
    args = p.parse_args()
    # warmup
    asyncio.run(static(args, warmup=True))
    asyncio.run(static(args, warmup=True))
    asyncio.run(static(args, warmup=True))
    # real deal
    asyncio.run(static(args))


if __name__ == "__main__":
    main(sys.argv)
