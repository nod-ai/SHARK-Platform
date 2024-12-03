# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from datetime import datetime as dt
import os
import sys
import time
import json
import argparse
import base64
import asyncio
import aiohttp
import requests

from PIL import Image

sample_request = {
    "prompt": [
        " A mountain with a halo cloud over it, Death Mountain, spooky, Zelda",
    ],
    "neg_prompt": ["Watermark, blurry, oversaturated, low resolution, pollution"],
    "height": [1024],
    "width": [1024],
    "steps": [50],
    "guidance_scale": [3.5],
    "seed": [0],
    "output_type": ["base64"],
    "rid": ["string"],
}


def bytes_to_img(in_bytes, outputdir, idx=0, width=1024, height=1024):
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    image = Image.frombytes(
        mode="RGB", size=(width, height), data=base64.b64decode(in_bytes)
    )
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    im_path = os.path.join(outputdir, f"shortfin_flux_output_{timestamp}_{idx}.png")
    image.save(im_path)
    print(f"Saved to {im_path}")


def get_batched(request, arg, idx):
    if isinstance(request[arg], list):
        # some args are broadcasted to each prompt, hence overriding idx for single-item entries
        if len(request[arg]) == 1:
            indexed = request[arg][0]
        else:
            indexed = request[arg][idx]
    else:
        indexed = request[arg]
    return indexed


async def send_request(session, rep, args, data):
    print("Sending request batch #", rep)
    url = f"{args.host}:{args.port}/generate"
    start = time.time()
    async with session.post(url, json=data) as response:
        end = time.time()
        # Check if the response was successful
        if response.status == 200:
            response.raise_for_status()  # Raise an error for bad responses
            res_json = await response.json(content_type=None)
            if args.save:
                for idx, item in enumerate(res_json["images"]):
                    width = get_batched(data, "width", idx)
                    height = get_batched(data, "height", idx)
                    print("Saving response as image...")
                    bytes_to_img(
                        item.encode("utf-8"), args.outputdir, idx, width, height
                    )
            latency = end - start
            print("Responses processed.")
            return latency, len(data["prompt"])
        print(f"Error: Received {response.status} from server")
        raise Exception


async def static(args):
    # Create an aiohttp session for sending requests
    async with aiohttp.ClientSession() as session:
        pending = []
        latencies = []
        sample_counts = []
        # Read the JSON file if supplied. Otherwise, get user input.
        try:
            if not args.file:
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
        start = time.time()

        async for i in async_range(args.reps):
            pending.append(asyncio.create_task(send_request(session, i, args, data)))
            await asyncio.sleep(1)  # Wait for 1 second before sending the next request
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.ALL_COMPLETED
            )
            for task in done:
                latency, num_samples = await task
                latencies.append(latency)
                sample_counts.append(num_samples)
        end = time.time()
        if not any(i is None for i in [latencies, sample_counts]):
            total_num_samples = sum(sample_counts)
            sps = str(total_num_samples / (end - start))
            # Until we have better measurements, don't report the throughput that includes saving images.
            if not args.save:
                print(f"Average throughput: {sps} samples per second")
        else:
            raise ValueError("Received error response from server.")


async def interactive(args):
    # Create an aiohttp session for sending requests
    async with aiohttp.ClientSession() as session:
        pending = []
        latencies = []
        sample_counts = []
        # Read the JSON file if supplied. Otherwise, get user input.
        try:
            if not args.file:
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
        while True:
            prompt = await ainput("Enter a prompt: ")
            data["prompt"] = [prompt]
            data["steps"] = [args.steps]
            print("Sending request with prompt: ", data["prompt"])

            async for i in async_range(args.reps):
                pending.append(
                    asyncio.create_task(send_request(session, i, args, data))
                )
                await asyncio.sleep(
                    1
                )  # Wait for 1 second before sending the next request
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.ALL_COMPLETED
                )
                for task in done:
                    _, _ = await task
            pending = []
            if any(i is None for i in [latencies, sample_counts]):
                raise ValueError("Received error response from server.")


async def ainput(prompt: str) -> str:
    return await asyncio.to_thread(input, f"{prompt} ")


async def async_range(count):
    for i in range(count):
        yield i
        await asyncio.sleep(0.0)


def check_health(url):
    ready = False
    print("Waiting for server.", end=None)
    while not ready:
        try:
            if requests.get(f"{url}/health", timeout=20).status_code == 200:
                print("Successfully connected to server.")
                ready = True
                return
            time.sleep(2)
            print(".", end=None)
        except:
            time.sleep(2)
            print(".", end=None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--file",
        type=str,
        default=None,
        help="A non-default request to send to the server.",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Number of times to duplicate each request in one second intervals.",
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
    p.add_argument(
        "--host", type=str, default="http://0.0.0.0", help="Server host address."
    )
    p.add_argument("--port", type=str, default="8000", help="Server port")
    p.add_argument(
        "--steps",
        type=int,
        default="20",
        help="Number of inference steps. More steps usually means a better image. Interactive only.",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Start as an example CLI client instead of sending static requests.",
    )
    args = p.parse_args()
    check_health(f"{args.host}:{args.port}")
    if args.interactive:
        asyncio.run(interactive(args))
    else:
        asyncio.run(static(args))


if __name__ == "__main__":
    main()