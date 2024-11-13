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


def bytes_to_img(bytes, idx=0, width=1024, height=1024):
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    image = Image.frombytes(
        mode="RGB", size=(width, height), data=base64.b64decode(bytes)
    )
    image.save(f"shortfin_sd_output_{timestamp}_{idx}.png")
    print(f"Saved to shortfin_sd_output_{timestamp}_{idx}.png")


async def send_json_file(args):
    async for rep in range(args.reps):
        # Send the data to the /generate endpoint
        try:
            time.sleep(1)
            start = time.time()
            print("Sending request batch #", rep)
            response = requests.post("http://0.0.0.0:8000/generate", json=data)
            response.raise_for_status()  # Raise an error for bad responses
            timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
            request = json.loads(response.request.body.decode("utf-8"))
            end = time.time()
            for idx, item in enumerate(response.json()["images"]):
                width = get_batched(request, "width", idx)
                height = get_batched(request, "height", idx)
                if args.save:
                    print("Saving response as image...")
                    bytes_to_img(item.encode("utf-8"), idx, width, height)
            latency = end - start
            print(f"Latency (" + str(len(data["prompt"])) + "samples) :")
            print(latency)
            print("Responses processed.")

        except requests.exceptions.RequestException as e:
            print(f"Error sending the request: {e}")


def get_batched(request, arg, idx):
    if isinstance(request[arg], list):
        if len(request[arg]) == 1:
            indexed = request[arg][0]
        else:
            indexed = request[arg][idx]
    else:
        indexed = request[arg]
    return indexed


async def send_request(session, rep, args, data):
    try:
        print("Sending request batch #", rep)
        url = f"http://0.0.0.0:{args.port}/generate"
        start = time.time()
        async with session.post(url, json=data) as response:
            end = time.time()
            # Check if the response was successful
            if response.status == 200:
                response.raise_for_status()  # Raise an error for bad responses
                timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
                res_json = await response.json(content_type=None)
                if args.save:
                    for idx, item in enumerate(res_json["images"]):
                        width = get_batched(data, "width", idx)
                        height = get_batched(data, "height", idx)
                        print("Saving response as image...")
                        bytes_to_img(item.encode("utf-8"), idx, width, height)
                latency = end - start
                print(f"Latency (" + str(len(data["prompt"])) + "samples) :")
                print(latency)
                print("Responses processed.")
                return latency, len(data["prompt"])
            else:
                print(f"Error: Received {response.status} from server")
                raise Exception
    except Exception as e:
        print(f"Request failed: {e}")
        raise Exception


async def main(args):
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
        if not any([i is None for i in [latencies, sample_counts]]):
            total_num_samples = sum(sample_counts)
            sps = str(total_num_samples / (end - start))
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
        print("Sending request with prompt: ", data["prompt"])
        while True:
            prompt = await ainput("Enter a prompt: ")
            data["prompt"] = [prompt]
            data["steps"] = [args.steps]
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
                    latency, num_samples = await task
            pending = []
            if any([i is None for i in [latencies, sample_counts]]):
                raise ValueError("Received error response from server.")


async def ainput(prompt: str) -> str:
    return await asyncio.to_thread(input, f"{prompt} ")


async def async_range(count):
    for i in range(count):
        yield (i)
        await asyncio.sleep(0.0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, default="default")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--save", action="store_true", help="save images")
    p.add_argument("--port", type=str, default="8000")
    p.add_argument("--steps", type=int, default="20")
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Start as an example client instead of sending static requests.",
    )
    args = p.parse_args()
    if args.interactive:
        asyncio.run(interactive(args))
    else:
        asyncio.run(main(args))
