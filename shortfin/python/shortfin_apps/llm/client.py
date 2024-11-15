# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import requests
import json
import uuid
import argparse
import time
from typing import Dict, Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LLM server")
    parser.add_argument("--text", default="1 2 3 4 5 ", help="Input text prompt")
    parser.add_argument(
        "--max_completion_tokens", type=int, default=50, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--stream", action="store_true", help="Enable response streaming"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="8000",
        help="Port that shortfin server is running on",
    )
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    data = {
        "text": args.text,
        "sampling_params": {
            "max_completion_tokens": args.max_completion_tokens,
            "temperature": args.temperature,
        },
        "rid": uuid.uuid4().hex,
        "return_logprob": False,
        "logprob_start_len": -1,
        "top_logprobs_num": 0,
        "return_text_in_logprobs": False,
        "stream": args.stream,
    }

    print(f"Testing LLM server at {base_url}")

    # Health check with exponential backoff
    backoff = 1
    while True:
        try:
            requests.get(f"{base_url}/health").raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            if backoff > 16:
                print("Health check failed, max retries exceeded")
                return
            print(f"Health check failed ({str(e)}), retrying in {backoff}s...")
            time.sleep(backoff)
            backoff *= 2

    # Generate request
    try:
        print("Prompt text:", data["text"])
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{base_url}/generate", headers=headers, json=data)
        response.raise_for_status()

        if response.text.startswith("data: "):
            text = response.text[6:].rstrip("\n")
            print("Generated text:", text)
            print("\nTest passed")
        else:
            print("\nTest failed: unexpected response format")

    except requests.exceptions.RequestException as e:
        print(f"\nTest failed: request error: {str(e)}")
    except KeyboardInterrupt:
        print("\nTest interrupted")


if __name__ == "__main__":
    main()
