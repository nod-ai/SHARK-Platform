import requests
import json
import uuid
import argparse
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LLM server")
    parser.add_argument("--text", default="1 2 3 4 5 ", help="Input text prompt")
    parser.add_argument(
        "--max_tokens", type=int, default=50, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--stream", action="store_true", help="Enable response streaming"
    )
    args = parser.parse_args()

    data = {
        "text": args.text,
        "sampling_params": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "rid": uuid.uuid4().hex,
        "return_logprob": False,
        "logprob_start_len": -1,
        "top_logprobs_num": 0,
        "return_text_in_logprobs": False,
        "stream": args.stream,
    }

    print(f"Testing LLM server at {BASE_URL}")

    # Health check with exponential backoff
    backoff = 1
    while True:
        try:
            requests.get(f"{BASE_URL}/health").raise_for_status()
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
        response = requests.post(f"{BASE_URL}/generate", headers=headers, json=data)
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
