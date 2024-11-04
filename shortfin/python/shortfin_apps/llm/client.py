import requests
import json
import uuid
import argparse
import time

BASE_URL = "http://localhost:8000"


def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check status code: {response.status_code}")
    response.raise_for_status()


def test_generate(prompt_text):
    headers = {"Content-Type": "application/json"}

    # Create a GenerateReqInput-like structure
    data = {
        "text": prompt_text,
        "sampling_params": {"max_tokens": 50, "temperature": 0.7},
        "rid": uuid.uuid4().hex,
        "return_logprob": False,
        "logprob_start_len": -1,
        "top_logprobs_num": 0,
        "return_text_in_logprobs": False,
        "stream": False,
    }

    print("Prompt text:")
    print(data["text"])

    response = requests.post(f"{BASE_URL}/generate", headers=headers, json=data)
    print(f"Generate endpoint status code: {response.status_code}")

    if response.status_code == 200:
        print("Generated text:")
        data = response.text
        assert data.startswith("data: ")
        data = data[6:]
        assert data.endswith("\n\n")
        data = data[:-2]
        print(data)
    else:
        print("Failed to generate text")
        print("Response content:")
        print(response.text)

    return response.status_code == 200


def main():
    parser = argparse.ArgumentParser(description="Test webapp with custom prompt")
    parser.add_argument(
        "--prompt",
        type=str,
        default="1,2,3,4,5,",
        help="The prompt text to send to the generate endpoint",
    )

    args = parser.parse_args()

    print(f"Testing shortfin llm server at {BASE_URL}")

    health_ok = False
    # previous backoff for fibonacci backoff
    prev_backoff = 0
    backoff = 1
    while not health_ok:
        try:
            health_ok = test_health()
        except requests.exceptions.ConnectionError:
            print(
                f"Health check failed. Waiting for {backoff} seconds before retrying."
            )
            time.sleep(backoff)
            prev_backoff, backoff = backoff, prev_backoff + backoff

    generate_ok = test_generate(args.prompt)

    if health_ok and generate_ok:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Please check the output above for details.")


if __name__ == "__main__":
    main()
