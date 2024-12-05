import concurrent.futures
import requests

url = "http://localhost:8003/generate"
payload = {
    "text": "Hello, how are you?",
    "sampling_params": {"max_completion_tokens": 50},
}


def fetch(url, payload):
    return requests.post(url, json=payload)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(fetch, url, payload) for _ in range(2)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result.status_code, result.text)
