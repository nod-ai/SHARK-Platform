import requests
import json
import uuid

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check status code: {response.status_code}")
    return response.status_code == 200

def test_generate():
    headers = {"Content-Type": "application/json"}
    
    # Create a GenerateReqInput-like structure
    data = {
        "text": "The meaning of life is",
        "sampling_params": {
            "max_tokens": 50,
            "temperature": 0.7
        },
        "rid": uuid.uuid4().hex,
        "return_logprob": False,
        "logprob_start_len": -1,
        "top_logprobs_num": 0,
        "return_text_in_logprobs": False,
        "stream": False
    }

    print("Prompt text:")
    print(data['text'])
    
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
    print("Testing webapp...")
    
    health_ok = test_health()
    generate_ok = test_generate()
    
    if health_ok and generate_ok:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Please check the output above for details.")

if __name__ == "__main__":
    main()
