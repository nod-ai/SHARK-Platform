import json
import requests
import argparse
import base64

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


def send_json_file(file_path):
    # Read the JSON file
    try:
        if file_path == "default":
            data = sample_request
        else:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
    except Exception as e:
        print(f"Error reading the JSON file: {e}")
        return

    # Send the data to the /generate endpoint
    try:
        response = requests.post("http://0.0.0.0:8000/generate", json=data)
        response.raise_for_status()  # Raise an error for bad responses
        print("Saving response as image...")
        timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        request = json.loads(response.request.body.decode("utf-8"))
        for idx, item in enumerate(response.json()["images"]):
            width = get_batched(request, "width", idx)
            height = get_batched(request, "height", idx)
            bytes_to_img(item.encode("utf-8"), idx, width, height)

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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, default="default")
    args = p.parse_args()
    send_json_file(args.file)
