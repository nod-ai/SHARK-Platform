import json
import requests
import argparse
import base64

from datetime import datetime as dt
from PIL import Image


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
            width = (
                request["width"][idx]
                if isinstance(request["height"], list)
                else request["height"]
            )
            height = (
                request["height"][idx]
                if isinstance(request["height"], list)
                else request["height"]
            )
            bytes_to_img(item.encode("ascii"), idx, width, height)

    except requests.exceptions.RequestException as e:
        print(f"Error sending the request: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("file", type=str)
    args = p.parse_args()
    send_json_file(args.file)
