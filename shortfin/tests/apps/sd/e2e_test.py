import json
import requests
import time
import base64
import pytest
import subprocess
import os
import socket
import sys
from contextlib import closing

from datetime import datetime as dt
from PIL import Image

BATCH_SIZES = [1]

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


def sd_artifacts(target: str = "gfx942"):
    return {
        "model_config": "sdxl_config_i8.json",
        "clip_vmfb": f"stable_diffusion_xl_base_1_0_bs1_64_fp16_text_encoder_{target}.vmfb",
        "scheduler_vmfb": f"stable_diffusion_xl_base_1_0_EulerDiscreteScheduler_bs1_1024x1024_fp16_20_{target}.vmfb",
        "unet_vmfb": f"stable_diffusion_xl_base_1_0_bs1_64_1024x1024_i8_punet_{target}.vmfb",
        "vae_vmfb": f"stable_diffusion_xl_base_1_0_bs1_1024x1024_fp16_vae_{target}.vmfb",
        "clip_params": "clip_splat_fp16.irpa",
        "unet_params": "punet_splat_i8.irpa",
        "vae_params": "vae_splat_fp16.irpa",
    }


cache = os.path.abspath("./tmp/sharktank/sd/")


@pytest.fixture(scope="module")
def sd_server():
    # Create necessary directories

    os.makedirs(cache, exist_ok=True)

    # Download model if it doesn't exist
    vmfbs_bucket = "https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl/vmfbs/"
    weights_bucket = (
        "https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl/weights/"
    )
    configs_bucket = (
        "https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl/configs/"
    )
    for artifact, path in sd_artifacts().items():
        if "vmfb" in artifact:
            bucket = vmfbs_bucket
        elif "params" in artifact:
            bucket = weights_bucket
        else:
            bucket = configs_bucket
        address = bucket + path
        local_file = os.path.join(cache, path)
        if not os.path.exists(local_file):
            print("Downloading artifact from " + address)
            r = requests.get(address, allow_redirects=True)
            with open(local_file, "wb") as lf:
                lf.write(r.content)
    # Start the server
    srv_args = [
        "python",
        "-m",
        "shortfin_apps.sd.server",
    ]
    for arg in sd_artifacts().keys():
        artifact_arg = f"--{arg}={cache}/{sd_artifacts()[arg]}"
        srv_args.extend([artifact_arg])
    runner = ServerRunner(srv_args)
    # Wait for server to start
    time.sleep(5)

    yield runner

    # Teardown: kill the server
    del runner


@pytest.mark.system("amdgpu")
def test_sd_server(sd_server):
    imgs, status_code = send_json_file(sd_server.url)
    assert len(imgs) == 1
    assert status_code == 200


class ServerRunner:
    def __init__(self, args):
        port = str(find_free_port())
        self.url = "http://0.0.0.0:" + port
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["HIP_VISIBLE_DEVICES"] = "0"
        self.process = subprocess.Popen(
            [
                *args,
                "--port=" + port,
                "--device=amdgpu",
            ],
            env=env,
            # TODO: Have a more robust way of forking a subprocess.
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print(self.process.args)
        self._wait_for_ready()

    def _wait_for_ready(self):
        start = time.time()
        while True:
            time.sleep(2)
            try:
                if requests.get(f"{self.url}/health").status_code == 200:
                    return
            except Exception as e:
                if self.process.errors is not None:
                    raise RuntimeError("API server processs terminated") from e
            time.sleep(1.0)
            if (time.time() - start) > 30:
                raise RuntimeError("Timeout waiting for server start")

    def __del__(self):
        try:
            process = self.process
        except AttributeError:
            pass
        else:
            process.terminate()
            process.wait()


def bytes_to_img(bytes, idx=0, width=1024, height=1024):
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    image = Image.frombytes(
        mode="RGB", size=(width, height), data=base64.b64decode(bytes)
    )
    img_fn = f"shortfin_sd_test_output_{timestamp}_{idx}.png"
    image.save(img_fn)
    return img_fn


def send_json_file(url="http://0.0.0.0:8000"):
    # Read the JSON file
    data = sample_request
    imgs = []
    # Send the data to the /generate endpoint
    try:
        response = requests.post(url + "/generate", json=data)
        response.raise_for_status()  # Raise an error for bad responses
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
            img_fn = bytes_to_img(item.encode("utf-8"), idx, width, height)
            imgs.append(img_fn)

    except requests.exceptions.RequestException as e:
        print(f"Error sending the request: {e}")

    return imgs, response.status_code


def find_free_port():
    """This tries to find a free port to run a server on for the test.

    Race conditions are possible - the port can be acquired between when this
    runs and when the server starts.

    https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
