import os
import sys
import copy
import json
import base64
import requests
import time
import socket
import subprocess
from contextlib import closing


import gradio as gr
import numpy as np
from PIL import Image

sdxl_request = {
    "prompt": [""],
    "neg_prompt": [""],
    "height": [1024],
    "width": [1024],
    "steps": [20],
    "guidance_scale": [7.5],
    "seed": [0],
    "output_type": ["base64"],
    "rid": ["string"],
}


def find_free_port():
    """This tries to find a free port to run a server on for the demo.

    Race conditions are possible - the port can be acquired between when this
    runs and when the server starts.

    https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def start_server(system):
    # Start the server
    srv_args = [
        "python",
        "-m",
        "shortfin_apps.sd.server",
    ]
    srv_args.extend(
        [
            f"--device={system}",
        ]
    )
    runner = ServerRunner(srv_args)
    # Wait for server to start
    time.sleep(3)
    return runner


class ServerRunner:
    def __init__(self, args):
        port = str(find_free_port())
        self.url = "http://0.0.0.0:" + port
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.process = subprocess.Popen(
            [
                *args,
                "--port=" + port,
                "--device=amdgpu",
                "--device_ids=0",
            ],
            env=env,
            # TODO: Have a more robust way of forking a subprocess.
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
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
                    raise RuntimeError("API server process terminated") from e
            time.sleep(1.0)
            if (time.time() - start) > 180:
                raise RuntimeError("Timeout waiting for server start")

    def __del__(self):
        try:
            process = self.process
        except AttributeError:
            pass
        else:
            process.terminate()
            process.wait()


def bytes_to_img(bytes, width=1024, height=1024):
    image = Image.frombytes(
        mode="RGB", size=(width, height), data=base64.b64decode(bytes)
    )
    return image


class SDXLClient:
    def __init__(self, system):
        self.runner = start_server(system)

    def send_request(self, request):
        imgs = []
        try:
            response = requests.post(self.runner.url + "/generate", json=request)
            response.raise_for_status()  # Raise an error for bad responses
            request = json.loads(response.request.body.decode("utf-8"))

            for idx, item in enumerate(response.json()["images"]):
                img = bytes_to_img(item.encode("utf-8"), 1024, 1024)
                imgs.extend([img])

        except requests.exceptions.RequestException as e:
            print(f"Error sending the request: {e}")

        return imgs

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        num_steps: int,
        guidance: float,
        seed: int,
        batch_size: int,
        batch_seed_increment: int = 20000,
    ):
        request = copy.deepcopy(sdxl_request)
        seed = int(seed)
        request["prompt"] = [prompt] * batch_size
        request["neg_prompt"] = [negative_prompt] * batch_size
        if seed == -1:
            seed_list = [
                np.random.randint(0, high=sys.maxsize, dtype=int)
                for i in range(batch_size)
            ]
        else:
            seed_list = [seed + i * batch_seed_increment for i in range(batch_size)]
        request["seed"] = seed_list
        request["num_steps"] = [num_steps]
        request["guidance_scale"] = [guidance]
        output_images = self.send_request(request)

        return output_images


def create_demo(system):
    generator = SDXLClient(system)

    with gr.Blocks() as demo:
        gr.Markdown("# SDXL Image Generation Demo - MI300x")

        prompt = gr.Textbox(
            label="Prompt",
            value="a single cybernetic shark jumping out of the waves set against a technicolor sunset",
        )
        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            value="",
        )

        with gr.Accordion("Advanced Options", open=False):
            num_steps = gr.Slider(1, 50, 20, step=1, label="Number of steps")
            guidance = gr.Slider(1.0, 10.0, 7.5, step=0.1, label="Guidance")
            seed = gr.Textbox(-1, label="Seed (-1 for random)")
            batch_size = gr.Slider(1, 16, step=1, label="Batch size")
            batch_seed_increment = gr.Slider(
                1, 10000000, step=1, label="Seed increment (for batch_size > 1)"
            )

        generate_btn = gr.Button("Generate")

        output_images = gr.Gallery(
            label="Generated images",
            show_label=False,
            columns=[3],
            rows=[1],
            object_fit="contain",
            preview=True,
            height="auto",
        )

        generate_btn.click(
            fn=generator.generate_image,
            inputs=[
                prompt,
                negative_prompt,
                num_steps,
                guidance,
                seed,
                batch_size,
                batch_seed_increment,
            ],
            outputs=[output_images],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SDXL")

    parser.add_argument("--system", type=str, default="amdgpu", help="Device to use")
    parser.add_argument(
        "--share", action="store_true", help="Create a public link to your demo"
    )
    args = parser.parse_args()

    demo = create_demo(args.system)
    demo.launch(share=args.share)
