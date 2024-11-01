# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any

import argparse
import logging
from pathlib import Path
import sys
import os
import io

from iree.build import *

import uvicorn.logging

# Import first as it does dep checking and reporting.
from shortfin.interop.fastapi import FastAPIResponder

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
import uvicorn


from .components.generate import ClientGenerateBatchProcess
from .components.config_struct import ModelParams
from .components.io_struct import GenerateReqInput
from .components.manager import SystemManager
from .components.service import GenerateService
from .components.tokenizer import Tokenizer
from .components.builders import sdxl


from shortfin.support.logging_setup import configure_main_logger

logger = configure_main_logger("server")

THIS_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    sysman.start()
    try:
        for service_name, service in services.items():
            logging.info("Initializing service '%s':", service_name)
            logging.info(str(service))
            service.start()
    except:
        sysman.shutdown()
        raise
    yield
    try:
        for service_name, service in services.items():
            logging.info("Shutting down service '%s'", service_name)
            service.shutdown()
    finally:
        sysman.shutdown()


sysman: SystemManager
services: dict[str, Any] = {}
app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


async def generate_request(gen_req: GenerateReqInput, request: Request):
    service = services["sd"]
    gen_req.post_init()
    responder = FastAPIResponder(request)
    ClientGenerateBatchProcess(service, gen_req, responder).launch()
    return await responder.response


app.post("/generate")(generate_request)
app.put("/generate")(generate_request)


def configure(args) -> SystemManager:
    # Setup system (configure devices, etc).
    sysman = SystemManager(args.device, args.device_ids)

    # Setup each service we are hosting.
    tokenizers = []
    for idx, tok_name in enumerate(args.tokenizers):
        subfolder = f"tokenizer_{idx + 1}" if idx > 0 else "tokenizer"
        tokenizers.append(Tokenizer.from_pretrained(tok_name, subfolder))

    model_params = ModelParams.load_json(args.model_config)
    sm = GenerateService(
        name="sd",
        sysman=sysman,
        tokenizers=tokenizers,
        model_params=model_params,
        fibers_per_device=args.fibers_per_device,
        prog_isolation=args.isolation,
        show_progress=args.show_progress,
        trace_execution=args.trace_execution,
    )
    vmfbs, params = get_modules(args)
    for key, vmfblist in vmfbs.items():
        for vmfb in vmfblist:
            sm.load_inference_module(vmfb, component=key)
    for key, datasets in params.items():
        sm.load_inference_parameters(*datasets, parameter_scope="model", component=key)
    services[sm.name] = sm
    return sysman


def get_modules(args):
    vmfbs = {"clip": [], "unet": [], "vae": [], "scheduler": []}
    params = {"clip": [], "unet": [], "vae": []}
    mod = load_build_module(os.path.join(THIS_DIR, "components", "builders.py"))
    out_file = io.StringIO()
    iree_build_main(
        mod,
        args=[
            f"--model_json={args.model_config}",
            f"--target={args.target}",
            f"--splat={args.splat}",
        ],
        stdout=out_file,
    )
    filenames = out_file.getvalue().strip().split("\n")
    for name in filenames:
        for key in vmfbs.keys():
            if key in name.lower():
                if any([x in name for x in [".irpa", ".safetensors", ".gguf"]]):
                    params[key].extend([name])
                elif "vmfb" in name:
                    vmfbs[key].extend([name])
    return vmfbs, params


def main(argv, log_config=uvicorn.config.LOGGING_CONFIG):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="Root path to use for installing behind path based proxy.",
    )
    parser.add_argument(
        "--timeout-keep-alive", type=int, default=5, help="Keep alive timeout"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["local-task", "hip", "amdgpu"],
        help="Primary inferencing device",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=False,
        default="gfx942",
        help="Primary inferencing device LLVM target arch.",
    )
    parser.add_argument(
        "--device_ids",
        type=int,
        nargs="*",
        default=None,
        help="Device IDs visible to the system builder. Defaults to None (full visibility). Can be an index or a sf device id like amdgpu:0:0@0",
    )
    parser.add_argument(
        "--tokenizers",
        type=Path,
        nargs="*",
        default=[
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ],
        help="HF repo from which to load tokenizer(s).",
    )
    parser.add_argument(
        "--model_config",
        type=Path,
        required=True,
        help="Path to the model config file",
    )
    parser.add_argument(
        "--fibers_per_device",
        type=int,
        default=1,
        help="Concurrency control -- how many fibers are created per device to run inference.",
    )
    parser.add_argument(
        "--isolation",
        type=str,
        default="per_fiber",
        choices=["per_fiber", "per_call", "none"],
        help="Concurrency control -- How to isolate programs.",
    )
    parser.add_argument(
        "--log_level", type=str, default="error", choices=["info", "debug", "error"]
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="enable tqdm progress for unet iterations.",
    )
    parser.add_argument(
        "--trace_execution",
        action="store_true",
        help="Enable tracing of program modules.",
    )
    parser.add_argument(
        "--splat",
        action="store_true",
        help="Use splat (empty) parameter files, usually for testing.",
    )
    log_levels = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "error": logging.ERROR,
    }

    args = parser.parse_args(argv)

    log_level = log_levels[args.log_level]
    logger.setLevel(log_level)
    logging.root.addHandler(logging.FileHandler("shortfin_sd.log"))
    global sysman
    sysman = configure(args)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=log_config,
        timeout_keep_alive=args.timeout_keep_alive,
    )


if __name__ == "__main__":
    main(
        sys.argv[1:],
        # Make logging defer to the default shortfin logging config.
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {},
            "handlers": {},
            "loggers": {},
        },
    )
