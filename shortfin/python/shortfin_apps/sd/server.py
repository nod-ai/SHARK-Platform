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


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    sysman.start()
    try:
        for service_name, service in services.items():
            logging.info("Initializing service '%s': %r", service_name, service)
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
    service = services["default"]
    gen_req.post_init()
    responder = FastAPIResponder(request)
    ClientGenerateBatchProcess(service, gen_req, responder).launch()
    return await responder.response


app.post("/generate")(generate_request)
app.put("/generate")(generate_request)


def configure(args) -> SystemManager:
    # Setup system (configure devices, etc).
    sysman = SystemManager()

    # Setup each service we are hosting.
    tokenizer = Tokenizer.from_tokenizer_json_file(args.tokenizer_json)
    model_params = ModelParams.load_json(args.model_config)
    sm = GenerateService(
        name="default", sysman=sysman, tokenizer=tokenizer, model_params=model_params
    )
    sm.load_inference_module(args.clip_vmfb)
    sm.load_inference_module(args.unet_vmfb)
    sm.load_inference_module(args.vae_vmfb)
    sm.load_inference_parameters(*args.parameters, parameter_scope="model")
    services[sm.name] = sm
    return sysman


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
        "--tokenizer_json",
        type=Path,
        required=True,
        help="Path to a tokenizer config file",
    )
    parser.add_argument(
        "--model_config",
        type=Path,
        required=True,
        help="Path to the model config file",
    )
    parser.add_argument(
        "--clip_vmfb",
        type=Path,
        required=True,
        help="Model VMFB to load",
    )
    parser.add_argument(
        "--unet_vmfb",
        type=Path,
        required=True,
        help="Model VMFB to load",
    )
    parser.add_argument(
        "--vae_vmfb",
        type=Path,
        required=True,
        help="Model VMFB to load",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        nargs="*",
        help="Parameter archives to load",
    )
    args = parser.parse_args(argv)
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
    from shortfin.support.logging_setup import configure_main_logger

    logger = configure_main_logger("server")
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
