# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import asyncio
from contextlib import asynccontextmanager
import threading
import sys

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import shortfin as sf
import uvicorn


class FastAPIResponder(sf.Message):
    """Bridge between FastAPI and shortfin that can be put on a queue and used to
    send a response back at an arbitrary point.

    This object is constructed in a FastAPI handler, capturing the current event loop
    used by the web server. Then it can be put on a shortfin Queue and once within
    a shortfin worker, an arbitrary worker can call `send_response` to send a simple
    FastAPI response back to the webserver loop and onto the client.

    """

    def __init__(self, request: Request):
        super().__init__()
        self.request = request
        # Capture the running loop so that we can send responses back.
        self._loop = asyncio.get_running_loop()
        self.response = asyncio.Future(loop=self._loop)
        self._responded = False
        self._streaming_queue: asyncio.Queue | None = None
        self.is_disconnected = False

    def send_response(self, response: Response):
        """Sends a response back for this transaction.

        This is intended for sending single part responses back. See
        start_response() for sending back a streaming, multi-part response.
        """
        assert not self._responded, "Response already sent"
        if self._loop.is_closed():
            raise IOError("Web server is shut down")
        self._responded = True
        self._loop.call_soon_threadsafe(self.response.set_result, response)

    def start_response(self, **kwargs):
        """Starts a streaming response, passing the given kwargs to the
        fastapi.responses.StreamingResponse constructor.

        This is appropriate to use for generating a sparse response stream as is
        typical of chat apps. As it will hop threads for each part, other means should
        be used for bulk transfer (i.e. by scheduling on the webserver loop
        directly).
        """
        assert not self._responded, "Response already sent"
        if self._loop.is_closed():
            raise IOError("Web server is shut down")
        self._responded = True
        self._streaming_queue = asyncio.Queue()

        async def gen():
            while True:
                if await self.request.is_disconnected():
                    self.is_disconnected = True
                part = await self._streaming_queue.get()
                if part is None:
                    break
                yield part

        def start():
            response = StreamingResponse(gen(), **kwargs)
            self.response.set_result(response)

        self._loop.call_soon_threadsafe(start)

    def stream_part(self, content: bytes | None):
        """Streams content to a response started with start_response().

        Streaming must be ended by sending None.
        """
        assert self._streaming_queue is not None, "start_response() not called"
        if self._loop.is_closed():
            raise IOError("Web server is shut down")
        self._loop.call_soon_threadsafe(self._streaming_queue.put_nowait, content)


class System:
    def __init__(self):
        self.ls = sf.host.CPUSystemBuilder().create_system()
        # TODO: Come up with an easier bootstrap thing than manually
        # running a thread.
        self.t = threading.Thread(target=lambda: self.ls.run(self.run()))
        self.request_queue = self.ls.create_queue("request")
        self.request_writer = self.request_queue.writer()

    def start(self):
        self.t.start()

    def shutdown(self):
        self.request_queue.close()

    async def run(self):
        print("*** Sytem Running ***")
        request_reader = self.request_queue.reader()
        while responder := await request_reader():
            print("Got request:", responder)
            # Can send a single response:
            #   request.send_response(JSONResponse({"answer": 42}))
            # Or stream:
            responder.start_response()
            for i in range(20):
                if responder.is_disconnected:
                    print("Cancelled!")
                    break
                responder.stream_part(f"Iteration {i}\n".encode())
                await asyncio.sleep(0.2)
            else:
                responder.stream_part(None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    system.start()
    yield
    print("Shutting down shortfin")
    system.shutdown()


system = System()
app = FastAPI(lifespan=lifespan)


@app.get("/predict")
async def predict(request: Request):
    transaction = FastAPIResponder(request)
    system.request_writer(transaction)
    return await transaction.response


def main(argv):
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
        "--testing-mock-service",
        action="store_true",
        help="Enable the mock testing service",
    )
    parser.add_argument(
        "--device-uri", type=str, default="local-task", help="Device URI to serve on"
    )

    args = parser.parse_args(argv)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=args.timeout_keep_alive,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
