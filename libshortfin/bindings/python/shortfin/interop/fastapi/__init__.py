# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio

try:
    from fastapi import Request, Response
    from fastapi.responses import StreamingResponse
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Shortfin fastapi interop requires fastapi to be installed"
    ) from e


class FastAPIResponder:
    """Bridge between FastAPI and shortfin that can be used to send out of band
    responses back to a waiting FastAPI async request.

    This isn't really shortfin specific and can be used to bridge to any non
    webserver owned loop.

    It is typically used by putting it in a Message that is sent to some processing
    queue. Then return/awaiting it from an API callback. Example:

    ```
    @app.get("/predict")
    async def predict(value: int, request: Request):
        message = RequestMessage(value, FastAPIResponder(request))
        system.request_writer(message)
        return await message.responder.response
    ```

    See: examples/python/fastapi/server.py
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

    def close_with_error(self):
        # Called in a failsafe fashion as part of exception handlers seeking to
        # shutdown the response. If not yet responded, this will response with
        # a status code of 500. If streaming, then None will be streamed.
        if self._responded:
            if self._streaming_queue:
                self.stream_part(None)
        else:
            self.send_response(Response(status_code=500))

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

        async def gen(request, streaming_queue):
            while True:
                if await request.is_disconnected():
                    self.is_disconnected = True
                part = await streaming_queue.get()
                if part is None:
                    break
                yield part

        def start(request, streaming_queue, response_future):
            response = StreamingResponse(gen(request, streaming_queue), **kwargs)
            response_future.set_result(response)

        self._loop.call_soon_threadsafe(
            start, self.request, self._streaming_queue, self.response
        )

    def stream_part(self, content: bytes | None):
        """Streams content to a response started with start_response().

        Streaming must be ended by sending None.
        """
        assert self._streaming_queue is not None, "start_response() not called"
        if self._loop.is_closed():
            raise IOError("Web server is shut down")
        self._loop.call_soon_threadsafe(self._streaming_queue.put_nowait, content)
        if content is None:
            self._streaming_queue = None
