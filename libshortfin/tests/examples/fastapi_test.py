# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import closing
import os
from pathlib import Path
import pytest
import requests
import socket
import subprocess
import sys
import time

project_dir = Path(__file__).resolve().parent.parent.parent
example_dir = project_dir / "examples" / "python"


@pytest.fixture(scope="session")
def server():
    runner = ServerRunner([])
    yield runner
    print("Sending kill signal")
    runner.process.terminate()
    print("Waiting for server to exit")
    runner.process.wait(20)


# Test error first to make sure it doesn't mess up the server.
def test_error_response(server):
    resp = requests.get(f"{server.url}/predict?value=0")
    assert resp.status_code == 500


def test_single_response(server):
    resp = requests.get(f"{server.url}/predict?value=1")
    resp.raise_for_status()
    full_contents = resp.content
    print(full_contents)
    assert full_contents == b'{"answer":1}'


def test_stream_response(server):
    resp = requests.get(f"{server.url}/predict?value=20")
    resp.raise_for_status()
    full_contents = resp.content
    print(full_contents)
    exp_contents = ("".join(['{"answer": %s}\n\x00' % i for i in range(21)])).encode()
    assert full_contents == exp_contents


class ServerRunner:
    def __init__(self, args):
        port = str(find_free_port())
        self.url = "http://localhost:" + port
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(example_dir / "fastapi" / "server.py"),
                "--port=" + port,
            ]
            + args,
            env=env,
            # TODO: Have a more robust way of forking a subprocess.
            cwd=str(example_dir),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._wait_for_ready()

    def _wait_for_ready(self):
        start = time.time()
        while True:
            try:
                if requests.get(f"{self.url}/health").status_code == 200:
                    return
            except Exception as e:
                if self.process.poll() is not None:
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
