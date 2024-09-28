# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import urllib.request

from shortfin.build.executor import BuildAction, BuildContext, BuildFile

__all__ = [
    "fetch_http",
]


def fetch_http(*, name: str, url: str) -> BuildFile:
    context = BuildContext.current()
    output_file = context.allocate_file(name)
    action = FetchHttpAction(
        url=url, output_file=output_file, desc=f"Fetch {url}", executor=context.executor
    )
    output_file.deps.add(action)
    return output_file


class FetchHttpAction(BuildAction):
    def __init__(self, url: str, output_file: BuildFile, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        self.output_file = output_file

    def invoke(self):
        path = self.output_file.get_fs_path()
        self.executor.write_status(f"Fetching URL: {self.url} -> {path}")
        urllib.request.urlretrieve(self.url, str(path))
