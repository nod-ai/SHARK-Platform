# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import contextlib
from pathlib import Path
import os
import shutil
import tempfile
import unittest
import torch
from typing import Any, Callable
from operator import eq
from collections.abc import Iterable
import gc

from ..types import *
from .math import cosine_similarity

# Range of torch.rand() is [0,1)
# Range of torch.rand() * 2 - 1 is [-1, 1), includes negative values
def make_rand_torch(shape: list[int], dtype: Optional[torch.dtype] = torch.float32):
    return torch.rand(shape, dtype=dtype) * 2 - 1


def make_random_mask(shape: tuple[int], dtype: Optional[torch.dtype] = None):
    mask = make_rand_torch(shape=shape, dtype=dtype)
    mask = (mask >= 0).to(dtype=dtype)
    return mask


class TempDirTestBase(unittest.TestCase):
    def setUp(self):
        self._temp_dir = Path(tempfile.mkdtemp(type(self).__qualname__))

    def tearDown(self):
        gc.collect()
        shutil.rmtree(self._temp_dir, ignore_errors=True)


class MainRunnerTestBase(TempDirTestBase):
    """Performs an in-process test of a `main(args)` func."""

    def get_file_path(self, name: str) -> Path:
        return self._temp_dir / name

    def get_irpa_path(self, name: str) -> Path:
        return self.get_file_path(f"{name}.irpa")

    def save_dataset(self, ds: Dataset, name: str) -> Path:
        p = self.get_irpa_path(name)
        ds.save(p)
        return p

    def run_main(self, main_func, *args):
        new_args = [str(arg) for arg in args]
        main_func(new_args)

    def assertFileWritten(self, p: Path):
        self.assertTrue(p.exists(), msg=f"Expected file {p} was not created")
        self.assertGreater(p.stat().st_size, 0, msg=f"Expected file {p} had zero size")


@contextlib.contextmanager
def temporary_directory(identifier: str):
    """Returns a context manager TemporaryDirectory suitable for testing.

    If the env var SHARKTANK_TEST_ASSETS_DIR is set then directories will be
    created under there, named by `identifier`. If the `identifier` subdirectory
    exists, it will be deleted first.

    This is useful for getting updated goldens and such.
    """
    explicit_dir = os.getenv("SHARKTANK_TEST_ASSETS_DIR", None)
    if explicit_dir is None:
        with tempfile.TemporaryDirectory(prefix=f"{identifier}_") as td:
            yield td
    else:
        explicit_path = Path(explicit_dir) / identifier
        if explicit_path.exists():
            shutil.rmtree(explicit_path)
        explicit_path.mkdir(parents=True, exist_ok=True)
        yield explicit_path


@contextlib.contextmanager
def override_debug_flags(flag_updates: dict):
    from .debugging import flags

    restore = {}
    try:
        for k, v in flag_updates.items():
            print(f"Overriding debug flag {k} = {v}")
            current_value = getattr(flags, k)
            restore[k] = current_value
            setattr(flags, k, v)
        yield
    finally:
        for k, v in restore.items():
            print(f"Restoring debug flag {k} = {v}")
            setattr(flags, k, v)


def get_best_torch_device() -> str:
    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda:0"
    return "cpu"


def assert_dicts_equal(
    dict1: dict, dict2: dict, *, values_equal: Callable[[Any, Any], bool] | None = None
) -> None:
    values_equal = values_equal or eq
    assert len(dict1) == len(
        dict2
    ), f"Dictionaries not equal. {dict1} and {dict2} have different number of elements {len(dict1)} != {len(dict2)}"
    for k, v1 in dict1.items():
        assert (
            k in dict2
        ), f"Dictionaries {dict1} and {dict2} not equal. Key {k} not found in {dict2}"
        v2 = dict2[k]
        assert values_equal(
            v1, dict2[k]
        ), f"Dictionaries {dict1} and {dict2} not equal for key {k}. Values {v1} and {v2} not equal"


def assert_equal(
    a: Any, b: Any, *, equal: Callable[[Any, Any], bool] | None = None
) -> None:
    equal = equal or eq
    assert equal(a, b), f"{a} and {b} are not equal"


def assert_golden_safetensors(actual_path, ref_path):
    """Asserts that actual and reference safetensors files are within tolerances."""
    from safetensors import safe_open
    import torch

    print(f"Asserting goldens: actual={actual_path}, ref={ref_path}")

    def print_stats(label, t):
        t = t.to(dtype=torch.float32)
        std, mean = torch.std_mean(t)
        print(
            f"    {label}: "
            f"MIN={torch.min(t)}, "
            f"MAX={torch.max(t)}, "
            f"MEAN={mean}, STD={std}"
        )

    with safe_open(actual_path, framework="pt") as actual_f, safe_open(
        ref_path, framework="pt"
    ) as ref_f:
        # Print all first.
        for name in ref_f.keys():
            actual = actual_f.get_tensor(name)
            ref = ref_f.get_tensor(name)

            print(f":: Comparing tensor {name}")
            print_stats(" REF", ref)
            print_stats(" ACT", actual)
            print_stats("DIFF", (ref - actual))
        # Then assert.
        for name in ref_f.keys():
            actual = actual_f.get_tensor(name)
            ref = ref_f.get_tensor(name)
            torch.testing.assert_close(actual, ref, msg=name)


def assert_iterables_equal(
    iterable1: Iterable,
    iterable2: Iterable,
    *,
    elements_equal: Callable[[Any, Any], bool] | None = None,
) -> None:
    elements_equal = elements_equal or eq
    for i, (v1, v2) in enumerate(zip(iterable1, iterable2, strict=True)):
        assert elements_equal(
            v1, v2
        ), f"Iterables not equal at index {i} for elements {v1} and {v2}"


def assert_text_encoder_state_close(
    actual: torch.Tensor, expected: torch.Tensor, atol: float
):
    """The cosine similarity has been suggested to compare encoder states.

    Dehua Peng, Zhipeng Gui, Huayi Wu -
    Interpreting the Curse of Dimensionality from Distance Concentration and Manifold
    Effect (2023)

    shows that cosine and all Minkowski distances suffer from the curse of
    dimensionality.
    The cosine similarity ignores the vector magnitudes. We can probably come up with a
    better metric, but this is maybe good enough.

    The functions expects that the last dimension is the features per token.
    It will compute the cosine similarity for each token.
    """
    cosine_similarity_per_token = cosine_similarity(
        actual,
        expected,
        dim=-1,
    )
    torch.testing.assert_close(
        cosine_similarity_per_token,
        torch.ones_like(cosine_similarity_per_token),
        atol=atol,
        rtol=0,
    )


SHARKTANK_TEST_SKIP_ENV_VAR = "SHARKTANK_TEST_SKIP"


def skip(*decorator_args, **decorator_kwargs):
    """Decorator to skip a test when SHARKTANK_TEST_SKIP env var is not set or != 0"""

    def decorator(test_item: Callable):
        if SHARKTANK_TEST_SKIP_ENV_VAR not in os.environ:
            should_skip = True
        else:
            should_skip = os.environ[SHARKTANK_TEST_SKIP_ENV_VAR] != "0"

        if should_skip:
            return unittest.skip(*decorator_args, **decorator_kwargs)(test_item)

        return test_item

    return decorator


test_prompts = [
    "Studies have been shown that owning a dog is good for you",
    "The horse went into the river",
    "We need at least one sentence long enough so that it spans more than one padding block which by default is of size 16.",
    "Make the batch size 4",
]
