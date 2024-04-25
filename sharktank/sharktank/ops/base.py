# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Sequence

import logging
from pathlib import Path
import textwrap

from jinja2 import Environment, PackageLoader, select_autoescape

from shark_turbine.support.ir_imports import (
    util_d,
    FlatSymbolRefAttr,
    FunctionType,
    MLIRError,
    Operation,
    RankedTensorType,
    StringAttr,
    TypeAttr,
    Value,
)

from shark_turbine.runtime.op_reg import (
    def_library,
    CustomOp,
    KernelBuilder,
    KernelSelection,
)

from shark_turbine.transforms.merger import Merger

from ..utils.logging import get_logger

LIBRARY = def_library("sharktank")
TEMPLATES_DIR = Path(__file__).parent / "templates"
logger = get_logger("sharktank.ops")

_JINJA2_ENVIRONMENT: Optional[Environment] = None


def _get_jinja2_env() -> Environment:
    global _JINJA2_ENVIRONMENT
    if _JINJA2_ENVIRONMENT is None:
        _JINJA2_ENVIRONMENT = Environment(loader=PackageLoader(__name__, "templates"))
    return _JINJA2_ENVIRONMENT


def call_function(target_function: Operation, *operands: Value) -> Sequence[Value]:
    target_symbol = FlatSymbolRefAttr.get(
        StringAttr(target_function.attributes["sym_name"]).value_bytes
    )
    ftype = FunctionType(TypeAttr(target_function.attributes["function_type"]).value)
    return Operation.create(
        "util.call",
        results=ftype.results,
        operands=operands,
        attributes={
            "callee": target_symbol,
        },
    ).results


def inline_template_function(
    kb: KernelBuilder,
    template_file: str,
    function_name: str,
    template_type: str = "format",
    **kwargs,
) -> Operation:
    """Inlines a template module by first expanding its ASM via **kwargs.

    Returns the inlined symbol `function_name`, which is expected to have been
    in the template.
    """
    try:
        return kb.symbol_table[function_name]
    except KeyError:
        pass
    if template_type == "format":
        source_module_op = load_mlir_template(kb, template_file, **kwargs)
    elif template_type == "jinja":
        source_module_op = load_jinja_template(kb, template_file, **kwargs)
    else:
        raise ValueError(f"Unsupported template_type: {template_type}")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Generated kernel IR %s:\n%s", function_name, str(source_module_op)
        )
    merger = Merger(
        source_module_op, kb.module_body.owner, target_symbol_table=kb.symbol_table
    )
    merger.merge()
    return kb.symbol_table[function_name]


def load_jinja_template(kb: KernelBuilder, template_file: str, **kwargs) -> Operation:
    """Loads an MLIR jinja-based template by name.

    The file is loaded relative to the templates/ directory. It is interpolated
    with **kwargs and loaded into the KernelBuilder.
    """
    # template_path = TEMPLATES_DIR / template_file
    # template_text = template_path.read_text()
    asm = _get_jinja2_env().get_template(template_file).render(**kwargs)
    try:
        module_op = Operation.parse(asm, context=kb.context)
    except MLIRError as e:
        raise RuntimeError(
            f"Error parsing generated op template:"
            f"\n{textwrap.indent(str(e), '  ')}"
            f"\n{textwrap.indent(asm, '    >> ')}"
        )
    return module_op.operation


def load_mlir_template(kb: KernelBuilder, template_file: str, **kwargs) -> Operation:
    """Loads an MLIR template by name.

    The file is loaded relative to the templates/ directory. It is interpolated
    with **kwargs and loaded into the KernelBuilder.
    """
    template_path = TEMPLATES_DIR / template_file
    template_text = template_path.read_text()
    asm = template_text.format(**kwargs)
    try:
        module_op = Operation.parse(
            asm, source_name=str(template_path), context=kb.context
        )
    except MLIRError as e:
        raise RuntimeError(
            f"Error parsing generated op template:"
            f"\n{textwrap.indent(str(e), '  ')}"
            f"\n{textwrap.indent(asm, '    >> ')}"
        )
    return module_op.operation
