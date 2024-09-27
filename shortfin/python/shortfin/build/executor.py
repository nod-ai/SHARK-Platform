# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Collection

import abc
import argparse
import concurrent.futures
import enum
import multiprocessing
import traceback
from pathlib import Path
import threading

_locals = threading.local()


class FileNamespace(enum.StrEnum):
    # Transient generated files go into the GEN namespace. These are typically
    # not packaged for distribution.
    GEN = enum.auto()

    # Distributable parameter files.
    PARAMS = enum.auto()

    # Distributable, platform-neutral binaries.
    BIN = enum.auto()

    # Distributable, platform specific binaries.
    PLATFORM_BIN = enum.auto()


FileNamespaceToPath = {
    FileNamespace.GEN: lambda executor: executor.output_dir / "genfiles",
    FileNamespace.PARAMS: lambda executor: executor.output_dir / "params",
    FileNamespace.BIN: lambda executor: executor.output_dir / "bin",
    # TODO: This isn't right. Need to resolve platform dynamically.
    FileNamespace.PLATFORM_BIN: lambda executor: executor.output_dir / "platform",
}


def join_namespace(prefix: str, suffix: str) -> str:
    """Joins two namespace components, taking care of the root namespace (empty)."""
    if not prefix:
        return suffix
    return f"{prefix}/{suffix}"


class Entrypoint:
    def __init__(self, name: str, wrapped: Callable):
        self.name = name
        self._wrapped = wrapped

    def __call__(self, *args, **kwargs):
        parent_context = BuildContext.current()
        bep = BuildEntrypoint(
            join_namespace(parent_context.path, self.name),
            parent_context.executor,
            self,
        )
        parent_context.executor.entrypoints.append(bep)
        with bep:
            results = self._wrapped(*args, **kwargs)
            if results is not None:
                bep.deps.update(bep.files(results))


class Scheduler:
    """Holds resources related to scheduling."""

    def __init__(self):
        self.thread_pool_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="shortfin.build"
        )
        self.process_pool_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=10, mp_context=multiprocessing.get_context("spawn")
        )

    def shutdown(self):
        self.thread_pool_executor.shutdown(cancel_futures=True)
        self.process_pool_executor.shutdown(cancel_futures=True)


class Executor:
    """Executor that all build contexts share."""

    def __init__(self, output_dir: Path, scheduler: Scheduler | None = None):
        self.output_dir = output_dir
        self.verbose_level = 5
        # Keyed by path
        self.all: dict[str, "BuildContext" | "BuildFile"] = {}
        self.entrypoints: list["BuildEntrypoint"] = []
        BuildContext("", self)
        self.scheduler = Scheduler() if scheduler is None else scheduler

    def check_path_not_exists(self, path: str, for_entity):
        existing = self.all.get(path)
        if existing is not None:
            formatted_stack = "".join(traceback.format_list(existing.def_stack))
            raise RuntimeError(
                f"Cannot add {for_entity} because an entity with that name was "
                f"already defined at:\n{formatted_stack}"
            )

    def get_context(self, path: str) -> "BuildContext":
        existing = self.all.get(path)
        if existing is None:
            raise RuntimeError(f"Context at path {path} not found")
        if not isinstance(existing, BuildContext):
            raise RuntimeError(
                f"Entity at path {path} is not a context. It is: {existing}"
            )
        return existing

    def get_file(self, path: str) -> "BuildFile":
        existing = self.all.get(path)
        if existing is None:
            raise RuntimeError(f"File at path {path} not found")
        if not isinstance(existing, BuildFile):
            raise RuntimeError(
                f"Entity at path {path} is not a file. It is: {existing}"
            )
        return existing

    def write_status(self, message: str):
        print(message)

    def get_root(self, namespace: FileNamespace) -> Path:
        return FileNamespaceToPath[namespace](self)

    def analyze(self, *entrypoints: Entrypoint):
        """Analyzes all entrypoints building the graph."""
        for entrypoint in entrypoints:
            if self.verbose_level > 1:
                self.write_status(f"Analyzing entrypoint {entrypoint.name}")
            with self.get_context("") as context:
                # TODO: Plumb args/kwargs/etc.
                entrypoint()

    def build(self, *initial_deps: "BuildDependency"):
        """Transitively builds the given deps."""
        # Invert the dependency chain so that we know when something is ready.
        in_flight: set["BuildDependency"] = set()
        producer_graph: dict["BuildDependency", list["BuildDependency"]] = dict()

        def add_dep(
            dep: "BuildDependency",
            produces: "BuildDependency",
            stack: set["BuildDependency"],
        ):
            if dep in stack:
                raise RuntimeError(
                    f"Circular dependency: '{dep}' depends on itself: {stack}"
                )
            plist = producer_graph.get(dep)
            if plist is None:
                plist = []
                producer_graph[dep] = plist
            plist.append(produces)
            next_stack = set(stack)
            next_stack.add(dep)
            if not dep.deps:
                # Terminal - depends on nothing.
                in_flight.add(dep)
            else:
                # Intermediate dep.
                for next_dep in dep.deps:
                    add_dep(next_dep, dep, next_stack)

        for entrypoint in initial_deps:
            stack = set()
            stack.add(entrypoint)
            for dep in entrypoint.deps:
                add_dep(dep, entrypoint, stack)

        # Schedule initial actions.
        for initial in in_flight:
            self._schedule_action(initial)

        def service():
            completed_deps: set["BuildDependency"] = set()
            try:
                for completed_fut in concurrent.futures.as_completed(
                    (d.future for d in in_flight), 0
                ):
                    completed_dep = completed_fut.result()
                    self.write_status(f"Completed {completed_dep}")
                    completed_deps.add(completed_dep)
            except TimeoutError:
                pass

            # Purge done from in-flight list.
            in_flight.difference_update(completed_deps)

            # Schedule any available.
            for completed_dep in completed_deps:
                ready_list = producer_graph.get(completed_dep)
                if ready_list is None:
                    continue
                del producer_graph[completed_dep]
                for ready_dep in ready_list:
                    self._schedule_action(ready_dep)
                    in_flight.add(ready_dep)

            # Do a blocking wait for at least one ready.
            concurrent.futures.wait(
                (d.future for d in in_flight),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

        while producer_graph:
            self.write_status(f"Servicing {len(producer_graph)} outstanding tasks")
            service()

    def _schedule_action(self, dep: "BuildDependency"):
        if isinstance(dep, BuildAction):
            # Schedule the action.
            self.write_status(f"Scheduling action: {dep}")
            if dep.future is None:

                def invoke():
                    dep.invoke()
                    return dep

                dep.future = self.scheduler.thread_pool_executor.submit(invoke)
        else:
            # Not schedulable. Just mark it as done.
            dep.future = concurrent.futures.Future()
            dep.future.set_result(dep)


class BuildDependency:
    """Base class of entities that can act as a build dependency."""

    def __init__(
        self, *, executor: Executor, deps: set["BuildDependency"] | None = None
    ):
        self.future: concurrent.futures.Future | None = None
        self.executor = executor
        self.deps: set[BuildDependency] = set()
        if deps:
            self.deps.update(deps)


class BuildFile(BuildDependency):
    """Generated file in the build tree."""

    def __init__(
        self,
        *,
        executor: Executor,
        path: str,
        namespace: FileNamespace = FileNamespace.GEN,
        deps: set[BuildDependency] | None = None,
    ):
        super().__init__(executor=executor, deps=deps)
        self.def_stack = traceback.extract_stack()[0:-2]
        self.executor = executor
        self.path = path
        self.namespace = namespace
        # Set of build files that must be made available to any transitive user
        # of this build file at runtime.
        self.runfiles: set["BuildFile"] = set()

        executor.check_path_not_exists(path, self)
        executor.all[path] = self

    def get_fs_path(self) -> Path:
        path = self.executor.get_root(self.namespace) / self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def __repr__(self):
        return f"BuildFile[{self.namespace}]({self.path})"


class ActionConcurrency(enum.StrEnum):
    THREAD = enum.auto()
    PROCESS = enum.auto()
    NONE = enum.auto()


class BuildAction(BuildDependency, abc.ABC):
    """An action that must be carried out."""

    def __init__(
        self,
        *,
        desc: str,
        executor: Executor,
        concurrency: ActionConcurrency = ActionConcurrency.THREAD,
        deps: set[BuildDependency] | None = None,
    ):
        super().__init__(executor=executor, deps=deps)
        self.desc = desc
        self.concurrnecy = concurrency

    def __str__(self):
        return self.desc

    def __repr__(self):
        return f"Action[{type(self).__name__}]('{self.desc}')"

    @abc.abstractmethod
    def invoke(self):
        ...


class BuildContext(BuildDependency):
    """Manages a build graph under construction."""

    def __init__(self, path: str, executor: Executor):
        super().__init__(executor=executor)
        self.def_stack = traceback.extract_stack()[0:-2]
        self.executor = executor
        self.path = path
        executor.check_path_not_exists(path, self)
        executor.all[path] = self
        self.analyzed = False

    def __repr__(self):
        return f"{type(self).__name__}(path='{self.path}')"

    def allocate_file(
        self, path: str, namespace: FileNamespace = FileNamespace.GEN
    ) -> BuildFile:
        """Allocates a file in the build tree with local path |path|.

        If |path| is absoluate (starts with '/'), then it is used as-is. Otherwise,
        it is joined with the path of this context.
        """
        if not path.startswith("/"):
            path = join_namespace(self.path, path)
        return BuildFile(executor=self.executor, path=path, namespace=namespace)

    def file(self, file: str | BuildFile) -> BuildFile:
        """Accesses a BuildFile by either string (path) or BuildFile.

        It must already exist.
        """
        if isinstance(file, BuildFile):
            return file
        path = file
        if not path.startswith("/"):
            path = join_namespace(self.path, path)
        existing = self.executor.all.get(path)
        if not isinstance(existing, BuildFile):
            all_files = [
                f.path for f in self.executor.all.values() if isinstance(f, BuildFile)
            ]
            raise RuntimeError(
                f"File with path '{path}' is not known in the build graph. Available:\n"
                f"  {'\n  '.join(all_files)}"
            )
        return existing

    def files(
        self, files: str | BuildFile | Collection[str | BuildFile]
    ) -> list[BuildFile]:
        """Accesses a collection of files (or single) as a list of BuildFiles."""
        if isinstance(files, (str, BuildFile)):
            return [self.file(files)]
        return [self.file(f) for f in files]

    @staticmethod
    def current() -> "BuildContext":
        try:
            return _locals.context_stack[-1]
        except (AttributeError, IndexError):
            raise RuntimeError(
                "The current code can only be evaluated within an active BuildContext"
            )

    def __enter__(self) -> "BuildContext":
        try:
            stack = _locals.context_stack
        except AttributeError:
            stack = _locals.context_stack = []
        stack.append(self)
        return self

    def __exit__(self, *args):
        try:
            stack = _locals.context_stack
        except AttributeError:
            raise AssertionError("BuildContext exit without enter")
        existing = stack.pop()
        assert existing is self, "Unbalanced BuildContext enter/exit"

    def populate_arg_parser(self, parser: argparse.ArgumentParser):
        ...


class BuildEntrypoint(BuildContext):
    def __init__(self, path: str, executor: Executor, entrypoint: Entrypoint):
        super().__init__(path, executor)
        self.entrypoint = entrypoint


# Type aliases.
BuildFileLike = BuildFile | str
