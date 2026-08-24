from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from vap.runners.base import CommandResult, ProcessHandle, ProcessStatus


class CommandExecutionError(RuntimeError):
    def __init__(self, command: Sequence[str], result: CommandResult):
        self.command = tuple(command)
        self.result = result
        detail = result.combined_text.strip()
        message = f"Command exited with code {result.exit_code}: {self.command!r}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class DockerCommandExecutor:
    def __init__(self, container: Any):
        self._container = container

    @property
    def container(self) -> Any:
        return self._container

    def run(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
        workdir: str | None = None,
        demux: bool = False,
    ) -> CommandResult:
        kwargs: dict[str, Any] = {"demux": demux}
        if environment is not None:
            kwargs["environment"] = dict(environment)
        if workdir is not None:
            kwargs["workdir"] = workdir
        exit_code, output = self._container.exec_run(
            list(command),
            **kwargs,
        )
        if demux:
            stdout, stderr = output if output else (b"", b"")
        else:
            stdout, stderr = output or b"", b""
        return CommandResult(
            exit_code=int(exit_code),
            stdout=stdout or b"",
            stderr=stderr or b"",
        )

    def start(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
        workdir: str | None = None,
    ) -> ProcessHandle:
        kwargs: dict[str, Any] = {}
        if environment is not None:
            kwargs["environment"] = dict(environment)
        if workdir is not None:
            kwargs["workdir"] = workdir
        created = self._container.client.api.exec_create(
            self._container.id,
            list(command),
            **kwargs,
        )
        exec_id = created["Id"]
        self._container.client.api.exec_start(exec_id, detach=True)
        return ProcessHandle(exec_id=exec_id, command=tuple(command))

    def inspect(self, handle: ProcessHandle) -> ProcessStatus:
        payload = self._container.client.api.exec_inspect(handle.exec_id)
        running = bool(payload.get("Running"))
        raw_exit_code = payload.get("ExitCode")
        raw_pid = payload.get("Pid")
        return ProcessStatus(
            running=running,
            exit_code=None if running or raw_exit_code is None else int(raw_exit_code),
            pid=int(raw_pid) if raw_pid else None,
        )


class ProcessAssistant:
    def __init__(self, executor: DockerCommandExecutor):
        self._executor = executor

    def run(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
        workdir: str | None = None,
        demux: bool = False,
        check: bool = False,
    ) -> CommandResult:
        result = self._executor.run(
            command,
            environment=environment,
            workdir=workdir,
            demux=demux,
        )
        if check and result.exit_code != 0:
            raise CommandExecutionError(command, result)
        return result

    def run_shell(
        self,
        command: str,
        *,
        environment: Mapping[str, str] | None = None,
        workdir: str | None = None,
        demux: bool = False,
        check: bool = False,
    ) -> CommandResult:
        return self.run(
            ["/bin/bash", "-c", command],
            environment=environment,
            workdir=workdir,
            demux=demux,
            check=check,
        )

    def start(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
        workdir: str | None = None,
    ) -> ProcessHandle:
        return self._executor.start(
            command,
            environment=environment,
            workdir=workdir,
        )

    def start_shell(
        self,
        command: str,
        *,
        environment: Mapping[str, str] | None = None,
        workdir: str | None = None,
    ) -> ProcessHandle:
        return self.start(
            ["/bin/bash", "-c", command],
            environment=environment,
            workdir=workdir,
        )

    def status(self, handle: ProcessHandle) -> ProcessStatus:
        return self._executor.inspect(handle)

    def wait(
        self,
        handle: ProcessHandle,
        *,
        timeout_sec: float | None = None,
        poll_interval_sec: float = 0.2,
    ) -> ProcessStatus:
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        while True:
            status = self.status(handle)
            if not status.running:
                return status
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Process did not exit within {timeout_sec:.1f}s: "
                    f"{handle.command!r}"
                )
            time.sleep(poll_interval_sec)

    def terminate(
        self,
        handle: ProcessHandle,
        *,
        timeout_sec: float = 5,
    ) -> ProcessStatus:
        status = self.status(handle)
        if not status.running or status.pid is None:
            return status
        self.run(["kill", "-TERM", str(status.pid)])
        try:
            return self.wait(handle, timeout_sec=timeout_sec)
        except TimeoutError:
            self.run(["kill", "-KILL", str(status.pid)])
            return self.wait(handle, timeout_sec=timeout_sec)
