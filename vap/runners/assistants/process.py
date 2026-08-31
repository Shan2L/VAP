from __future__ import annotations

import logging
import signal
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from vap.runners.base import CommandResult, ProcessHandle, ProcessStatus

logger = logging.getLogger("VAP")

# vLLM with Ray workers can take a while to drain after SIGTERM.
TERMINATE_GRACE_SEC = 60.0
KILL_WAIT_SEC = 10.0

# Walk /proc and signal the exec process plus descendants. Docker exec_inspect
# Pid is a host PID, so kill from inside the container must use the container PID.
_SIGNAL_TREE_SCRIPT = r"""
import os
import sys

root = int(sys.argv[1])
sig = int(sys.argv[2])

def children(ppid):
    found = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return found
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", encoding="utf-8") as fh:
                stat = fh.read()
        except OSError:
            continue
        rparen = stat.rfind(")")
        if rparen < 0:
            continue
        fields = stat[rparen + 2 :].split()
        try:
            if int(fields[1]) == ppid:
                found.append(int(name))
        except (IndexError, ValueError):
            continue
    return found

def descendants(pid):
    result = []
    for child in children(pid):
        result.extend(descendants(child))
        result.append(child)
    return result

for pid in descendants(root) + [root]:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass
"""


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
        original_command = tuple(command)
        pid_file = f"/tmp/vap-proc-{uuid.uuid4().hex}.pid"
        wrapped = [
            "/bin/bash",
            "-c",
            'echo $$ > "$1"; shift; exec "$@"',
            "vap-proc",
            pid_file,
            *original_command,
        ]
        created = self._container.client.api.exec_create(
            self._container.id,
            wrapped,
            **kwargs,
        )
        exec_id = created["Id"]
        self._container.client.api.exec_start(exec_id, detach=True)
        return ProcessHandle(
            exec_id=exec_id,
            command=tuple(original_command),
            pid_file=pid_file,
        )

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
        timeout_sec: float = TERMINATE_GRACE_SEC,
    ) -> ProcessStatus:
        status = self.status(handle)
        if not status.running:
            return status
        pid = self._container_pid(handle)
        if pid is None:
            raise TimeoutError(
                f"Process is running but has no container PID: {handle.command!r}"
            )
        self._signal_tree(pid, signal.SIGTERM)
        try:
            return self.wait(handle, timeout_sec=timeout_sec)
        except TimeoutError:
            self._signal_tree(pid, signal.SIGKILL)
            return self.wait(handle, timeout_sec=min(timeout_sec, KILL_WAIT_SEC))

    def _container_pid(self, handle: ProcessHandle) -> int | None:
        if handle.pid_file:
            result = self.run(["cat", handle.pid_file])
            text = result.stdout_text.strip()
            if result.exit_code == 0 and text.isdigit():
                return int(text)
            logger.warning(
                "Container pidfile %s is unreadable; not using host exec PID",
                handle.pid_file,
            )
            return None
        # exec_inspect Pid is a host PID. Without a container pidfile it is
        # not safe to signal from inside the container.
        return None

    def _signal_tree(self, pid: int, sig: signal.Signals) -> None:
        result = self.run(
            ["python3", "-c", _SIGNAL_TREE_SCRIPT, str(pid), str(int(sig))]
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to signal pid {pid} with {int(sig)}: "
                f"{result.combined_text.strip()}"
            )
