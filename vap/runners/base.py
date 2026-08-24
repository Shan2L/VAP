from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: bytes = b""
    stderr: bytes = b""

    @property
    def stdout_text(self) -> str:
        return self.stdout.decode(errors="replace")

    @property
    def stderr_text(self) -> str:
        return self.stderr.decode(errors="replace")

    @property
    def combined_text(self) -> str:
        return (self.stderr or self.stdout).decode(errors="replace")


@dataclass(frozen=True)
class ProcessHandle:
    exec_id: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class ProcessStatus:
    running: bool
    exit_code: int | None
    pid: int | None


@dataclass(frozen=True)
class MountSpec:
    target: str
    source: str
    type: str = "bind"
    read_only: bool = False
    create_source: bool = False


@dataclass(frozen=True)
class UlimitSpec:
    name: str
    soft: int
    hard: int


@dataclass(frozen=True)
class ContainerSpec:
    image: str
    name: str
    mounts: tuple[MountSpec, ...] = ()
    devices: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    command: tuple[str, ...] = ("/bin/bash", "-c", "sleep infinity")
    entrypoint: tuple[str, ...] = ()
    ipc_mode: str = "host"
    network_mode: str = "host"
    cap_add: tuple[str, ...] = ()
    group_add: tuple[str, ...] = ()
    security_opt: tuple[str, ...] = ()
    ulimits: tuple[UlimitSpec, ...] = ()
    shm_size: str | int | None = None
    detach: bool = True
    remove: bool = False


@dataclass(frozen=True)
class DockerTarget:
    hostname: str | None = None

    @property
    def label(self) -> str:
        return self.hostname or "local"
