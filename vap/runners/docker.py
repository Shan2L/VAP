from __future__ import annotations

import logging
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any

import docker
from docker.types import Mount, Ulimit

from vap.runners.assistants.files import FileAssistant
from vap.runners.assistants.network import NetworkAssistant
from vap.runners.assistants.process import (
    DockerCommandExecutor,
    ProcessAssistant,
)
from vap.runners.base import ContainerSpec, DockerTarget
from vap.runners.ssh_identity import resolve_ssh_identity

logger = logging.getLogger("VAP")

_RUNNING_CONTAINER_STATUSES = frozenset({"created", "running"})
VAP_MANAGED_LABEL = "com.vap.managed"
VAP_RUN_LABEL = "com.vap.run_id"
VAP_KIND_LABEL = "com.vap.kind"
DEFAULT_DOCKER_TIMEOUT_SECONDS = 30


def _container_log_text(container: Any) -> str:
    try:
        logs = container.logs()
    except Exception:
        return ""
    if isinstance(logs, bytes):
        return logs.decode("utf-8", errors="replace")
    return str(logs or "")


_REAL_SSH = shutil.which("ssh") or "/usr/bin/ssh"
_SSH_WRAPPER_DIR: str | None = None


def _noninteractive_ssh_wrapper_dir() -> str:
    """PATH dir whose `ssh` rejects password prompts instead of blocking vap start."""
    global _SSH_WRAPPER_DIR
    if _SSH_WRAPPER_DIR is not None:
        return _SSH_WRAPPER_DIR
    directory = Path(tempfile.mkdtemp(prefix="vap-ssh-"))
    wrapper = directory / "ssh"
    real_ssh = shlex.quote(_REAL_SSH)
    wrapper.write_text(
        f"""#!/bin/sh
if [ -n "$VAP_SSH_IDENTITY" ]; then
  exec {real_ssh} \\
    -o BatchMode=yes \\
    -o ConnectTimeout=5 \\
    -o NumberOfPasswordPrompts=0 \\
    -o PasswordAuthentication=no \\
    -o KbdInteractiveAuthentication=no \\
    -o PreferredAuthentications=publickey \\
    -o StrictHostKeyChecking=yes \\
    -i "$VAP_SSH_IDENTITY" \\
    -o IdentitiesOnly=yes \\
    "$@"
fi
exec {real_ssh} \\
  -o BatchMode=yes \\
  -o ConnectTimeout=5 \\
  -o NumberOfPasswordPrompts=0 \\
  -o PasswordAuthentication=no \\
  -o KbdInteractiveAuthentication=no \\
  -o PreferredAuthentications=publickey \\
  -o StrictHostKeyChecking=yes \\
  "$@"
""",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    _SSH_WRAPPER_DIR = str(directory)
    return _SSH_WRAPPER_DIR


def _prepare_noninteractive_docker_ssh(ssh_key: str | None) -> None:
    wrapper_dir = _noninteractive_ssh_wrapper_dir()
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if path_parts[:1] != [wrapper_dir]:
        os.environ["PATH"] = os.pathsep.join([wrapper_dir, *path_parts])
    os.environ["SSH_ASKPASS_REQUIRE"] = "never"
    if ssh_key:
        identity = resolve_ssh_identity(ssh_key)
        if identity:
            os.environ["VAP_SSH_IDENTITY"] = identity
        else:
            logger.warning(
                "SSH key %s is not a readable file; using the default SSH identity",
                ssh_key,
            )
            os.environ.pop("VAP_SSH_IDENTITY", None)
    else:
        os.environ.pop("VAP_SSH_IDENTITY", None)


def create_docker_client(
    target: DockerTarget,
    *,
    timeout_seconds: int | None = None,
) -> Any:
    kwargs = {
        "timeout": (
            timeout_seconds
            if timeout_seconds is not None
            else DEFAULT_DOCKER_TIMEOUT_SECONDS
        )
    }
    if target.hostname is None:
        return docker.from_env(**kwargs)
    _prepare_noninteractive_docker_ssh(target.ssh_key)
    return docker.DockerClient(
        base_url=f"ssh://{target.hostname}",
        use_ssh_client=True,
        **kwargs,
    )


def _container_name(container: Any) -> str:
    return str(getattr(container, "name", "") or "").lstrip("/")


def orphaned_vap_containers(containers: list[Any]) -> list[Any]:
    """Return managed containers not protected by an active runner run."""
    managed = [
        container
        for container in containers
        if (getattr(container, "labels", {}) or {}).get(VAP_MANAGED_LABEL) == "true"
    ]
    active_run_ids = {
        str((getattr(container, "labels", {}) or {}).get(VAP_RUN_LABEL))
        for container in managed
        if getattr(container, "status", None) in _RUNNING_CONTAINER_STATUSES
        and (getattr(container, "labels", {}) or {}).get(VAP_KIND_LABEL) == "runner"
    }
    orphaned: list[Any] = []
    for container in managed:
        labels = getattr(container, "labels", {}) or {}
        running = getattr(container, "status", None) in _RUNNING_CONTAINER_STATUSES
        run_id = str(labels.get(VAP_RUN_LABEL))
        if running and (
            labels.get(VAP_KIND_LABEL) == "runner" or run_id in active_run_ids
        ):
            continue
        orphaned.append(container)
    return orphaned


def sweep_vap_containers(
    target: DockerTarget,
) -> None:
    """Force-remove containers explicitly labeled as managed by VAP."""
    client = create_docker_client(target)
    try:
        containers = client.containers.list(
            all=True,
            filters={"label": f"{VAP_MANAGED_LABEL}=true"},
        )
        for container in orphaned_vap_containers(containers):
            name = _container_name(container)
            try:
                container.remove(force=True)
                logger.info("Removed leftover container %s on %s", name, target.label)
            except Exception as exc:
                logger.warning(
                    "Failed to remove leftover container %s on %s: %s",
                    name,
                    target.label,
                    exc,
                )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


class DockerRunner:
    """Execution context for one local or remote Docker daemon."""

    def __init__(
        self,
        target: DockerTarget | None = None,
        *,
        client: Any = None,
        auto_network_refresh: bool = True,
    ):
        self._target = target or DockerTarget()
        self._client = client
        self._auto_network_refresh = auto_network_refresh
        self._container: Any = None
        self._process: ProcessAssistant | None = None
        self._files: FileAssistant | None = None
        self._network: NetworkAssistant | None = None

    @property
    def target(self) -> DockerTarget:
        return self._target

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = create_docker_client(self._target)
        return self._client

    @property
    def container(self) -> Any:
        if self._container is None:
            raise RuntimeError(f"Docker runner {self.target.label} is not started")
        return self._container

    @property
    def process(self) -> ProcessAssistant:
        if self._process is None:
            raise RuntimeError(f"Docker runner {self.target.label} is not started")
        return self._process

    @property
    def files(self) -> FileAssistant:
        if self._files is None:
            raise RuntimeError(f"Docker runner {self.target.label} is not started")
        return self._files

    @property
    def network(self) -> NetworkAssistant:
        if self._network is None:
            raise RuntimeError(f"Docker runner {self.target.label} is not started")
        return self._network

    @property
    def is_started(self) -> bool:
        return self._container is not None

    def start(self, spec: ContainerSpec) -> None:
        if self._container is not None:
            raise RuntimeError(f"Docker runner {self.target.label} is already started")
        self.client.images.get(spec.image)
        logger.info(
            "Docker image %s is available on %s",
            spec.image,
            self.target.label,
        )
        mounts = [
            Mount(
                target=mount.target,
                source=mount.source,
                type=mount.type,
                read_only=mount.read_only,
            )
            for mount in spec.mounts
            if not mount.create_source
        ]
        volumes = {
            mount.source: {
                "bind": mount.target,
                "mode": "ro" if mount.read_only else "rw",
            }
            for mount in spec.mounts
            if mount.create_source
        }
        ulimits = [
            Ulimit(name=limit.name, soft=limit.soft, hard=limit.hard)
            for limit in spec.ulimits
        ]
        try:
            self._container = self.client.containers.run(
                image=spec.image,
                name=spec.name,
                ipc_mode=spec.ipc_mode,
                network_mode=spec.network_mode,
                cap_add=list(spec.cap_add),
                devices=list(spec.devices),
                ulimits=ulimits,
                shm_size=spec.shm_size,
                group_add=list(spec.group_add),
                security_opt=list(spec.security_opt),
                mounts=mounts,
                volumes=volumes,
                environment=dict(spec.environment),
                labels=dict(spec.labels),
                entrypoint=list(spec.entrypoint),
                command=list(spec.command),
                detach=spec.detach,
                remove=spec.remove,
            )
            self._ensure_running(spec)
            executor = DockerCommandExecutor(self.container)
            self._process = ProcessAssistant(executor)
            self._files = FileAssistant(executor)
            self._network = NetworkAssistant(
                executor,
                self.target.label,
                auto_refresh=self._auto_network_refresh,
            )
        except Exception:
            self.cleanup()
            raise
        logger.info(
            "Started runner container %s (%s) on %s",
            spec.name,
            self.container.id,
            self.target.label,
        )

    def _ensure_running(self, spec: ContainerSpec) -> None:
        container = self._container
        if container is None:
            return
        try:
            container.reload()
        except Exception:
            pass
        status = getattr(container, "status", None)
        if not isinstance(status, str) or status in _RUNNING_CONTAINER_STATUSES:
            return
        detail = _container_log_text(container).strip()[-4000:]
        raise RuntimeError(
            f"Container {spec.name} on {self.target.label} is not running "
            f"(status={status})" + (f": {detail}" if detail else "")
        )

    def cleanup(self) -> None:
        container = self._container
        client = self._client
        self._process = None
        self._files = None
        self._network = None
        self._container = None
        try:
            if container is not None:
                container.remove(force=True)
        except Exception as exc:
            logger.warning(
                "Failed to remove container on %s: %s",
                self.target.label,
                exc,
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
            self._client = None
