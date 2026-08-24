from __future__ import annotations

import logging
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

logger = logging.getLogger("VAP")


def create_docker_client(target: DockerTarget) -> Any:
    if target.hostname is None:
        return docker.from_env()
    return docker.DockerClient(
        base_url=f"ssh://{target.hostname}",
        use_ssh_client=True,
    )


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
                entrypoint=list(spec.entrypoint),
                command=list(spec.command),
                detach=spec.detach,
                remove=spec.remove,
            )
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

    def cleanup(self) -> None:
        container = self._container
        self._process = None
        self._files = None
        self._network = None
        self._container = None
        if container is None:
            return
        try:
            container.stop()
        except Exception as exc:
            logger.warning(
                "Failed to stop container on %s: %s",
                self.target.label,
                exc,
            )
        try:
            container.remove()
        except Exception as exc:
            logger.warning(
                "Failed to remove container on %s: %s",
                self.target.label,
                exc,
            )
