from __future__ import annotations

import glob
import logging
import os
import socket
import subprocess
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, TypeVar

from vap.clock_probe.execution.host import (
    inspect_node_hardware_timestamping,
    ssh_command,
)
from vap.config import VAPConfig
from vap.runners import DockerTarget
from vap.runners.docker import create_docker_client, sweep_vap_containers
from vap.runners.ssh_identity import resolve_ssh_identity
from vap.server import settings
from vap.validation import validate_config_payload

logger = logging.getLogger("VAP")

DOCKER_IMAGE_CHECK_TIMEOUT_SEC = 8
CLOCK_PROBE_CHECK_TIMEOUT_SEC = 12

T = TypeVar("T")
R = TypeVar("R")


def collect_with_timeout(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    timeout_sec: float,
    on_timeout: Callable[[T], R],
) -> list[R]:
    """Run checks in parallel and return after timeout without waiting on hung SSH."""
    item_list = list(items)
    if not item_list:
        return []
    executor = ThreadPoolExecutor(max_workers=min(8, len(item_list)))
    try:
        futures = [(item, executor.submit(func, item)) for item in item_list]
        deadline = time.monotonic() + timeout_sec
        results: list[R] = []
        for item, future in futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                results.append(on_timeout(item))
                continue
            try:
                results.append(future.result(timeout=remaining))
            except FutureTimeoutError:
                results.append(on_timeout(item))
        return results
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _vap_run_is_active() -> bool:
    with settings.RUN_LOCK:
        process = settings.RUN_STATE["process"]
        if not settings.RUN_STATE["running"]:
            return False
        if process is None:
            return True
        return process.poll() is None


def _sweep_local_vap_containers_if_idle() -> None:
    """Drop leftover host-network VAP containers before UI port checks.

    Ray GCS (6379) and vLLM (8080) stay bound after a SIGTERM timeout; Validate
    used to fail even though Run would have swept them.
    """
    if _vap_run_is_active():
        return
    try:
        sweep_vap_containers(DockerTarget())
    except Exception as exc:
        logger.warning("Failed to sweep leftover VAP containers: %s", exc)


def describe_listening_port(port: int) -> str | None:
    """Best-effort description of a local TCP LISTEN occupant."""
    hexport = f"{int(port):04X}"
    inodes: set[str] = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local = parts[1]
            state = parts[3]
            inode = parts[9]
            if state == "0A" and local.upper().endswith(f":{hexport}"):
                inodes.add(inode)
    if not inodes:
        return None
    for fd_path in glob.glob("/proc/[0-9]*/fd/[0-9]*"):
        try:
            target = os.readlink(fd_path)
        except OSError:
            continue
        if not target.startswith("socket:[") or target[8:-1] not in inodes:
            continue
        pid = fd_path.split("/")[2]
        try:
            command = (
                Path(f"/proc/{pid}/cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", errors="replace")
                .strip()
            )
        except OSError:
            command = ""
        label = command or f"pid {pid}"
        return f"{label[:120]} (pid {pid})"
    return f"TCP LISTEN inode {next(iter(inodes))}"


def is_local_port_available(port: int) -> bool:
    """True when IPv4 0.0.0.0 can be bound (LISTEN occupied => False).

    SO_REUSEADDR ignores TIME_WAIT leftovers from a previous Ray/vLLM process.
    Dual-stack IPv6 binds are not required: this machine has
    net.ipv6.bindv6only=0, so binding :: would also grab IPv4 and make a
    follow-up ::1 bind fail even when the port is free.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def check_config_ports(payload: dict[str, Any]) -> dict[str, Any]:
    validation = validate_config_payload(payload)
    if not validation["valid"]:
        return {"valid": False, "ports": [], "errors": validation["errors"]}

    config = VAPConfig.model_validate(payload)
    _sweep_local_vap_containers_if_idle()
    ports = [
        {
            "name": "vLLM service port",
            "port": config.vllm_port,
            "available": is_local_port_available(config.vllm_port),
            "blocking": True,
        },
    ]
    if config.profiler_cfg.enable:
        ports.extend(
            [
                {
                    "name": "TensorBoard port",
                    "port": config.profiler_cfg.tensorboard_port,
                    "available": is_local_port_available(
                        config.profiler_cfg.tensorboard_port
                    ),
                    "blocking": True,
                },
                {
                    "name": "Perfetto Trace Processor port",
                    "port": settings.PERFETTO_PORT,
                    "available": is_local_port_available(settings.PERFETTO_PORT),
                    "blocking": False,
                },
            ]
        )
    distributed = config.distributed_cfg
    if distributed is not None and distributed.enable:
        ports.append(
            {
                "name": "Ray head port",
                "port": distributed.ray_port,
                "available": is_local_port_available(distributed.ray_port),
                "blocking": True,
            },
        )
    if (
        distributed is not None
        and distributed.enable
        and config.clock_probe_cfg.enabled
    ):
        ports.append(
            {
                "name": "Clock probe port",
                "port": config.clock_probe_cfg.port,
                "available": is_local_port_available(config.clock_probe_cfg.port),
                "blocking": True,
            },
        )
    for item in ports:
        if item["available"]:
            item["message"] = f"Local port {item['port']} is available"
        elif item["blocking"]:
            occupant = describe_listening_port(item["port"])
            suffix = f" ({occupant})" if occupant else ""
            item["message"] = (
                f"Local port {item['port']} is already in use or cannot be bound"
                f"{suffix}"
            )
        else:
            item["message"] = (
                f"Local port {item['port']} is unavailable; Perfetto visualization "
                "will be skipped"
            )
    return {"valid": True, "ports": ports}


def check_config_machines(payload: dict[str, Any]) -> dict[str, Any]:
    validation = validate_config_payload(payload)
    if not validation["valid"]:
        return {"valid": False, "machines": [], "errors": validation["errors"]}

    config = VAPConfig.model_validate(payload)
    distributed = config.distributed_cfg
    if distributed is None or not distributed.enable:
        return {
            "valid": True,
            "machines": [],
            "message": "Distributed machines are not enabled in the current config.",
        }

    nodes = list(dict.fromkeys(distributed.worker_nodes))
    machines = collect_with_timeout(
        lambda node: check_machine(node, [{"label": "SSH", "port": 22}]),
        nodes,
        timeout_sec=6,
        on_timeout=lambda node: {
            "node": node,
            "reachable": False,
            "ip": None,
            "checks": [],
            "message": f"{node}: machine check timed out",
        },
    )
    return {"valid": True, "machines": machines}


def check_config_resources(payload: dict[str, Any]) -> dict[str, Any]:
    checks = [
        *check_config_model_resources(payload)["checks"],
        *check_config_docker_resources(payload)["checks"],
        *check_config_worker_docker_resources(payload)["checks"],
        *check_config_container_resources(payload)["checks"],
    ]
    return {"valid": all(check["ok"] for check in checks), "checks": checks}


def check_config_model_resources(payload: dict[str, Any]) -> dict[str, Any]:
    model_cfg = payload.get("model_cfg") or {}
    model_root = str(model_cfg.get("model_path") or "")
    model_name = str(model_cfg.get("model_name") or "")
    model_path = str(Path(model_root) / model_name) if model_root and model_name else ""
    checks = [
        check_path(
            "Model root",
            model_root,
            expect_dir=True,
            required=True,
        ),
        check_path(
            "Model weight path",
            model_path,
            expect_dir=True,
            required=True,
        ),
    ]
    return {"valid": all(check["ok"] for check in checks), "checks": checks}


def check_config_docker_resources(payload: dict[str, Any]) -> dict[str, Any]:
    container_cfg = payload.get("container_cfg") or {}
    image_name = str(container_cfg.get("image_name") or "")
    image_tag = str(container_cfg.get("image_tag") or "")
    docker_image = f"{image_name}:{image_tag}" if image_name and image_tag else ""
    checks = collect_with_timeout(
        lambda _: check_docker_image(docker_image),
        ["local"],
        timeout_sec=DOCKER_IMAGE_CHECK_TIMEOUT_SEC,
        on_timeout=lambda _: {
            "name": "Docker image (local)",
            "ok": False,
            "message": (
                "Local Docker daemon timed out after "
                f"{DOCKER_IMAGE_CHECK_TIMEOUT_SEC}s"
            ),
            "image": docker_image,
            "node": "local",
        },
    )
    return {"valid": all(check["ok"] for check in checks), "checks": checks}


def check_config_worker_docker_resources(payload: dict[str, Any]) -> dict[str, Any]:
    container_cfg = payload.get("container_cfg") or {}
    image_name = str(container_cfg.get("image_name") or "")
    image_tag = str(container_cfg.get("image_tag") or "")
    docker_image = f"{image_name}:{image_tag}" if image_name and image_tag else ""
    distributed_cfg = payload.get("distributed_cfg") or {}
    worker_nodes = (
        distributed_cfg.get("worker_nodes")
        if distributed_cfg.get("enable")
        and isinstance(distributed_cfg.get("worker_nodes"), list)
        else []
    )
    unique_workers = list(dict.fromkeys(str(node) for node in worker_nodes if node))
    ssh_key = distributed_cfg.get("sshkey_path") or None
    if not unique_workers:
        return {
            "valid": True,
            "checks": [],
            "message": "Distributed workers are not enabled",
        }
    checks = collect_with_timeout(
        lambda node: check_docker_image(docker_image, node, ssh_key=ssh_key),
        unique_workers,
        timeout_sec=DOCKER_IMAGE_CHECK_TIMEOUT_SEC,
        on_timeout=lambda node: {
            "name": f"Docker image ({node})",
            "ok": False,
            "message": (
                f"Docker daemon on {node} timed out after "
                f"{DOCKER_IMAGE_CHECK_TIMEOUT_SEC}s"
            ),
            "image": docker_image,
            "node": node,
        },
    )
    return {"valid": all(check["ok"] for check in checks), "checks": checks}


def check_config_container_resources(payload: dict[str, Any]) -> dict[str, Any]:
    container_cfg = payload.get("container_cfg") or {}
    checks: list[dict[str, Any]] = []
    devices = container_cfg.get("devices") or []
    if not isinstance(devices, list):
        checks.append(
            {
                "name": "Device files",
                "ok": False,
                "message": "devices must be an array or null",
                "path": "",
            }
        )
        devices = []
    for index, device in enumerate(devices):
        checks.append(
            check_path(
                f"Device file devices[{index}]",
                str(device),
                expect_dir=False,
                required=True,
            )
        )

    mounts = container_cfg.get("mounts") or []
    if not isinstance(mounts, list):
        checks.append(
            {
                "name": "Mount sources",
                "ok": False,
                "message": "mounts must be an array or null",
                "path": "",
            }
        )
        mounts = []
    for index, mount in enumerate(mounts):
        if not isinstance(mount, dict):
            checks.append(
                {
                    "name": f"Mount source mounts[{index}].source",
                    "ok": False,
                    "message": "mount must be an object",
                    "path": "",
                }
            )
            continue
        checks.append(
            check_path(
                f"Mount source mounts[{index}].source",
                str(mount.get("source") or ""),
                expect_dir=None,
                required=True,
            )
        )

    return {"valid": all(check["ok"] for check in checks), "checks": checks}


def check_config_clock_probe(payload: dict[str, Any]) -> dict[str, Any]:
    probe_cfg = payload.get("clock_probe_cfg") or {}
    distributed_cfg = payload.get("distributed_cfg") or {}
    enabled = bool(probe_cfg.get("enabled"))
    mode = str(probe_cfg.get("mode") or "hardware")
    if not enabled:
        return {
            "valid": True,
            "mode": mode,
            "nodes": [],
            "message": "Clock probe is disabled",
        }
    if not distributed_cfg.get("enable"):
        return {
            "valid": False,
            "mode": mode,
            "nodes": [],
            "message": "Clock probe requires distributed_cfg.enable=true",
        }
    if mode == "software":
        return {
            "valid": True,
            "mode": mode,
            "nodes": [],
            "message": "Software mode does not require PHC hardware",
        }

    ssh_key = distributed_cfg.get("sshkey_path")
    preferred_interface = probe_cfg.get("hardware_interface") or None
    preferred_phc = probe_cfg.get("hardware_phc_device") or None
    nodes = ["local", *list(dict.fromkeys(distributed_cfg.get("worker_nodes") or []))]
    results = collect_with_timeout(
        lambda node: inspect_node_hardware_timestamping(
            node=node,
            hostname=None if node == "local" else str(node),
            ssh_key=ssh_key,
            preferred_interface=preferred_interface,
            preferred_phc_device=preferred_phc,
            capture=False,
            require_readable=False,
        ),
        nodes,
        timeout_sec=CLOCK_PROBE_CHECK_TIMEOUT_SEC,
        on_timeout=lambda node: {
            "node": node,
            "usable": False,
            "message": f"{node}: hardware timestamp check timed out",
        },
    )

    failures = [item for item in results if not item.get("usable")]
    warnings: list[str] = []
    if mode == "auto":
        warnings.extend(
            item.get("message")
            or f"{item.get('node')}: hardware timestamping unavailable"
            for item in failures
        )
    return {
        "valid": not failures if mode == "hardware" else True,
        "mode": mode,
        "nodes": results,
        "warnings": warnings,
        "message": (
            "\n".join(item.get("message", "") for item in results)
            if results
            else "No nodes were checked"
        ),
    }


def check_path(
    name: str,
    raw_path: str,
    *,
    expect_dir: bool | None,
    required: bool,
) -> dict[str, Any]:
    if not raw_path:
        return {
            "name": name,
            "ok": not required,
            "message": "Path is empty" if required else "Not configured, skipped",
            "path": raw_path,
        }

    path = Path(raw_path)
    if not path.is_absolute():
        return {
            "name": name,
            "ok": False,
            "message": "Path must be an absolute path",
            "path": raw_path,
        }
    if not path.exists():
        return {
            "name": name,
            "ok": False,
            "message": "Path does not exist",
            "path": raw_path,
        }
    if expect_dir is True and not path.is_dir():
        return {
            "name": name,
            "ok": False,
            "message": "Path exists but is not a directory",
            "path": raw_path,
        }
    return {
        "name": name,
        "ok": True,
        "message": "Exists and is accessible",
        "path": raw_path,
    }


def check_remote_docker_image(
    image: str,
    hostname: str,
    ssh_key: str | None = None,
) -> dict[str, Any]:
    check_name = f"Docker image ({hostname})"
    identity = resolve_ssh_identity(ssh_key)
    key_note = ""
    if ssh_key and identity is None:
        key_note = (
            f" Configured SSH key {ssh_key} is missing; "
            "used the default SSH identity instead."
        )
    try:
        result = subprocess.run(
            ssh_command(
                hostname,
                ["docker", "image", "inspect", "--format", "{{.Id}}", image],
                identity,
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=DOCKER_IMAGE_CHECK_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": check_name,
            "ok": False,
            "message": f"SSH Docker check on {hostname} timed out.{key_note}",
            "image": image,
            "node": hostname,
        }

    detail = (result.stderr or result.stdout).strip()
    if result.returncode == 0:
        message = f"Docker image exists on {hostname}"
        if key_note:
            message += f".{key_note}"
        return {
            "name": check_name,
            "ok": True,
            "message": message,
            "image": image,
            "node": hostname,
        }

    lowered = detail.lower()
    if "no such image" in lowered or "no such object" in lowered:
        message = (
            f"Docker image does not exist on {hostname}. "
            "Pull it on that worker, not only on an SSH jump host."
        )
    elif any(
        token in lowered
        for token in (
            "permission denied",
            "could not resolve",
            "connection refused",
            "connection timed out",
            "host key verification",
            "no matching host key",
            "identity file",
            "not accessible",
        )
    ):
        message = (
            f"SSH to {hostname} failed before Docker could be checked: "
            f"{detail}.{key_note}"
        )
    else:
        message = f"Docker image check failed on {hostname}: {detail}.{key_note}"
    return {
        "name": check_name,
        "ok": False,
        "message": message.strip(),
        "image": image,
        "node": hostname,
    }


def check_docker_image(
    image: str,
    hostname: str | None = None,
    ssh_key: str | None = None,
) -> dict[str, Any]:
    node = hostname or "local"
    check_name = f"Docker image ({node})"
    if not image:
        return {
            "name": check_name,
            "ok": False,
            "message": "Docker image config is empty",
            "image": image,
            "node": node,
        }
    try:
        from docker.errors import DockerException, ImageNotFound
    except Exception as exc:
        return {
            "name": check_name,
            "ok": False,
            "message": f"Cannot import Docker SDK: {exc}",
            "image": image,
            "node": node,
        }

    client = None
    try:
        if hostname:
            return check_remote_docker_image(image, hostname, ssh_key=ssh_key)
        client = create_docker_client(
            DockerTarget(hostname=hostname, ssh_key=ssh_key),
            timeout_seconds=5,
        )
        client.images.get(image)
        return {
            "name": check_name,
            "ok": True,
            "message": f"Docker image exists on {node}",
            "image": image,
            "node": node,
        }
    except ImageNotFound:
        return {
            "name": check_name,
            "ok": False,
            "message": f"Docker image does not exist on {node}. Pull or build it first.",
            "image": image,
            "node": node,
        }
    except DockerException as exc:
        return {
            "name": check_name,
            "ok": False,
            "message": f"Docker daemon on {node} is unavailable: {exc}",
            "image": image,
            "node": node,
        }
    except Exception as exc:
        return {
            "name": check_name,
            "ok": False,
            "message": f"Docker image check failed on {node}: {exc}",
            "image": image,
            "node": node,
        }
    finally:
        if client is not None:
            client.close()


def check_machine(node: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        resolved_ip = socket.gethostbyname(node)
    except OSError as exc:
        return {
            "node": node,
            "reachable": False,
            "ip": None,
            "checks": [],
            "message": f"DNS resolution failed: {exc}",
        }

    port_results = []
    for check in checks:
        port = int(check["port"])
        label = str(check["label"])
        try:
            with socket.create_connection((node, port), timeout=1.5):
                port_results.append(
                    {
                        "label": label,
                        "port": port,
                        "reachable": True,
                        "message": f"{label} port {port} is reachable",
                    }
                )
        except OSError as exc:
            port_results.append(
                {
                    "label": label,
                    "port": port,
                    "reachable": False,
                    "message": f"{label} port {port} is unreachable: {exc}",
                }
            )

    reachable = any(result["reachable"] for result in port_results)
    return {
        "node": node,
        "reachable": reachable,
        "ip": resolved_ip,
        "checks": port_results,
        "message": (
            "Machine is reachable"
            if reachable
            else "Machine DNS resolved, but none of the checked ports are reachable"
        ),
    }
