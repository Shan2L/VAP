from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from vap.config import VAPConfig
from vap.runners import DockerTarget
from vap.runners.docker import create_docker_client
from vap.server import settings
from vap.validation import validate_config_payload


def is_local_port_available(port: int) -> bool:
    bind_targets = [
        (socket.AF_INET, "0.0.0.0"),
        (socket.AF_INET, "127.0.0.1"),
    ]
    if socket.has_ipv6:
        bind_targets.extend(
            [
                (socket.AF_INET6, "::"),
                (socket.AF_INET6, "::1"),
            ]
        )

    for family, host in bind_targets:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.bind((host, port))
            except OSError:
                return False
    return True


def check_config_ports(payload: dict[str, Any]) -> dict[str, Any]:
    validation = validate_config_payload(payload)
    if not validation["valid"]:
        return {"valid": False, "ports": [], "errors": validation["errors"]}

    config = VAPConfig.model_validate(payload)
    ports = [
        {
            "name": "vLLM service port",
            "port": config.vllm_port,
            "available": is_local_port_available(config.vllm_port),
            "blocking": True,
        },
        {
            "name": "TensorBoard port",
            "port": config.profiler_cfg.tensorboard_port,
            "available": is_local_port_available(config.profiler_cfg.tensorboard_port),
            "blocking": True,
        },
        {
            "name": "Perfetto Trace Processor port",
            "port": settings.PERFETTO_PORT,
            "available": is_local_port_available(settings.PERFETTO_PORT),
            "blocking": False,
        },
    ]
    distributed = config.distributed_cfg
    if distributed is not None and distributed.enable:
        ports.insert(
            2,
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
        ports.insert(
            3,
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
            item["message"] = (
                f"Local port {item['port']} is already in use or cannot be bound"
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
    with ThreadPoolExecutor(max_workers=min(8, len(nodes))) as executor:
        machines = list(
            executor.map(
                lambda node: check_machine(
                    node,
                    [{"label": "SSH", "port": 22}],
                ),
                nodes,
            )
        )
    return {"valid": True, "machines": machines}


def check_config_resources(payload: dict[str, Any]) -> dict[str, Any]:
    model_cfg = payload.get("model_cfg") or {}
    container_cfg = payload.get("container_cfg") or {}
    model_root = str(model_cfg.get("model_path") or "")
    model_name = str(model_cfg.get("model_name") or "")
    image_name = str(container_cfg.get("image_name") or "")
    image_tag = str(container_cfg.get("image_tag") or "")
    model_path = str(Path(model_root) / model_name) if model_root and model_name else ""
    docker_image = f"{image_name}:{image_tag}" if image_name and image_tag else ""

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
        check_docker_image(docker_image),
    ]
    distributed_cfg = payload.get("distributed_cfg") or {}
    worker_nodes = (
        distributed_cfg.get("worker_nodes")
        if distributed_cfg.get("enable")
        and isinstance(distributed_cfg.get("worker_nodes"), list)
        else []
    )
    unique_workers = list(dict.fromkeys(str(node) for node in worker_nodes if node))
    if unique_workers:
        with ThreadPoolExecutor(max_workers=min(8, len(unique_workers))) as executor:
            checks.extend(
                executor.map(
                    lambda node: check_docker_image(docker_image, node),
                    unique_workers,
                )
            )

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

    return {"valid": True, "checks": checks}


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


def check_docker_image(
    image: str,
    hostname: str | None = None,
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
        client = create_docker_client(DockerTarget(hostname=hostname))
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
