from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from config import TORCH_PROFILER_DIR, VAPConfig

SHELL_UNSAFE_PATTERN = re.compile(r"[\n\r;&|`$<>]")
ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_VISIBLE_DEVICE_COUNT = 8
PERFETTO_PORT = 9001


def validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        config = VAPConfig.model_validate(payload)
        errors = validate_runtime_config(config)
        warnings = build_security_warnings(config)
        if errors:
            return {
                "valid": False,
                "message": "Config validation failed.",
                "errors": errors,
                "warnings": warnings,
            }
        return {
            "valid": True,
            "message": "Config is valid and can be used for a VAP run.",
            "summary": build_config_summary(config),
            "warnings": warnings,
        }
    except Exception as exc:
        return {
            "valid": False,
            "message": "Config validation failed.",
            "errors": format_validation_error(exc),
            "warnings": [],
        }


def validate_config_or_raise(config: VAPConfig) -> list[dict[str, str]]:
    errors = validate_runtime_config(config)
    if errors:
        message = "; ".join(
            f"{item.get('path', 'root')}: {item.get('message', item)}"
            for item in errors
        )
        raise ValueError(f"Config validation failed: {message}")
    return build_security_warnings(config)


def validate_runtime_config(config: VAPConfig) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    deploy_host = config.vllm_deploy_cfg.get("--host")
    bench_host = config.vllm_bench_cfg.get("--host")
    deploy_port = config.vllm_deploy_cfg.get("--port")
    bench_port = config.vllm_bench_cfg.get("--port")

    if config.profiler_cfg.torch_profiler_dir != TORCH_PROFILER_DIR:
        errors.append(
            {
                "path": "profiler_cfg.torch_profiler_dir",
                "message": (
                    "torch_profiler_dir is immutable and must remain "
                    f"{TORCH_PROFILER_DIR}"
                ),
            }
        )

    if deploy_host is None:
        errors.append(
            {"path": "vllm_deploy_cfg.--host", "message": "Missing vLLM deploy host"}
        )
    if bench_host is None:
        errors.append(
            {"path": "vllm_bench_cfg.--host", "message": "Missing vLLM benchmark host"}
        )
    if deploy_host is not None and bench_host is not None and deploy_host != bench_host:
        errors.append(
            {
                "path": "vllm_deploy_cfg.--host",
                "message": "Deploy host must match benchmark host",
            }
        )

    if deploy_port is None:
        errors.append(
            {"path": "vllm_deploy_cfg.--port", "message": "Missing vLLM deploy port"}
        )
    if bench_port is None:
        errors.append(
            {"path": "vllm_bench_cfg.--port", "message": "Missing vLLM benchmark port"}
        )
    if deploy_port is not None and bench_port is not None and deploy_port != bench_port:
        errors.append(
            {
                "path": "vllm_deploy_cfg.--port",
                "message": "Deploy port must match benchmark port",
            }
        )
    if deploy_port is not None and not is_valid_port(deploy_port):
        errors.append(
            {
                "path": "vllm_deploy_cfg.--port",
                "message": "Port must be an integer from 1 to 65535",
            }
        )

    if not is_valid_port(config.profiler_cfg.tensorboard_port):
        errors.append(
            {
                "path": "profiler_cfg.tensorboard_port",
                "message": "TensorBoard port must be an integer from 1 to 65535",
            }
        )

    local_vllm_port = (
        deploy_port
        if is_valid_port(deploy_port) and deploy_port == bench_port
        else None
    )
    errors.extend(validate_local_service_port_conflicts(config, local_vllm_port))
    errors.extend(validate_tensor_parallel_devices(config))
    errors.extend(validate_risky_config(config))
    return errors


def validate_local_service_port_conflicts(
    config: VAPConfig,
    local_vllm_port: int | None,
) -> list[dict[str, str]]:
    ports = [
        (
            "profiler_cfg.tensorboard_port",
            "TensorBoard port",
            config.profiler_cfg.tensorboard_port,
        ),
    ]
    if local_vllm_port is not None:
        ports.insert(
            0, ("vllm_deploy_cfg.--port", "vLLM service port", local_vllm_port)
        )

    errors: list[dict[str, str]] = []
    seen: dict[int, tuple[str, str]] = {}
    for path, name, port in ports:
        if not is_valid_port(port):
            continue
        if port in seen:
            previous_path, previous_name = seen[port]
            errors.append(
                {
                    "path": path,
                    "message": f"{name} conflicts with {previous_name}; both use local port {port}",
                }
            )
            errors.append(
                {
                    "path": previous_path,
                    "message": f"{previous_name} conflicts with {name}; both use local port {port}",
                }
            )
        else:
            seen[port] = (path, name)
    return errors


def validate_tensor_parallel_devices(config: VAPConfig) -> list[dict[str, str]]:
    tp_value = config.vllm_deploy_cfg.get("-tp")
    if tp_value is None:
        return []

    try:
        tensor_parallel_size = int(tp_value)
    except (TypeError, ValueError):
        return [
            {
                "path": "vllm_deploy_cfg.-tp",
                "message": "-tp must be a positive integer",
            }
        ]

    if tensor_parallel_size < 1:
        return [
            {
                "path": "vllm_deploy_cfg.-tp",
                "message": "-tp must be a positive integer",
            }
        ]

    devices = config.container_cfg.devices or []
    visible_device_count = len(devices) if devices else DEFAULT_VISIBLE_DEVICE_COUNT
    if tensor_parallel_size > visible_device_count:
        return [
            {
                "path": "vllm_deploy_cfg.-tp",
                "message": (
                    f"-tp={tensor_parallel_size} exceeds visible GPU device count "
                    f"{visible_device_count}. Empty devices means all "
                    f"{DEFAULT_VISIBLE_DEVICE_COUNT} GPUs are visible."
                ),
            }
        ]
    return []


def validate_risky_config(config: VAPConfig) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    model_name = config.model_cfg.model_name
    if os.path.isabs(model_name) or ".." in Path(model_name).parts:
        errors.append(
            {
                "path": "model_cfg.model_name",
                "message": "Model name cannot be an absolute path or contain '..'",
            }
        )
    if has_shell_unsafe_chars(model_name):
        errors.append(
            {
                "path": "model_cfg.model_name",
                "message": "Model name contains shell-unsafe characters",
            }
        )

    image = config.docker_image
    if has_shell_unsafe_chars(image) or any(ch.isspace() for ch in image):
        errors.append(
            {
                "path": "container_cfg.image_name",
                "message": "Docker image name or tag cannot contain whitespace or shell-unsafe characters",
            }
        )

    for cfg_name, cfg in (
        ("vllm_deploy_cfg", config.vllm_deploy_cfg),
        ("vllm_bench_cfg", config.vllm_bench_cfg),
        ("profiler_cfg", config.build_profiler_cli_args_dict()),
    ):
        errors.extend(validate_cli_args(cfg_name, cfg))

    for key, value in (config.container_cfg.env_vars or {}).items():
        if not ENV_KEY_PATTERN.match(key):
            errors.append(
                {
                    "path": f"container_cfg.env_vars.{key}",
                    "message": "Environment variable keys can only contain letters, digits, and underscores, and cannot start with a digit",
                }
            )
        if has_shell_unsafe_chars(value):
            errors.append(
                {
                    "path": f"container_cfg.env_vars.{key}",
                    "message": "Environment variable value contains newlines or shell-unsafe characters",
                }
            )

    for index, mount in enumerate(config.container_cfg.mounts or []):
        if not os.path.isabs(mount.source):
            errors.append(
                {
                    "path": f"container_cfg.mounts.{index}.source",
                    "message": "Host mount source must be an absolute path",
                }
            )
        if not os.path.isabs(mount.target):
            errors.append(
                {
                    "path": f"container_cfg.mounts.{index}.target",
                    "message": "Container mount target must be an absolute path",
                }
            )
        if has_shell_unsafe_chars(mount.source) or has_shell_unsafe_chars(mount.target):
            errors.append(
                {
                    "path": f"container_cfg.mounts.{index}",
                    "message": "Mount path contains shell-unsafe characters",
                }
            )

    for index, device in enumerate(config.container_cfg.devices or []):
        if not os.path.isabs(device):
            errors.append(
                {
                    "path": f"container_cfg.devices.{index}",
                    "message": "Device path must be an absolute path",
                }
            )
        if has_shell_unsafe_chars(device):
            errors.append(
                {
                    "path": f"container_cfg.devices.{index}",
                    "message": "Device path contains shell-unsafe characters",
                }
            )

    return errors


def validate_cli_args(cfg_name: str, cfg: dict[str, Any]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for key, value in cfg.items():
        if not key.startswith("-"):
            errors.append(
                {
                    "path": f"{cfg_name}.{key}",
                    "message": "CLI argument key must start with '-'",
                }
            )
        if has_shell_unsafe_chars(key) or any(ch.isspace() for ch in key):
            errors.append(
                {
                    "path": f"{cfg_name}.{key}",
                    "message": "CLI argument key cannot contain whitespace or shell-unsafe characters",
                }
            )
        if value is not None and isinstance(value, str):
            if has_shell_unsafe_chars(value):
                errors.append(
                    {
                        "path": f"{cfg_name}.{key}",
                        "message": "CLI argument value contains shell-unsafe characters",
                    }
                )
            if any(ch.isspace() for ch in value):
                errors.append(
                    {
                        "path": f"{cfg_name}.{key}",
                        "message": "CLI argument value does not currently support whitespace",
                    }
                )
    return errors


def build_security_warnings(config: VAPConfig) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    conflicting_services: list[str] = []
    if config.vllm_deploy_cfg.get("--port") == PERFETTO_PORT:
        conflicting_services.append("vLLM")
    if config.profiler_cfg.tensorboard_port == PERFETTO_PORT:
        conflicting_services.append("TensorBoard")
    if conflicting_services:
        warnings.append(
            {
                "path": "perfetto.port",
                "message": (
                    f"Perfetto port {PERFETTO_PORT} conflicts with "
                    f"{' and '.join(conflicting_services)}; Perfetto visualization "
                    "will be skipped."
                ),
            }
        )
    if config.distributed_cfg is not None:
        warnings.append(
            {
                "path": "distributed_cfg",
                "message": "Distributed mode is unavailable; VAP will run locally.",
            }
        )
    if "--trust-remote-code" in config.vllm_deploy_cfg:
        warnings.append(
            {
                "path": "vllm_deploy_cfg.--trust-remote-code",
                "message": "--trust-remote-code is enabled; trust the model source.",
            }
        )
    warnings.append(
        {
            "path": "container_cfg.runtime",
            "message": "Host networking and elevated container permissions are enabled.",
        }
    )
    return warnings


def has_shell_unsafe_chars(value: Any) -> bool:
    return isinstance(value, str) and bool(SHELL_UNSAFE_PATTERN.search(value))


def is_valid_port(port: Any) -> bool:
    return isinstance(port, int) and 1 <= port <= 65535


def format_validation_error(exc: Exception) -> list[dict[str, str]]:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        return [
            {
                "path": ".".join(str(part) for part in item.get("loc", [])) or "root",
                "message": item.get("msg", str(exc)),
            }
            for item in errors()
        ]
    return [{"path": "root", "message": str(exc)}]


def build_config_summary(config: VAPConfig) -> dict[str, Any]:
    distributed = config.distributed_cfg
    return {
        "model": config.model_cfg.model_name,
        "model_path": config.model_path,
        "docker_image": config.docker_image,
        "vllm_host": config.vllm_host,
        "vllm_port": config.vllm_port,
        "distributed": bool(distributed),
        "node_count": distributed.num_nodes if distributed else 1,
    }
