import os
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TORCH_PROFILER_DIR = "/app/VAP/log/vllm-profile"
PARALLEL_SIZE_ALIASES = {
    "tensor": ("-tp", "--tensor-parallel-size"),
    "pipeline": ("-pp", "--pipeline-parallel-size"),
    "data": ("-dp", "--data-parallel-size"),
}
RUN_NAME_DEFAULT = "{model}_{parallel}"
RUN_NAME_PLACEHOLDERS = ("{model}", "{parallel}")
RUN_NAME_MAX_LENGTH = 80


def _positive_size(deploy: Dict[str, Any], aliases: tuple[str, ...]) -> int:
    for key in aliases:
        value = deploy.get(key)
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 1


def parallel_sizes(deploy: Dict[str, Any]) -> tuple[int, int, int, int]:
    """TP, PP, DP and EP sizes from vLLM deploy arguments; 1 when unset."""
    tp, pp, dp = (
        _positive_size(deploy, PARALLEL_SIZE_ALIASES[name])
        for name in ("tensor", "pipeline", "data")
    )
    ep = tp * dp if "--enable-expert-parallel" in deploy or "-ep" in deploy else 1
    return tp, pp, dp, ep


def expand_run_name(payload: Dict[str, Any]) -> str:
    """Suffix of a run's log directory: run_name with {model} (the last part of
    model_name) and {parallel} (for example tp2pp2) filled in, reduced to
    letters, digits, '.', '_' and '-'. Tolerates an invalid config."""
    template = payload.get("run_name", RUN_NAME_DEFAULT)
    if not isinstance(template, str):
        return ""
    model_cfg = payload.get("model_cfg")
    model = model_cfg.get("model_name") if isinstance(model_cfg, dict) else None
    deploy = payload.get("vllm_deploy_cfg")
    tp, pp, dp, ep = parallel_sizes(deploy if isinstance(deploy, dict) else {})
    parallel = (
        "".join(
            f"{name}{size}"
            for name, size in (("tp", tp), ("pp", pp), ("dp", dp))
            if size > 1
        )
        or "tp1"
    )
    if ep > 1:
        parallel += f"ep{ep}"
    name = template.replace("{model}", str(model or "").rstrip("/").rsplit("/", 1)[-1])
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name.replace("{parallel}", parallel))
    return name.strip("-._")[:RUN_NAME_MAX_LENGTH].strip("-._")


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictBaseModel):
    model_name: str
    model_path: str


LEGACY_DISTRIBUTED_FIELDS = ("num_nodes", "head_node")


def is_legacy_distributed_cfg(data: Any) -> bool:
    """distributed_cfg as written before distributed runs were implemented."""
    return (
        isinstance(data, dict)
        and "enable" not in data
        and any(key in data for key in LEGACY_DISTRIBUTED_FIELDS)
    )


class DistributedConfig(StrictBaseModel):
    enable: bool
    ray_port: int = Field(ge=1, le=65535)
    worker_nodes: List[str]
    sshkey_path: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_format(cls, data: Any) -> Any:
        if not is_legacy_distributed_cfg(data):
            return data
        # Those versions always ran on the local node only.
        upgraded = {
            key: value
            for key, value in data.items()
            if key not in LEGACY_DISTRIBUTED_FIELDS
        }
        return {**upgraded, "enable": False}


class ClockProbeConfig(StrictBaseModel):
    enabled: bool = False
    apply_clc_on_warning: bool = False
    mode: Literal["auto", "hardware", "software"] = "hardware"
    required: bool = False
    ray_address: str = "auto"
    port: int = Field(default=31990, ge=1, le=65535)
    interval_ms: float = Field(default=100.0, gt=0)
    hardware_interval_ms: float = Field(default=50.0, gt=0)
    hardware_interface: Optional[str] = None
    hardware_phc_device: Optional[str] = None
    hardware_ptp_logs: Dict[str, str] = Field(default_factory=dict)


class ProfilerConfig(StrictBaseModel):
    enable: bool = True
    profiler: str
    torch_profiler_dir: str
    torch_profiler_record_shapes: bool
    torch_profiler_with_stack: bool
    torch_profiler_with_memory: bool
    torch_profiler_with_flops: bool
    torch_profiler_use_gzip: bool
    delay_iterations: int = 0
    max_iterations: int = 0
    tensorboard_port: int = 6006


class MountConfig(StrictBaseModel):
    target: str
    source: str
    type: Optional[str] = "bind"


class DockerConfig(StrictBaseModel):
    image_name: str
    image_tag: str
    devices: Optional[List[str]] = None
    mounts: Optional[List[MountConfig]] = None
    env_vars: Optional[Dict[str, str]] = None


class VAPConfig(StrictBaseModel):
    model_cfg: ModelConfig
    # Log directory suffix: <start time>_<run_name>; empty keeps the start time.
    run_name: str = RUN_NAME_DEFAULT
    distributed_cfg: Optional[DistributedConfig] = None
    clock_probe_cfg: ClockProbeConfig = Field(default_factory=ClockProbeConfig)
    vllm_deploy_cfg: Dict[str, Any]
    vllm_bench_cfg: Dict[str, Any]
    profiler_cfg: ProfilerConfig
    container_cfg: DockerConfig

    @field_validator("run_name")
    @classmethod
    def check_run_name(cls, value: str) -> str:
        literal = value
        for placeholder in RUN_NAME_PLACEHOLDERS:
            literal = literal.replace(placeholder, "")
        if len(value) > RUN_NAME_MAX_LENGTH or not re.fullmatch(
            r"[A-Za-z0-9._-]*", literal
        ):
            raise ValueError(
                "run_name may only use letters, digits, '.', '_', '-' and the "
                f"placeholders {{model}} and {{parallel}}, at most {RUN_NAME_MAX_LENGTH} characters"
            )
        return value

    @property
    def docker_image(self) -> str:
        return f"{self.container_cfg.image_name}:{self.container_cfg.image_tag}"

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_cfg.model_path, self.model_cfg.model_name)

    @property
    def vllm_host(self) -> str:
        if "--host" not in self.vllm_deploy_cfg:
            raise ValueError("vllm_deploy_cfg.--host is required")
        if "--host" not in self.vllm_bench_cfg:
            raise ValueError("vllm_bench_cfg.--host is required")
        if self.vllm_deploy_cfg["--host"] != self.vllm_bench_cfg["--host"]:
            raise ValueError("vLLM deploy and benchmark hosts must match")
        return self.vllm_deploy_cfg["--host"]

    @property
    def vllm_port(self) -> int:
        if "--port" not in self.vllm_deploy_cfg:
            raise ValueError("vllm_deploy_cfg.--port is required")
        if "--port" not in self.vllm_bench_cfg:
            raise ValueError("vllm_bench_cfg.--port is required")
        if self.vllm_deploy_cfg["--port"] != self.vllm_bench_cfg["--port"]:
            raise ValueError("vLLM deploy and benchmark ports must match")
        return self.vllm_deploy_cfg["--port"]

    def build_profiler_cli_args_dict(self) -> dict[str, object]:
        args: dict[str, object] = {}
        if not self.profiler_cfg.enable:
            return args
        for k, v in self.profiler_cfg.model_dump(
            exclude={"enable", "tensorboard_port"}
        ).items():
            if isinstance(v, bool):
                v = str(v).lower()
            args[f"--profiler-config.{k}"] = v
        return args

    def vllm_deploy_args(self) -> list[str]:
        deploy_args = dict(self.vllm_deploy_cfg)
        deploy_args.update(self.build_profiler_cli_args_dict())
        return self.build_cli_args(deploy_args)

    def vllm_bench_args(self) -> list[str]:
        return self.build_cli_args(dict(self.vllm_bench_cfg))

    @property
    def parallel_world_size(self) -> int:
        world_size = 1
        for dimension, aliases in PARALLEL_SIZE_ALIASES.items():
            configured = [
                self.vllm_deploy_cfg[key]
                for key in aliases
                if self.vllm_deploy_cfg.get(key) is not None
            ]
            if not configured:
                continue
            sizes = {int(value) for value in configured}
            if len(sizes) != 1:
                raise ValueError(
                    f"Conflicting {dimension} parallel size aliases: {aliases}"
                )
            size = sizes.pop()
            if size < 1:
                raise ValueError(
                    f"{dimension} parallel size must be a positive integer"
                )
            world_size *= size
        return world_size

    def build_cli_args(self, args_dict: Dict[str, Any]) -> list[str]:
        args: list[str] = []
        for key, value in args_dict.items():
            args.append(str(key))
            if value is not None:
                if isinstance(value, bool):
                    value = str(value).lower()
                args.append(str(value))
        return args
