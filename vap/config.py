import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

TORCH_PROFILER_DIR = "/app/VAP/log/vllm-profile"
PARALLEL_SIZE_ALIASES = {
    "tensor": ("-tp", "--tensor-parallel-size"),
    "pipeline": ("-pp", "--pipeline-parallel-size"),
    "data": ("-dp", "--data-parallel-size"),
}


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
    distributed_cfg: Optional[DistributedConfig] = None
    clock_probe_cfg: ClockProbeConfig = Field(default_factory=ClockProbeConfig)
    vllm_deploy_cfg: Dict[str, Any]
    vllm_bench_cfg: Dict[str, Any]
    profiler_cfg: ProfilerConfig
    container_cfg: DockerConfig

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
