import os
import shlex
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

TORCH_PROFILER_DIR = "/app/VAP/log/vllm-profile"


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictBaseModel):
    model_name: str
    model_path: str


class DistributedConfig(StrictBaseModel):
    num_nodes: int
    ray_port: int
    head_node: Literal["localhost"]
    worker_nodes: List[str]


class ProfilerConfig(StrictBaseModel):
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
        for k, v in self.profiler_cfg.model_dump(exclude={"tensorboard_port"}).items():
            if isinstance(v, bool):
                v = str(v).lower()
            args[f"--profiler-config.{k}"] = v
        return args

    def vllm_deploy_args(self) -> list[str]:
        deploy_args = dict(self.vllm_deploy_cfg)
        deploy_args.update(self.build_profiler_cli_args_dict())
        return self.build_cli_args(deploy_args)

    def vllm_deploy_args_str(self) -> str:
        return shlex.join(self.vllm_deploy_args())

    def vllm_bench_args(self) -> list[str]:
        return self.build_cli_args(dict(self.vllm_bench_cfg))

    def vllm_bench_args_str(self) -> str:
        return shlex.join(self.vllm_bench_args())

    def build_cli_args(self, args_dict: Dict[str, Any]) -> list[str]:
        args: list[str] = []
        for key, value in args_dict.items():
            args.append(str(key))
            if value is not None:
                if isinstance(value, bool):
                    value = str(value).lower()
                args.append(str(value))
        return args

    def build_cli_rgs_str(self, args_dict: Dict[str, Any]) -> str:
        return shlex.join(self.build_cli_args(args_dict))
