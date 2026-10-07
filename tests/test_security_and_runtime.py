from __future__ import annotations

import io
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import types
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlparse


def install_docker_stub() -> None:
    if "docker" in sys.modules:
        return

    docker_module = types.ModuleType("docker")
    docker_module.__path__ = []
    docker_types = types.ModuleType("docker.types")
    docker_errors = types.ModuleType("docker.errors")
    docker_models = types.ModuleType("docker.models")
    docker_models.__path__ = []
    docker_containers = types.ModuleType("docker.models.containers")

    class DockerClient:
        pass

    class DockerException(Exception):
        pass

    class ImageNotFound(DockerException):
        pass

    class Mount:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Ulimit:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    docker_module.DockerClient = DockerClient
    docker_module.from_env = Mock()
    docker_errors.DockerException = DockerException
    docker_errors.ImageNotFound = ImageNotFound
    docker_module.errors = docker_errors
    docker_containers.Container = object
    docker_models.containers = docker_containers
    docker_module.models = docker_models
    docker_types.Mount = Mount
    docker_types.Ulimit = Ulimit
    sys.modules["docker"] = docker_module
    sys.modules["docker.types"] = docker_types
    sys.modules["docker.errors"] = docker_errors
    sys.modules["docker.models"] = docker_models
    sys.modules["docker.models.containers"] = docker_containers


install_docker_stub()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from vap import cli, config, main, runtime_paths, validation, visualization
from vap.agent import analysis as agent_analysis
from vap.agent import runtime as agent_runtime
from vap.agent import tools as agent_tools
from vap.clock_probe.execution import ray as clock_ray
from vap.pipelines import torch_profiling_pipeline as torch_pipeline
from vap.pipelines.services import benchmark as benchmark_service
from vap.pipelines.services import clock_probe as clock_probe_service
from vap.pipelines.services import profiler as profiler_service
from vap.pipelines.services import ray_cluster as ray_cluster_service
from vap.pipelines.services import trace_postprocessor as trace_service
from vap.runners import CommandResult
from vap.server import artifacts as server_artifacts
from vap.server import auth as server_auth
from vap.server import checks as server_checks
from vap.server import handler as server_handler
from vap.server import settings as server_settings
from vap.server import state as server_state


def example_payload() -> dict:
    return json.loads(
        (PROJECT_ROOT / "example-config.json").read_text(encoding="utf-8")
    )


class ConfigSecurityTests(unittest.TestCase):
    def test_cli_values_are_preserved_as_single_tokens(self) -> None:
        payload = example_payload()
        payload["profiler_cfg"]["profiler"] = "torch; touch /tmp/not-created"
        parsed = config.VAPConfig.model_validate(payload)

        tokens = parsed.vllm_deploy_args()

        index = tokens.index("--profiler-config.profiler")
        self.assertEqual(tokens[index + 1], "torch; touch /tmp/not-created")

    def test_unknown_config_fields_are_rejected(self) -> None:
        payload = example_payload()
        payload["profiler_cfg"]["tensorboard_poort"] = 7777

        result = validation.validate_config_payload(payload)

        self.assertFalse(result["valid"])
        self.assertTrue(
            any("tensorboard_poort" in error["path"] for error in result["errors"])
        )

    def test_profiler_shell_characters_are_rejected(self) -> None:
        payload = example_payload()
        payload["profiler_cfg"]["torch_profiler_dir"] = "/tmp/profile;touch /tmp/pwn"

        result = validation.validate_config_payload(payload)

        self.assertFalse(result["valid"])
        self.assertTrue(
            any("shell-unsafe" in error["message"] for error in result["errors"])
        )

    def test_torch_profiler_dir_is_immutable(self) -> None:
        payload = example_payload()
        payload["profiler_cfg"]["torch_profiler_dir"] = "/tmp/other-profile"

        result = validation.validate_config_payload(payload)

        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                error["path"] == "profiler_cfg.torch_profiler_dir"
                and "immutable" in error["message"]
                for error in result["errors"]
            )
        )

    def test_distributed_config_is_supported(self) -> None:
        payload = example_payload()
        payload["distributed_cfg"] = {
            "enable": True,
            "ray_port": 6379,
            "worker_nodes": ["worker.example"],
        }

        result = validation.validate_config_payload(payload)

        self.assertTrue(result["valid"])
        self.assertTrue(result["summary"]["distributed"])
        self.assertFalse(
            any(warning["path"] == "distributed_cfg" for warning in result["warnings"])
        )

    def test_clock_probe_hardware_mode_does_not_require_static_phc_fields(self) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = True
        payload["clock_probe_cfg"]["mode"] = "hardware"
        payload["clock_probe_cfg"]["hardware_interface"] = None
        payload["clock_probe_cfg"]["hardware_phc_device"] = None

        result = validation.validate_config_payload(payload)

        self.assertTrue(result["valid"])

    def test_model_resource_check_does_not_wait_on_worker_docker(self) -> None:
        payload = example_payload()
        with patch.object(server_checks, "check_docker_image") as check_image:
            result = server_checks.check_config_model_resources(payload)

        check_image.assert_not_called()
        self.assertTrue(
            all(check["name"].startswith("Model") for check in result["checks"])
        )

    def test_clock_probe_check_skips_hardware_scan_when_disabled(self) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = False
        with patch.object(
            server_checks,
            "inspect_node_hardware_timestamping",
        ) as inspect:
            result = server_checks.check_config_clock_probe(payload)

        inspect.assert_not_called()
        self.assertTrue(result["valid"])
        self.assertEqual(result["message"], "Clock probe is disabled")

    def test_clock_probe_hardware_check_blocks_unusable_nodes(self) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = True
        payload["clock_probe_cfg"]["mode"] = "hardware"

        def inspect(*, node: str, **kwargs):
            return {
                "node": node,
                "usable": node == "local",
                "message": f"{node}: ok" if node == "local" else f"{node}: no PHC",
            }

        with patch.object(
            server_checks,
            "inspect_node_hardware_timestamping",
            side_effect=inspect,
        ):
            result = server_checks.check_config_clock_probe(payload)

        self.assertFalse(result["valid"])
        self.assertTrue(any("no PHC" in item["message"] for item in result["nodes"]))
        self.assertEqual(result["warnings"], [])

    def test_clock_probe_hardware_unreadable_phc_still_passes(self) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = True
        payload["clock_probe_cfg"]["mode"] = "hardware"

        def inspect(*, node: str, **kwargs):
            self.assertFalse(kwargs["require_readable"])
            self.assertFalse(kwargs["capture"])
            return {
                "node": node,
                "usable": True,
                "readable": False,
                "phc_device": "/dev/ptp0",
                "interface": "enp4s0f0np0",
                "message": (
                    f"{node}: enp4s0f0np0 -> /dev/ptp0 "
                    "(not readable as this user; the run container will mount it)"
                ),
            }

        with patch.object(
            server_checks,
            "inspect_node_hardware_timestamping",
            side_effect=inspect,
        ):
            result = server_checks.check_config_clock_probe(payload)

        self.assertTrue(result["valid"])
        self.assertEqual(result["warnings"], [])
        self.assertIn("not readable as this user", result["message"])

    def test_clock_probe_auto_mode_does_not_block_on_hardware_failure(self) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = True
        payload["clock_probe_cfg"]["mode"] = "auto"

        with patch.object(
            server_checks,
            "inspect_node_hardware_timestamping",
            return_value={"node": "local", "usable": False, "message": "no PHC"},
        ):
            result = server_checks.check_config_clock_probe(payload)

        self.assertTrue(result["valid"])
        self.assertEqual(result["mode"], "auto")

    def test_clock_probe_check_requires_distributed_enable(self) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = True
        payload["distributed_cfg"]["enable"] = False

        result = server_checks.check_config_clock_probe(payload)

        self.assertFalse(result["valid"])
        self.assertIn("distributed_cfg.enable=true", result["message"])

    def test_worker_container_keeps_that_host_phc_devices(self) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = True
        parsed = config.VAPConfig.model_validate(payload)
        pipeline = torch_pipeline.TorchProfilingPipeline.__new__(
            torch_pipeline.TorchProfilingPipeline
        )
        pipeline.config = parsed
        pipeline.date_str = "20260101"
        pipeline.clock_probe = Mock()
        pipeline.clock_probe.enabled = True

        with tempfile.TemporaryDirectory() as tmp:
            pipeline.log_path = tmp
            with patch.object(
                torch_pipeline,
                "list_node_phc_devices",
                return_value=["/dev/ptp2"],
            ) as list_phc:
                spec = pipeline._container_spec(
                    include_clock_source=False,
                    hostname="cse-ai-10.amd.com",
                )

        self.assertIn("/dev/ptp2", spec.devices)
        list_phc.assert_called_once_with(
            "cse-ai-10.amd.com",
            ssh_key=parsed.distributed_cfg.sshkey_path,
        )

    def test_clock_probe_requires_distributed_mode(self) -> None:
        payload = example_payload()
        payload["distributed_cfg"] = None
        payload["clock_probe_cfg"]["enabled"] = True

        result = validation.validate_config_payload(payload)

        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                error["path"] == "clock_probe_cfg.enabled" for error in result["errors"]
            )
        )

    def test_parallel_world_size_supports_long_aliases(self) -> None:
        payload = example_payload()
        payload["vllm_deploy_cfg"].pop("-tp", None)
        payload["vllm_deploy_cfg"].pop("-pp", None)
        payload["vllm_deploy_cfg"]["--tensor-parallel-size"] = 2
        payload["vllm_deploy_cfg"]["--pipeline-parallel-size"] = 3
        payload["vllm_deploy_cfg"]["--data-parallel-size"] = 2

        parsed = config.VAPConfig.model_validate(payload)

        self.assertEqual(parsed.parallel_world_size, 12)

    def test_conflicting_parallel_aliases_are_rejected(self) -> None:
        payload = example_payload()
        payload["vllm_deploy_cfg"]["--tensor-parallel-size"] = 4

        result = validation.validate_config_payload(payload)

        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                error["path"] == "vllm_deploy_cfg.parallel_sizes"
                for error in result["errors"]
            )
        )

    def test_security_warnings_do_not_block_compatible_config(self) -> None:
        result = validation.validate_config_payload(example_payload())

        self.assertTrue(result["valid"])
        self.assertGreaterEqual(len(result["warnings"]), 2)
        self.assertTrue(
            any(
                "trust-remote-code" in warning["path"] for warning in result["warnings"]
            )
        )

    def test_perfetto_port_conflict_is_warning_only(self) -> None:
        payload = example_payload()
        payload["vllm_deploy_cfg"]["--port"] = validation.PERFETTO_PORT
        payload["vllm_bench_cfg"]["--port"] = validation.PERFETTO_PORT

        result = validation.validate_config_payload(payload)

        self.assertTrue(result["valid"])
        self.assertTrue(
            any(warning["path"] == "perfetto.port" for warning in result["warnings"])
        )


class ServerAuthorizationTests(unittest.TestCase):
    def make_handler(self, headers: dict[str, str]):
        handler = server_handler.VAPConfigHandler.__new__(
            server_handler.VAPConfigHandler
        )
        handler.headers = headers
        handler.send_json = Mock()
        return handler

    def test_valid_header_token_and_same_origin_are_allowed(self) -> None:
        handler = self.make_handler(
            {
                "X-VAP-Token": server_settings.SERVER_AUTH_TOKEN,
                "Origin": "http://127.0.0.1:8899",
                "Host": "127.0.0.1:8899",
                "Content-Type": "application/json",
            }
        )

        allowed = handler.require_authorized(
            urlparse("/api/run"),
            require_json=True,
            require_same_origin=True,
        )

        self.assertTrue(allowed)
        handler.send_json.assert_not_called()

    def test_bad_token_is_rejected(self) -> None:
        handler = self.make_handler({"X-VAP-Token": "wrong"})

        allowed = handler.require_authorized(urlparse("/api/run"))

        self.assertFalse(allowed)
        handler.send_json.assert_called_once()

    def test_cross_origin_write_is_rejected(self) -> None:
        handler = self.make_handler(
            {
                "X-VAP-Token": server_settings.SERVER_AUTH_TOKEN,
                "Origin": "https://attacker.example",
                "Host": "127.0.0.1:8899",
                "Content-Type": "application/json",
            }
        )

        allowed = handler.require_authorized(
            urlparse("/api/run"),
            require_json=True,
            require_same_origin=True,
        )

        self.assertFalse(allowed)
        handler.send_json.assert_called_once()

    def test_cookie_parser_handles_multiple_values(self) -> None:
        cookies = server_auth.parse_cookie_header(
            f"one=1; {server_settings.SERVER_COOKIE_NAME}=secret; two=2"
        )
        self.assertEqual(cookies[server_settings.SERVER_COOKIE_NAME], "secret")

    def test_wildcard_bind_prints_local_hostname_and_ip_candidates(self) -> None:
        with patch.object(
            server_auth,
            "discover_network_hosts",
            return_value=["vap-host.example", "10.0.0.8"],
        ):
            urls = server_auth.build_session_urls("0.0.0.0", 8899, "session-token")

        self.assertEqual(
            urls,
            [
                (
                    "Local",
                    "http://127.0.0.1:8899/?token=session-token",
                ),
                (
                    "Network candidate",
                    "http://vap-host.example:8899/?token=session-token",
                ),
                (
                    "Network candidate",
                    "http://10.0.0.8:8899/?token=session-token",
                ),
            ],
        )

    def test_perfetto_port_check_is_non_blocking(self) -> None:
        with (
            patch.object(
                server_checks,
                "is_local_port_available",
                side_effect=lambda port: port != server_settings.PERFETTO_PORT,
            ),
            patch.object(server_checks, "sweep_vap_containers"),
        ):
            result = server_checks.check_config_ports(example_payload())

        perfetto = next(
            port
            for port in result["ports"]
            if port["name"] == "Perfetto Trace Processor port"
        )
        self.assertTrue(result["valid"])
        self.assertFalse(perfetto["available"])
        self.assertFalse(perfetto["blocking"])
        self.assertIn("will be skipped", perfetto["message"])

    def test_disabled_profiler_skips_visualization_port_checks(self) -> None:
        payload = example_payload()
        payload["profiler_cfg"]["enable"] = False
        with (
            patch.object(server_checks, "is_local_port_available", return_value=True),
            patch.object(server_checks, "sweep_vap_containers"),
        ):
            result = server_checks.check_config_ports(payload)

        names = [port["name"] for port in result["ports"]]
        self.assertTrue(result["valid"])
        self.assertIn("vLLM service port", names)
        self.assertNotIn("TensorBoard port", names)
        self.assertNotIn("Perfetto Trace Processor port", names)

    def test_idle_port_check_sweeps_leftover_vap_containers(self) -> None:
        payload = example_payload()
        payload["profiler_cfg"]["enable"] = False
        with (
            patch.object(server_checks, "_vap_run_is_active", return_value=False),
            patch.object(server_checks, "sweep_vap_containers") as sweep,
            patch.object(server_checks, "is_local_port_available", return_value=True),
        ):
            result = server_checks.check_config_ports(payload)

        sweep.assert_called_once()
        self.assertTrue(result["valid"])

    def test_active_run_port_check_does_not_sweep_containers(self) -> None:
        payload = example_payload()
        payload["profiler_cfg"]["enable"] = False
        with (
            patch.object(server_checks, "_vap_run_is_active", return_value=True),
            patch.object(server_checks, "sweep_vap_containers") as sweep,
            patch.object(server_checks, "is_local_port_available", return_value=True),
        ):
            server_checks.check_config_ports(payload)

        sweep.assert_not_called()

    def test_local_port_available_detects_listener(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", 0))
        port = listener.getsockname()[1]
        listener.listen(1)
        try:
            self.assertFalse(server_checks.is_local_port_available(port))
            occupant = server_checks.describe_listening_port(port)
            self.assertIsNotNone(occupant)
            self.assertIn(str(os.getpid()), occupant or "")
        finally:
            listener.close()
        self.assertTrue(server_checks.is_local_port_available(port))

    def test_remote_docker_image_uses_ssh_inspect(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=0,
            stdout="sha256:abc\n",
            stderr="",
        )
        with patch.object(
            server_checks.subprocess, "run", return_value=completed
        ) as run:
            result = server_checks.check_docker_image(
                "example/image:tag",
                "worker.example",
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["node"], "worker.example")
        self.assertIn("docker", run.call_args.args[0])
        self.assertIn("image", run.call_args.args[0])
        self.assertIn("inspect", run.call_args.args[0])
        self.assertIn("worker.example", run.call_args.args[0])

    def test_remote_docker_ssh_failure_is_not_reported_as_missing_image(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=255,
            stdout="",
            stderr="Permission denied (publickey,password).\n",
        )
        with patch.object(server_checks.subprocess, "run", return_value=completed):
            result = server_checks.check_docker_image(
                "example/image:tag",
                "worker.example",
            )

        self.assertFalse(result["ok"])
        self.assertIn("SSH to worker.example failed", result["message"])
        self.assertNotIn("does not exist", result["message"])

    def test_missing_ssh_key_falls_back_to_default_identity(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=0,
            stdout="sha256:abc\n",
            stderr="",
        )
        with patch.object(
            server_checks.subprocess, "run", return_value=completed
        ) as run:
            result = server_checks.check_docker_image(
                "example/image:tag",
                "worker.example",
                ssh_key="/missing/id_ed25519",
            )

        self.assertTrue(result["ok"])
        self.assertIn("SSH key /missing/id_ed25519 is missing", result["message"])
        self.assertNotIn("-i", run.call_args.args[0])

    def test_resource_checks_include_all_worker_docker_images(self) -> None:
        payload = example_payload()

        def image_result(image: str, hostname: str | None = None, **kwargs) -> dict:
            node = hostname or "local"
            return {
                "name": f"Docker image ({node})",
                "ok": True,
                "message": f"Docker image exists on {node}",
                "image": image,
                "node": node,
            }

        with patch.object(
            server_checks,
            "check_docker_image",
            side_effect=image_result,
        ):
            result = server_checks.check_config_resources(payload)

        image_nodes = {
            check["node"]
            for check in result["checks"]
            if check["name"].startswith("Docker image")
        }
        self.assertEqual(
            image_nodes,
            {"local", *payload["distributed_cfg"]["worker_nodes"]},
        )

    def test_local_docker_check_does_not_wait_on_workers(self) -> None:
        payload = example_payload()
        with patch.object(
            server_checks,
            "check_docker_image",
            return_value={
                "name": "Docker image (local)",
                "ok": True,
                "message": "Docker image exists on local",
                "image": "example:tag",
                "node": "local",
            },
        ) as check_image:
            result = server_checks.check_config_docker_resources(payload)

        self.assertEqual(len(result["checks"]), 1)
        self.assertEqual(result["checks"][0]["node"], "local")
        for call in check_image.call_args_list:
            self.assertTrue(len(call.args) < 2 or call.args[1] is None)
            self.assertIsNone(call.kwargs.get("hostname"))

    def test_worker_docker_timeout_does_not_block_validation(self) -> None:
        payload = example_payload()

        def hang(image: str, hostname: str | None = None, **kwargs):
            time.sleep(5)
            return {
                "name": f"Docker image ({hostname})",
                "ok": True,
                "message": "late",
                "image": image,
                "node": hostname,
            }

        with (
            patch.object(server_checks, "DOCKER_IMAGE_CHECK_TIMEOUT_SEC", 0.2),
            patch.object(server_checks, "check_docker_image", side_effect=hang),
        ):
            started = time.monotonic()
            result = server_checks.check_config_worker_docker_resources(payload)
            elapsed = time.monotonic() - started

        self.assertLess(elapsed, 1.5)
        self.assertFalse(result["valid"])
        self.assertTrue(
            all("timed out" in check["message"] for check in result["checks"])
        )

    def test_machine_checks_include_all_enabled_workers(self) -> None:
        payload = example_payload()

        def machine_result(node: str, checks: list[dict]) -> dict:
            return {
                "node": node,
                "reachable": True,
                "ip": "192.168.0.10",
                "checks": checks,
                "message": "Machine is reachable",
            }

        with patch.object(
            server_checks,
            "check_machine",
            side_effect=machine_result,
        ) as check_machine:
            result = server_checks.check_config_machines(payload)

        self.assertTrue(result["valid"])
        self.assertEqual(
            {machine["node"] for machine in result["machines"]},
            set(payload["distributed_cfg"]["worker_nodes"]),
        )
        self.assertEqual(check_machine.call_count, 2)

    def test_query_token_is_only_accepted_on_session_entrypoint(self) -> None:
        entry_handler = self.make_handler({})
        api_handler = self.make_handler({})

        self.assertTrue(
            entry_handler.is_authenticated(
                urlparse(f"/?token={server_settings.SERVER_AUTH_TOKEN}")
            )
        )
        self.assertFalse(
            api_handler.is_authenticated(
                urlparse(f"/api/run/status?token={server_settings.SERVER_AUTH_TOKEN}")
            )
        )

    def test_start_lock_rejects_concurrent_start(self) -> None:
        self.assertTrue(server_settings.RUN_START_LOCK.acquire(blocking=False))
        try:
            with self.assertRaisesRegex(RuntimeError, "already starting"):
                server_state.start_vap_run()
        finally:
            server_settings.RUN_START_LOCK.release()

    def test_stop_request_is_remembered_during_startup(self) -> None:
        with patch.dict(
            server_settings.RUN_STATE,
            {
                "process": None,
                "running": True,
                "run_dir": None,
                "output": "",
                "stop_requested": False,
            },
        ):
            result = server_state.stop_vap_run()

            self.assertTrue(result["stop_requested"])
            self.assertIn("as soon as it starts", result["message"])

    def test_second_stop_does_not_signal_again(self) -> None:
        process = Mock()
        process.pid = 4321
        process.poll.return_value = None
        with (
            patch.dict(
                server_settings.RUN_STATE,
                {
                    "process": process,
                    "pid": process.pid,
                    "running": True,
                    "run_dir": None,
                    "output": "",
                    "stop_requested": False,
                },
            ),
            patch.object(server_state, "terminate_run_process") as terminate,
            patch.object(server_state, "force_kill_process_group_later") as force_kill,
            patch.object(server_state, "terminate_recorded_visualization_pids"),
        ):
            first = server_state.stop_vap_run()
            second = server_state.stop_vap_run()

        terminate.assert_called_once_with(process)
        force_kill.assert_called_once_with(
            process, timeout_sec=server_state.STOP_CLEANUP_GRACE_SEC
        )
        self.assertIn("Stop signal sent", first["message"])
        self.assertIn("already in progress", second["message"])
        self.assertTrue(second["stop_requested"])

    def test_server_exit_uses_full_cleanup_grace_period(self) -> None:
        process = Mock()
        process.pid = 4321
        process.poll.return_value = None
        with (
            patch.object(server_settings, "SHUTDOWN_CLEANUP_DONE", False),
            patch.dict(
                server_settings.RUN_STATE,
                {
                    "process": process,
                    "running": True,
                    "run_dir": None,
                    "output": "",
                    "stop_requested": False,
                },
            ),
            patch.object(
                server_state,
                "stop_process_group_sync",
                return_value=True,
            ) as stop_group,
            patch.object(server_state, "terminate_recorded_visualization_pids"),
            patch.object(server_state, "clear_active_run_record") as clear_record,
        ):
            server_state.cleanup_active_run_on_server_exit()

        stop_group.assert_called_once_with(
            process, timeout_sec=server_state.STOP_CLEANUP_GRACE_SEC
        )
        clear_record.assert_called_once()

    def test_server_exit_keeps_active_record_when_cleanup_times_out(self) -> None:
        process = Mock()
        process.pid = 4321
        process.poll.return_value = None
        with (
            patch.object(server_settings, "SHUTDOWN_CLEANUP_DONE", False),
            patch.dict(
                server_settings.RUN_STATE,
                {
                    "process": process,
                    "running": True,
                    "run_dir": None,
                    "output": "",
                    "stop_requested": False,
                },
            ),
            patch.object(
                server_state,
                "stop_process_group_sync",
                return_value=False,
            ),
            patch.object(server_state, "terminate_recorded_visualization_pids"),
            patch.object(server_state, "clear_active_run_record") as clear_record,
        ):
            server_state.cleanup_active_run_on_server_exit()

        clear_record.assert_not_called()

    def test_agent_cannot_start_with_modified_torch_profiler_dir(self) -> None:
        payload = example_payload()
        payload["profiler_cfg"]["torch_profiler_dir"] = "/tmp/other-profile"

        with (
            patch.object(agent_tools, "save_temp_config") as save_temp,
            patch.object(agent_tools, "start_vap_run") as start_run,
            self.assertRaisesRegex(ValueError, "torch_profiler_dir.*immutable"),
        ):
            agent_tools.start_agent_run({"config": payload})

        save_temp.assert_not_called()
        start_run.assert_not_called()


class FileBoundaryTests(unittest.TestCase):
    def test_read_only_artifacts_fall_back_to_latest_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp) / "logs"
            run_dir = logs_dir / "run"
            profile_dir = run_dir / "vllm-profile"
            profile_dir.mkdir(parents=True)
            (run_dir / "vap_log.txt").write_text("latest", encoding="utf-8")
            (profile_dir / "trace.json").write_text("{}", encoding="utf-8")
            with (
                patch.object(server_settings, "LOGS_DIR", logs_dir),
                patch.object(
                    server_artifacts,
                    "get_run_state_snapshot",
                    return_value={"run_dir": None},
                ),
            ):
                log_info = server_artifacts.read_current_log_file("vap_log.txt")
                archive_info = server_artifacts.profile_archive_info()

        self.assertEqual(log_info["content"], "latest")
        self.assertEqual(archive_info["run_dir"], str(run_dir.resolve()))

    def test_starting_run_does_not_fall_back_to_previous_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp) / "logs"
            old_run_dir = logs_dir / "old-run"
            old_run_dir.mkdir(parents=True)
            (old_run_dir / "vap_log.txt").write_text("stale", encoding="utf-8")
            snapshot = {
                "run_dir": None,
                "running": True,
                "started_at": "2026-08-24 12:00:00",
            }
            with (
                patch.object(server_settings, "LOGS_DIR", logs_dir),
                patch.object(
                    server_artifacts,
                    "get_run_state_snapshot",
                    return_value=snapshot,
                ),
            ):
                log_info = server_artifacts.read_current_log_file("vap_log.txt")
                with self.assertRaisesRegex(ValueError, "refusing.*previous run"):
                    server_artifacts.resolve_profile_archive_run_dir()

        self.assertFalse(log_info["exists"])
        self.assertNotEqual(log_info["content"], "stale")

    def test_old_profile_archive_temp_files_are_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = (
                Path(tmp) / f"{server_artifacts.PROFILE_ARCHIVE_TEMP_PREFIX}old.zip"
            )
            archive.write_bytes(b"zip")
            old = time.time() - server_artifacts.PROFILE_ARCHIVE_MAX_AGE_SEC - 1
            os.utime(archive, (old, old))
            with patch.object(
                server_artifacts.tempfile,
                "gettempdir",
                return_value=tmp,
            ):
                server_artifacts.cleanup_old_profile_archives()

            self.assertFalse(archive.exists())

    def test_status_log_reads_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp) / "logs"
            run_dir = logs_dir / "run"
            run_dir.mkdir(parents=True)
            (run_dir / "vap_log.txt").write_text("0123456789", encoding="utf-8")
            with (
                patch.object(server_settings, "LOGS_DIR", logs_dir),
                patch.object(
                    server_artifacts,
                    "get_run_state_snapshot",
                    return_value={"run_dir": str(run_dir)},
                ),
            ):
                result = server_artifacts.read_current_log_file(
                    "vap_log.txt",
                    max_bytes=4,
                )

        self.assertEqual(result["content"], "6789")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["start_offset"], 6)

    def test_log_file_handler_applies_status_byte_limit(self) -> None:
        handler = server_handler.VAPConfigHandler.__new__(
            server_handler.VAPConfigHandler
        )
        handler.send_json = Mock()
        payload = {
            "exists": True,
            "content": "tail",
            "truncated": True,
        }

        with patch.object(
            server_handler,
            "read_current_log_file",
            return_value=payload,
        ) as mock_read:
            handler.handle_get_log_file("name=vllm_deploy.log")

        mock_read.assert_called_once_with(
            "vllm_deploy.log",
            max_bytes=server_settings.MAX_STATUS_LOG_BYTES,
        )
        handler.send_json.assert_called_once_with(payload)

    def test_agent_prefers_aligned_merged_trace_over_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp) / "logs"
            run_dir = logs_dir / "run"
            profile_dir = run_dir / "vllm-profile"
            aligned_dir = profile_dir / "aligned"
            aligned_dir.mkdir(parents=True)
            selected = profile_dir / "run-aligned-merged_trace.json"
            selected.write_text('{"traceEvents": []}', encoding="utf-8")
            (profile_dir / "run-merged_trace.json").write_text(
                '{"traceEvents": []}',
                encoding="utf-8",
            )
            (aligned_dir / "manifest.json").write_text("{}", encoding="utf-8")

            with patch.object(server_settings, "LOGS_DIR", logs_dir):
                result = agent_analysis.inspect_latest_trace({"run_dir": str(run_dir)})

        self.assertEqual(result["trace_path"], str(selected))
        self.assertNotIn("manifest.json", result["candidates"])

    def test_profile_archive_skips_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_dir = root / "logs"
            run_dir = logs_dir / "run"
            profile_dir = run_dir / "vllm-profile"
            profile_dir.mkdir(parents=True)
            (profile_dir / "trace.json").write_text("{}", encoding="utf-8")
            clock_dir = run_dir / "clock-probe"
            clock_dir.mkdir()
            (clock_dir / "clock-session.json").write_text(
                '{"status": "PASS"}',
                encoding="utf-8",
            )
            secret = root / "secret.txt"
            secret.write_text("do-not-archive", encoding="utf-8")
            (profile_dir / "secret-link").symlink_to(secret)

            with patch.object(server_settings, "LOGS_DIR", logs_dir):
                _, content = server_artifacts.build_current_profile_archive(
                    str(run_dir)
                )

            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = archive.namelist()
                self.assertIn("vllm-profile/trace.json", names)
                self.assertIn("clock-probe/clock-session.json", names)
                self.assertNotIn("vllm-profile/secret-link", names)

    def test_profile_archive_excludes_decompressed_alignment_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp) / "logs"
            run_dir = logs_dir / "run"
            profile_dir = run_dir / "vllm-profile"
            raw_dir = profile_dir / "aligned" / "raw"
            raw_dir.mkdir(parents=True)
            (profile_dir / "rank0.pt.trace.json.gz").write_bytes(b"compressed")
            (raw_dir / "rank-0.trace.json").write_bytes(b"decompressed")
            (profile_dir / "aligned" / "manifest.json").write_text(
                "{}",
                encoding="utf-8",
            )

            with patch.object(server_settings, "LOGS_DIR", logs_dir):
                _, content = server_artifacts.build_current_profile_archive(
                    str(run_dir)
                )

            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = archive.namelist()
                self.assertIn("vllm-profile/rank0.pt.trace.json.gz", names)
                self.assertIn("vllm-profile/aligned/manifest.json", names)
                self.assertNotIn(
                    "vllm-profile/aligned/raw/rank-0.trace.json",
                    names,
                )

    def test_temp_config_is_private(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_config_dir = Path(tmp) / "configs"
            with (
                patch.object(server_settings, "TEMP_CONFIG_DIR", temp_config_dir),
                patch.object(server_artifacts, "ensure_vap_home"),
            ):
                path = server_artifacts.save_temp_config({"secret": "value"})

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(temp_config_dir.stat().st_mode), 0o700)

    def test_runtime_directories_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".vap"
            replacements = {
                "VAP_HOME": root,
                "VAP_BIN_DIR": root / "bin",
                "VAP_LOGS_DIR": root / "logs",
                "VAP_TMP_DIR": root / "tmp",
                "VAP_TEMP_CONFIG_DIR": root / "tmp" / "configs",
                "VAP_PERFETTO_HOME": root / "perfetto-home",
                "VAP_CACHE_DIR": root / "cache",
            }
            with ExitStack() as stack:
                for name, value in replacements.items():
                    stack.enter_context(patch.object(runtime_paths, name, value))
                runtime_paths.ensure_vap_home()

            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            self.assertEqual(
                stat.S_IMODE((root / "tmp" / "configs").stat().st_mode), 0o700
            )


class RuntimeAndCliTests(unittest.TestCase):
    def test_sigterm_exits_with_conventional_failure_code(self) -> None:
        handlers = {}

        def register_handler(signum, handler):
            handlers[signum] = handler

        def trigger_sigterm():
            handlers[signal.SIGTERM](signal.SIGTERM, None)

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")
            args = types.SimpleNamespace(
                config=str(config_path),
                visualization_host="127.0.0.1",
            )
            with (
                patch.object(main, "load_config", return_value=Mock()),
                patch.object(main, "setup_logging", return_value=Mock()),
                patch.object(main.signal, "signal", side_effect=register_handler),
                patch.object(main, "TorchProfilingPipeline") as pipeline_class,
            ):
                pipeline_class.return_value.run_pipeline.side_effect = trigger_sigterm
                with self.assertRaises(SystemExit) as raised:
                    main.run(args, tmp)

        self.assertEqual(raised.exception.code, 128 + signal.SIGTERM)
        pipeline_class.return_value.cleanup.assert_called_once()

    def test_run_directory_is_named_after_the_start_time_and_run_name(self) -> None:
        payload = {
            "model_cfg": {
                "model_name": "ibm-granite/granite-4.1-8b",
                "model_path": "/m",
            },
            "vllm_deploy_cfg": {"-tp": 2, "--pipeline-parallel-size": 2},
        }
        names = {}
        for label, run_name in (
            ("default", None),
            ("custom", "baseline_{parallel}"),
            ("empty", ""),
        ):
            if run_name is not None:
                payload["run_name"] = run_name
            with tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / "config.json"
                config_path.write_text(json.dumps(payload), encoding="utf-8")
                logs = Path(tmp) / "logs"
                with (
                    patch.object(main, "load_config", return_value=Mock()),
                    patch.object(main, "setup_logging", return_value=Mock()),
                    patch.object(main, "TorchProfilingPipeline") as pipeline_class,
                ):
                    main.run(
                        types.SimpleNamespace(
                            config=str(config_path), visualization_host="127.0.0.1"
                        ),
                        str(logs),
                    )
                (run_dir,) = logs.iterdir()
                names[label] = run_dir.name
                # The start time alone stays the run id of containers and probes.
                self.assertEqual(pipeline_class.call_args.args[2], run_dir.name[:15])
        self.assertRegex(names["default"], r"^\d{8}_\d{6}_granite-4\.1-8b_tp2pp2$")
        self.assertRegex(names["custom"], r"^\d{8}_\d{6}_baseline_tp2pp2$")
        self.assertRegex(names["empty"], r"^\d{8}_\d{6}$")

    def test_run_name_fills_in_the_parallel_layout_and_rejects_unsafe_names(
        self,
    ) -> None:
        def name(deploy: dict, run_name: str = "{model}_{parallel}") -> str:
            return config.expand_run_name(
                {
                    "model_cfg": {"model_name": "Qwen/Qwen3-30B-A3B"},
                    "run_name": run_name,
                    "vllm_deploy_cfg": deploy,
                }
            )

        self.assertEqual(name({}), "Qwen3-30B-A3B_tp1")
        self.assertEqual(name({"-tp": "8"}), "Qwen3-30B-A3B_tp8")
        self.assertEqual(
            name({"-tp": 2, "-dp": 2, "--enable-expert-parallel": True}),
            "Qwen3-30B-A3B_tp2dp2ep4",
        )
        self.assertEqual(name({}, "a b/../c"), "a-b-..-c")
        self.assertEqual(
            config.VAPConfig.model_validate(example_payload()).run_name,
            "{model}_{parallel}",
        )
        for bad in ("../x", "a b", "{gpus}", "x" * 81):
            payload = example_payload()
            payload["run_name"] = bad
            with self.assertRaises(ValueError, msg=bad):
                config.VAPConfig.model_validate(payload)
        self.assertEqual(
            main.run_dir_name("20261007_120000", "/nonexistent/config.json"),
            "20261007_120000",
        )

    def test_pipeline_cleanup_is_idempotent(self) -> None:
        pipeline = torch_pipeline.TorchProfilingPipeline.__new__(
            torch_pipeline.TorchProfilingPipeline
        )
        pipeline.clock_probe = Mock()
        pipeline.clock_probe.active = False
        pipeline.ptp4l = Mock()
        pipeline.ptp4l.active = False
        pipeline.profiler = Mock()
        pipeline.profiler.active = False
        pipeline._vllm_process = None
        pipeline.ray_cluster = Mock()
        pipeline.ray_cluster.processes = []
        pipeline.worker_runners = [Mock()]
        pipeline.master_runner = Mock()

        pipeline.cleanup()
        pipeline.cleanup()

        pipeline.ptp4l.cleanup.assert_called_once()
        pipeline.master_runner.cleanup.assert_called_once()
        pipeline.worker_runners[0].cleanup.assert_called_once()

    def test_pipeline_cleanup_stops_containers_if_services_fail(self) -> None:
        pipeline = torch_pipeline.TorchProfilingPipeline.__new__(
            torch_pipeline.TorchProfilingPipeline
        )
        pipeline.clock_probe = Mock()
        pipeline.clock_probe.active = True
        pipeline.clock_probe.cleanup.side_effect = RuntimeError("probe failed")
        pipeline.ptp4l = Mock()
        pipeline.ptp4l.active = True
        pipeline.profiler = Mock()
        pipeline.profiler.active = False
        pipeline._vllm_process = None
        pipeline.ray_cluster = Mock()
        pipeline.ray_cluster.processes = []
        pipeline.worker_runners = []
        pipeline.master_runner = Mock()

        with self.assertLogs("VAP", level="ERROR"):
            pipeline.cleanup()

        pipeline.ptp4l.cleanup.assert_called_once()
        pipeline.master_runner.cleanup.assert_called_once()

    def test_clock_coordinator_is_killed_when_stop_fails(self) -> None:
        ray_module = Mock()
        coordinator = Mock()
        ray_module.get.side_effect = RuntimeError("stop failed")
        args = types.SimpleNamespace(output="/tmp/unused-clock-session.json")
        with (
            patch.object(clock_ray, "_connect_ray", return_value=ray_module),
            patch.object(
                clock_ray,
                "_get_active_coordinator",
                return_value=coordinator,
            ),
            self.assertRaisesRegex(RuntimeError, "stop failed"),
        ):
            clock_ray.stop_manual_calibration(args)

        ray_module.kill.assert_called_once_with(coordinator, no_restart=True)
        ray_module.shutdown.assert_called_once()

    def test_server_recovers_recorded_orphan_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record_path = Path(tmp) / "active-run.json"
            record_path.write_text(
                json.dumps(
                    {
                        "pid": 1234,
                        "pgid": 1234,
                        "start_ticks": 99,
                        "run_dir": None,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(server_settings, "ACTIVE_RUN_PATH", record_path),
                patch.object(
                    server_state,
                    "process_cmdline",
                    return_value="/venv/bin/python -m vap.main run --config x",
                ),
                patch.object(
                    server_state,
                    "process_start_ticks",
                    side_effect=[99, None, None],
                ),
                patch.object(server_state.os, "killpg") as killpg,
            ):
                recovered = server_state.recover_orphaned_run()
                self.assertFalse(record_path.exists())

        self.assertTrue(recovered)
        killpg.assert_called_once_with(1234, signal.SIGTERM)

    def test_perfetto_port_unavailable_does_not_block_run(self) -> None:
        parsed = config.VAPConfig.model_validate(example_payload())
        with (
            patch.object(
                torch_pipeline,
                "is_port_available",
                side_effect=lambda port: port != validation.PERFETTO_PORT,
            ),
            self.assertLogs("VAP", level="WARNING") as logs,
        ):
            torch_pipeline.check_port_availability(parsed)

        self.assertTrue(
            any(
                "Perfetto visualization will be skipped" in line for line in logs.output
            )
        )

    def test_perfetto_visualization_is_skipped_when_port_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            trace_path = log_dir / "trace.json"
            trace_path.write_text('{"traceEvents": []}\n', encoding="utf-8")
            with (
                patch.object(
                    visualization,
                    "find_trace_processor",
                    return_value="/bin/trace_processor",
                ),
                patch.object(visualization, "is_port_available", return_value=False),
                patch.object(visualization.subprocess, "Popen") as popen,
                self.assertLogs("VAP", level="WARNING") as logs,
            ):
                process = visualization.start_perfetto(
                    str(trace_path),
                    "127.0.0.1",
                )

        self.assertIsNone(process)
        popen.assert_not_called()
        self.assertTrue(
            any("skip Perfetto visualization" in line for line in logs.output)
        )

    def test_profile_stop_runs_after_benchmark_failure(self) -> None:
        parsed = config.VAPConfig.model_validate(example_payload())
        pipeline = torch_pipeline.TorchProfilingPipeline.__new__(
            torch_pipeline.TorchProfilingPipeline
        )
        pipeline.config = parsed
        pipeline.master_runner = Mock()
        pipeline.master_runner.network.http_status.side_effect = [200, 204]
        pipeline.master_runner.process.run_shell.side_effect = RuntimeError(
            "benchmark crashed"
        )
        pipeline.profiler = profiler_service.ProfilerLifecycle(
            parsed, pipeline.master_runner
        )
        pipeline.benchmark = benchmark_service.BenchmarkLifecycle(
            parsed, pipeline.master_runner
        )

        with self.assertRaisesRegex(RuntimeError, "benchmark crashed"):
            pipeline._run_workload()

        self.assertEqual(
            pipeline.master_runner.network.http_status.call_count,
            2,
        )
        start_url = pipeline.master_runner.network.http_status.call_args_list[0].args[0]
        stop_url = pipeline.master_runner.network.http_status.call_args_list[1].args[0]
        self.assertIn("start_profile", start_url)
        self.assertIn("stop_profile", stop_url)

    def test_profile_stop_failure_is_fatal(self) -> None:
        parsed = config.VAPConfig.model_validate(example_payload())
        pipeline = torch_pipeline.TorchProfilingPipeline.__new__(
            torch_pipeline.TorchProfilingPipeline
        )
        pipeline.config = parsed
        pipeline.master_runner = Mock()
        pipeline.master_runner.network.http_status.side_effect = [200, 500]
        pipeline.master_runner.process.run_shell.return_value = CommandResult(0)
        pipeline.profiler = profiler_service.ProfilerLifecycle(
            parsed, pipeline.master_runner
        )
        pipeline.benchmark = benchmark_service.BenchmarkLifecycle(
            parsed, pipeline.master_runner
        )

        with self.assertRaisesRegex(RuntimeError, "Failed to stop"):
            pipeline._run_workload()

    def test_disabled_profiler_runs_benchmark_only(self) -> None:
        payload = example_payload()
        payload["profiler_cfg"]["enable"] = False
        parsed = config.VAPConfig.model_validate(payload)
        tokens = parsed.vllm_deploy_args()
        self.assertFalse(
            any(token.startswith("--profiler-config.") for token in tokens)
        )

        pipeline = torch_pipeline.TorchProfilingPipeline.__new__(
            torch_pipeline.TorchProfilingPipeline
        )
        pipeline.config = parsed
        pipeline.master_runner = Mock()
        pipeline.master_runner.process.run_shell.return_value = CommandResult(0)
        pipeline.profiler = profiler_service.ProfilerLifecycle(
            parsed, pipeline.master_runner
        )
        pipeline.benchmark = benchmark_service.BenchmarkLifecycle(
            parsed, pipeline.master_runner
        )

        pipeline._run_workload()

        pipeline.master_runner.network.http_status.assert_not_called()
        pipeline.master_runner.process.run_shell.assert_called_once()
        self.assertFalse(pipeline.profiler.active)

    def test_clock_stop_does_not_mask_profiling_failure(self) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = True
        parsed = config.VAPConfig.model_validate(payload)
        pipeline = torch_pipeline.TorchProfilingPipeline.__new__(
            torch_pipeline.TorchProfilingPipeline
        )
        pipeline.config = parsed
        pipeline.master_runner = Mock()
        pipeline.worker_runners = [Mock()]
        pipeline.ray_cluster = Mock()
        pipeline.clock_probe = Mock()
        pipeline.clock_probe.enabled = True
        pipeline.ptp4l = Mock()
        pipeline.ptp4l.enabled = False
        pipeline.trace_postprocessor = Mock()
        pipeline._check_model_weights = Mock()
        pipeline._deploy_model = Mock()
        pipeline._wait_for_vllm_ready = Mock()
        pipeline._run_workload = Mock(side_effect=RuntimeError("benchmark failed"))
        pipeline.clock_probe.stop.side_effect = RuntimeError("clock stop failed")
        pipeline.cleanup = Mock()

        with (
            patch.object(torch_pipeline, "check_port_availability"),
            patch.object(torch_pipeline.os.path, "exists", return_value=True),
            patch.object(pipeline, "_container_spec", return_value=Mock()),
            self.assertLogs("VAP", level="ERROR"),
            self.assertRaisesRegex(RuntimeError, "benchmark failed"),
        ):
            pipeline.run_pipeline()

        pipeline.clock_probe.start.assert_called_once()
        pipeline.clock_probe.stop.assert_called_once()
        pipeline.ptp4l.stop.assert_called_once()

    def test_ptp4l_starts_before_clock_probe(self) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = True
        payload["clock_probe_cfg"]["mode"] = "hardware"
        parsed = config.VAPConfig.model_validate(payload)
        pipeline = torch_pipeline.TorchProfilingPipeline.__new__(
            torch_pipeline.TorchProfilingPipeline
        )
        pipeline.config = parsed
        pipeline.master_runner = Mock()
        pipeline.worker_runners = [Mock()]
        pipeline.ray_cluster = Mock()
        pipeline.ray_cluster.runner_node_ids = {
            "local": "192.168.0.9",
            "cse-ai-6.amd.com": "192.168.0.6",
        }
        pipeline.clock_probe = Mock()
        pipeline.clock_probe.enabled = True
        pipeline.ptp4l = Mock()
        pipeline.ptp4l.enabled = True
        pipeline.ptp4l.container_log_paths = {
            "cse-ai-9": "/app/VAP/log/clock-probe/ptp4l/run/cse-ai-9.log"
        }
        pipeline.ptp4l.node_descriptors = {
            "192.168.0.9": {
                "interface": "eth0",
                "phc_device": "/dev/ptp0",
                "ptp_log": "/app/VAP/log/clock-probe/ptp4l/run/cse-ai-9.log",
            }
        }
        pipeline.profiler = Mock()
        pipeline.profiler.enabled = False
        pipeline.trace_postprocessor = Mock()
        pipeline._check_model_weights = Mock()
        pipeline._deploy_model = Mock()
        pipeline._wait_for_vllm_ready = Mock()
        pipeline._run_workload = Mock()
        pipeline.cleanup = Mock()
        order: list[str] = []
        pipeline.ptp4l.start.side_effect = lambda *args, **kwargs: order.append(
            "ptp4l.start"
        )
        pipeline.clock_probe.start.side_effect = lambda: order.append(
            "clock_probe.start"
        )
        pipeline._run_workload.side_effect = lambda: order.append("workload")
        pipeline.clock_probe.stop.side_effect = lambda: order.append("clock_probe.stop")
        pipeline.ptp4l.stop.side_effect = lambda: order.append("ptp4l.stop")

        with (
            patch.object(torch_pipeline, "check_port_availability"),
            patch.object(torch_pipeline.os.path, "exists", return_value=True),
            patch.object(pipeline, "_container_spec", return_value=Mock()),
            patch.object(torch_pipeline, "start_tensorboard", return_value=Mock()),
            patch.object(torch_pipeline, "start_perfetto", return_value=Mock()),
            patch.object(torch_pipeline, "write_visualization_pids"),
            patch.object(torch_pipeline, "wait_for_visualizations"),
            patch.object(torch_pipeline, "stop_visualizations"),
        ):
            pipeline.run_pipeline()

        self.assertEqual(
            order,
            [
                "ptp4l.start",
                "clock_probe.start",
                "workload",
                "clock_probe.stop",
                "ptp4l.stop",
            ],
        )
        pipeline.clock_probe.set_ptp_logs.assert_called_once_with(
            pipeline.ptp4l.container_log_paths
        )
        pipeline.clock_probe.set_ptp_nodes.assert_called_once_with(
            pipeline.ptp4l.node_descriptors
        )
        pipeline.ptp4l.start.assert_called_once_with(
            node_addresses=pipeline.ray_cluster.runner_node_ids
        )
        pipeline.master_runner.network.ensure_inventory.assert_called()
        pipeline.worker_runners[0].network.ensure_inventory.assert_called()

    def test_ptp4l_failure_does_not_abort_workload_when_not_required(self) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = True
        payload["clock_probe_cfg"]["mode"] = "hardware"
        payload["clock_probe_cfg"]["required"] = False
        parsed = config.VAPConfig.model_validate(payload)
        pipeline = torch_pipeline.TorchProfilingPipeline.__new__(
            torch_pipeline.TorchProfilingPipeline
        )
        pipeline.config = parsed
        pipeline.master_runner = Mock()
        pipeline.worker_runners = [Mock()]
        pipeline.ray_cluster = Mock()
        pipeline.ray_cluster.runner_node_ids = {
            "local": "192.168.0.9",
            "worker.example": "192.168.0.10",
        }
        pipeline.clock_probe = Mock()
        pipeline.clock_probe.enabled = True
        pipeline.clock_probe.required = False
        pipeline.ptp4l = Mock()
        pipeline.ptp4l.enabled = True
        pipeline.ptp4l.start.side_effect = RuntimeError(
            "ptp4l on local did not reach MASTER or GRAND_MASTER"
        )
        pipeline.profiler = Mock()
        pipeline.profiler.enabled = False
        pipeline.trace_postprocessor = Mock()
        pipeline._check_model_weights = Mock()
        pipeline._deploy_model = Mock()
        pipeline._wait_for_vllm_ready = Mock()
        pipeline._run_workload = Mock()
        pipeline.cleanup = Mock()

        with (
            patch.object(torch_pipeline, "check_port_availability"),
            patch.object(torch_pipeline.os.path, "exists", return_value=True),
            patch.object(pipeline, "_container_spec", return_value=Mock()),
            patch.object(torch_pipeline, "start_tensorboard", return_value=Mock()),
            patch.object(torch_pipeline, "start_perfetto", return_value=Mock()),
            patch.object(torch_pipeline, "write_visualization_pids"),
            patch.object(torch_pipeline, "wait_for_visualizations"),
            patch.object(torch_pipeline, "stop_visualizations"),
            self.assertLogs("VAP", level="WARNING"),
        ):
            pipeline.run_pipeline()

        pipeline.ptp4l.start.assert_called_once_with(
            node_addresses=pipeline.ray_cluster.runner_node_ids
        )
        pipeline.clock_probe.set_ptp_logs.assert_not_called()
        pipeline.clock_probe.start.assert_not_called()
        pipeline._run_workload.assert_called_once()
        pipeline.ptp4l.stop.assert_called_once()

    def test_clock_probe_start_and_stop_produce_session(self) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = True
        payload["clock_probe_cfg"]["mode"] = "software"
        payload["vllm_deploy_cfg"]["-tp"] = 1
        payload["vllm_deploy_cfg"]["-pp"] = 2
        parsed = config.VAPConfig.model_validate(payload)
        master_runner = Mock()
        master_runner.process.run.side_effect = [
            CommandResult(0, stdout=b'{"status": "running"}'),
            CommandResult(0, stdout=b'{"status": "PASS"}'),
        ]
        master_runner.files.is_file.return_value = True

        with tempfile.TemporaryDirectory() as tmp:
            clock_probe = clock_probe_service.ClockProbeLifecycle(
                parsed,
                master_runner,
                tmp,
                "20260820_160000",
            )
            session_path = Path(tmp) / "clock-probe" / "clock-session.json"
            session_path.parent.mkdir()
            session_path.write_text(
                json.dumps(
                    {
                        "clock_source": "udp_software",
                        "status": "PASS",
                        "execution": {
                            "requested_mode": "software",
                            "selected_mode": "software",
                        },
                        "models": [
                            {
                                "model_type": "identity",
                                "status": "PASS",
                                "source": {"hostname": "master"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertLogs("VAP", level="INFO") as logs:
                clock_probe.start()
                clock_probe.stop()

        self.assertFalse(clock_probe.active)
        master_runner.files.ensure_directory.assert_called_once_with(
            clock_probe_service.CLOCK_PROBE_CONTAINER_DIR
        )
        written_config = master_runner.files.write_text.call_args.args[1]
        self.assertIn('"mode": "software"', written_config)
        start_command = master_runner.process.run.call_args_list[0].args[0]
        stop_command = master_runner.process.run.call_args_list[1].args[0]
        self.assertIn("start", start_command)
        self.assertIn("stop", stop_command)
        self.assertTrue(
            any(value.endswith("clock-session.json") for value in stop_command)
        )
        self.assertTrue(
            any("Clock probe fitting summary" in line for line in logs.output)
        )

    def test_clock_probe_fail_session_logs_reasons_as_warning(self) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = True
        payload["clock_probe_cfg"]["mode"] = "hardware"
        parsed = config.VAPConfig.model_validate(payload)
        master_runner = Mock()
        master_runner.process.run.return_value = CommandResult(
            0, stdout=b'{"status": "FAIL"}'
        )
        master_runner.files.is_file.return_value = True

        with tempfile.TemporaryDirectory() as tmp:
            clock_probe = clock_probe_service.ClockProbeLifecycle(
                parsed,
                master_runner,
                tmp,
                "20260830_123214",
            )
            clock_probe._active = True
            session_path = Path(tmp) / "clock-probe" / "clock-session.json"
            session_path.parent.mkdir()
            session_path.write_text(
                json.dumps(
                    {
                        "clock_source": "ptp_hardware",
                        "status": "FAIL",
                        "models": [],
                        "failures": [
                            {
                                "hostname": "cse-ai-6",
                                "reasons": [
                                    "ptp4l rms p95 4284.4 ns exceeds 1000.0 ns"
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertLogs("VAP", level="WARNING") as logs:
                clock_probe.stop()

        output = "\n".join(logs.output)
        self.assertIn("Clock probe calibration FAILED", output)
        self.assertIn("cse-ai-6", output)
        self.assertIn("4284.4 ns exceeds 1000.0 ns", output)

    def test_required_clock_probe_fail_session_warns_and_disables_alignment(
        self,
    ) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = True
        payload["clock_probe_cfg"]["mode"] = "hardware"
        payload["clock_probe_cfg"]["required"] = True
        parsed = config.VAPConfig.model_validate(payload)
        master_runner = Mock()
        master_runner.process.run.return_value = CommandResult(
            0, stdout=b'{"status": "FAIL"}'
        )
        master_runner.files.is_file.return_value = True

        with tempfile.TemporaryDirectory() as tmp:
            clock_probe = clock_probe_service.ClockProbeLifecycle(
                parsed,
                master_runner,
                tmp,
                "20260830_123214",
            )
            clock_probe._active = True
            session_path = Path(tmp) / "clock-probe" / "clock-session.json"
            session_path.parent.mkdir()
            session_path.write_text(
                json.dumps(
                    {
                        "clock_source": "ptp_hardware",
                        "status": "FAIL",
                        "models": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertLogs("VAP", level="WARNING") as logs:
                clock_probe.stop()

            self.assertTrue(clock_probe.has_session)
            self.assertFalse(clock_probe.alignment_ready)
            self.assertTrue(
                any("Clock probe calibration FAILED" in line for line in logs.output)
            )

    def test_failed_clock_session_skips_alignment_nccl_and_clc(self) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = True
        payload["clock_probe_cfg"]["mode"] = "software"
        payload["clock_probe_cfg"]["required"] = True
        parsed = config.VAPConfig.model_validate(payload)
        pipeline = torch_pipeline.TorchProfilingPipeline.__new__(
            torch_pipeline.TorchProfilingPipeline
        )
        pipeline.config = parsed
        pipeline.log_path = "/tmp/vap-test"
        pipeline.date_str = "20260901_090000"
        pipeline.visualization_host = "127.0.0.1"
        pipeline.master_runner = Mock()
        pipeline.master_runner.target.label = "local"
        worker = Mock()
        worker.target.label = "worker.example"
        pipeline.worker_runners = [worker]
        pipeline.ray_cluster = Mock()
        pipeline.ray_cluster.runner_node_ids = {
            "local": "head-node-id",
            "worker.example": "worker-node-id",
        }
        pipeline.clock_probe = Mock()
        pipeline.clock_probe.enabled = True
        pipeline.clock_probe.alignment_ready = False
        pipeline.ptp4l = Mock()
        pipeline.ptp4l.enabled = False
        pipeline.profiler = Mock()
        pipeline.profiler.enabled = True
        pipeline.trace_postprocessor = Mock()
        pipeline.trace_postprocessor.profile_dir = "/tmp/vap-test/vllm-profile"
        pipeline.trace_postprocessor.distributed_trace_inputs.return_value = {
            0: types.SimpleNamespace(path=Path("/tmp/rank-0.json")),
            1: types.SimpleNamespace(path=Path("/tmp/rank-1.json")),
        }
        pipeline.trace_postprocessor.fuse_trace_files.return_value = (
            "/tmp/vap-test/vllm-profile/raw-fallback-merged_trace.json"
        )
        pipeline._check_model_weights = Mock()
        pipeline._deploy_model = Mock()
        pipeline._wait_for_vllm_ready = Mock()
        pipeline._run_workload = Mock()
        pipeline.cleanup = Mock()

        with (
            patch.object(torch_pipeline, "check_port_availability"),
            patch.object(torch_pipeline.os.path, "exists", return_value=True),
            patch.object(pipeline, "_container_spec", return_value=Mock()),
            patch.object(torch_pipeline, "start_tensorboard", return_value=Mock()),
            patch.object(torch_pipeline, "start_perfetto", return_value=Mock()),
            patch.object(torch_pipeline, "write_visualization_pids"),
            patch.object(torch_pipeline, "wait_for_visualizations"),
            patch.object(torch_pipeline, "stop_visualizations"),
            self.assertLogs("VAP", level="WARNING") as logs,
        ):
            pipeline.run_pipeline()

        pipeline.trace_postprocessor.align_trace_files.assert_not_called()
        pipeline.trace_postprocessor.fuse_trace_files.assert_called_once()
        self.assertTrue(
            pipeline.trace_postprocessor.fuse_trace_files.call_args.kwargs[
                "raw_fallback"
            ]
        )
        output = "\n".join(logs.output)
        self.assertIn(
            "CLOCK ALIGNMENT SKIPPED: calibration precision gate did not PASS",
            output,
        )
        self.assertIn(
            "Raw rank traces will still be fused; visualization and download "
            "remain available",
            output,
        )

    def test_passed_clock_gate_makes_downstream_validation_strict(self) -> None:
        pipeline = torch_pipeline.TorchProfilingPipeline.__new__(
            torch_pipeline.TorchProfilingPipeline
        )
        pipeline.clock_probe = Mock()
        pipeline.clock_probe.enabled = True
        pipeline.clock_probe.alignment_ready = True
        pipeline.clock_probe.local_session_path = Path("/tmp/clock-session.json")
        pipeline.ray_cluster = Mock()
        pipeline.ray_cluster.runner_node_ids = {
            "local": "head-node-id",
            "worker.example": "worker-node-id",
        }
        pipeline.trace_postprocessor = Mock()
        pipeline.trace_postprocessor.distributed_trace_inputs.return_value = {
            0: types.SimpleNamespace(path=Path("/tmp/rank-0.json")),
            1: types.SimpleNamespace(path=Path("/tmp/rank-1.json")),
        }
        pipeline.trace_postprocessor.align_trace_files.side_effect = (
            trace_service.AlignmentValidationError(
                "NCCL causal validation failed",
                manifest_path="/tmp/aligned/manifest.json",
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Clock calibration passed, but strict NCCL validation failed",
        ):
            pipeline._postprocess_distributed_traces("/tmp/vllm-profile")

        pipeline.trace_postprocessor.fuse_trace_files.assert_not_called()

    def test_clock_probe_is_disabled_for_single_node_runs(self) -> None:
        payload = example_payload()
        payload["distributed_cfg"]["enable"] = False
        payload["clock_probe_cfg"]["enabled"] = True
        parsed = config.VAPConfig.model_validate(payload)

        clock_probe = clock_probe_service.ClockProbeLifecycle(
            parsed,
            Mock(),
            "/tmp/vap-test",
            "20260820_160000",
        )
        self.assertFalse(clock_probe.enabled)

    def test_model_weights_are_checked_on_every_runner(self) -> None:
        parsed = config.VAPConfig.model_validate(example_payload())
        pipeline = torch_pipeline.TorchProfilingPipeline.__new__(
            torch_pipeline.TorchProfilingPipeline
        )
        pipeline.config = parsed
        pipeline.master_runner = Mock()
        pipeline.master_runner.target.label = "local"
        pipeline.master_runner.files.exists.return_value = True
        worker = Mock()
        worker.target.label = "worker.example"
        worker.files.exists.return_value = False
        pipeline.worker_runners = [worker]

        with self.assertRaisesRegex(FileNotFoundError, "worker.example"):
            pipeline._check_model_weights()

    def test_missing_expected_trace_rank_is_rejected(self) -> None:
        parsed = config.VAPConfig.model_validate(example_payload())
        postprocessor = trace_service.TracePostprocessor(
            parsed,
            "/tmp/vap-test",
            Mock(),
            [],
        )

        with self.assertRaisesRegex(RuntimeError, "missing=\\[3\\]"):
            postprocessor.validate_trace_ranks({0: object(), 1: object(), 2: object()})

    def test_trace_files_must_have_stable_sizes(self) -> None:
        payload = example_payload()
        payload["vllm_deploy_cfg"]["-tp"] = 1
        payload["vllm_deploy_cfg"]["-pp"] = 2
        parsed = config.VAPConfig.model_validate(payload)
        master_runner = Mock()
        master_runner.target.label = "local"
        master_runner.files.directory_snapshot.side_effect = [
            {"rank0.pt.trace.json.gz": (100, 1)},
            {"rank0.pt.trace.json.gz": (100, 1)},
        ]
        worker = Mock()
        worker.target.label = "worker.example"
        worker.files.directory_snapshot.side_effect = [
            {"rank1.pt.trace.json.gz": (200, 1)},
            {"rank1.pt.trace.json.gz": (200, 1)},
        ]
        postprocessor = trace_service.TracePostprocessor(
            parsed,
            "/tmp/vap-test",
            master_runner,
            [worker],
        )

        postprocessor.wait_for_trace_files(
            timeout_sec=1,
            poll_interval_sec=0,
            stable_checks=1,
        )

        self.assertEqual(
            master_runner.files.directory_snapshot.call_count,
            2,
        )

    def test_non_rank_trace_does_not_block_rank_trace_finalization(self) -> None:
        payload = example_payload()
        payload["vllm_deploy_cfg"]["-tp"] = 1
        payload["vllm_deploy_cfg"]["-pp"] = 2
        parsed = config.VAPConfig.model_validate(payload)
        master_runner = Mock()
        master_runner.target.label = "local"
        master_runner.files.directory_snapshot.side_effect = [
            {
                "rank0.pt.trace.json": (100, 1),
                "node.async_llm.1.pt.trace.json": (0, 1),
            },
            {
                "rank0.pt.trace.json": (100, 1),
                "node.async_llm.1.pt.trace.json": (50, 2),
            },
        ]
        worker = Mock()
        worker.target.label = "worker.example"
        worker.files.directory_snapshot.side_effect = [
            {"rank1.pt.trace.json": (200, 1)},
            {"rank1.pt.trace.json": (200, 1)},
        ]
        postprocessor = trace_service.TracePostprocessor(
            parsed,
            "/tmp/vap-test",
            master_runner,
            [worker],
        )

        postprocessor.wait_for_trace_files(
            timeout_sec=1,
            poll_interval_sec=0,
            stable_checks=1,
        )

        self.assertEqual(master_runner.files.directory_snapshot.call_count, 2)

    def test_ray_cluster_starts_head_and_workers(self) -> None:
        parsed = config.VAPConfig.model_validate(example_payload())
        master_runner = Mock()
        master_runner.target.label = "local"
        master_runner.network.select_vllm_host_ip.return_value = "192.168.0.9"
        master_runner.process.start.return_value = Mock()
        master_runner.process.run.return_value = CommandResult(0)
        workers = []
        for index, hostname in enumerate(parsed.distributed_cfg.worker_nodes, start=10):
            worker = Mock()
            worker.target.hostname = hostname
            worker.target.label = hostname
            worker.network.select_vllm_host_ip.return_value = f"192.168.0.{index}"
            worker.process.start.return_value = Mock()
            worker.process.run.return_value = CommandResult(0)
            workers.append(worker)
        ray_cluster = ray_cluster_service.RayClusterLifecycle(
            parsed,
            master_runner,
            workers,
        )

        ray_cluster.start()

        self.assertEqual(len(ray_cluster.processes), 1 + len(workers))
        self.assertEqual(ray_cluster.master_ip, "192.168.0.9")
        self.assertEqual(
            set(ray_cluster.runner_node_ids),
            {"local", *(worker.target.label for worker in workers)},
        )
        master_runner.process.start.assert_called_once()
        for worker in workers:
            worker.process.start.assert_called_once()

    def test_ray_cluster_cleanup_terminates_foreground_processes(self) -> None:
        parsed = config.VAPConfig.model_validate(example_payload())
        master_runner = Mock()
        master_runner.is_started = True
        master_runner.target.label = "local"
        worker = Mock()
        worker.is_started = True
        worker.target.label = "worker.example"
        ray_cluster = ray_cluster_service.RayClusterLifecycle(
            parsed,
            master_runner,
            [worker],
        )
        head = Mock()
        worker_proc = Mock()
        ray_cluster._processes = [
            (master_runner, head),
            (worker, worker_proc),
        ]

        ray_cluster.cleanup()

        worker.process.run.assert_not_called()
        master_runner.process.run.assert_not_called()
        worker.process.terminate.assert_called_once_with(worker_proc, timeout_sec=5)
        master_runner.process.terminate.assert_called_once_with(head, timeout_sec=5)
        self.assertEqual(ray_cluster.processes, ())

    def test_worker_traces_are_collected_by_node(self) -> None:
        parsed = config.VAPConfig.model_validate(example_payload())
        with tempfile.TemporaryDirectory() as tmp:
            worker = Mock()
            worker.target.label = "worker.example"
            worker.files.is_directory.return_value = True
            trace = (
                Path(tmp)
                / "vllm-profile"
                / "workers"
                / "worker.example"
                / "rank-1.trace.json.gz"
            )
            merged = trace.parent / "stale-merged_trace.json.gz"
            worker.files.directory_snapshot.return_value = {
                trace.name: (100, 1),
                merged.name: (200, 1),
            }
            worker.files.download_files.return_value = [trace]
            postprocessor = trace_service.TracePostprocessor(
                parsed,
                tmp,
                Mock(),
                [worker],
            )

            collected = postprocessor.collect_worker_traces()

        self.assertEqual(
            collected,
            {"worker.example": [str(trace)]},
        )
        self.assertEqual(
            worker.files.download_files.call_args.args[1],
            [trace.name],
        )

    def test_aligned_distributed_traces_are_fused(self) -> None:
        payload = example_payload()
        payload["clock_probe_cfg"]["enabled"] = True
        payload["clock_probe_cfg"]["apply_clc_on_warning"] = True
        payload["vllm_deploy_cfg"]["-tp"] = 1
        payload["vllm_deploy_cfg"]["-pp"] = 2
        parsed = config.VAPConfig.model_validate(payload)
        master_runner = Mock()
        master_runner.target.label = "local"
        worker = Mock()
        worker.target.label = "worker.example"
        runner_node_ids = {
            "local": "192.168.0.9",
            "worker.example": "192.168.0.10",
        }

        with tempfile.TemporaryDirectory() as tmp:
            postprocessor = trace_service.TracePostprocessor(
                parsed,
                tmp,
                master_runner,
                [worker],
            )
            profile_dir = Path(tmp) / "vllm-profile"
            profile_dir.mkdir()
            master_trace = profile_dir / "dp0_rank0.1.pt.trace.json"
            async_llm_trace = profile_dir / "cse-ai-9_13880.async_llm.1.pt.trace.json"
            worker_trace = (
                profile_dir / "workers" / "worker" / "dp0_rank1.1.pt.trace.json"
            )
            master_trace.write_text("{}", encoding="utf-8")
            async_llm_trace.write_text("{}", encoding="utf-8")
            worker_trace.parent.mkdir(parents=True)
            worker_trace.write_text("{}", encoding="utf-8")
            worker_trace_files = {
                "worker.example": [str(worker_trace)],
            }
            manifest = types.SimpleNamespace(
                primary_timeline="aligned",
                aligned={
                    0: str(profile_dir / "aligned" / "rank-0.aligned.json"),
                    1: str(profile_dir / "aligned" / "rank-1.aligned.json"),
                },
                clc={},
            )
            with (
                patch.object(
                    trace_service, "align_traces", return_value=manifest
                ) as align,
                patch.object(
                    trace_service,
                    "fuse_traces",
                    return_value=str(profile_dir / "aligned-merged.json.gz"),
                ) as fuse,
                self.assertLogs("VAP", level="INFO") as logs,
            ):
                trace_inputs = postprocessor.distributed_trace_inputs(
                    str(profile_dir),
                    runner_node_ids,
                    worker_trace_files,
                )
                aligned_traces = postprocessor.align_trace_files(
                    trace_inputs,
                    str(profile_dir),
                    Path(tmp) / "clock-probe" / "clock-session.json",
                )
                result = postprocessor.fuse_trace_files(
                    aligned_traces,
                    str(profile_dir),
                    aligned=True,
                )

        self.assertEqual(result, str(profile_dir / "aligned-merged.json.gz"))
        alignment_inputs = align.call_args.args[0]
        self.assertEqual(alignment_inputs[0].source_node, "192.168.0.9")
        self.assertEqual(alignment_inputs[1].source_node, "192.168.0.10")
        self.assertTrue(
            any(
                "Skipping non-rank profiler trace" in line and "async_llm" in line
                for line in logs.output
            )
        )
        self.assertTrue(align.call_args.kwargs["apply_clc_on_warning"])
        self.assertEqual(fuse.call_args.args[0], manifest.aligned)

    def test_single_node_multi_gpu_traces_are_fused_without_alignment(self) -> None:
        payload = example_payload()
        payload["distributed_cfg"]["enable"] = False
        payload["vllm_deploy_cfg"]["-tp"] = 2
        payload["vllm_deploy_cfg"]["-pp"] = 1
        parsed = config.VAPConfig.model_validate(payload)

        with tempfile.TemporaryDirectory() as tmp:
            postprocessor = trace_service.TracePostprocessor(
                parsed,
                tmp,
                Mock(),
                [],
            )
            profile_dir = Path(tmp)
            rank_zero = profile_dir / "dp0_rank0.1.pt.trace.json"
            rank_one = profile_dir / "dp0_rank1.1.pt.trace.json"
            rank_zero.write_text("{}", encoding="utf-8")
            rank_one.write_text("{}", encoding="utf-8")
            with (
                patch.object(
                    postprocessor,
                    "fuse_trace_files",
                    return_value=str(profile_dir / "merged.json.gz"),
                ) as fuse,
                patch.object(trace_service, "align_traces") as align,
            ):
                result = postprocessor.prepare_single_node_trace(str(profile_dir))

        self.assertEqual(result, str(profile_dir / "merged.json.gz"))
        self.assertEqual(
            fuse.call_args.args[0],
            {0: str(rank_zero), 1: str(rank_one)},
        )
        self.assertFalse(fuse.call_args.kwargs["aligned"])
        align.assert_not_called()

    def test_single_node_postprocessing_skips_non_rank_trace(self) -> None:
        payload = example_payload()
        payload["distributed_cfg"]["enable"] = False
        payload["vllm_deploy_cfg"]["-tp"] = 1
        payload["vllm_deploy_cfg"]["-pp"] = 1
        parsed = config.VAPConfig.model_validate(payload)

        with tempfile.TemporaryDirectory() as tmp:
            postprocessor = trace_service.TracePostprocessor(
                parsed,
                tmp,
                Mock(),
                [],
            )
            profile_dir = Path(tmp)
            rank_trace = profile_dir / "dp0_rank0.1.pt.trace.json"
            auxiliary = profile_dir / "node.async_llm.1.pt.trace.json"
            rank_trace.write_text("{}", encoding="utf-8")
            auxiliary.write_text("{}", encoding="utf-8")

            result = postprocessor.prepare_single_node_trace(str(profile_dir))

        self.assertEqual(result, str(rank_trace))

    def test_pipeline_orchestrates_master_runner_and_cleans_up(self) -> None:
        payload = example_payload()
        payload["distributed_cfg"]["enable"] = False
        parsed = config.VAPConfig.model_validate(payload)
        pipeline = torch_pipeline.TorchProfilingPipeline.__new__(
            torch_pipeline.TorchProfilingPipeline
        )
        pipeline.config = parsed
        pipeline.log_path = "/tmp/vap-test"
        pipeline.date_str = "20260820_140000"
        pipeline.visualization_host = "127.0.0.1"
        pipeline.master_runner = Mock()
        pipeline.worker_runners = []
        pipeline._vllm_process = None
        pipeline.ray_cluster = Mock()
        pipeline.ray_cluster.runner_node_ids = {}
        pipeline.ray_cluster.processes = []
        pipeline.clock_probe = Mock()
        pipeline.clock_probe.enabled = False
        pipeline.clock_probe.active = False
        pipeline.clock_probe.required = False
        pipeline.clock_probe.local_session_path = Path(
            "/tmp/vap-test/clock-probe/clock-session.json"
        )
        pipeline.ptp4l = Mock()
        pipeline.ptp4l.enabled = False
        pipeline.ptp4l.active = False
        pipeline.profiler = Mock()
        pipeline.profiler.enabled = True
        pipeline.profiler.active = False
        pipeline.trace_postprocessor = Mock()
        pipeline.trace_postprocessor.profile_dir = "/tmp/vap-test/vllm-profile"
        pipeline.trace_postprocessor.prepare_single_node_trace.return_value = (
            "/tmp/trace.json"
        )
        pipeline._deploy_model = Mock()
        pipeline._wait_for_vllm_ready = Mock()
        pipeline._check_model_weights = Mock()
        pipeline._run_workload = Mock()
        tensorboard_process = Mock()
        perfetto_process = Mock()

        with (
            patch.object(torch_pipeline, "check_port_availability"),
            patch.object(torch_pipeline.os.path, "exists", return_value=True),
            patch.object(
                torch_pipeline,
                "start_tensorboard",
                return_value=tensorboard_process,
            ),
            patch.object(
                torch_pipeline,
                "start_perfetto",
                return_value=perfetto_process,
            ),
            patch.object(torch_pipeline, "write_visualization_pids") as write_pids,
            patch.object(torch_pipeline, "wait_for_visualizations") as wait,
            patch.object(torch_pipeline, "stop_visualizations") as stop,
            patch.object(pipeline, "_container_spec", return_value=Mock()) as spec,
        ):
            pipeline.run_pipeline()

        pipeline.master_runner.start.assert_called_once_with(spec.return_value)
        pipeline._check_model_weights.assert_called_once()
        pipeline.ray_cluster.start.assert_not_called()
        pipeline._deploy_model.assert_called_once()
        pipeline._wait_for_vllm_ready.assert_called_once()
        pipeline.clock_probe.start.assert_not_called()
        pipeline.clock_probe.stop.assert_not_called()
        pipeline._run_workload.assert_called_once()
        pipeline.trace_postprocessor.wait_for_trace_files.assert_called_once()
        pipeline.trace_postprocessor.collect_worker_traces.assert_not_called()
        pipeline.trace_postprocessor.prepare_single_node_trace.assert_called_once()
        pipeline.clock_probe.cleanup.assert_not_called()
        pipeline.ray_cluster.cleanup.assert_not_called()
        pipeline.master_runner.cleanup.assert_called_once()
        wait.assert_called_once()
        stop.assert_called_once()
        self.assertEqual(write_pids.call_count, 2)

    def test_disabled_profiler_skips_trace_postprocessing(self) -> None:
        payload = example_payload()
        payload["distributed_cfg"]["enable"] = False
        payload["profiler_cfg"]["enable"] = False
        parsed = config.VAPConfig.model_validate(payload)
        pipeline = torch_pipeline.TorchProfilingPipeline.__new__(
            torch_pipeline.TorchProfilingPipeline
        )
        pipeline.config = parsed
        pipeline.log_path = "/tmp/vap-test"
        pipeline.date_str = "20260820_140000"
        pipeline.visualization_host = "127.0.0.1"
        pipeline.master_runner = Mock()
        pipeline.worker_runners = []
        pipeline._vllm_process = None
        pipeline.ray_cluster = Mock()
        pipeline.ray_cluster.processes = []
        pipeline.clock_probe = Mock()
        pipeline.clock_probe.enabled = False
        pipeline.clock_probe.active = False
        pipeline.ptp4l = Mock()
        pipeline.ptp4l.enabled = False
        pipeline.ptp4l.active = False
        pipeline.profiler = Mock()
        pipeline.profiler.enabled = False
        pipeline.profiler.active = False
        pipeline.trace_postprocessor = Mock()
        pipeline._deploy_model = Mock()
        pipeline._wait_for_vllm_ready = Mock()
        pipeline._check_model_weights = Mock()
        pipeline._run_workload = Mock()

        with (
            patch.object(torch_pipeline, "check_port_availability"),
            patch.object(torch_pipeline.os.path, "exists", return_value=True),
            patch.object(torch_pipeline, "start_tensorboard") as start_tb,
            patch.object(torch_pipeline, "start_perfetto") as start_perfetto,
            patch.object(pipeline, "_container_spec", return_value=Mock()),
            self.assertLogs("VAP", level="INFO"),
        ):
            pipeline.run_pipeline()

        pipeline._run_workload.assert_called_once()
        pipeline.trace_postprocessor.wait_for_trace_files.assert_not_called()
        start_tb.assert_not_called()
        start_perfetto.assert_not_called()

    def test_cli_run_supplies_visualization_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            with (
                patch.object(cli, "VAP_CONFIG_PATH", config_path),
                patch.object(cli, "ensure_vap_home"),
                patch.object(cli.vap_workflow, "run") as run,
            ):
                cli.main(["run", "--visualization-host", "localhost"])

        args, _ = run.call_args.args
        self.assertEqual(args.visualization_host, "localhost")

    def test_cli_start_binds_all_interfaces_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")
            with (
                patch.object(cli, "VAP_CONFIG_PATH", config_path),
                patch.object(cli, "ensure_vap_home"),
                patch.object(cli, "start_server") as start_server,
            ):
                cli.main(["start"])

        start_server.assert_called_once_with(["--host", "0.0.0.0", "--port", "8899"])

    def test_cli_uninstall_forwards_options_to_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app_dir = Path(tmp)
            (app_dir / "uninstall.sh").write_text("#!/usr/bin/env bash\n")
            with (
                patch.object(cli, "ASSET_DIR", app_dir),
                patch.object(cli, "ensure_vap_home") as ensure_vap_home,
                patch.object(cli.os, "execvp") as execvp,
            ):
                cli.main(["uninstall", "--purge", "--remove-source", "--yes"])

        execvp.assert_called_once_with(
            "bash",
            [
                "bash",
                str(app_dir / "uninstall.sh"),
                "--purge",
                "--remove-source",
                "--yes",
            ],
        )
        ensure_vap_home.assert_not_called()

    def test_uninstall_removes_legacy_managed_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            vap_home = home / ".vap"
            venv_vap = vap_home / "venv" / "bin" / "vap"
            wrapper = home / ".local" / "bin" / "vap"
            venv_vap.parent.mkdir(parents=True)
            wrapper.parent.mkdir(parents=True)
            (vap_home / "logs").mkdir()
            (vap_home / "config.json").write_text("{}\n", encoding="utf-8")
            (vap_home / ".vap-installed").write_text(
                "installed_from=/source\n", encoding="utf-8"
            )
            venv_vap.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            venv_vap.chmod(0o755)
            wrapper.write_text(
                f'#!/usr/bin/env bash\nexec "{venv_vap}" "$@"\n',
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
            env = {
                **os.environ,
                "HOME": str(home),
                "VAP_HOME": str(vap_home),
                "XDG_DATA_HOME": str(home / ".local" / "share"),
            }

            result = subprocess.run(
                ["bash", str(PROJECT_ROOT / "uninstall.sh"), "--yes"],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(wrapper.exists())
            self.assertFalse((vap_home / "venv").exists())
            self.assertFalse((vap_home / ".vap-installed").exists())
            self.assertTrue((vap_home / "config.json").exists())
            self.assertTrue((vap_home / "logs").exists())
            self.assertIn("Removed command:", result.stdout)

    def test_uninstall_preserves_unmanaged_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            vap_home = home / ".vap"
            wrapper = home / ".local" / "bin" / "vap"
            wrapper.parent.mkdir(parents=True)
            vap_home.mkdir(parents=True)
            (vap_home / ".vap-installed").write_text(
                "installed_from=/source\n", encoding="utf-8"
            )
            wrapper.write_text(
                "#!/usr/bin/env bash\necho unrelated\n", encoding="utf-8"
            )
            wrapper.chmod(0o755)
            env = {
                **os.environ,
                "HOME": str(home),
                "VAP_HOME": str(vap_home),
                "XDG_DATA_HOME": str(home / ".local" / "share"),
            }

            result = subprocess.run(
                ["bash", str(PROJECT_ROOT / "uninstall.sh"), "--yes"],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(wrapper.exists())
            self.assertIn("Keeping unmanaged command:", result.stderr)

    def test_clean_refuses_non_vap_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = root / "logs"
            outside = root / "important"
            outside.mkdir()
            with patch.object(main, "VAP_LOGS_DIR", allowed):
                with self.assertRaisesRegex(ValueError, "Refusing"):
                    main.clean(str(outside))
            self.assertTrue(outside.is_dir())


class FrontendFallbackTests(unittest.TestCase):
    def test_distributed_config_is_editable(self) -> None:
        html = (PROJECT_ROOT / "public" / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="dist-enable"', html)
        self.assertIn('id="dist-ray-port"', html)
        self.assertIn('id="dist-sshkey"', html)
        self.assertIn("sshkey_path:", html)
        self.assertIn('id="worker-nodes-list"', html)
        self.assertIn('id="add-worker"', html)
        self.assertIn("enable: distributedEnabled", html)
        self.assertIn('id="distributed-panel"', html)
        self.assertIn('id="distributed-settings"', html)
        self.assertIn('classList.toggle("distributed-disabled", !enabled)', html)
        self.assertIn('id="profiler-panel"', html)
        self.assertIn('id="profiler-enable"', html)
        self.assertIn("Enable Torch Profiler", html)
        self.assertIn('id="clock-probe-panel"', html)
        self.assertIn("Clock Probe Hardware Capability", html)
        self.assertIn("ptp4l sidecar", html)
        self.assertNotIn('id="clock-probe-ptp-logs"', html)
        self.assertNotIn('id="add-clock-probe-ptp-log"', html)
        self.assertIn('fetch("/api/check-clock-probe"', html)
        self.assertIn("checkResourceScopeOnServer", html)
        self.assertIn("/api/check-${scope}-resources", html)
        self.assertIn('checkResourceScopeOnServer(config, "model")', html)
        self.assertIn('checkResourceScopeOnServer(config, "docker")', html)
        self.assertIn('checkResourceScopeOnServer(config, "worker-docker")', html)
        self.assertIn("Worker Docker Images Exist", html)
        self.assertIn('checkResourceScopeOnServer(config, "container")', html)
        self.assertIn('id="clock-probe-clc"', html)
        self.assertIn("apply_clc_on_warning:", html)
        self.assertIn("Download Profile Archive", html)
        self.assertIn("fetch(`/api/log-file/download?name=", html)
        self.assertIn('data.stop_requested ? "Stopping" : "Running"', html)
        self.assertIn("let stopInFlight = false", html)
        self.assertIn("if (stopInFlight) return", html)
        self.assertIn("btnStop.disabled = !isRunning || isStopping", html)
        self.assertNotIn(
            'data.message?.content || "", { local: true });\n        await syncAfterAgentAction',
            html,
        )
        self.assertIn('.join("\\n") || "No ports need to be checked"', html)
        self.assertIn('fetch("/api/check-machines"', html)
        self.assertNotIn(
            "Distributed machine connectivity checks are not supported", html
        )
        self.assertNotIn("Backend Pydantic Model Validation", html)

    def test_perfetto_fallback_dialog_supports_manual_trace_import(self) -> None:
        html = (PROJECT_ROOT / "public" / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="perfetto-fallback-modal"', html)
        self.assertIn('id="perfetto-fallback-download"', html)
        self.assertIn("https://ui.perfetto.dev/", html)
        self.assertIn("async function downloadCurrentTrace()", html)
        self.assertIn("if (perfettoPortUnavailable)", html)
        self.assertIn("port.blocking === false", html)

    def test_log_switching_uses_bounded_status_payloads(self) -> None:
        html = (PROJECT_ROOT / "public" / "index.html").read_text(encoding="utf-8")

        self.assertIn("function applyStatusLogs(", html)
        self.assertIn("function formatLogInfo(", html)
        self.assertIn('formatLogInfo(logs["vllm_deploy.log"])', html)
        self.assertIn('formatLogInfo(logs["vllm_bench.log"])', html)
        self.assertIn('if (name === "deploy" && !deployLogOutput.textContent)', html)
        self.assertIn('if (name === "bench" && !benchLogOutput.textContent)', html)
        self.assertNotIn("function refreshFileLogs(", html)
        self.assertIn("contain: content", html)


class TraceSkillSchemaTests(unittest.TestCase):
    def test_default_agent_model_is_gpt_5_6_sol(self) -> None:
        self.assertEqual(agent_runtime.DEFAULT_AGENT_MODEL, "gpt-5.6-sol")

    def test_registered_query_enum_matches_skill_queries(self) -> None:
        runtime = agent_runtime.VAPAgentRuntime()

        agent_tools.register_vap_agent_tools(runtime)

        schema_enum = runtime._tools["run_perfetto_sql"].parameters["properties"][
            "query_name"
        ]["enum"]
        self.assertEqual(schema_enum, sorted(agent_analysis.load_skill_queries()))

    def test_agent_prompt_marks_torch_profiler_dir_immutable(self) -> None:
        runtime = agent_runtime.VAPAgentRuntime()

        self.assertIn(
            "profiler_cfg.torch_profiler_dir field is immutable",
            runtime._system_prompt(),
        )


if __name__ == "__main__":
    unittest.main()
