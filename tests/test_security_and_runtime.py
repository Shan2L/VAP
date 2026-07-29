from __future__ import annotations

import io
import json
import shlex
import stat
import sys
import tempfile
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
    docker_types = types.ModuleType("docker.types")

    class DockerClient:
        pass

    class Mount:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Ulimit:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    docker_module.DockerClient = DockerClient
    docker_module.from_env = Mock()
    docker_module.errors = types.SimpleNamespace(
        ImageNotFound=type("ImageNotFound", (Exception,), {})
    )
    docker_module.models = types.SimpleNamespace(
        containers=types.SimpleNamespace(Container=object)
    )
    docker_types.Mount = Mount
    docker_types.Ulimit = Ulimit
    sys.modules["docker"] = docker_module
    sys.modules["docker.types"] = docker_types


install_docker_stub()

import agent_runtime
import cli
import config
import main
import runtime_paths
import server
import validation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def example_payload() -> dict:
    return json.loads(
        (PROJECT_ROOT / "example-config.json").read_text(encoding="utf-8")
    )


class ConfigSecurityTests(unittest.TestCase):
    def test_cli_values_are_shell_quoted_as_single_tokens(self) -> None:
        payload = example_payload()
        payload["profiler_cfg"]["profiler"] = "torch; touch /tmp/not-created"
        parsed = config.VAPConfig.model_validate(payload)

        tokens = shlex.split(parsed.vllm_deploy_args_str())

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

    def test_distributed_config_is_explicitly_rejected(self) -> None:
        payload = example_payload()
        payload["distributed_cfg"] = {
            "num_nodes": 2,
            "ray_port": 6379,
            "head_node": "localhost",
            "worker_nodes": ["worker.example"],
        }

        result = validation.validate_config_payload(payload)

        self.assertFalse(result["valid"])
        self.assertTrue(
            any(error["path"] == "distributed_cfg" for error in result["errors"])
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


class ServerAuthorizationTests(unittest.TestCase):
    def make_handler(self, headers: dict[str, str]):
        handler = server.VAPConfigHandler.__new__(server.VAPConfigHandler)
        handler.headers = headers
        handler.send_json = Mock()
        return handler

    def test_valid_header_token_and_same_origin_are_allowed(self) -> None:
        handler = self.make_handler(
            {
                "X-VAP-Token": server.SERVER_AUTH_TOKEN,
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
                "X-VAP-Token": server.SERVER_AUTH_TOKEN,
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
        cookies = server.parse_cookie_header(
            f"one=1; {server.SERVER_COOKIE_NAME}=secret; two=2"
        )
        self.assertEqual(cookies[server.SERVER_COOKIE_NAME], "secret")

    def test_query_token_is_only_accepted_on_session_entrypoint(self) -> None:
        entry_handler = self.make_handler({})
        api_handler = self.make_handler({})

        self.assertTrue(
            entry_handler.is_authenticated(
                urlparse(f"/?token={server.SERVER_AUTH_TOKEN}")
            )
        )
        self.assertFalse(
            api_handler.is_authenticated(
                urlparse(f"/api/run/status?token={server.SERVER_AUTH_TOKEN}")
            )
        )

    def test_start_lock_rejects_concurrent_start(self) -> None:
        self.assertTrue(server.RUN_START_LOCK.acquire(blocking=False))
        try:
            with self.assertRaisesRegex(RuntimeError, "already starting"):
                server.start_vap_run()
        finally:
            server.RUN_START_LOCK.release()

    def test_stop_request_is_remembered_during_startup(self) -> None:
        with patch.dict(
            server.RUN_STATE,
            {
                "process": None,
                "running": True,
                "run_dir": None,
                "output": "",
                "stop_requested": False,
            },
        ):
            result = server.stop_vap_run()

            self.assertTrue(result["stop_requested"])
            self.assertIn("as soon as it starts", result["message"])

    def test_agent_cannot_start_with_modified_torch_profiler_dir(self) -> None:
        payload = example_payload()
        payload["profiler_cfg"]["torch_profiler_dir"] = "/tmp/other-profile"

        with (
            patch.object(server, "save_temp_config") as save_temp,
            patch.object(server, "start_vap_run") as start_run,
            self.assertRaisesRegex(ValueError, "torch_profiler_dir.*immutable"),
        ):
            server.start_agent_run({"config": payload})

        save_temp.assert_not_called()
        start_run.assert_not_called()


class FileBoundaryTests(unittest.TestCase):
    def test_profile_archive_skips_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_dir = root / "logs"
            run_dir = logs_dir / "run"
            profile_dir = run_dir / "vllm-profile"
            profile_dir.mkdir(parents=True)
            (profile_dir / "trace.json").write_text("{}", encoding="utf-8")
            secret = root / "secret.txt"
            secret.write_text("do-not-archive", encoding="utf-8")
            (profile_dir / "secret-link").symlink_to(secret)

            with patch.object(server, "LOGS_DIR", logs_dir):
                _, content = server.build_current_profile_archive(str(run_dir))

            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = archive.namelist()
                self.assertIn("vllm-profile/trace.json", names)
                self.assertNotIn("vllm-profile/secret-link", names)

    def test_temp_config_is_private(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_config_dir = Path(tmp) / "configs"
            with (
                patch.object(server, "TEMP_CONFIG_DIR", temp_config_dir),
                patch.object(server, "ensure_vap_home"),
            ):
                path = server.save_temp_config({"secret": "value"})

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
    def test_profile_stop_runs_after_benchmark_failure(self) -> None:
        parsed = config.VAPConfig.model_validate(example_payload())
        start_response = Mock()
        stop_response = Mock()
        container = Mock()
        container.exec_run.side_effect = RuntimeError("benchmark crashed")

        with patch.object(
            main.requests, "post", side_effect=[start_response, stop_response]
        ) as post:
            with self.assertRaisesRegex(RuntimeError, "benchmark crashed"):
                main.bench_and_profile(parsed, container)

        self.assertEqual(post.call_count, 2)
        start_response.raise_for_status.assert_called_once()
        stop_response.raise_for_status.assert_called_once()

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


class TraceSkillSchemaTests(unittest.TestCase):
    def test_registered_query_enum_matches_skill_queries(self) -> None:
        runtime = agent_runtime.VAPAgentRuntime()

        server.register_vap_agent_tools(runtime)

        schema_enum = runtime._tools["run_perfetto_sql"].parameters["properties"][
            "query_name"
        ]["enum"]
        self.assertEqual(schema_enum, sorted(server.load_skill_queries()))

    def test_agent_prompt_marks_torch_profiler_dir_immutable(self) -> None:
        runtime = agent_runtime.VAPAgentRuntime()

        self.assertIn(
            "profiler_cfg.torch_profiler_dir field is immutable",
            runtime._system_prompt(),
        )


if __name__ == "__main__":
    unittest.main()
