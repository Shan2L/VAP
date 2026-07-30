from __future__ import annotations

import io
import json
import os
import shlex
import stat
import subprocess
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

    def test_distributed_config_is_reported_as_warning(self) -> None:
        payload = example_payload()
        payload["distributed_cfg"] = {
            "num_nodes": 2,
            "ray_port": 6379,
            "head_node": "localhost",
            "worker_nodes": ["worker.example"],
        }

        result = validation.validate_config_payload(payload)

        self.assertTrue(result["valid"])
        self.assertTrue(
            any(warning["path"] == "distributed_cfg" for warning in result["warnings"])
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

    def test_wildcard_bind_prints_local_hostname_and_ip_candidates(self) -> None:
        with patch.object(
            server,
            "discover_network_hosts",
            return_value=["vap-host.example", "10.0.0.8"],
        ):
            urls = server.build_session_urls("0.0.0.0", 8899, "session-token")

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
        with patch.object(
            server,
            "is_local_port_available",
            side_effect=lambda port: port != server.PERFETTO_PORT,
        ):
            result = server.check_config_ports(example_payload())

        perfetto = next(
            port
            for port in result["ports"]
            if port["name"] == "Perfetto Trace Processor port"
        )
        self.assertTrue(result["valid"])
        self.assertFalse(perfetto["available"])
        self.assertFalse(perfetto["blocking"])
        self.assertIn("will be skipped", perfetto["message"])

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
    def test_perfetto_port_unavailable_does_not_block_run(self) -> None:
        parsed = config.VAPConfig.model_validate(example_payload())
        with (
            patch.object(
                main,
                "is_port_available",
                side_effect=lambda port: port != main.PERFETTO_PORT,
            ),
            self.assertLogs("VAP", level="WARNING") as logs,
        ):
            main.check_port_availability(parsed)

        self.assertTrue(
            any(
                "Perfetto visualization will be skipped" in line for line in logs.output
            )
        )

    def test_perfetto_visualization_is_skipped_when_port_is_unavailable(self) -> None:
        parsed = config.VAPConfig.model_validate(example_payload())
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            trace_path = log_dir / "trace.json"
            trace_path.write_text('{"traceEvents": []}\n', encoding="utf-8")
            with (
                patch.object(main, "find_tensorboard_command", return_value=None),
                patch.object(main, "find_perfetto_trace", return_value=str(trace_path)),
                patch.object(
                    main, "find_trace_processor", return_value="/bin/trace_processor"
                ),
                patch.object(main, "is_port_available", return_value=False),
                patch.object(main.subprocess, "Popen") as popen,
                self.assertLogs("VAP", level="WARNING") as logs,
            ):
                main.visualize_profile(parsed, str(log_dir), "127.0.0.1")

        popen.assert_not_called()
        self.assertTrue(
            any("skip Perfetto visualization" in line for line in logs.output)
        )

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

    def test_cli_start_binds_all_interfaces_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")
            with (
                patch.object(cli, "VAP_CONFIG_PATH", config_path),
                patch.object(cli, "ensure_vap_home"),
                patch.object(cli.vap_server, "main") as start_server,
            ):
                cli.main(["start"])

        start_server.assert_called_once_with(["--host", "0.0.0.0", "--port", "8899"])

    def test_cli_uninstall_forwards_options_to_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app_dir = Path(tmp)
            (app_dir / "uninstall.sh").write_text("#!/usr/bin/env bash\n")
            with (
                patch.object(cli, "APP_DIR", app_dir),
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
    def test_perfetto_fallback_dialog_supports_manual_trace_import(self) -> None:
        html = (PROJECT_ROOT / "public" / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="perfetto-fallback-modal"', html)
        self.assertIn('id="perfetto-fallback-download"', html)
        self.assertIn("https://ui.perfetto.dev/", html)
        self.assertIn("async function downloadCurrentTrace()", html)
        self.assertIn("if (perfettoPortUnavailable)", html)
        self.assertIn("port.blocking === false", html)


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
