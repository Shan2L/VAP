from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from vap.runners import ContainerSpec, DockerRunner, DockerTarget, MountSpec
from vap.runners.assistants.files import FileAssistant
from vap.runners.assistants.network import NetworkAssistant
from vap.runners.assistants.process import (
    DockerCommandExecutor,
    ProcessAssistant,
)
from vap.runners.base import CommandResult, ProcessHandle
from vap.runners.docker import (
    DEFAULT_DOCKER_TIMEOUT_SECONDS,
    VAP_KIND_LABEL,
    VAP_MANAGED_LABEL,
    VAP_RUN_LABEL,
    create_docker_client,
    sweep_vap_containers,
)


def network_payload() -> bytes:
    interfaces = [
        {
            "name": "enp4s0f0np0",
            "index": 2,
            "mac": "90:5a:08:98:59:e6",
            "ipv4": [],
            "ipv6": [],
            "mtu": 1500,
            "link_up": True,
            "speed_mbps": 2500,
            "driver": "i40e",
            "pci_address": "0000:04:00.0",
            "interface_type": "ethernet",
            "ptp_device": "/dev/ptp0",
        },
        {
            "name": "enp4s0f1np1",
            "index": 3,
            "mac": "90:5a:08:98:59:e7",
            "ipv4": ["10.67.93.244"],
            "ipv6": [],
            "mtu": 1500,
            "link_up": True,
            "speed_mbps": 1000,
            "driver": "i40e",
            "pci_address": "0000:04:00.1",
            "interface_type": "ethernet",
            "ptp_device": "/dev/ptp1",
        },
        {
            "name": "enp196s0f0np0",
            "index": 4,
            "mac": "cc:40:f3:f5:7c:14",
            "ipv4": [],
            "ipv6": [],
            "mtu": 1500,
            "link_up": False,
            "speed_mbps": None,
            "driver": "mlx5_core",
            "pci_address": "0000:c4:00.0",
            "interface_type": "ethernet",
            "ptp_device": "/dev/ptp2",
        },
        {
            "name": "enp196s0f1np1",
            "index": 5,
            "mac": "cc:40:f3:f5:7c:15",
            "ipv4": ["192.168.0.9"],
            "ipv6": [],
            "mtu": 1500,
            "link_up": True,
            "speed_mbps": 25000,
            "driver": "mlx5_core",
            "pci_address": "0000:c4:00.1",
            "interface_type": "ethernet",
            "ptp_device": "/dev/ptp3",
        },
    ]
    return json.dumps(interfaces).encode()


class ProcessAssistantTests(unittest.TestCase):
    def test_demuxed_command_result_is_normalized(self) -> None:
        container = Mock()
        container.exec_run.return_value = (7, (b"stdout", b"stderr"))
        assistant = ProcessAssistant(DockerCommandExecutor(container))

        result = assistant.run(["command"], demux=True)

        self.assertEqual(result.exit_code, 7)
        self.assertEqual(result.stdout, b"stdout")
        self.assertEqual(result.stderr, b"stderr")

    def test_managed_process_exposes_status(self) -> None:
        container = Mock(id="container-id")
        container.client.api.exec_create.return_value = {"Id": "exec-id"}
        container.client.api.exec_inspect.return_value = {
            "Running": True,
            "ExitCode": None,
            "Pid": 123,
        }
        assistant = ProcessAssistant(DockerCommandExecutor(container))

        handle = assistant.start(["sleep", "infinity"])
        status = assistant.status(handle)

        self.assertEqual(handle.exec_id, "exec-id")
        self.assertTrue(handle.pid_file.startswith("/tmp/vap-proc-"))
        self.assertTrue(status.running)
        self.assertEqual(status.pid, 123)
        created_command = container.client.api.exec_create.call_args.args[1]
        self.assertEqual(created_command[0], "/bin/bash")
        self.assertIn("echo $$", created_command[2])
        self.assertIn("sleep", created_command)
        self.assertIn("infinity", created_command)
        container.client.api.exec_start.assert_called_once_with(
            "exec-id",
            detach=True,
        )

    def test_managed_process_can_be_terminated(self) -> None:
        container = Mock(id="container-id")
        container.client.api.exec_inspect.side_effect = [
            {"Running": True, "ExitCode": None, "Pid": 999999},
            {"Running": False, "ExitCode": 143, "Pid": 0},
        ]
        container.exec_run.side_effect = [
            (0, b"42\n"),
            (0, b""),
        ]
        assistant = ProcessAssistant(DockerCommandExecutor(container))

        status = assistant.terminate(
            ProcessHandle(
                exec_id="exec-id",
                command=("vllm",),
                pid_file="/tmp/vap-proc-test.pid",
            )
        )

        self.assertFalse(status.running)
        self.assertEqual(status.exit_code, 143)
        self.assertEqual(
            container.exec_run.call_args_list[0].args[0],
            ["cat", "/tmp/vap-proc-test.pid"],
        )
        kill_command = container.exec_run.call_args_list[1].args[0]
        self.assertEqual(kill_command[0], "python3")
        self.assertEqual(kill_command[-2:], ["42", "15"])

    def test_terminate_waits_one_minute_before_giving_up(self) -> None:
        from vap.runners.assistants.process import TERMINATE_GRACE_SEC

        self.assertEqual(TERMINATE_GRACE_SEC, 60.0)


class FileAssistantTests(unittest.TestCase):
    def test_file_queries_and_directory_creation_use_target_container(self) -> None:
        executor = Mock()
        executor.run.side_effect = [
            CommandResult(0),
            CommandResult(0),
        ]
        assistant = FileAssistant(executor)

        self.assertTrue(assistant.exists("/data/model"))
        assistant.ensure_directory("/data/logs")

        self.assertEqual(
            executor.run.call_args_list[0].args[0], ["test", "-e", "/data/model"]
        )
        self.assertEqual(
            executor.run.call_args_list[1].args[0],
            ["mkdir", "-p", "--", "/data/logs"],
        )

    def test_directory_snapshot_returns_root_file_sizes(self) -> None:
        executor = Mock()
        executor.run.return_value = CommandResult(
            0,
            stdout=b'{"rank0.pt.trace.json.gz": [1234, 99]}',
        )
        assistant = FileAssistant(executor)

        snapshot = assistant.directory_snapshot(
            "/app/VAP/log/vllm-profile",
            (".pt.trace.json.gz",),
        )

        self.assertEqual(snapshot, {"rank0.pt.trace.json.gz": (1234, 99)})

    def test_directory_download_skips_links(self) -> None:
        archive_buffer = io.BytesIO()
        with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
            content = b'{"traceEvents": []}'
            trace = tarfile.TarInfo("vllm-profile/rank-1.trace.json")
            trace.size = len(content)
            archive.addfile(trace, io.BytesIO(content))
            link = tarfile.TarInfo("vllm-profile/unsafe-link")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            archive.addfile(link)

        container = Mock()
        container.get_archive.return_value = ([archive_buffer.getvalue()], {})
        assistant = FileAssistant(DockerCommandExecutor(container))
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = assistant.download_directory(
                "/app/VAP/log/vllm-profile",
                tmp,
            )

            self.assertEqual(
                downloaded,
                [Path(tmp) / "rank-1.trace.json"],
            )
            self.assertFalse((Path(tmp) / "unsafe-link").exists())

    def test_download_files_requests_only_named_container_files(self) -> None:
        archive_buffer = io.BytesIO()
        content = b'{"traceEvents": []}'
        with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
            trace = tarfile.TarInfo("rank0.pt.trace.json.gz")
            trace.size = len(content)
            archive.addfile(trace, io.BytesIO(content))

        container = Mock()
        container.get_archive.return_value = ([archive_buffer.getvalue()], {})
        assistant = FileAssistant(DockerCommandExecutor(container))
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = assistant.download_files(
                "/app/VAP/log/vllm-profile",
                ["rank0.pt.trace.json.gz"],
                tmp,
            )

        self.assertEqual(
            downloaded,
            [Path(tmp) / "rank0.pt.trace.json.gz"],
        )
        container.get_archive.assert_called_once_with(
            "/app/VAP/log/vllm-profile/rank0.pt.trace.json.gz"
        )


class NetworkAssistantTests(unittest.TestCase):
    def test_inventory_is_captured_and_queryable(self) -> None:
        executor = Mock()
        executor.run.return_value = CommandResult(0, stdout=network_payload())

        network = NetworkAssistant(executor, "node-1")

        self.assertEqual(len(network.interfaces()), 4)
        self.assertEqual(
            network.get_interface("enp196s0f1np1").speed_mbps,
            25000,
        )
        self.assertEqual(
            network.find_by_ip("192.168.0.9").name,
            "enp196s0f1np1",
        )
        self.assertEqual(len(network.ptp_interfaces()), 4)

    def test_ensure_inventory_probes_once_when_cache_is_empty(self) -> None:
        executor = Mock()
        executor.run.return_value = CommandResult(0, stdout=network_payload())
        network = NetworkAssistant(executor, "node-1", auto_refresh=False)

        self.assertFalse(network.has_inventory)
        first = network.ensure_inventory()
        second = network.ensure_inventory()

        self.assertTrue(network.has_inventory)
        self.assertEqual(len(first), 4)
        self.assertEqual(first, second)
        self.assertEqual(executor.run.call_count, 1)

    def test_select_ptp_interface_prefers_fastest_up_ethernet_with_phc(self) -> None:
        executor = Mock()
        executor.run.return_value = CommandResult(0, stdout=network_payload())
        network = NetworkAssistant(executor, "node-1", auto_refresh=False)

        selected = network.select_ptp_interface()

        self.assertEqual(selected.name, "enp196s0f1np1")
        self.assertEqual(selected.ptp_device, "/dev/ptp3")
        self.assertEqual(executor.run.call_count, 1)

    def test_select_ptp_interface_honors_preferred_interface(self) -> None:
        executor = Mock()
        executor.run.return_value = CommandResult(0, stdout=network_payload())
        network = NetworkAssistant(executor, "node-1")

        selected = network.select_ptp_interface(preferred_interface="enp4s0f0np0")

        self.assertEqual(selected.name, "enp4s0f0np0")
        self.assertEqual(selected.ptp_device, "/dev/ptp0")

    def test_ip_selection_refreshes_inventory_and_uses_private_candidate(self) -> None:
        executor = Mock()
        executor.run.side_effect = [
            CommandResult(0, stdout=network_payload()),
            CommandResult(0, stdout=network_payload()),
            CommandResult(0, stdout=b"192.168.0.9\n"),
        ]
        network = NetworkAssistant(executor, "node-1")

        selected = network.select_vllm_host_ip("worker.example")

        self.assertEqual(selected, "192.168.0.9")
        selection_command = executor.run.call_args_list[2].args[0]
        self.assertEqual(selection_command[0:2], ["python3", "-c"])
        self.assertIn("worker.example", selection_command)
        self.assertIn("192.168.0.9", selection_command)


class DockerRunnerTests(unittest.TestCase):
    def test_remote_client_uses_docker_ssh_transport(self) -> None:
        with patch("vap.runners.docker.docker.DockerClient") as client:
            create_docker_client(DockerTarget(hostname="worker.example"))

        client.assert_called_once_with(
            base_url="ssh://worker.example",
            use_ssh_client=True,
            timeout=DEFAULT_DOCKER_TIMEOUT_SECONDS,
        )

    def test_runner_composes_assistants_and_cleans_up(self) -> None:
        client = Mock()
        container = Mock(id="container-id")
        client.containers.run.return_value = container
        runner = DockerRunner(
            client=client,
            auto_network_refresh=False,
        )

        runner.start(
            ContainerSpec(
                image="example/image:tag",
                name="vap-test",
                mounts=(
                    MountSpec(
                        source="/remote/logs",
                        target="/app/VAP/log",
                        create_source=True,
                    ),
                ),
            )
        )

        self.assertIsInstance(runner.process, ProcessAssistant)
        self.assertIsInstance(runner.files, FileAssistant)
        self.assertIsInstance(runner.network, NetworkAssistant)
        self.assertEqual(
            client.containers.run.call_args.kwargs["volumes"],
            {"/remote/logs": {"bind": "/app/VAP/log", "mode": "rw"}},
        )
        runner.cleanup()
        container.stop.assert_not_called()
        container.remove.assert_called_once_with(force=True)
        client.close.assert_called_once()
        self.assertFalse(runner.is_started)

    def test_start_raises_if_container_exits_immediately(self) -> None:
        client = Mock()
        container = Mock(id="dead-id", status="exited")
        container.logs.return_value = b"ptp4l is not installed in this image\n"
        client.containers.run.return_value = container
        runner = DockerRunner(client=client)

        with self.assertRaisesRegex(
            RuntimeError,
            r"not running \(status=exited\).*ptp4l is not installed",
        ):
            runner.start(ContainerSpec(image="example/image:tag", name="vap-ptp4l"))

        container.exec_run.assert_not_called()
        container.remove.assert_called_once_with(force=True)
        self.assertFalse(runner.is_started)

    def test_runner_cleanup_force_removes_without_blocking_stop(self) -> None:
        client = Mock()
        container = Mock(id="container-id")
        client.containers.run.return_value = container
        runner = DockerRunner(client=client, auto_network_refresh=False)
        runner.start(ContainerSpec(image="example/image:tag", name="vap-test"))

        runner.cleanup()

        container.stop.assert_not_called()
        container.remove.assert_called_once_with(force=True)
        client.close.assert_called_once()
        self.assertFalse(runner.is_started)

    def test_sweep_vap_containers_removes_only_managed_labels(self) -> None:
        leftover = Mock()
        leftover.name = (
            "/vap_deepseek-ai_DeepSeek-R1-Distill-Llama-70B__20260829_220044"
        )
        leftover.labels = {VAP_MANAGED_LABEL: "true"}
        sidecar = Mock()
        sidecar.name = "vap_ptp4l_20260829_cse-ai-9_58584e9d"
        sidecar.labels = {VAP_MANAGED_LABEL: "true"}
        other = Mock()
        other.name = "vap_unlabeled"
        other.labels = {}
        client = Mock()
        client.containers.list.return_value = [leftover, sidecar, other]
        with patch("vap.runners.docker.create_docker_client", return_value=client):
            sweep_vap_containers(DockerTarget())
        leftover.remove.assert_called_once_with(force=True)
        sidecar.remove.assert_called_once_with(force=True)
        other.remove.assert_not_called()
        client.containers.list.assert_called_once_with(
            all=True,
            filters={"label": f"{VAP_MANAGED_LABEL}=true"},
        )
        client.close.assert_called_once()

    def test_sweep_preserves_containers_for_an_active_run(self) -> None:
        active_runner = Mock(status="running")
        active_runner.name = "vap_active_runner"
        active_runner.labels = {
            VAP_MANAGED_LABEL: "true",
            VAP_RUN_LABEL: "active",
            VAP_KIND_LABEL: "runner",
        }
        active_sidecar = Mock(status="running")
        active_sidecar.name = "vap_active_ptp4l"
        active_sidecar.labels = {
            VAP_MANAGED_LABEL: "true",
            VAP_RUN_LABEL: "active",
            VAP_KIND_LABEL: "ptp4l",
        }
        orphan = Mock(status="exited")
        orphan.name = "vap_old_runner"
        orphan.labels = {
            VAP_MANAGED_LABEL: "true",
            VAP_RUN_LABEL: "old",
            VAP_KIND_LABEL: "runner",
        }
        client = Mock()
        client.containers.list.return_value = [
            active_runner,
            active_sidecar,
            orphan,
        ]

        with patch("vap.runners.docker.create_docker_client", return_value=client):
            sweep_vap_containers(DockerTarget())

        active_runner.remove.assert_not_called()
        active_sidecar.remove.assert_not_called()
        orphan.remove.assert_called_once_with(force=True)

    def test_missing_ssh_identity_is_ignored(self) -> None:
        from vap.runners.ssh_identity import resolve_ssh_identity

        self.assertIsNone(resolve_ssh_identity("/missing/id_ed25519"))
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "id_ed25519"
            key.write_text("dummy", encoding="utf-8")
            self.assertEqual(resolve_ssh_identity(str(key)), str(key))


if __name__ == "__main__":
    unittest.main()
