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
from vap.runners.base import CommandResult
from vap.runners.docker import create_docker_client


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
        self.assertTrue(status.running)
        self.assertEqual(status.pid, 123)
        container.client.api.exec_start.assert_called_once_with(
            "exec-id",
            detach=True,
        )

    def test_managed_process_can_be_terminated(self) -> None:
        container = Mock(id="container-id")
        container.client.api.exec_inspect.side_effect = [
            {"Running": True, "ExitCode": None, "Pid": 123},
            {"Running": False, "ExitCode": 143, "Pid": 0},
        ]
        container.exec_run.return_value = (0, b"")
        assistant = ProcessAssistant(DockerCommandExecutor(container))

        status = assistant.terminate(
            Mock(exec_id="exec-id", command=("vllm",)),
        )

        self.assertFalse(status.running)
        self.assertEqual(status.exit_code, 143)
        self.assertEqual(
            container.exec_run.call_args.args[0],
            ["kill", "-TERM", "123"],
        )


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
        container.stop.assert_called_once()
        container.remove.assert_called_once()
        self.assertFalse(runner.is_started)


if __name__ == "__main__":
    unittest.main()
