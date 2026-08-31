from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vap.clock_probe.calibration.ptp_health import (
    evaluate_ptp_health,
    parse_ptp4l_log,
)
from vap.config import VAPConfig
from vap.pipelines.services import ptp4l as ptp4l_mod
from vap.pipelines.services.clock_probe import ClockProbeLifecycle
from vap.pipelines.services.ptp4l import (
    PTP4L_APT_PACKAGE,
    PTP4L_MISSING_MESSAGE,
    Ptp4lLifecycle,
    container_ptp4l_log_path,
    create_ptp4l_runner,
    render_ptp4l_conf,
    render_sidecar_shell,
    sweep_ptp4l_containers,
)
from vap.runners import DockerRunner, DockerTarget
from vap.runners.assistants.network import NetworkInterfaceInfo
from vap.runners.base import CommandResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def hardware_config() -> VAPConfig:
    payload = json.loads(
        (PROJECT_ROOT / "example-config.json").read_text(encoding="utf-8")
    )
    payload["clock_probe_cfg"]["enabled"] = True
    payload["clock_probe_cfg"]["mode"] = "hardware"
    payload["distributed_cfg"]["enable"] = True
    payload["distributed_cfg"]["worker_nodes"] = ["cse-ai-10.amd.com"]
    return VAPConfig.model_validate(payload)


class FakeSidecarRunner:
    def __init__(self, target: DockerTarget):
        self.target = target
        self.spec = None
        self.cleaned = False
        self.container = SimpleNamespace(
            status="running",
            reload=lambda: None,
            logs=lambda: b"",
        )
        self.files = SimpleNamespace(read_text=lambda path: "")

    def start(self, spec) -> None:
        self.spec = spec

    def cleanup(self) -> None:
        self.cleaned = True


class FakeInventory:
    def __init__(
        self,
        target: DockerTarget,
        hostname: str,
        interface: str,
        phc: str,
        ipv4: tuple[str, ...] = (),
    ):
        self.target = target
        self.ensure_calls = 0
        self.published: list[tuple[str, str]] = []
        iface = NetworkInterfaceInfo(
            name=interface,
            index=5,
            mac="",
            ipv4=ipv4,
            ipv6=(),
            mtu=1500,
            link_up=True,
            speed_mbps=25000,
            driver="mlx5_core",
            pci_address=None,
            interface_type="ethernet",
            ptp_device=phc,
        )

        def ensure_inventory():
            self.ensure_calls += 1
            return (iface,)

        self.network = SimpleNamespace(
            ensure_inventory=ensure_inventory,
            select_ptp_interface=lambda **kwargs: iface,
            find_by_ip=lambda address: iface if address in iface.ipv4 else None,
        )
        self.process = SimpleNamespace(
            run=lambda command, **kwargs: CommandResult(
                0, stdout=f"{hostname}\n".encode()
            )
        )
        self.files = SimpleNamespace(
            ensure_directory=lambda path: None,
            write_text=lambda path, content: self.published.append((path, content)),
        )


class Ptp4lConfigTests(unittest.TestCase):
    def test_render_gm_and_slave_configs(self) -> None:
        gm = render_ptp4l_conf("enp4s0f0np0", grandmaster=True)
        slave = render_ptp4l_conf("enp4s0f0np0", grandmaster=False)

        self.assertIn("[enp4s0f0np0]", gm)
        self.assertIn("priority1                 100", gm)
        self.assertNotIn("clientOnly", gm)
        self.assertIn("clientOnly                1", slave)
        self.assertIn("time_stamping             hardware", gm)
        self.assertNotIn("unicast_listen", gm)
        self.assertNotIn("serverOnly", gm)
        self.assertNotIn("serverOnly", slave)
        self.assertNotIn("sanity_freq_limit", gm)
        self.assertNotIn("sanity_freq_limit", slave)
        self.assertNotIn("[unicast_master_table]", slave)

    def test_container_log_path_includes_run_id_and_hostname(self) -> None:
        self.assertEqual(
            container_ptp4l_log_path("abc123", "cse-ai-9"),
            "/app/VAP/log/clock-probe/ptp4l/abc123/cse-ai-9.log",
        )

    def test_default_sidecar_runner_skips_network_refresh(self) -> None:
        runner = create_ptp4l_runner(DockerTarget())
        self.assertIsInstance(runner, DockerRunner)
        self.assertFalse(runner._auto_network_refresh)

    def test_sidecar_shell_installs_linuxptp_when_ptp4l_is_missing(self) -> None:
        conf = render_ptp4l_conf("enp4s0f0np0", grandmaster=True)
        script = render_sidecar_shell(
            "cse-ai-9.conf",
            "cse-ai-9.log",
            conf_text=conf,
            local_clock_id="cc40f3.fffe.f57c15",
        )
        self.assertIn("vap ptp local-clock-id cc40f3.fffe.f57c15", script)
        self.assertIn("cat > /ptp/cse-ai-9.conf", script)
        self.assertIn("[enp4s0f0np0]", script)
        self.assertNotIn("linuxptp.deb", script)
        self.assertIn("/etc/os-release", script)
        self.assertIn("dpkg --print-architecture", script)
        self.assertIn("apt-get", script)
        self.assertIn(
            f"apt-get install -y --no-install-recommends {PTP4L_APT_PACKAGE}", script
        )
        self.assertIn("command -v ptp4l", script)
        self.assertIn(PTP4L_MISSING_MESSAGE, script)
        self.assertIn("exec ptp4l -f /ptp/cse-ai-9.conf", script)


class Ptp4lLifecycleTests(unittest.TestCase):
    def test_software_mode_does_not_start_sidecars(self) -> None:
        config = hardware_config()
        config.clock_probe_cfg.mode = "software"
        with tempfile.TemporaryDirectory() as tmp:
            lifecycle = Ptp4lLifecycle(config, tmp, "20260828_160000")
            lifecycle.start()
        self.assertFalse(lifecycle.active)
        self.assertEqual(lifecycle.sessions, [])

    def test_start_launches_first_worker_as_gm_then_ray_head_as_slave(self) -> None:
        config = hardware_config()
        runners: list[FakeSidecarRunner] = []

        def factory(target: DockerTarget) -> FakeSidecarRunner:
            runner = FakeSidecarRunner(target)
            runners.append(runner)
            return runner

        master_inventory = FakeInventory(
            DockerTarget(),
            "cse-ai-9",
            "enp4s0f0np0",
            "/dev/ptp0",
            ipv4=("192.168.0.9",),
        )
        worker_inventory = FakeInventory(
            DockerTarget(hostname="cse-ai-10.amd.com"),
            "cse-ai-10",
            "ens3f0",
            "/dev/ptp2",
            ipv4=("192.168.0.10",),
        )
        swept: list[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            lifecycle = Ptp4lLifecycle(
                config,
                tmp,
                "20260828_160000",
                master_inventory=master_inventory,
                worker_inventories=[worker_inventory],
                runner_factory=factory,
                sidecar_sweeper=lambda target: swept.append(target.label),
                run_id="runuuid12",
            )
            with patch.object(Ptp4lLifecycle, "_wait_until_locked"):
                lifecycle.start(
                    node_addresses={
                        "local": "192.168.0.9",
                        "cse-ai-10.amd.com": "192.168.0.10",
                    }
                )

            self.assertGreaterEqual(master_inventory.ensure_calls, 1)
            self.assertGreaterEqual(worker_inventory.ensure_calls, 1)
            self.assertTrue(lifecycle.active)
            self.assertEqual(len(runners), 2)
            gm_conf = (
                Path(tmp) / "clock-probe" / "ptp4l" / "runuuid12" / "cse-ai-10.conf"
            )
            slave_conf = (
                Path(tmp) / "clock-probe" / "ptp4l" / "runuuid12" / "cse-ai-9.conf"
            )
            self.assertTrue(gm_conf.is_file())
            self.assertTrue(slave_conf.is_file())
            self.assertIn("priority1                 100", gm_conf.read_text())
            self.assertNotIn("unicast_listen", gm_conf.read_text())
            self.assertIn("clientOnly                1", slave_conf.read_text())
            self.assertNotIn("UDPv4 192.168.0.9", slave_conf.read_text())
            self.assertEqual(runners[0].spec.devices, ("/dev/ptp2",))
            self.assertEqual(runners[1].spec.devices, ("/dev/ptp0",))
            self.assertIn("NET_ADMIN", runners[0].spec.cap_add)
            sidecar_cmd = " ".join(runners[0].spec.command)
            self.assertEqual(runners[0].spec.command[0], "/bin/bash")
            self.assertIn("cat > /ptp/cse-ai-10.conf", sidecar_cmd)
            self.assertIn("[ens3f0]", sidecar_cmd)
            self.assertNotIn("linuxptp.deb", sidecar_cmd)
            self.assertIn("/etc/os-release", sidecar_cmd)
            self.assertIn("dpkg --print-architecture", sidecar_cmd)
            self.assertIn("apt-get", sidecar_cmd)
            self.assertIn(PTP4L_APT_PACKAGE, sidecar_cmd)
            self.assertIn(PTP4L_MISSING_MESSAGE, sidecar_cmd)
            self.assertEqual(
                lifecycle.container_log_paths,
                {
                    "cse-ai-9": "/app/VAP/log/clock-probe/ptp4l/runuuid12/cse-ai-9.log",
                    "cse-ai-10": "/app/VAP/log/clock-probe/ptp4l/runuuid12/cse-ai-10.log",
                },
            )
            self.assertEqual(
                set(lifecycle.node_descriptors),
                {"192.168.0.9", "192.168.0.10"},
            )
            self.assertEqual(
                lifecycle.node_descriptors["192.168.0.10"]["phc_device"],
                "/dev/ptp2",
            )
            lifecycle.stop()
            self.assertTrue(all(runner.cleaned for runner in runners))
            self.assertFalse(lifecycle.active)
            self.assertEqual(
                swept,
                [
                    "local",
                    "cse-ai-10.amd.com",
                    "local",
                    "cse-ai-10.amd.com",
                ],
            )

    def test_wait_until_locked_accepts_expected_port_state(self) -> None:
        config = hardware_config()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "node.log"
            log_path.write_text(
                "ptp4l[1.0]: port 1 (enp4s0f0np0): LISTENING to MASTER "
                "on ANNOUNCE_RECEIPT_TIMEOUT_EXPIRES\n",
                encoding="utf-8",
            )
            lifecycle = Ptp4lLifecycle(config, tmp, "20260828_160000", run_id="x")
            session = ptp4l_mod.Ptp4lNodeSession(
                label="local",
                os_hostname="cse-ai-9",
                role="master",
                interface="enp4s0f0np0",
                phc_device="/dev/ptp0",
                runner=FakeSidecarRunner(DockerTarget()),
                host_log=log_path,
                container_log="/ptp/cse-ai-9.log",
            )
            lifecycle._wait_until_locked(session, timeout_sec=0.2)

    def test_wait_until_locked_accepts_grand_master(self) -> None:
        config = hardware_config()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "node.log"
            log_path.write_text(
                "ptp4l[1.0]: port 1 (enp196s0f1np1): LISTENING to GRAND_MASTER "
                "on RS_GRAND_MASTER\n",
                encoding="utf-8",
            )
            lifecycle = Ptp4lLifecycle(config, tmp, "20260828_160000", run_id="x")
            session = ptp4l_mod.Ptp4lNodeSession(
                label="local",
                os_hostname="cse-ai-9",
                role="master",
                interface="enp196s0f1np1",
                phc_device="/dev/ptp3",
                runner=FakeSidecarRunner(DockerTarget()),
                host_log=log_path,
                container_log="/ptp/cse-ai-9.log",
            )
            lifecycle._wait_until_locked(session, timeout_sec=0.2)

    def test_ptp4l_log_parser_treats_grand_master_as_locked_gm(self) -> None:
        text = (
            "ptp4l[111618.047]: port 1 (enp196s0f1np1): new foreign master "
            "cc40f3.fffe.f58385-1\n"
            "ptp4l[111620.047]: selected best master clock cc40f3.fffe.f58385\n"
            "ptp4l[111620.048]: port 1 (enp196s0f1np1): assuming the grand master role\n"
            "ptp4l[111620.048]: port 1 (enp196s0f1np1): LISTENING to GRAND_MASTER "
            "on RS_GRAND_MASTER\n"
        )
        parsed = parse_ptp4l_log(text)
        self.assertEqual(parsed["states"][-1]["state"], "GRAND_MASTER")
        health = evaluate_ptp_health(text, role="master")
        self.assertEqual(health.port_state, "GRAND_MASTER")
        self.assertTrue(health.lock_ok)
        self.assertIsNone(health.grandmaster_clock_id)

    def test_gm_uses_vap_local_clock_id_not_foreign_bmc(self) -> None:
        text = (
            "vap ptp local-clock-id cc40f3.fffe.f57c15\n"
            "ptp4l[115978.872]: port 1 (enp196s0f1np1): new foreign master "
            "cc40f3.fffe.f58385-1\n"
            "ptp4l[115978.872]: selected best master clock cc40f3.fffe.f58385\n"
            "ptp4l[115978.872]: port 1 (enp196s0f1np1): assuming the grand master role\n"
            "ptp4l[115978.872]: port 1 (enp196s0f1np1): LISTENING to GRAND_MASTER "
            "on RS_GRAND_MASTER\n"
        )
        health = evaluate_ptp_health(text, role="master")
        self.assertTrue(health.lock_ok)
        self.assertEqual(health.grandmaster_clock_id, "cc40f3.fffe.f57c15")

    def test_clock_identity_from_mac_inserts_fffe(self) -> None:
        from vap.clock_probe.calibration.ptp_health import clock_identity_from_mac

        self.assertEqual(
            clock_identity_from_mac("cc:40:f3:f5:7c:15"),
            "cc40f3.fffe.f57c15",
        )

    def test_parser_reads_linuxptp4_signed_freq_rms(self) -> None:
        text = (
            "ptp4l[1509257.665]: port 1 (enp196s0f1np1): UNCALIBRATED to SLAVE "
            "on MASTER_CLOCK_SELECTED\n"
            "ptp4l[1509257.665]: selected best master clock cc40f3.fffe.f57c15\n"
            "ptp4l[1509258.291]: rms 2372126 max 4744784 freq  +3675 +/- 2786 delay  2664 +/-   0\n"
            + "".join(
                f"ptp4l[{1509259.291 + idx:.3f}]: rms   30 max   50 freq  +4600 +/-  40"
                f"{' delay  2700 +/-   0' if idx % 2 == 0 else ''}\n"
                for idx in range(12)
            )
            + "ptp4l[1509272.000]: clockcheck: clock frequency changed unexpectedly!\n"
        )
        parsed = parse_ptp4l_log(text)
        self.assertGreaterEqual(len(parsed["summaries"]), 12)
        self.assertEqual(parsed["summaries"][0].freq_ppb, 3675)
        health = evaluate_ptp_health(text, role="slave")
        self.assertFalse(health.lock_ok)
        self.assertGreater(health.clockcheck_count, 0)
        self.assertGreater(health.late_clockcheck_count, 0)
        self.assertTrue(
            any("clockcheck warning" in reason for reason in health.reasons)
        )

    def test_parser_reads_master_offset_when_rms_is_absent(self) -> None:
        text = (
            "ptp4l[1.0]: port 1 (enp0): UNCALIBRATED to SLAVE on MASTER_CLOCK_SELECTED\n"
            "ptp4l[1.0]: selected best master clock aabb.ccdd.eeff00\n"
            + "".join(
                f"ptp4l[{2.0 + idx}]: master offset       -{20 + idx} s2 freq   +1234 "
                "path delay      2700\n"
                for idx in range(12)
            )
        )
        health = evaluate_ptp_health(text, role="slave")
        self.assertTrue(health.lock_ok, health.reasons)
        self.assertGreaterEqual(health.summary_count, 12)

    def test_wait_until_locked_slave_waits_for_rms(self) -> None:
        config = hardware_config()
        header = (
            "ptp4l[1.0]: port 1 (enp196s0f1np1): UNCALIBRATED to SLAVE "
            "on MASTER_CLOCK_SELECTED\n"
            "ptp4l[1.0]: selected best master clock cc40f3.fffe.f57c15\n"
        )
        ready = header + "".join(
            f"ptp4l[{2.0 + idx}]: rms   28 max   50 freq  +4600 +/-  40 delay  2700 +/-   0\n"
            for idx in range(12)
        )
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "node.log"
            log_path.write_text(header, encoding="utf-8")
            lifecycle = Ptp4lLifecycle(config, tmp, "20260828_160000", run_id="x")
            session = ptp4l_mod.Ptp4lNodeSession(
                label="cse-ai-6.amd.com",
                os_hostname="cse-ai-6",
                role="slave",
                interface="enp196s0f1np1",
                phc_device="/dev/ptp3",
                runner=FakeSidecarRunner(DockerTarget()),
                host_log=log_path,
                container_log="/ptp/cse-ai-6.log",
            )
            with (
                patch.object(
                    lifecycle,
                    "_read_log",
                    side_effect=[header, ready, ready],
                ),
                patch.object(ptp4l_mod, "PTP4L_LOCK_POLL_SEC", 0.01),
            ):
                lifecycle._wait_until_locked(session, timeout_sec=1.0)

    def test_wait_until_locked_reports_install_hang(self) -> None:
        config = hardware_config()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "node.log"
            log_path.write_text(
                "Installing linuxptp (provides ptp4l)\n", encoding="utf-8"
            )
            lifecycle = Ptp4lLifecycle(config, tmp, "20260828_160000", run_id="x")
            session = ptp4l_mod.Ptp4lNodeSession(
                label="cse-ai-6.amd.com",
                os_hostname="cse-ai-6",
                role="slave",
                interface="enp196s0f1np1",
                phc_device="/dev/ptp3",
                runner=FakeSidecarRunner(DockerTarget()),
                host_log=log_path,
                container_log="/ptp/cse-ai-6.log",
            )
            with patch.object(ptp4l_mod, "PTP4L_LOCK_POLL_SEC", 0.01):
                with self.assertRaisesRegex(
                    RuntimeError, "ptp4l never started.*Installing linuxptp"
                ):
                    lifecycle._wait_until_locked(session, timeout_sec=0.05)

    def test_publish_does_not_overwrite_live_log(self) -> None:
        config = hardware_config()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "node.log"
            log_path.write_text("sidecar-live\n", encoding="utf-8")
            inventory = FakeInventory(
                DockerTarget(),
                "cse-ai-6",
                "enp196s0f1np1",
                "/dev/ptp3",
            )
            inventory.files = SimpleNamespace(
                ensure_directory=lambda path: None,
                is_file=lambda path: True,
                read_text=lambda path: "already-mounted-live-log\n",
                write_text=lambda path, content: inventory.published.append(
                    (path, content)
                ),
            )
            lifecycle = Ptp4lLifecycle(config, tmp, "20260828_160000", run_id="x")
            session = ptp4l_mod.Ptp4lNodeSession(
                label="cse-ai-6.amd.com",
                os_hostname="cse-ai-6",
                role="slave",
                interface="enp196s0f1np1",
                phc_device="/dev/ptp3",
                runner=FakeSidecarRunner(DockerTarget()),
                host_log=log_path,
                container_log="/app/VAP/log/clock-probe/ptp4l/x/cse-ai-6.log",
            )
            lifecycle._publish_ptp_log(session, inventory)
            self.assertEqual(inventory.published, [])

    def test_read_log_prefers_ptp4l_over_longer_apt_output(self) -> None:
        config = hardware_config()
        with tempfile.TemporaryDirectory() as tmp:
            host_log = Path(tmp) / "cse-ai-6.log"
            host_log.write_text(
                "Installing linuxptp\n" + ("x" * 5000), encoding="utf-8"
            )
            runner = FakeSidecarRunner(DockerTarget())
            runner.files = SimpleNamespace(
                read_text=lambda path: (
                    "ptp4l[1.0]: port 1 (enp4s0f0np0): "
                    "LISTENING to SLAVE on MASTER_CLOCK_SELECTED\n"
                )
            )
            lifecycle = Ptp4lLifecycle(config, tmp, "20260828_160000", run_id="x")
            session = ptp4l_mod.Ptp4lNodeSession(
                label="cse-ai-6.amd.com",
                os_hostname="cse-ai-6",
                role="slave",
                interface="enp4s0f0np0",
                phc_device="/dev/ptp0",
                runner=runner,
                host_log=host_log,
                container_log="/ptp/cse-ai-6.log",
            )
            text = lifecycle._read_log(session)
        self.assertIn("SLAVE", text)
        self.assertNotIn("Installing linuxptp", text)

    def test_read_log_uses_container_file_when_host_log_is_empty(self) -> None:
        config = hardware_config()
        with tempfile.TemporaryDirectory() as tmp:
            host_log = Path(tmp) / "cse-ai-6.log"
            host_log.write_text("", encoding="utf-8")
            runner = FakeSidecarRunner(DockerTarget())
            runner.files = SimpleNamespace(
                read_text=lambda path: "ptp4l[1.0]: port 1 (enp4s0f0np0): "
                "LISTENING to SLAVE on MASTER_CLOCK_SELECTED\n"
            )
            lifecycle = Ptp4lLifecycle(config, tmp, "20260828_160000", run_id="x")
            session = ptp4l_mod.Ptp4lNodeSession(
                label="cse-ai-6.amd.com",
                os_hostname="cse-ai-6",
                role="slave",
                interface="enp4s0f0np0",
                phc_device="/dev/ptp0",
                runner=runner,
                host_log=host_log,
                container_log="/ptp/cse-ai-6.log",
            )
            text = lifecycle._read_log(session)
        self.assertIn("SLAVE", text)

    def test_cleanup_sweeps_leftover_sidecars_without_sessions(self) -> None:
        config = hardware_config()
        swept: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            lifecycle = Ptp4lLifecycle(
                config,
                tmp,
                "20260828_160000",
                worker_targets=[DockerTarget(hostname="cse-ai-10.amd.com")],
                sidecar_sweeper=lambda target: swept.append(target.label),
            )
            lifecycle.cleanup()
        self.assertEqual(swept, ["local", "cse-ai-10.amd.com"] * 2)

    def test_start_failure_stops_started_sidecars(self) -> None:
        config = hardware_config()
        runners: list[FakeSidecarRunner] = []

        def factory(target: DockerTarget) -> FakeSidecarRunner:
            runner = FakeSidecarRunner(target)
            runners.append(runner)
            return runner

        with tempfile.TemporaryDirectory() as tmp:
            lifecycle = Ptp4lLifecycle(
                config,
                tmp,
                "20260828_160000",
                master_inventory=FakeInventory(
                    DockerTarget(),
                    "cse-ai-9",
                    "enp4s0f0np0",
                    "/dev/ptp0",
                    ipv4=("192.168.0.9",),
                ),
                worker_inventories=[
                    FakeInventory(
                        DockerTarget(hostname="cse-ai-10.amd.com"),
                        "cse-ai-10",
                        "enp4s0f0np0",
                        "/dev/ptp0",
                        ipv4=("192.168.0.10",),
                    )
                ],
                runner_factory=factory,
                sidecar_sweeper=lambda target: None,
            )
            with (
                patch.object(
                    Ptp4lLifecycle,
                    "_wait_until_locked",
                    side_effect=[None, RuntimeError("slave did not lock")],
                ),
                self.assertRaisesRegex(RuntimeError, "slave did not lock"),
            ):
                lifecycle.start()

        self.assertTrue(all(runner.cleaned for runner in runners))
        self.assertFalse(lifecycle.active)
        self.assertEqual(lifecycle.sessions, [])

    def test_hardware_start_requires_inventory_runners(self) -> None:
        config = hardware_config()
        with tempfile.TemporaryDirectory() as tmp:
            lifecycle = Ptp4lLifecycle(config, tmp, "20260828_160000")
            with self.assertRaisesRegex(RuntimeError, "NetworkAssistant inventory"):
                lifecycle.start()

    def test_sweep_ptp4l_containers_removes_only_labeled_sidecars(self) -> None:
        sidecar = Mock()
        sidecar.name = "/vap_ptp4l_20260829_cse-ai-9_429b6949"
        sidecar.labels = {
            ptp4l_mod.VAP_MANAGED_LABEL: "true",
            ptp4l_mod.VAP_KIND_LABEL: "ptp4l",
        }
        other = Mock()
        other.name = "vap_ptp4l_unlabeled"
        other.labels = {}
        client = Mock()
        client.containers.list.return_value = [sidecar, other]
        with patch.object(ptp4l_mod, "create_docker_client", return_value=client):
            sweep_ptp4l_containers(DockerTarget())
        sidecar.remove.assert_called_once_with(force=True)
        other.remove.assert_not_called()
        client.close.assert_called_once()

    def test_clock_probe_uses_sidecar_log_override(self) -> None:
        config = hardware_config()
        probe = ClockProbeLifecycle(config, SimpleNamespace(), "/tmp/logs", "run")
        probe.set_ptp_logs(
            {"cse-ai-9": "/app/VAP/log/clock-probe/ptp4l/run/cse-ai-9.log"}
        )
        self.assertEqual(
            probe.probe_config().hardware_ptp_logs["cse-ai-9"],
            "/app/VAP/log/clock-probe/ptp4l/run/cse-ai-9.log",
        )


if __name__ == "__main__":
    unittest.main()
