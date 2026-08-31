from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vap.clock_probe.execution import host as host_mod
from vap.clock_probe.sampling import phc as phc_mod
from vap.clock_probe.sampling.phc import (
    HardwareTimestamping,
    discover_hardware_timestamp_phc,
    inspect_interface_hardware,
    inspect_interface_phc_capability,
    order_hardware_interface_candidates,
    parse_ethtool_hardware,
)

ETHTOOL_HARDWARE = """\
Time stamping parameters for enp4s0f1np1:
Capabilities:
	hardware-transmit
	software-transmit
	hardware-receive
	software-receive
	hardware-raw-clock
PTP Hardware Clock: 1
"""


class HardwarePhcDiscoveryTests(unittest.TestCase):
    def test_parse_ethtool_hardware_requires_tx_rx_raw_and_index(self) -> None:
        hardware = parse_ethtool_hardware(ETHTOOL_HARDWARE, "enp4s0f1np1")

        self.assertTrue(hardware.usable)
        self.assertEqual(hardware.ptp_clock_index, 1)
        self.assertEqual(hardware.interface, "enp4s0f1np1")

    def test_inspect_falls_back_to_sysfs_when_ethtool_is_missing(self) -> None:
        with (
            patch.object(
                phc_mod.subprocess,
                "run",
                side_effect=FileNotFoundError("ethtool"),
            ),
            patch.object(phc_mod, "sysfs_ptp_index", return_value=3),
        ):
            hardware = inspect_interface_hardware("enp196s0f1np1")

        self.assertTrue(hardware.usable)
        self.assertEqual(hardware.ptp_clock_index, 3)
        self.assertEqual(hardware.interface, "enp196s0f1np1")

    def test_inspect_sysfs_skips_nics_without_a_phc(self) -> None:
        with (
            patch.object(
                phc_mod.subprocess,
                "run",
                side_effect=FileNotFoundError("ethtool"),
            ),
            patch.object(phc_mod, "sysfs_ptp_index", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "no sysfs PHC"):
                inspect_interface_hardware("docker0")

    def test_order_prefers_the_interface_that_owns_the_node_address(self) -> None:
        ordered = order_hardware_interface_candidates(
            [
                {
                    "name": "eth0",
                    "index": 2,
                    "is_up": True,
                    "is_loopback": False,
                    "ipv4_addresses": ["10.0.0.2"],
                },
                {
                    "name": "eth1",
                    "index": 3,
                    "is_up": True,
                    "is_loopback": False,
                    "ipv4_addresses": ["10.0.0.1"],
                },
            ],
            node_address="10.0.0.1",
        )

        self.assertEqual(ordered[0]["name"], "eth1")

    def test_discover_selects_the_first_usable_interface(self) -> None:
        interfaces = [
            {
                "name": "eth0",
                "index": 2,
                "is_up": True,
                "is_loopback": False,
                "ipv4_addresses": ["10.0.0.2"],
            },
            {
                "name": "eth1",
                "index": 3,
                "is_up": True,
                "is_loopback": False,
                "ipv4_addresses": ["10.0.0.1"],
            },
        ]

        def inspect(name: str, **kwargs):
            return {
                "interface": name,
                "phc_device": "/dev/ptp1",
                "capture_method": None,
                "hardware_timestamping": {"usable": True},
            }

        with patch.object(
            phc_mod,
            "inspect_interface_phc_capability",
            side_effect=inspect,
        ) as inspect_mock:
            selected, diagnostics = discover_hardware_timestamp_phc(
                interfaces,
                node_address="10.0.0.1",
            )

        self.assertEqual(selected["interface"], "eth1")
        self.assertEqual(selected["phc_device"], "/dev/ptp1")
        self.assertTrue(diagnostics[0]["usable"])
        inspect_mock.assert_called_once_with(
            "eth1",
            preferred_phc_device=None,
            capture=False,
            require_readable=True,
        )

    def test_discover_fails_when_no_interface_has_hardware_timestamping(self) -> None:
        with patch.object(
            phc_mod,
            "inspect_interface_phc_capability",
            side_effect=RuntimeError("no hardware timestamping"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "No hardware-timestamp-capable PHC"
            ):
                discover_hardware_timestamp_phc(
                    [
                        {
                            "name": "eth0",
                            "index": 2,
                            "is_up": True,
                            "is_loopback": False,
                            "ipv4_addresses": ["10.0.0.2"],
                        }
                    ]
                )


class HardwareHostInspectTests(unittest.TestCase):
    def test_local_inspect_uses_discovered_phc(self) -> None:
        interface = SimpleNamespace(
            name="eth1",
            index=3,
            is_up=True,
            is_loopback=False,
            ipv4_addresses=[SimpleNamespace(address="10.0.0.1")],
        )
        selected = {
            "interface": "eth1",
            "phc_device": "/dev/ptp1",
            "capture_method": "gettime",
            "hardware_timestamping": {"usable": True},
        }
        with (
            patch.object(host_mod, "list_network_interfaces", return_value=[interface]),
            patch.object(
                host_mod,
                "discover_hardware_timestamp_phc",
                return_value=(selected, [{"usable": True, **selected}]),
            ) as discover,
        ):
            result = host_mod.inspect_node_hardware_timestamping(
                node="local",
                capture=True,
            )

        self.assertTrue(result["usable"])
        self.assertEqual(result["interface"], "eth1")
        self.assertIn("/dev/ptp1", result["message"])
        discover.assert_called_once()
        self.assertTrue(discover.call_args.kwargs["capture"])
        self.assertFalse(discover.call_args.kwargs["require_readable"])
        self.assertIsNone(discover.call_args.kwargs["node_address"])

    def test_local_inspect_forwards_node_address(self) -> None:
        interface = SimpleNamespace(
            name="eth1",
            index=3,
            is_up=True,
            is_loopback=False,
            ipv4_addresses=[SimpleNamespace(address="192.168.0.9")],
        )
        selected = {
            "interface": "eth1",
            "phc_device": "/dev/ptp1",
            "capture_method": None,
            "hardware_timestamping": {"usable": True},
        }
        with (
            patch.object(host_mod, "list_network_interfaces", return_value=[interface]),
            patch.object(
                host_mod,
                "discover_hardware_timestamp_phc",
                return_value=(selected, [{"usable": True, **selected}]),
            ) as discover,
        ):
            host_mod.inspect_node_hardware_timestamping(
                node="local",
                node_address="192.168.0.9",
            )
        self.assertEqual(discover.call_args.kwargs["node_address"], "192.168.0.9")

    def test_remote_inspect_does_not_use_local_ethtool(self) -> None:
        payload = {
            "usable": True,
            "selected": {
                "interface": "ens3f0",
                "phc_device": "/dev/ptp2",
                "capture_method": None,
                "hardware_timestamping": {"usable": True},
            },
            "os_hostname": "worker-a",
            "diagnostics": [],
        }
        with (
            patch.object(
                host_mod, "run_ssh_python", return_value=json.dumps(payload)
            ) as ssh,
            patch.object(host_mod, "list_network_interfaces") as local_ifaces,
            patch.object(host_mod, "discover_hardware_timestamp_phc") as discover,
        ):
            result = host_mod.inspect_node_hardware_timestamping(
                node="worker-a",
                hostname="worker-a.example.com",
                ssh_key="/tmp/id_ed25519",
                node_address="192.168.0.10",
            )

        self.assertTrue(result["usable"])
        self.assertEqual(result["phc_device"], "/dev/ptp2")
        self.assertEqual(result["os_hostname"], "worker-a")
        ssh.assert_called_once()
        self.assertEqual(ssh.call_args.kwargs["extra_env"]["VAP_REQUIRE_READABLE"], "0")
        self.assertEqual(
            ssh.call_args.kwargs["extra_env"]["VAP_PREFERRED_IP"], "192.168.0.10"
        )
        local_ifaces.assert_not_called()
        discover.assert_not_called()

    def test_remote_inspect_returns_unusable_on_ssh_failure(self) -> None:
        with patch.object(
            host_mod,
            "run_ssh_python",
            side_effect=RuntimeError("Host key verification failed"),
        ):
            result = host_mod.inspect_node_hardware_timestamping(
                node="worker-a",
                hostname="worker-a.example.com",
            )

        self.assertFalse(result["usable"])
        self.assertIn("Host key verification failed", result["message"])

    def test_remote_inspect_warns_when_phc_is_not_readable(self) -> None:
        payload = {
            "usable": True,
            "selected": {
                "interface": "enp4s0f0np0",
                "phc_device": "/dev/ptp0",
                "capture_method": None,
                "readable": False,
                "hardware_timestamping": {"usable": True},
            },
            "diagnostics": [],
        }
        with patch.object(host_mod, "run_ssh_python", return_value=json.dumps(payload)):
            result = host_mod.inspect_node_hardware_timestamping(
                node="cse-ai-6.amd.com",
                hostname="cse-ai-6.amd.com",
            )

        self.assertTrue(result["usable"])
        self.assertFalse(result["readable"])
        self.assertIn("not readable as this user", result["message"])

    def test_run_ssh_python_pipes_script_on_stdin(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")
        with patch.object(host_mod.subprocess, "run", return_value=completed) as run:
            output = host_mod.run_ssh_python(
                "cse-ai-6.amd.com",
                "\nimport json\nprint(json.dumps({'ok': True}))\n",
            )

        self.assertEqual(output, '{"ok": true}')
        command = run.call_args.args[0]
        self.assertEqual(command[-2:], ["python3", "-"])
        self.assertNotIn("-c", command)
        self.assertIn("RequestTTY=no", command)
        self.assertIn("import json", run.call_args.kwargs["input"])


class HardwarePhcReadabilityTests(unittest.TestCase):
    def _hardware(self) -> HardwareTimestamping:
        return HardwareTimestamping(
            interface="enp4s0f0np0",
            hardware_transmit=True,
            hardware_receive=True,
            hardware_raw_clock=True,
            ptp_clock_index=0,
        )

    def test_unreadable_phc_is_accepted_when_not_required(self) -> None:
        with tempfile.NamedTemporaryFile() as tmp:
            phc = Path(tmp.name)
            with (
                patch.object(
                    phc_mod,
                    "inspect_interface_hardware",
                    return_value=self._hardware(),
                ),
                patch.object(phc_mod, "phc_device_for_index", return_value=phc),
                patch.object(phc_mod.os, "access", return_value=False),
            ):
                selected = inspect_interface_phc_capability(
                    "enp4s0f0np0",
                    require_readable=False,
                )

        self.assertFalse(selected["readable"])
        self.assertEqual(selected["phc_device"], str(phc))

    def test_unreadable_phc_is_rejected_when_required(self) -> None:
        with tempfile.NamedTemporaryFile() as tmp:
            phc = Path(tmp.name)
            with (
                patch.object(
                    phc_mod,
                    "inspect_interface_hardware",
                    return_value=self._hardware(),
                ),
                patch.object(phc_mod, "phc_device_for_index", return_value=phc),
                patch.object(phc_mod.os, "access", return_value=False),
            ):
                with self.assertRaisesRegex(PermissionError, "not readable"):
                    inspect_interface_phc_capability(
                        "enp4s0f0np0",
                        require_readable=True,
                    )


class PhcBridgeCalibrationTests(unittest.TestCase):
    def test_trailing_short_group_is_coalesced_not_skipped(self) -> None:
        from vap.clock_probe.calibration.clock_bridge import (
            ClockBridgeConfig,
            build_clock_bridge,
        )

        interval_ns = 50_000_000
        samples = []
        for index in range(23):
            monotonic_ns = index * interval_ns
            samples.append(
                {
                    "bridge_monotonic_ns": monotonic_ns,
                    "bridge_realtime_ns": monotonic_ns + 1_000,
                    "bridge_read_span_ns": 500,
                }
            )

        bridge = build_clock_bridge(
            samples,
            boot_id="boot",
            config=ClockBridgeConfig(segment_seconds=1.0, min_segment_samples=10),
        )

        self.assertEqual(bridge["health"]["skipped_group_count"], 0)
        self.assertEqual(bridge["health"]["segment_count"], 1)
        self.assertEqual(bridge["status"], "PASS")

    def test_auto_phc_bridge_survives_leftover_affine_tail(self) -> None:
        from vap.clock_probe.calibration.clock_bridge import ClockBridgeConfig
        from vap.clock_probe.calibration.phc_bridge import build_phc_bridge

        interval_ns = 50_000_000
        samples = []
        for index in range(80):
            phc_ns = 1_000_000_000 + index * interval_ns
            realtime_ns = phc_ns + 10_000
            samples.append(
                {
                    "bridge_phc_ns": phc_ns,
                    "bridge_realtime_ns": realtime_ns,
                    "bridge_read_span_ns": 500,
                }
            )

        bridge = build_phc_bridge(
            samples,
            boot_id="boot",
            config=ClockBridgeConfig(
                segment_seconds=1.0,
                max_validation_p95_us=1.0,
                min_segment_samples=10,
            ),
            method="auto",
            candidate_segment_seconds=(0.5, 1.0),
            candidate_sample_strides=(1,),
            tuning_fraction=0.6,
        )

        self.assertEqual(bridge["status"], "PASS")
        self.assertIn(
            bridge["model_selection"]["selected_method"],
            {"interpolation", "piecewise_affine"},
        )

    def test_select_candidate_tries_next_when_validation_score_raises(self) -> None:
        from vap.clock_probe.calibration.core import Candidate, select_candidate

        samples = [{"t": index, "v": index} for index in range(6)]

        def build(name: str, selected: list[dict]) -> dict:
            return {"status": "PASS", "name": name, "n": len(selected)}

        def score(payload: dict) -> dict[str, float]:
            if payload["name"] == "affine" and payload["n"] < 3:
                raise ValueError("PHC bridge has skipped affine groups")
            return {"uncertainty_us": 0.5 if payload["name"] == "affine" else 0.8}

        payload, selection = select_candidate(
            samples,
            [
                Candidate("affine", {"method": "affine"}, complexity=0),
                Candidate("interp", {"method": "interp"}, complexity=1),
            ],
            tuning_fraction=0.6,
            time_key=lambda sample: int(sample["t"]),
            build=build,
            score=score,
            objective_key="uncertainty_us",
        )

        self.assertEqual(payload["name"], "interp")
        self.assertEqual(selection["selected"]["method"], "interp")
        self.assertEqual(selection["validation_status"], "PASS")

    def test_hardware_model_fits_bridge_even_when_ptp_is_over_budget(self) -> None:
        from vap.clock_probe.calibration import hardware as hardware_mod
        from vap.clock_probe.calibration.ptp_health import PtpHealth

        health = PtpHealth(
            role="slave",
            port_state="SLAVE",
            grandmaster_clock_id="cc40f3.fffe.f57c15",
            assuming_grandmaster=False,
            summary_count=100,
            clockcheck_count=0,
            late_clockcheck_count=0,
            offset_rms_p50_ns=92.0,
            offset_rms_p95_ns=4164.0,
            offset_max_ns=412_740,
            path_delay_ns=2760.0,
            freq_ppb=4133.0,
            lock_ok=False,
            status="FAIL",
            reasons=["ptp4l rms p95 4164.2 ns exceeds 1000.0 ns"],
        )
        bridge = {
            "status": "PASS",
            "uncertainty_us": 0.5,
            "valid_from_phc_ns": 1_000,
            "segments": [
                {"valid_from_phc_ns": 1_000, "status": "PASS", "uncertainty_us": 0.5}
            ],
        }
        with patch.object(
            hardware_mod, "build_phc_bridge", return_value=bridge
        ) as build_bridge:
            model = hardware_mod.build_hardware_model(
                [{"bridge_realtime_ns": 1}],
                role="slave",
                ptp_health=health,
                source={"hostname": "cse-ai-6", "boot_id": "boot-a"},
            )

        self.assertEqual(model["status"], "FAIL")
        self.assertEqual(model["realtime_phc_bridge"], bridge)
        self.assertIsNone(build_bridge.call_args.kwargs["max_uncertainty_us"])
        self.assertTrue(
            any("4164.2 ns exceeds" in reason for reason in model["fail_reasons"])
        )

    def test_hardware_preflight_uses_ptp_sidecar_interface_and_phc(self) -> None:
        from vap.clock_probe.execution import ray as ray_mod

        node = {
            "node_id": "node-a",
            "name": "worker-a",
            "address": "192.168.0.10",
            "is_head": False,
            "resources": {},
        }
        log = (
            "ptp4l[1.0]: selected best master clock cc40f3.fffe.f57c15\n"
            "ptp4l[1.0]: port 1 (eth9): UNCALIBRATED to SLAVE on X\n"
            + "".join(
                f"ptp4l[{2 + index}.0]: rms 20 max 30 freq +1 +/- 1 "
                "delay 100 +/- 1\n"
                for index in range(12)
            )
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(ray_mod, "list_network_interfaces", return_value=[]),
            patch.object(ray_mod, "read_boot_id", return_value="boot-a"),
        ):
            log_path = Path(tmp) / "ptp4l.log"
            log_path.write_text(log, encoding="utf-8")
            agent = ray_mod.ClockNodeAgent(
                node=node,
                reference=node,
                session_id="session",
                raw_output_root=tmp,
            )
            selected = {
                "interface": "eth9",
                "phc_device": "/dev/ptp9",
                "hardware_timestamping": {},
                "capture_method": "precise",
            }
            with patch.object(
                agent,
                "_discover_hardware_phc",
                return_value=(selected, []),
            ) as discover:
                result = agent.hardware_preflight(
                    {
                        "ptp_nodes": {
                            "192.168.0.10": {
                                "interface": "eth9",
                                "phc_device": "/dev/ptp9",
                                "ptp_log": str(log_path),
                            }
                        },
                        "model_config": {},
                    }
                )

        self.assertTrue(result["usable"], result)
        preflight_config = discover.call_args.args[0]
        self.assertEqual(preflight_config["interface"], "eth9")
        self.assertEqual(preflight_config["phc_device"], "/dev/ptp9")

    def test_hardware_session_rejects_missing_master_clock_identity(self) -> None:
        from vap.clock_probe.calibration.hardware import build_hardware_session

        models = [
            {
                "status": "PASS",
                "uncertainty_us": 0.5,
                "source": {"hostname": "master"},
                "ptp": {"role": "master", "grandmaster_clock_id": None},
                "realtime_phc_bridge": {"valid_from_phc_ns": 1},
            },
            {
                "status": "PASS",
                "uncertainty_us": 0.5,
                "source": {"hostname": "slave"},
                "ptp": {
                    "role": "slave",
                    "grandmaster_clock_id": "cc40f3.fffe.f57c15",
                },
                "realtime_phc_bridge": {"valid_from_phc_ns": 1},
            },
        ]

        with self.assertRaisesRegex(ValueError, "no local grandmaster clock id"):
            build_hardware_session(models)

    def test_hardware_session_uses_nodes_that_have_phc_segments(self) -> None:
        from vap.clock_probe.calibration.hardware import build_hardware_session

        master = {
            "status": "PASS",
            "uncertainty_us": 0.6,
            "ptp_uncertainty_us": 0.0,
            "source": {"hostname": "cse-ai-9"},
            "ptp": {
                "role": "master",
                "grandmaster_clock_id": "cc40f3.fffe.f57c15",
            },
            "realtime_phc_bridge": {
                "status": "PASS",
                "valid_from_phc_ns": 3_600_000_000_000,
                "segments": [{"valid_from_phc_ns": 3_600_000_000_000}],
            },
            "collection": {"started_at": "2026-08-30T04:07:16+00:00"},
        }
        slave = {
            "status": "FAIL",
            "uncertainty_us": 4.44,
            "ptp_uncertainty_us": 4.44,
            "source": {"hostname": "cse-ai-6"},
            "ptp": {
                "role": "slave",
                "grandmaster_clock_id": "cc40f3.fffe.f57c15",
            },
            "realtime_phc_bridge": {},
            "fail_reasons": ["ptp4l rms p95 4164.2 ns exceeds 1000.0 ns"],
            "collection": {"started_at": "2026-08-30T04:07:16+00:00"},
        }

        session = build_hardware_session([master, slave], session_id="run-1")

        self.assertEqual(session["status"], "FAIL")
        self.assertEqual(session["target_base_time_ns"], 3_600_000_000_000)
        self.assertEqual(session["failures"][0]["hostname"], "cse-ai-6")


if __name__ == "__main__":
    unittest.main()
