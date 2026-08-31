"""Inspect hardware-timestamp PHC capability on the local host or via SSH."""

from __future__ import annotations

import json
import socket
import subprocess
from typing import Any

from vap.clock_probe.execution.network import list_network_interfaces
from vap.clock_probe.sampling.phc import (
    discover_hardware_timestamp_phc,
    list_phc_device_paths,
)
from vap.runners.ssh_identity import resolve_ssh_identity

SSH_TIMEOUT_SEC = 12
REMOTE_HARDWARE_PROBE = r"""
import json
import os
import re
import socket
import subprocess
from pathlib import Path

PTP_CLOCK_INDEX_RE = re.compile(r"PTP Hardware Clock:\s*(?P<index>\d+)")
preferred_interface = os.environ.get("VAP_PREFERRED_IFACE") or None
preferred_phc = os.environ.get("VAP_PREFERRED_PHC") or None
preferred_ip = os.environ.get("VAP_PREFERRED_IP") or None
require_readable = os.environ.get("VAP_REQUIRE_READABLE") == "1"

def ethtool_hardware(name):
    result = subprocess.run(
        ["ethtool", "-T", name],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ethtool failed")
    capabilities = {
        line.strip() for line in result.stdout.splitlines() if line.startswith(("\t", " "))
    }
    match = PTP_CLOCK_INDEX_RE.search(result.stdout)
    hardware = {
        "interface": name,
        "hardware_transmit": "hardware-transmit" in capabilities,
        "hardware_receive": "hardware-receive" in capabilities,
        "hardware_raw_clock": "hardware-raw-clock" in capabilities,
        "ptp_clock_index": int(match.group("index")) if match else None,
    }
    hardware["usable"] = (
        hardware["hardware_transmit"]
        and hardware["hardware_receive"]
        and hardware["hardware_raw_clock"]
        and hardware["ptp_clock_index"] is not None
    )
    return hardware

records = json.loads(
    subprocess.run(
        ["ip", "-j", "address", "show"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout
)
interfaces = []
for record in records:
    flags = set(record.get("flags", []))
    interfaces.append(
        {
            "name": record["ifname"],
            "index": int(record["ifindex"]),
            "is_up": "UP" in flags,
            "is_loopback": "LOOPBACK" in flags,
            "ipv4_addresses": [
                str(address["local"])
                for address in record.get("addr_info", [])
                if address.get("family") == "inet"
            ],
        }
    )
candidates = [
    item for item in interfaces if item["is_up"] and not item["is_loopback"]
]
if preferred_interface:
    candidates = [item for item in candidates if item["name"] == preferred_interface]
    if not candidates:
        raise SystemExit(
            json.dumps(
                {
                    "usable": False,
                    "reason": "configured interface is not UP",
                    "interface": preferred_interface,
                    "phc_devices": [str(path) for path in sorted(Path("/dev").glob("ptp[0-9]*")) if path.exists()],
                    "diagnostics": [],
                }
            )
        )
else:
    candidates.sort(
        key=lambda item: (
            0 if preferred_ip and preferred_ip in (item.get("ipv4_addresses") or []) else 1,
            item["index"],
            item["name"],
        )
    )

diagnostics = []
selected = None
for item in candidates:
    try:
        hardware = ethtool_hardware(item["name"])
        if not hardware["usable"]:
            raise RuntimeError("does not advertise hardware TX/RX/raw timestamping")
        phc_device = Path(f"/dev/ptp{hardware['ptp_clock_index']}")
        if preferred_phc and str(phc_device) != preferred_phc:
            raise RuntimeError(f"advertised PHC is {phc_device}, not {preferred_phc}")
        if not phc_device.exists():
            raise FileNotFoundError(f"PHC device {phc_device} is missing")
        if not os.access(phc_device, os.R_OK):
            if require_readable:
                raise PermissionError(f"PHC device {phc_device} is not readable")
            selected = {
                "interface": item["name"],
                "phc_device": str(phc_device),
                "capture_method": None,
                "readable": False,
                "hardware_timestamping": hardware,
            }
            diagnostics.append({"usable": True, **selected})
            break
        selected = {
            "interface": item["name"],
            "phc_device": str(phc_device),
            "capture_method": None,
            "readable": True,
            "hardware_timestamping": hardware,
        }
        diagnostics.append({"usable": True, **selected})
        break
    except Exception as error:
        diagnostics.append(
            {"usable": False, "interface": item["name"], "reason": str(error)}
        )

print(
    json.dumps(
        {
            "usable": selected is not None,
            "selected": selected,
            "os_hostname": socket.gethostname(),
            "diagnostics": diagnostics,
            "phc_devices": [
                str(path)
                for path in sorted(Path("/dev").glob("ptp[0-9]*"))
                if path.exists()
            ],
            "reason": None
            if selected is not None
            else "No hardware-timestamp-capable PHC was found",
        },
        sort_keys=True,
    )
)
"""


def ssh_command(
    hostname: str,
    remote_command: list[str],
    ssh_key: str | None,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "RequestTTY=no",
    ]
    if ssh_key:
        identity = resolve_ssh_identity(ssh_key)
        if identity:
            command.extend(["-i", identity, "-o", "IdentitiesOnly=yes"])
    command.append(hostname)
    if extra_env:
        command.append("env")
        command.extend(f"{key}={value}" for key, value in extra_env.items())
    command.extend(remote_command)
    return command


def run_ssh_python(
    hostname: str,
    script: str,
    *,
    ssh_key: str | None = None,
    extra_env: dict[str, str] | None = None,
    timeout_sec: int = SSH_TIMEOUT_SEC,
) -> str:
    result = subprocess.run(
        ssh_command(
            hostname,
            ["python3", "-"],
            ssh_key,
            extra_env=extra_env,
        ),
        check=False,
        capture_output=True,
        text=True,
        input=script.lstrip("\n"),
        timeout=timeout_sec,
    )
    if result.returncode != 0 and not result.stdout.strip():
        detail = result.stderr.strip() or "ssh failed"
        raise RuntimeError(f"SSH to {hostname} failed: {detail}")
    return result.stdout


def list_node_phc_devices(
    hostname: str | None = None,
    *,
    ssh_key: str | None = None,
) -> list[str]:
    if not hostname:
        return list_phc_device_paths()
    payload = json.loads(
        run_ssh_python(hostname, REMOTE_HARDWARE_PROBE, ssh_key=ssh_key)
    )
    devices = payload.get("phc_devices") or []
    if not isinstance(devices, list):
        raise RuntimeError(f"{hostname} returned an invalid PHC device list")
    return [str(device) for device in devices]


def inspect_node_hardware_timestamping(
    *,
    node: str,
    hostname: str | None = None,
    ssh_key: str | None = None,
    preferred_interface: str | None = None,
    preferred_phc_device: str | None = None,
    node_address: str | None = None,
    capture: bool = False,
    require_readable: bool = False,
) -> dict[str, Any]:
    try:
        if hostname:
            extra_env = {
                "VAP_REQUIRE_READABLE": "1" if require_readable else "0",
            }
            if preferred_interface:
                extra_env["VAP_PREFERRED_IFACE"] = preferred_interface
            if preferred_phc_device:
                extra_env["VAP_PREFERRED_PHC"] = preferred_phc_device
            if node_address:
                extra_env["VAP_PREFERRED_IP"] = node_address
            payload = json.loads(
                run_ssh_python(
                    hostname,
                    REMOTE_HARDWARE_PROBE,
                    ssh_key=ssh_key,
                    extra_env=extra_env,
                )
            )
            selected = payload.get("selected")
            if not payload.get("usable") or not isinstance(selected, dict):
                raise RuntimeError(
                    payload.get("reason")
                    or json.dumps(payload.get("diagnostics") or [], sort_keys=True)
                )
            readable = selected.get("readable", True)
            suffix = ""
            if readable is False:
                suffix = " (not readable as this user; the run container will mount it)"
            return {
                "node": node,
                "usable": True,
                "readable": readable,
                "message": (
                    f"{node}: {selected['interface']} -> {selected['phc_device']}"
                    + suffix
                ),
                "diagnostics": payload.get("diagnostics") or [],
                **selected,
                "os_hostname": payload.get("os_hostname") or socket.gethostname(),
            }

        interfaces = [
            {
                "name": item.name,
                "index": item.index,
                "is_up": item.is_up,
                "is_loopback": item.is_loopback,
                "ipv4_addresses": [address.address for address in item.ipv4_addresses],
            }
            for item in list_network_interfaces()
        ]
        selected, diagnostics = discover_hardware_timestamp_phc(
            interfaces,
            preferred_interface=preferred_interface,
            preferred_phc_device=preferred_phc_device,
            node_address=node_address,
            hostname=node,
            capture=capture,
            require_readable=require_readable,
        )
        readable = selected.get("readable", True)
        suffix = ""
        if selected.get("capture_method"):
            suffix += f" ({selected['capture_method']})"
        if readable is False:
            suffix += " (not readable as this user; the run container will mount it)"
        return {
            "node": node,
            "usable": True,
            "readable": readable,
            "message": (
                f"{node}: {selected['interface']} -> {selected['phc_device']}" + suffix
            ),
            "diagnostics": diagnostics,
            **selected,
            "os_hostname": socket.gethostname(),
        }
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as error:
        return {
            "node": node,
            "usable": False,
            "message": f"{node}: {error}",
            "reason": str(error),
            "diagnostics": [],
        }
