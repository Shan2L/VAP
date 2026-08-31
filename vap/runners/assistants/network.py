from __future__ import annotations

import ipaddress
import json
import logging
import threading
import time
from dataclasses import dataclass

from vap.runners.assistants.process import (
    CommandExecutionError,
    DockerCommandExecutor,
)

logger = logging.getLogger("VAP")

_NETWORK_PROBE = r"""
import fcntl
import json
import socket
import struct
from pathlib import Path

SIOCGIFADDR = 0x8915
VIRTUAL_PREFIXES = (
    "docker", "br-", "veth", "virbr", "cni", "flannel",
    "cali", "kube", "tun", "tap",
)

def read(path, default=""):
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return default

def ipv4_of(name):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        request = struct.pack("256s", name.encode()[:15])
        result = fcntl.ioctl(sock.fileno(), SIOCGIFADDR, request)
        return socket.inet_ntoa(result[20:24])
    except OSError:
        return None
    finally:
        sock.close()

ipv6_by_name = {}
try:
    for line in Path("/proc/net/if_inet6").read_text().splitlines():
        raw, _, prefix, _, _, name = line.split()
        value = socket.inet_ntop(socket.AF_INET6, bytes.fromhex(raw))
        ipv6_by_name.setdefault(name, []).append(f"{value}/{int(prefix, 16)}")
except OSError:
    pass

interfaces = []
for path in sorted(Path("/sys/class/net").iterdir()):
    name = path.name
    if name == "lo" or name.lower().startswith(VIRTUAL_PREFIXES):
        continue
    device = path / "device"
    if not device.exists():
        continue

    type_value = read(path / "type")
    kind = "infiniband" if type_value == "32" or name.lower().startswith("ib") else "ethernet"
    speed_text = read(path / "speed")
    try:
        speed = int(speed_text)
        if speed < 0:
            speed = None
    except ValueError:
        speed = None

    driver_path = device / "driver"
    try:
        driver = driver_path.resolve().name if driver_path.exists() else None
    except OSError:
        driver = None
    try:
        device_name = device.resolve().name
        pci_address = device_name if ":" in device_name else None
    except OSError:
        pci_address = None

    ptp_names = sorted(
        candidate.name
        for candidate in (device / "ptp").glob("ptp*")
        if candidate.name.startswith("ptp")
    )
    ipv4 = ipv4_of(name)
    interfaces.append(
        {
            "name": name,
            "index": int(read(path / "ifindex", "0")),
            "mac": read(path / "address"),
            "ipv4": [ipv4] if ipv4 else [],
            "ipv6": sorted(ipv6_by_name.get(name, [])),
            "mtu": int(read(path / "mtu", "0")),
            "link_up": read(path / "operstate") == "up",
            "speed_mbps": speed,
            "driver": driver,
            "pci_address": pci_address,
            "interface_type": kind,
            "ptp_device": f"/dev/{ptp_names[0]}" if ptp_names else None,
        }
    )

print(json.dumps(interfaces, sort_keys=True))
"""

_SELECT_IP = r"""
import ipaddress
import socket
import sys

def is_rfc1918(value):
    address = ipaddress.IPv4Address(value)
    return (
        address in ipaddress.IPv4Network("10.0.0.0/8")
        or address in ipaddress.IPv4Network("172.16.0.0/12")
        or address in ipaddress.IPv4Network("192.168.0.0/16")
    )

peer = sys.argv[1]
peer_ip = socket.getaddrinfo(peer, None, socket.AF_INET, socket.SOCK_DGRAM)[0][4][0]
for candidate in sys.argv[2:]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((candidate, 0))
        sock.connect((peer_ip, 1))
        print(candidate)
        raise SystemExit(0)
    except OSError:
        pass
    finally:
        sock.close()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.connect((peer_ip, 1))
    routed = sock.getsockname()[0]
finally:
    sock.close()
if is_rfc1918(routed):
    print(routed)
    raise SystemExit(0)
raise SystemExit(f"no private source address can reach {peer}")
"""

_HTTP_STATUS = r"""
import sys
import urllib.error
import urllib.request

request = urllib.request.Request(sys.argv[1], method=sys.argv[2])
try:
    with urllib.request.urlopen(request, timeout=float(sys.argv[3])) as response:
        print(response.status)
except urllib.error.HTTPError as exc:
    print(exc.code)
"""


@dataclass(frozen=True)
class NetworkInterfaceInfo:
    name: str
    index: int
    mac: str
    ipv4: tuple[str, ...]
    ipv6: tuple[str, ...]
    mtu: int
    link_up: bool
    speed_mbps: int | None
    driver: str | None
    pci_address: str | None
    interface_type: str
    ptp_device: str | None


class NetworkAssistant:
    def __init__(
        self,
        executor: DockerCommandExecutor,
        target_label: str,
        *,
        auto_refresh: bool = True,
    ):
        self._executor = executor
        self._target_label = target_label
        self._lock = threading.RLock()
        self._interfaces: dict[str, NetworkInterfaceInfo] = {}
        if auto_refresh:
            self.refresh()

    @property
    def has_inventory(self) -> bool:
        with self._lock:
            return bool(self._interfaces)

    def ensure_inventory(self) -> tuple[NetworkInterfaceInfo, ...]:
        """Return the cached NIC snapshot, probing the container if it is empty."""
        with self._lock:
            if self._interfaces:
                return self._snapshot()
        logger.info(
            "NetworkAssistant inventory is empty on %s; probing NICs now",
            self._target_label,
        )
        return self.refresh()

    def refresh(self) -> tuple[NetworkInterfaceInfo, ...]:
        command = ["python3", "-c", _NETWORK_PROBE]
        result = self._executor.run(command)
        if result.exit_code != 0:
            raise CommandExecutionError(command, result)
        try:
            payload = json.loads(result.stdout_text)
            interfaces = {
                item["name"]: NetworkInterfaceInfo(
                    name=item["name"],
                    index=int(item["index"]),
                    mac=item["mac"],
                    ipv4=tuple(item["ipv4"]),
                    ipv6=tuple(item["ipv6"]),
                    mtu=int(item["mtu"]),
                    link_up=bool(item["link_up"]),
                    speed_mbps=item["speed_mbps"],
                    driver=item["driver"],
                    pci_address=item["pci_address"],
                    interface_type=item["interface_type"],
                    ptp_device=item["ptp_device"],
                )
                for item in payload
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid network inventory from {self._target_label}: "
                f"{result.stdout_text!r}"
            ) from exc
        with self._lock:
            self._interfaces = interfaces
            snapshot = self._snapshot()
        logger.info(
            "Discovered %d physical network interfaces on %s",
            len(snapshot),
            self._target_label,
        )
        return snapshot

    def interfaces(self) -> tuple[NetworkInterfaceInfo, ...]:
        with self._lock:
            return self._snapshot()

    def get_interface(self, name: str) -> NetworkInterfaceInfo | None:
        with self._lock:
            return self._interfaces.get(name)

    def find_by_ip(self, ip: str) -> NetworkInterfaceInfo | None:
        needle = ip.split("/", 1)[0]
        with self._lock:
            for interface in self._interfaces.values():
                addresses = (*interface.ipv4, *interface.ipv6)
                if any(value.split("/", 1)[0] == needle for value in addresses):
                    return interface
        return None

    def ptp_interfaces(self) -> tuple[NetworkInterfaceInfo, ...]:
        return tuple(
            interface
            for interface in self.interfaces()
            if interface.ptp_device is not None
        )

    def select_ptp_interface(
        self,
        *,
        preferred_interface: str | None = None,
        preferred_phc_device: str | None = None,
    ) -> NetworkInterfaceInfo:
        """Pick an UP NIC that already has a PHC in the saved inventory."""
        self.ensure_inventory()
        candidates = [
            interface for interface in self.ptp_interfaces() if interface.link_up
        ]
        if preferred_interface:
            selected = next(
                (
                    interface
                    for interface in candidates
                    if interface.name == preferred_interface
                ),
                None,
            )
            if selected is None:
                raise RuntimeError(
                    f"{self._target_label}: interface {preferred_interface!r} "
                    "is not UP with a PHC in the NetworkAssistant inventory"
                )
        elif preferred_phc_device:
            selected = next(
                (
                    interface
                    for interface in candidates
                    if interface.ptp_device == preferred_phc_device
                ),
                None,
            )
            if selected is None:
                raise RuntimeError(
                    f"{self._target_label}: PHC {preferred_phc_device!r} "
                    "is not on an UP NIC in the NetworkAssistant inventory"
                )
        else:
            if not candidates:
                raise RuntimeError(
                    f"{self._target_label}: no UP NIC with a PHC in the "
                    "NetworkAssistant inventory"
                )
            selected = sorted(
                candidates,
                key=lambda interface: (
                    0 if interface.interface_type == "ethernet" else 1,
                    0 if interface.ipv4 else 1,
                    -(interface.speed_mbps if interface.speed_mbps is not None else -1),
                    interface.index,
                    interface.name,
                ),
            )[0]
        if preferred_phc_device and selected.ptp_device != preferred_phc_device:
            raise RuntimeError(
                f"{self._target_label}: {selected.name} has {selected.ptp_device}, "
                f"not {preferred_phc_device}"
            )
        logger.info(
            "Selected PTP NIC %s (%s, %s Mb/s) on %s from NetworkAssistant inventory",
            selected.name,
            selected.ptp_device,
            selected.speed_mbps,
            self._target_label,
        )
        return selected

    def select_vllm_host_ip(self, peer: str) -> str:
        interfaces = self.refresh()
        candidates: list[tuple[int, str, str]] = []
        for interface in interfaces:
            rank = 0 if interface.interface_type == "infiniband" else 1
            for value in interface.ipv4:
                if _is_rfc1918(value):
                    candidates.append((rank, interface.name, value))
        candidates.sort()
        command = [
            "python3",
            "-c",
            _SELECT_IP,
            peer,
            *(value for _, _, value in candidates),
        ]
        result = self._executor.run(command)
        if result.exit_code != 0:
            raise CommandExecutionError(command, result)
        selected = result.stdout_text.strip().splitlines()[-1]
        if not _is_rfc1918(selected):
            raise RuntimeError(
                f"{self._target_label} selected non-private VLLM_HOST_IP: {selected}"
            )
        logger.info(
            "Selected VLLM_HOST_IP=%s on %s toward %s",
            selected,
            self._target_label,
            peer,
        )
        return selected

    def http_status(
        self,
        url: str,
        *,
        method: str = "GET",
        timeout_sec: float = 5,
    ) -> int | None:
        result = self._executor.run(
            [
                "python3",
                "-c",
                _HTTP_STATUS,
                url,
                method,
                str(timeout_sec),
            ]
        )
        if result.exit_code != 0:
            return None
        try:
            return int(result.stdout_text.strip())
        except ValueError:
            return None

    def wait_http(
        self,
        url: str,
        *,
        expected_status: int = 200,
        timeout_sec: float = 1800,
        poll_interval_sec: float = 5,
    ) -> None:
        deadline = time.monotonic() + timeout_sec
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            status = self.http_status(url)
            if status == expected_status:
                logger.info(
                    "%s ready at %s (attempt %d)", self._target_label, url, attempt
                )
                return
            logger.info(
                "Waiting for %s: %s returned %s (attempt %d)",
                self._target_label,
                url,
                status if status is not None else "no response",
                attempt,
            )
            time.sleep(poll_interval_sec)
        raise TimeoutError(
            f"{url} did not return HTTP {expected_status} within {timeout_sec:.0f}s"
        )

    def _snapshot(self) -> tuple[NetworkInterfaceInfo, ...]:
        return tuple(self._interfaces[name] for name in sorted(self._interfaces))


def _is_rfc1918(value: str) -> bool:
    address = ipaddress.ip_address(value.split("/", 1)[0])
    return (
        address in ipaddress.ip_network("10.0.0.0/8")
        or address in ipaddress.ip_network("172.16.0.0/12")
        or address in ipaddress.ip_network("192.168.0.0/16")
    )
