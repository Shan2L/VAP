"""Linux interface discovery for software-timestamped clock traffic."""

from __future__ import annotations

import fcntl
import ipaddress
import json
import socket
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

SIOCGIFADDR = 0x8915
SIOCGIFNETMASK = 0x891B


@dataclass(frozen=True)
class InterfaceAddress:
    """One IPv4 address assigned to an interface."""

    address: str
    prefix_length: int
    scope: str


@dataclass(frozen=True)
class NetworkInterface:  # pylint: disable=too-many-instance-attributes
    """Interface state and Linux timestamping capabilities."""

    name: str
    index: int
    is_up: bool
    is_loopback: bool
    software_transmit: bool
    software_receive: bool
    ipv4_addresses: tuple[InterfaceAddress, ...]
    capability_error: str | None = None

    @property
    def supports_software_timestamping(self) -> bool:
        """Whether both required software timestamp directions are present."""
        return self.software_transmit and self.software_receive

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/Ray-serializable representation."""
        return asdict(self) | {
            "supports_software_timestamping": (self.supports_software_timestamping)
        }


def parse_timestamping_capabilities(output: str) -> tuple[bool, bool]:
    """Parse ``ethtool -T`` software TX/RX capability lines."""
    capabilities = {
        line.strip() for line in output.splitlines() if line.startswith(("\t", " "))
    }
    return (
        "software-transmit" in capabilities,
        "software-receive" in capabilities,
    )


def _ethtool_capabilities(interface_name: str) -> tuple[bool, bool, str | None]:
    try:
        result = subprocess.run(
            ["ethtool", "-T", interface_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return False, False, repr(error)
    if result.returncode != 0:
        return False, False, result.stderr.strip() or result.stdout.strip()
    transmit, receive = parse_timestamping_capabilities(result.stdout)
    return transmit, receive, None


def parse_interface_records(
    records: Sequence[dict[str, Any]],
    capabilities: dict[str, tuple[bool, bool, str | None]],
) -> list[NetworkInterface]:
    """Convert ``ip -j address`` records into normalized interfaces."""
    interfaces = []
    for record in records:
        name = str(record["ifname"])
        transmit, receive, capability_error = capabilities.get(
            name,
            (False, False, "timestamp capabilities were not inspected"),
        )
        addresses = tuple(
            InterfaceAddress(
                address=str(address["local"]),
                prefix_length=int(address["prefixlen"]),
                scope=str(address.get("scope", "unknown")),
            )
            for address in record.get("addr_info", [])
            if address.get("family") == "inet"
        )
        flags = set(record.get("flags", []))
        interfaces.append(
            NetworkInterface(
                name=name,
                index=int(record["ifindex"]),
                is_up="UP" in flags,
                is_loopback="LOOPBACK" in flags,
                software_transmit=transmit,
                software_receive=receive,
                ipv4_addresses=addresses,
                capability_error=capability_error,
            )
        )
    return interfaces


def list_network_interfaces() -> list[NetworkInterface]:
    """Inspect local IPv4 addresses and software timestamp support."""
    records = _ip_address_records()
    if records is None:
        records = _sysfs_address_records()
    capabilities = {
        str(record["ifname"]): _ethtool_capabilities(str(record["ifname"]))
        for record in records
    }
    return parse_interface_records(records, capabilities)


def _ip_address_records() -> list[dict[str, Any]] | None:
    try:
        result = subprocess.run(
            ["ip", "-j", "address", "show"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    return payload


def _sysfs_address_records() -> list[dict[str, Any]]:
    """Build ``ip -j address`` shaped records without iproute2."""
    records: list[dict[str, Any]] = []
    try:
        names = socket.if_nameindex()
    except OSError as error:
        raise RuntimeError(f"Unable to inspect Linux interfaces: {error!r}") from error
    for index, name in names:
        records.append(
            {
                "ifname": name,
                "ifindex": int(index),
                "flags": _sysfs_flags(name),
                "addr_info": _ioctl_addr_info(name),
            }
        )
    return records


def _sysfs_flags(name: str) -> list[str]:
    flags: list[str] = []
    try:
        value = int((Path("/sys/class/net") / name / "flags").read_text(), 16)
    except (OSError, ValueError):
        return flags
    if value & 0x1:
        flags.append("UP")
    if value & 0x8:
        flags.append("LOOPBACK")
    if value & 0x40:
        flags.append("RUNNING")
    return flags


def _ioctl_addr_info(name: str) -> list[dict[str, Any]]:
    request = struct.pack("256s", name.encode()[:15])
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        try:
            addr_raw = fcntl.ioctl(sock.fileno(), SIOCGIFADDR, request)
            mask_raw = fcntl.ioctl(sock.fileno(), SIOCGIFNETMASK, request)
        except OSError:
            return []
        address = socket.inet_ntoa(addr_raw[20:24])
        netmask = socket.inet_ntoa(mask_raw[20:24])
        prefix_length = ipaddress.IPv4Network(f"0.0.0.0/{netmask}").prefixlen
        return [
            {
                "family": "inet",
                "local": address,
                "prefixlen": prefix_length,
                "scope": _ipv4_scope(address),
            }
        ]
    finally:
        sock.close()


def _ipv4_scope(address: str) -> str:
    parsed = ipaddress.ip_address(address)
    if parsed.is_loopback:
        return "host"
    if parsed.is_link_local:
        return "link"
    return "global"


def reference_candidates(
    interfaces: Sequence[NetworkInterface],
    *,
    preferred_address: str | None = None,
    preferred_interface: str | None = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Return eligible Head addresses in deterministic preference order."""
    candidates: list[dict[str, Any]] = []
    for interface in interfaces:
        if (
            not interface.is_up
            or interface.is_loopback
            or not interface.supports_software_timestamping
        ):
            continue
        for address in interface.ipv4_addresses:
            ip_address = ipaddress.ip_address(address.address)
            if (
                address.scope != "global"
                or ip_address.is_loopback
                or ip_address.is_link_local
            ):
                continue
            candidates.append(
                {
                    "interface": interface.name,
                    "address": address.address,
                    "prefix_length": address.prefix_length,
                }
            )

    if (
        strict
        and preferred_address is not None
        and not any(
            candidate["address"] == preferred_address for candidate in candidates
        )
    ):
        raise RuntimeError(
            f"Requested reference address {preferred_address!r} is not on an "
            "UP interface with software-transmit and software-receive"
        )
    if (
        strict
        and preferred_interface is not None
        and not any(
            candidate["interface"] == preferred_interface for candidate in candidates
        )
    ):
        raise RuntimeError(
            f"Requested reference interface {preferred_interface!r} has no "
            "eligible IPv4 address with software timestamping"
        )

    return sorted(
        (
            candidate
            for candidate in candidates
            if (
                not strict
                or preferred_address is None
                or candidate["address"] == preferred_address
            )
            and (
                not strict
                or preferred_interface is None
                or candidate["interface"] == preferred_interface
            )
        ),
        key=lambda candidate: (
            candidate["address"] != preferred_address,
            candidate["interface"] != preferred_interface,
            candidate["interface"],
            ipaddress.ip_address(candidate["address"]),
        ),
    )


def route_to(
    destination: str,
    interfaces: Sequence[NetworkInterface],
) -> dict[str, Any]:
    """Inspect the local route and timestamp capability to a destination."""
    try:
        result = subprocess.run(
            ["ip", "-j", "route", "get", destination],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        records = json.loads(result.stdout)
    except FileNotFoundError:
        return _route_to_via_socket(destination, interfaces)
    except (
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        return {
            "destination": destination,
            "usable": False,
            "reason": f"route lookup failed: {error!r}",
        }

    if not records:
        return {
            "destination": destination,
            "usable": False,
            "reason": "route lookup returned no route",
        }
    route = records[0]
    return _describe_route(
        destination,
        interfaces,
        interface_name=route.get("dev"),
        source_address=route.get("prefsrc") or route.get("src"),
        gateway=route.get("gateway"),
    )


def _route_to_via_socket(
    destination: str,
    interfaces: Sequence[NetworkInterface],
) -> dict[str, Any]:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((destination, 1))
            source_address = sock.getsockname()[0]
        finally:
            sock.close()
    except OSError as error:
        return {
            "destination": destination,
            "usable": False,
            "reason": f"route lookup failed: {error!r}",
        }
    interface = next(
        (
            candidate
            for candidate in interfaces
            if source_address
            in {address.address for address in candidate.ipv4_addresses}
        ),
        None,
    )
    return _describe_route(
        destination,
        interfaces,
        interface_name=None if interface is None else interface.name,
        source_address=source_address,
        gateway=None,
    )


def _describe_route(
    destination: str,
    interfaces: Sequence[NetworkInterface],
    *,
    interface_name: str | None,
    source_address: str | None,
    gateway: str | None,
) -> dict[str, Any]:
    interface = next(
        (candidate for candidate in interfaces if candidate.name == interface_name),
        None,
    )
    if interface is None:
        reason = f"route uses unknown interface {interface_name!r}"
    elif not interface.is_up:
        reason = f"route interface {interface.name!r} is not UP"
    elif not interface.supports_software_timestamping:
        reason = (
            f"route interface {interface.name!r} lacks software TX/RX " "timestamping"
        )
    elif not source_address:
        reason = "route did not provide a source IPv4 address"
    elif source_address not in {
        address.address for address in interface.ipv4_addresses
    }:
        reason = (
            f"route source {source_address!r} is not assigned to " f"{interface.name!r}"
        )
    else:
        reason = None

    return {
        "destination": destination,
        "usable": reason is None,
        "interface": interface_name,
        "source_address": source_address,
        "gateway": gateway,
        "reason": reason,
    }
