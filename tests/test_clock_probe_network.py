from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from vap.clock_probe.execution.network import (
    InterfaceAddress,
    NetworkInterface,
    list_network_interfaces,
    route_to,
)


def interface(name: str, address: str) -> NetworkInterface:
    return NetworkInterface(
        name=name,
        index=1,
        is_up=True,
        is_loopback=False,
        software_transmit=True,
        software_receive=True,
        ipv4_addresses=(InterfaceAddress(address, 24, "global"),),
    )


class ClockProbeNetworkTests(unittest.TestCase):
    def test_list_network_interfaces_falls_back_when_ip_is_missing(self) -> None:
        records = [
            {
                "ifname": "enp196s0f1np1",
                "ifindex": 5,
                "flags": ["UP", "RUNNING"],
                "addr_info": [
                    {
                        "family": "inet",
                        "local": "192.168.0.9",
                        "prefixlen": 24,
                        "scope": "global",
                    }
                ],
            }
        ]
        with (
            patch(
                "vap.clock_probe.execution.network.subprocess.run",
                side_effect=FileNotFoundError("ip"),
            ),
            patch(
                "vap.clock_probe.execution.network._sysfs_address_records",
                return_value=records,
            ),
            patch(
                "vap.clock_probe.execution.network._ethtool_capabilities",
                return_value=(True, True, None),
            ),
        ):
            interfaces = list_network_interfaces()

        self.assertEqual(len(interfaces), 1)
        self.assertEqual(interfaces[0].name, "enp196s0f1np1")
        self.assertEqual(interfaces[0].ipv4_addresses[0].address, "192.168.0.9")

    def test_route_falls_back_to_socket_when_ip_is_missing(self) -> None:
        class FakeSocket:
            def connect(self, addr):
                return None

            def getsockname(self):
                return ("10.67.93.244", 0)

            def close(self):
                return None

        with (
            patch(
                "vap.clock_probe.execution.network.subprocess.run",
                side_effect=FileNotFoundError("ip"),
            ),
            patch(
                "vap.clock_probe.execution.network.socket.socket",
                return_value=FakeSocket(),
            ),
        ):
            route = route_to("10.67.91.123", [interface("mgmt0", "10.67.93.244")])

        self.assertTrue(route["usable"])
        self.assertEqual(route["interface"], "mgmt0")
        self.assertEqual(route["source_address"], "10.67.93.244")

    def test_route_lookup_failure_is_still_reported(self) -> None:
        with patch(
            "vap.clock_probe.execution.network.subprocess.run",
            side_effect=subprocess.CalledProcessError(2, "ip"),
        ):
            route = route_to("10.67.91.123", [interface("mgmt0", "10.67.93.244")])
        self.assertFalse(route["usable"])
        self.assertIn("route lookup failed", route["reason"])


if __name__ == "__main__":
    unittest.main()
