from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def parse_cookie_header(header: str | None) -> dict[str, str]:
    cookies: dict[str, str] = {}
    if not header:
        return cookies
    for part in header.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        cookies[key.strip()] = value.strip()
    return cookies


def same_origin_allowed(origin: str | None, host: str | None) -> bool:
    if not origin:
        return True
    parsed = urlparse(origin)
    if parsed.scheme not in {"http", "https"}:
        return False
    if host and parsed.netloc == host:
        return True
    port = host.rsplit(":", 1)[1] if host and ":" in host else ""
    local_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
    return parsed.netloc in local_hosts


def discover_network_hosts() -> list[str]:
    """Return hostname/IP candidates that may be reachable from the LAN."""
    candidates: list[str] = []

    def add(host: str | None) -> None:
        value = (host or "").strip()
        if not value or any(ch.isspace() for ch in value):
            return
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            if value.lower() == "localhost":
                return
        else:
            if address.is_loopback or address.is_unspecified:
                return
        if value not in candidates:
            candidates.append(value)

    hostname = socket.getfqdn() or socket.gethostname()
    add(hostname)
    try:
        for info in socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM
        ):
            add(info[4][0])
    except OSError:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            add(probe.getsockname()[0])
    except OSError:
        pass
    return candidates


def build_session_urls(bind_host: str, port: int, token: str) -> list[tuple[str, str]]:
    hosts: list[tuple[str, str]]
    if bind_host in {"0.0.0.0", "::"}:
        hosts = [("Local", "127.0.0.1")]
        hosts.extend(("Network candidate", host) for host in discover_network_hosts())
    else:
        hosts = [("Session", bind_host)]

    urls: list[tuple[str, str]] = []
    for label, host in hosts:
        url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
        urls.append((label, f"http://{url_host}:{port}/?token={token}"))
    return urls
