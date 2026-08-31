from __future__ import annotations

from pathlib import Path


def resolve_ssh_identity(ssh_key: str | None) -> str | None:
    """Return an existing private key path, or None to use the default SSH identity."""
    if not ssh_key:
        return None
    path = Path(ssh_key).expanduser()
    if path.is_file():
        return str(path)
    return None
