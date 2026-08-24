from __future__ import annotations

import base64
import json
import posixpath
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from vap.runners.assistants.process import (
    CommandExecutionError,
    DockerCommandExecutor,
)

DIRECTORY_SNAPSHOT_SCRIPT = """
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
suffixes = tuple(sys.argv[2:])
snapshot = {
    path.name: [path.stat().st_size, path.stat().st_mtime_ns]
    for path in root.iterdir()
    if path.is_file() and path.name.endswith(suffixes)
}
print(json.dumps(snapshot, sort_keys=True))
"""


class FileAssistant:
    """File operations scoped to the target runner container."""

    def __init__(self, executor: DockerCommandExecutor):
        self._executor = executor

    def exists(self, path: str) -> bool:
        result = self._executor.run(["test", "-e", path])
        return result.exit_code == 0

    def is_file(self, path: str) -> bool:
        result = self._executor.run(["test", "-f", path])
        return result.exit_code == 0

    def is_directory(self, path: str) -> bool:
        result = self._executor.run(["test", "-d", path])
        return result.exit_code == 0

    def ensure_directory(self, path: str) -> None:
        command = ["mkdir", "-p", "--", path]
        result = self._executor.run(command)
        if result.exit_code != 0:
            raise CommandExecutionError(command, result)

    def remove(self, path: str, *, recursive: bool = False) -> None:
        command = ["rm", "-rf" if recursive else "-f", "--", path]
        result = self._executor.run(command)
        if result.exit_code != 0:
            raise CommandExecutionError(command, result)

    def read_text(self, path: str) -> str:
        command = [
            "python3",
            "-c",
            "from pathlib import Path; import sys; print(Path(sys.argv[1]).read_text(), end='')",
            path,
        ]
        result = self._executor.run(command)
        if result.exit_code != 0:
            raise CommandExecutionError(command, result)
        return result.stdout_text

    def write_text(self, path: str, content: str) -> None:
        encoded = base64.b64encode(content.encode()).decode()
        command = [
            "python3",
            "-c",
            (
                "import base64,sys; from pathlib import Path; "
                "Path(sys.argv[1]).write_bytes(base64.b64decode(sys.argv[2]))"
            ),
            path,
            encoded,
        ]
        result = self._executor.run(command)
        if result.exit_code != 0:
            raise CommandExecutionError(command, result)

    def directory_snapshot(
        self,
        path: str,
        suffixes: tuple[str, ...],
    ) -> dict[str, tuple[int, int]]:
        command = [
            "python3",
            "-c",
            DIRECTORY_SNAPSHOT_SCRIPT,
            path,
            *suffixes,
        ]
        result = self._executor.run(command)
        if result.exit_code != 0:
            raise CommandExecutionError(command, result)
        payload = json.loads(result.stdout_text)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid directory snapshot for {path}")
        return {
            str(name): (int(metadata[0]), int(metadata[1]))
            for name, metadata in payload.items()
        }

    def download_directory(
        self,
        source: str,
        destination: str | Path,
    ) -> list[Path]:
        """Download a container directory without extracting links or devices."""
        chunks, _ = self._executor.container.get_archive(source)
        target_root = Path(destination)
        target_root.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []

        with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024) as archive_file:
            for chunk in chunks:
                archive_file.write(chunk)
            archive_file.seek(0)
            with tarfile.open(fileobj=archive_file, mode="r:*") as archive:
                for member in archive:
                    relative = PurePosixPath(member.name)
                    parts = relative.parts[1:]
                    if (
                        not parts
                        or relative.is_absolute()
                        or ".." in parts
                        or member.issym()
                        or member.islnk()
                        or member.isdev()
                    ):
                        continue
                    target = target_root.joinpath(*parts)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    with extracted, target.open("wb") as output:
                        shutil.copyfileobj(extracted, output)
                    downloaded.append(target)
        return downloaded

    def download_files(
        self,
        source_directory: str,
        file_names: list[str],
        destination: str | Path,
    ) -> list[Path]:
        target_root = Path(destination)
        target_root.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        for file_name in file_names:
            if Path(file_name).name != file_name:
                raise ValueError(f"Invalid container file name: {file_name}")
            chunks, _ = self._executor.container.get_archive(
                posixpath.join(source_directory, file_name)
            )
            with tempfile.SpooledTemporaryFile(
                max_size=64 * 1024 * 1024
            ) as archive_file:
                for chunk in chunks:
                    archive_file.write(chunk)
                archive_file.seek(0)
                with tarfile.open(fileobj=archive_file, mode="r:*") as archive:
                    member = next(
                        (
                            item
                            for item in archive
                            if item.isfile()
                            and not item.issym()
                            and not item.islnk()
                            and not item.isdev()
                        ),
                        None,
                    )
                    if member is None:
                        raise ValueError(
                            f"Container archive has no regular file: {file_name}"
                        )
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError(f"Cannot extract container file: {file_name}")
                    target = target_root / file_name
                    with extracted, target.open("wb") as output:
                        shutil.copyfileobj(extracted, output)
                    downloaded.append(target)
        return downloaded
