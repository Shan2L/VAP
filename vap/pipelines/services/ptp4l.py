"""Per-node ptp4l sidecar lifecycle for hardware Clock Probe."""

from __future__ import annotations

import io
import logging
import re
import socket
import tarfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from vap.clock_probe.calibration.ptp_health import (
    clock_identity_from_mac,
    evaluate_ptp_health,
    parse_ptp4l_log,
)
from vap.config import VAPConfig
from vap.pipelines.services.clock_probe import (
    CLOCK_PROBE_CONTAINER_DIR,
    clock_probe_enabled,
)
from vap.runners import ContainerSpec, DockerRunner, DockerTarget, MountSpec
from vap.runners.docker import (
    VAP_KIND_LABEL,
    VAP_MANAGED_LABEL,
    VAP_RUN_LABEL,
    create_docker_client,
    orphaned_vap_containers,
)

logger = logging.getLogger("VAP")

PTP4L_CONTAINER_DIR = f"{CLOCK_PROBE_CONTAINER_DIR}/ptp4l"
PTP4L_LOCK_TIMEOUT_SEC = 180
PTP4L_LOCK_POLL_SEC = 1.0
PTP4L_WAIT_LOG_SEC = 15.0
GM_LOCKED_STATES = {"MASTER", "GRAND_MASTER"}
SLAVE_LOCKED_STATES = {"SLAVE"}
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
PTP4L_MISSING_MESSAGE = "ptp4l is not installed in this image"
PTP4L_APT_PACKAGE = "linuxptp"


def render_sidecar_shell(
    conf_name: str,
    log_name: str,
    *,
    conf_text: str,
    local_clock_id: str | None = None,
) -> str:
    """Write ptp4l.conf inside the container, install linuxptp if needed, then exec."""
    log = f"/ptp/{log_name}"
    conf = f"/ptp/{conf_name}"
    marker = "VAP_PTP4L_CONF"
    while marker in conf_text:
        marker += "_X"
    body = conf_text if conf_text.endswith("\n") else f"{conf_text}\n"
    identity = ""
    if local_clock_id:
        identity = f"echo 'vap ptp local-clock-id {local_clock_id}' | tee -a {log}\n"
    return (
        "set -eu\n"
        "set -o pipefail\n"
        'export PATH="/usr/sbin:/sbin:${PATH:-/usr/bin:/bin}"\n'
        f"cat > {conf} << '{marker}'\n"
        f"{body}"
        f"{marker}\n"
        f"{identity}"
        "if ! command -v ptp4l >/dev/null 2>&1; then\n"
        f"  echo 'Installing {PTP4L_APT_PACKAGE} (provides ptp4l)' | tee -a {log}\n"
        "  export DEBIAN_FRONTEND=noninteractive\n"
        f"  test -r /etc/os-release || {{ echo 'unsupported image: no /etc/os-release' | tee -a {log} >&2; exit 126; }}\n"
        "  . /etc/os-release\n"
        f'  case "${{ID:-}}" in ubuntu|debian) ;; *) echo "unsupported image distribution: ${{ID:-unknown}}" | tee -a {log} >&2; exit 126 ;; esac\n'
        f"  command -v dpkg >/dev/null 2>&1 || {{ echo 'unsupported image: dpkg is missing' | tee -a {log} >&2; exit 126; }}\n"
        '  arch="$(dpkg --print-architecture)"\n'
        f'  case "$arch" in amd64|arm64) ;; *) echo "unsupported image architecture: $arch" | tee -a {log} >&2; exit 126 ;; esac\n'
        "  apt-get -o Acquire::ForceIPv4=true"
        " -o Acquire::http::Timeout=20"
        " -o Acquire::https::Timeout=20"
        " -o Acquire::Retries=2 update 2>&1 | tee -a "
        f"{log}\n"
        f"  apt-get install -y --no-install-recommends {PTP4L_APT_PACKAGE}"
        f" 2>&1 | tee -a {log}\n"
        "fi\n"
        "if ! command -v ptp4l >/dev/null 2>&1; then\n"
        f"  echo '{PTP4L_MISSING_MESSAGE}' | tee -a {log} >&2\n"
        "  exit 127\n"
        "fi\n"
        f"exec ptp4l -f {conf} -m >> {log} 2>&1\n"
    )


def _archive_file_text(container: Any, path: str) -> str:
    get_archive = getattr(container, "get_archive", None)
    if not callable(get_archive):
        return ""
    try:
        stream, _info = get_archive(path)
        payload = b"".join(stream)
    except Exception:
        return ""
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            member = next((item for item in archive if item.isfile()), None)
            if member is None:
                return ""
            extracted = archive.extractfile(member)
            if extracted is None:
                return ""
            return extracted.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def create_ptp4l_runner(target: DockerTarget) -> DockerRunner:
    """Sidecars do not need NIC inventory; skip exec that 409s if ptp4l exits."""
    return DockerRunner(target, auto_network_refresh=False)


def _sidecar_container_name(container: Any) -> str:
    return str(getattr(container, "name", "") or "").lstrip("/")


def sweep_ptp4l_containers(target: DockerTarget) -> None:
    """Stop and remove containers explicitly labeled as VAP ptp4l sidecars."""
    client = create_docker_client(target)
    try:
        containers = client.containers.list(
            all=True,
            filters={"label": f"{VAP_MANAGED_LABEL}=true"},
        )
        for container in orphaned_vap_containers(containers):
            name = _sidecar_container_name(container)
            labels = getattr(container, "labels", {}) or {}
            if (
                labels.get(VAP_MANAGED_LABEL) != "true"
                or labels.get(VAP_KIND_LABEL) != "ptp4l"
            ):
                continue
            try:
                container.remove(force=True)
                logger.info("Removed ptp4l sidecar %s on %s", name, target.label)
            except Exception as exc:
                logger.warning(
                    "Failed to remove ptp4l sidecar %s on %s: %s",
                    name,
                    target.label,
                    exc,
                )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def ptp4l_enabled(config: VAPConfig) -> bool:
    return clock_probe_enabled(config) and config.clock_probe_cfg.mode in {
        "hardware",
        "auto",
    }


def container_ptp4l_dir(run_id: str) -> str:
    return f"{PTP4L_CONTAINER_DIR}/{run_id}"


def container_ptp4l_log_path(run_id: str, os_hostname: str) -> str:
    return f"{container_ptp4l_dir(run_id)}/{os_hostname}.log"


def host_ptp4l_dir(log_path: str, run_id: str) -> Path:
    return Path(log_path) / "clock-probe" / "ptp4l" / run_id


def render_ptp4l_conf(
    interface: str,
    *,
    grandmaster: bool,
) -> str:
    """Render the validated clock_probe ptp4l configs. Never run phc2sys.

    These match ``projects/clock_probe/configs/ptp/ptp4l-{gm,slave}.conf``:
    multicast UDPv4 on domain 42.
    """
    if grandmaster:
        role_lines = (
            "priority1                 100\n"
            "priority2                 100\n"
            "twoStepFlag               1\n"
        )
    else:
        role_lines = (
            "priority1                 255\n"
            "priority2                 255\n"
            "clientOnly                1\n"
            "twoStepFlag               1\n"
        )
    return (
        "[global]\n"
        "domainNumber              42\n"
        "network_transport         UDPv4\n"
        "delay_mechanism           E2E\n"
        "time_stamping             hardware\n"
        "clock_servo               pi\n"
        "delay_filter              moving_median\n"
        "delay_filter_length       10\n"
        "tx_timestamp_timeout      100\n"
        "logging_level             6\n"
        "summary_interval          0\n"
        "logSyncInterval           -3\n"
        "logAnnounceInterval       0\n"
        "logMinDelayReqInterval    0\n"
        f"{role_lines}"
        "# Never touch CLOCK_REALTIME; only the NIC PHC is steered.\n"
        f"[{interface}]\n"
    )


def _runner_os_hostname(runner: DockerRunner) -> str:
    for command in (["hostname", "-s"], ["hostname"]):
        try:
            result = runner.process.run(command)
        except Exception:
            continue
        if result.exit_code != 0:
            continue
        name = result.stdout_text.strip().splitlines()
        if name and name[0].strip():
            return name[0].strip()
    return socket.gethostname().split(".", 1)[0]


def _safe_name(value: str) -> str:
    cleaned = SAFE_NAME_RE.sub("-", value).strip("-.")
    return cleaned or "node"


def _latest_port_state(text: str) -> str | None:
    states = parse_ptp4l_log(text).get("states") or []
    if not states:
        return None
    return str(states[-1].get("state") or "") or None


def _ptp4l_started(text: str) -> bool:
    return "ptp4l[" in text


def _slave_probe_ready(text: str) -> bool:
    """True when a slave log has parseable rms/offset samples after SLAVE lock."""
    health = evaluate_ptp_health(text, role="slave")
    return bool(health.lock_ok)


@dataclass
class Ptp4lNodeSession:
    label: str
    os_hostname: str
    role: str
    interface: str
    phc_device: str
    runner: DockerRunner
    host_log: Path
    container_log: str
    ray_address: str | None = None
    local_clock_id: str | None = None


class Ptp4lLifecycle:
    """Start host-network ptp4l sidecars, then stop them after Clock Probe."""

    def __init__(
        self,
        config: VAPConfig,
        log_path: str,
        date_str: str,
        *,
        master_target: DockerTarget | None = None,
        worker_targets: list[DockerTarget] | None = None,
        master_inventory: DockerRunner | None = None,
        worker_inventories: Sequence[DockerRunner] | None = None,
        runner_factory: Callable[[DockerTarget], DockerRunner] = create_ptp4l_runner,
        sidecar_sweeper: Callable[[DockerTarget], None] = sweep_ptp4l_containers,
        run_id: str | None = None,
    ):
        self.config = config
        self.log_path = log_path
        self.date_str = date_str
        self.master_inventory = master_inventory
        self.worker_inventories = list(worker_inventories or [])
        self.master_target = (
            master_inventory.target
            if master_inventory is not None
            else (master_target or DockerTarget())
        )
        self.worker_targets = (
            [runner.target for runner in self.worker_inventories]
            if worker_inventories is not None
            else list(worker_targets or [])
        )
        self._runner_factory = runner_factory
        self._sidecar_sweeper = sidecar_sweeper
        self.run_id = run_id or uuid.uuid4().hex
        self.sessions: list[Ptp4lNodeSession] = []
        self._active = False

    @property
    def enabled(self) -> bool:
        return ptp4l_enabled(self.config)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def container_log_paths(self) -> dict[str, str]:
        return {session.os_hostname: session.container_log for session in self.sessions}

    @property
    def node_descriptors(self) -> dict[str, dict[str, str]]:
        descriptors: dict[str, dict[str, str]] = {}
        for session in self.sessions:
            if not session.ray_address:
                continue
            descriptor = {
                "ray_address": session.ray_address,
                "hostname": session.os_hostname,
                "interface": session.interface,
                "phc_device": session.phc_device,
                "ptp_log": session.container_log,
            }
            if session.local_clock_id:
                descriptor["local_clock_id"] = session.local_clock_id
            descriptors[session.ray_address] = descriptor
        return descriptors

    def start(
        self,
        node_addresses: Mapping[str, str] | None = None,
    ) -> None:
        if not self.enabled:
            return
        if self._active:
            raise RuntimeError("ptp4l sidecars are already running")
        inventories = self._inventory_runners()
        if not inventories:
            raise RuntimeError(
                "ptp4l sidecars need a started runner NetworkAssistant inventory"
            )
        preferred_interface = self.config.clock_probe_cfg.hardware_interface
        preferred_phc = self.config.clock_probe_cfg.hardware_phc_device
        work_dir = host_ptp4l_dir(self.log_path, self.run_id)
        work_dir.mkdir(parents=True, exist_ok=True)
        self._sweep_leftover_sidecars()

        started: list[Ptp4lNodeSession] = []
        addresses = dict(node_addresses or {})
        try:
            for runner, grandmaster in inventories:
                runner.network.ensure_inventory()
                ray_address = addresses.get(runner.target.label)
                if preferred_interface or preferred_phc:
                    interface = runner.network.select_ptp_interface(
                        preferred_interface=preferred_interface,
                        preferred_phc_device=preferred_phc,
                    )
                elif ray_address:
                    interface = runner.network.find_by_ip(ray_address)
                    if (
                        interface is None
                        or not interface.link_up
                        or interface.ptp_device is None
                    ):
                        raise RuntimeError(
                            f"{runner.target.label}: Ray/VLLM address {ray_address} "
                            "is not on an UP NIC with a PHC"
                        )
                    logger.info(
                        "Selected PTP NIC %s (%s) on %s from Ray/VLLM address %s",
                        interface.name,
                        interface.ptp_device,
                        runner.target.label,
                        ray_address,
                    )
                else:
                    interface = runner.network.select_ptp_interface()
                if interface.ptp_device is None:
                    raise RuntimeError(
                        f"{runner.target.label}: selected NIC {interface.name} "
                        "has no PHC"
                    )
                session = self._start_node(
                    info={
                        "node": runner.target.label,
                        "interface": interface.name,
                        "phc_device": interface.ptp_device,
                        "os_hostname": _runner_os_hostname(runner),
                        "ray_address": ray_address,
                        "mac": getattr(interface, "mac", "") or "",
                    },
                    target=runner.target,
                    grandmaster=grandmaster,
                    work_dir=work_dir,
                )
                started.append(session)
                self._wait_until_locked(session)
                self._publish_ptp_log(session, runner)
        except Exception:
            for session in reversed(started):
                try:
                    self._persist_host_log(session, self._read_log(session))
                except Exception:
                    pass
                try:
                    self._stop_session(session)
                except Exception as exc:
                    logger.warning(
                        "Failed to stop ptp4l sidecar on %s after start error: %s",
                        session.label,
                        exc,
                    )
            self._sweep_leftover_sidecars()
            raise

        self.sessions = started
        self._active = True
        logger.info(
            "ptp4l sidecars locked for run %s on %s",
            self.run_id,
            ", ".join(
                f"{session.os_hostname}/{session.role}" for session in self.sessions
            ),
        )

    def stop(self) -> None:
        errors: list[str] = []
        remaining: list[Ptp4lNodeSession] = []
        for session in reversed(self.sessions):
            try:
                self._persist_host_log(session, self._read_log(session))
            except Exception as exc:
                logger.warning(
                    "Failed to persist ptp4l log for %s before stop: %s",
                    session.label,
                    exc,
                )
            try:
                self._stop_session(session)
            except Exception as exc:
                errors.append(f"{session.label}: {exc}")
                remaining.append(session)
        self.sessions = remaining
        self._active = bool(remaining)
        try:
            self._sweep_leftover_sidecars()
        except Exception as exc:
            errors.append(f"sweep: {exc}")
        if errors:
            raise RuntimeError("Failed to stop ptp4l sidecars: " + "; ".join(errors))
        logger.info("ptp4l sidecars stopped for run %s", self.run_id)

    def cleanup(self) -> None:
        try:
            self.stop()
        except Exception as exc:
            logger.warning("Failed to stop ptp4l sidecars during cleanup: %s", exc)
        finally:
            self._active = False
            try:
                self._sweep_leftover_sidecars()
            except Exception as exc:
                logger.warning(
                    "Failed to sweep leftover ptp4l sidecars during cleanup: %s",
                    exc,
                )

    def _inventory_runners(self) -> list[tuple[DockerRunner, bool]]:
        if self.master_inventory is None:
            return []
        # PTP and Ray roles are independent. The first configured worker is the
        # GM; the Ray head and remaining workers are slaves. Start the GM first.
        if self.worker_inventories:
            grandmaster, *remaining_workers = self.worker_inventories
            return [
                (grandmaster, True),
                (self.master_inventory, False),
                *((runner, False) for runner in remaining_workers),
            ]
        return [
            (self.master_inventory, True),
        ]

    def _start_node(
        self,
        *,
        info: dict[str, Any],
        target: DockerTarget,
        grandmaster: bool,
        work_dir: Path,
    ) -> Ptp4lNodeSession:
        os_hostname = str(info.get("os_hostname") or "").strip() or socket.gethostname()
        interface = str(info["interface"])
        phc_device = str(info["phc_device"])
        ray_address = str(info.get("ray_address") or "").strip() or None
        role = "master" if grandmaster else "slave"
        conf_name = f"{os_hostname}.conf"
        log_name = f"{os_hostname}.log"
        conf_text = render_ptp4l_conf(
            interface,
            grandmaster=grandmaster,
        )
        (work_dir / conf_name).write_text(conf_text, encoding="utf-8")
        (work_dir / log_name).write_text("", encoding="utf-8")
        local_clock_id = clock_identity_from_mac(str(info.get("mac") or ""))
        runner = self._runner_factory(target)
        spec = ContainerSpec(
            image=self.config.docker_image,
            name=f"vap_ptp4l_{self.date_str}_{_safe_name(os_hostname)}_{self.run_id[:8]}",
            mounts=(
                MountSpec(
                    target="/ptp",
                    source=str(work_dir),
                    create_source=True,
                ),
            ),
            devices=(phc_device,),
            cap_add=("SYS_ADMIN", "NET_ADMIN", "NET_RAW", "SYS_TIME"),
            labels={
                VAP_MANAGED_LABEL: "true",
                VAP_RUN_LABEL: self.date_str,
                VAP_KIND_LABEL: "ptp4l",
            },
            security_opt=("seccomp=unconfined",),
            command=(
                "/bin/bash",
                "-c",
                render_sidecar_shell(
                    conf_name,
                    log_name,
                    conf_text=conf_text,
                    local_clock_id=local_clock_id,
                ),
            ),
        )
        runner.start(spec)
        return Ptp4lNodeSession(
            label=str(info.get("node") or target.label),
            os_hostname=os_hostname,
            role=role,
            interface=interface,
            phc_device=phc_device,
            runner=runner,
            host_log=work_dir / log_name,
            container_log=f"{container_ptp4l_dir(self.run_id)}/{log_name}",
            ray_address=ray_address,
            local_clock_id=local_clock_id,
        )

    def _wait_until_locked(
        self,
        session: Ptp4lNodeSession,
        *,
        timeout_sec: float = PTP4L_LOCK_TIMEOUT_SEC,
    ) -> None:
        locked = GM_LOCKED_STATES if session.role == "master" else SLAVE_LOCKED_STATES
        expected = "MASTER or GRAND_MASTER" if session.role == "master" else "SLAVE"
        deadline = time.monotonic() + timeout_sec
        last_state = None
        last_report = 0.0
        last_text = ""
        while time.monotonic() < deadline:
            container = session.runner.container
            container.reload()
            if container.status not in {"created", "running"}:
                detail = self._read_log(session).strip()[-4000:]
                self._persist_host_log(session, detail)
                raise RuntimeError(
                    f"ptp4l sidecar on {session.label} exited before {expected} lock"
                    + (f": {detail}" if detail else "")
                )
            text = self._read_log(session)
            last_text = text
            self._persist_host_log(session, text)
            last_state = _latest_port_state(text)
            started = _ptp4l_started(text)
            now = time.monotonic()
            if now - last_report >= PTP4L_WAIT_LOG_SEC:
                if started:
                    phase = f"last state={last_state}"
                elif "Installing" in text or "apt-get" in text or "linuxptp" in text:
                    phase = "installing linuxptp"
                else:
                    phase = "no ptp4l log yet"
                logger.info(
                    "Waiting for ptp4l %s on %s (%s, %d log bytes)",
                    expected,
                    session.label,
                    phase,
                    len(text),
                )
                last_report = now
            if last_state in locked:
                if session.role != "slave" or _slave_probe_ready(text):
                    logger.info(
                        "ptp4l on %s locked as %s (%s -> %s)",
                        session.os_hostname,
                        last_state,
                        session.interface,
                        session.phc_device,
                    )
                    return
            time.sleep(PTP4L_LOCK_POLL_SEC)
        self._persist_host_log(session, last_text)
        detail = f"last state={last_state}"
        if not _ptp4l_started(last_text):
            detail = "ptp4l never started (still installing linuxptp or log unreadable)"
        elif session.role == "slave":
            health = evaluate_ptp_health(last_text, role="slave")
            if health.reasons:
                detail = "; ".join(health.reasons)
        tail = last_text.strip()[-1500:]
        if tail:
            detail = f"{detail}: {tail}"
        raise RuntimeError(
            f"ptp4l on {session.label} did not reach {expected} within "
            f"{int(timeout_sec)}s ({detail})"
        )

    def _publish_ptp_log(
        self, session: Ptp4lNodeSession, inventory: DockerRunner
    ) -> None:
        """Make the sidecar log visible to Ray actors in the vLLM container.

        The vLLM container already bind-mounts the run log directory, and the
        sidecar writes into that tree. Overwriting via write_text truncates
        the live ptp4l log (O_APPEND then continues past the snapshot), which
        is how preflight previously saw a SLAVE lock with no rms lines.
        """
        try:
            inventory.files.ensure_directory(container_ptp4l_dir(self.run_id))
        except Exception as exc:
            logger.warning(
                "Failed to create ptp4l log dir for %s in runner %s: %s",
                session.os_hostname,
                inventory.target.label,
                exc,
            )
            return
        is_file = getattr(inventory.files, "is_file", None)
        read_text = getattr(inventory.files, "read_text", None)
        try:
            if callable(is_file) and is_file(session.container_log):
                existing = (
                    read_text(session.container_log) if callable(read_text) else ""
                )
                if str(existing or "").strip():
                    return
        except Exception:
            pass
        text = self._read_log(session)
        if not text.strip():
            logger.warning(
                "ptp4l log for %s is empty after lock; Ray preflight will fail",
                session.os_hostname,
            )
            return
        try:
            inventory.files.write_text(session.container_log, text)
        except Exception as exc:
            logger.warning(
                "Failed to copy ptp4l log for %s into runner %s: %s",
                session.os_hostname,
                inventory.target.label,
                exc,
            )

    def _read_log(self, session: Ptp4lNodeSession) -> str:
        candidates: list[str] = []
        container_path = f"/ptp/{session.os_hostname}.log"
        try:
            text = session.runner.files.read_text(container_path)
            if text.strip():
                candidates.append(text)
        except Exception:
            pass
        try:
            text = _archive_file_text(session.runner.container, container_path)
            if text.strip():
                candidates.append(text)
        except Exception:
            pass
        try:
            text = session.host_log.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                candidates.append(text)
        except OSError:
            pass
        try:
            logs = session.runner.container.logs()
            if isinstance(logs, bytes):
                text = logs.decode("utf-8", errors="replace")
            else:
                text = str(logs or "")
            if text.strip():
                candidates.append(text)
        except Exception:
            pass
        if not candidates:
            return ""
        with_ptp = [text for text in candidates if _ptp4l_started(text)]
        return max(with_ptp or candidates, key=len)

    def _persist_host_log(self, session: Ptp4lNodeSession, text: str) -> None:
        if not text.strip():
            return
        try:
            current = ""
            if session.host_log.is_file():
                current = session.host_log.read_text(encoding="utf-8", errors="replace")
            if len(text) <= len(current):
                return
            session.host_log.parent.mkdir(parents=True, exist_ok=True)
            session.host_log.write_text(text, encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "Failed to persist ptp4l log for %s: %s",
                session.label,
                exc,
            )

    def _stop_session(self, session: Ptp4lNodeSession) -> None:
        session.runner.cleanup()

    def _sweep_leftover_sidecars(self) -> None:
        for target in (self.master_target, *self.worker_targets):
            try:
                self._sidecar_sweeper(target)
            except Exception as exc:
                logger.warning(
                    "Failed to sweep ptp4l sidecars on %s: %s",
                    target.label,
                    exc,
                )
