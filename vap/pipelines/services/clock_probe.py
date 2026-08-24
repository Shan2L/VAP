from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from vap.clock_probe import ProbeConfig
from vap.clock_probe.summary import format_clock_summary
from vap.config import VAPConfig
from vap.runners import DockerRunner

logger = logging.getLogger("VAP")

CLOCK_PROBE_CONTAINER_DIR = "/app/VAP/log/clock-probe"
CLOCK_PROBE_SOURCE_DIR = "/opt/vap"
CLOCK_PROBE_CONFIG_PATH = f"{CLOCK_PROBE_CONTAINER_DIR}/config.json"
CLOCK_PROBE_SESSION_PATH = f"{CLOCK_PROBE_CONTAINER_DIR}/clock-session.json"


def clock_probe_enabled(config: VAPConfig) -> bool:
    distributed = config.distributed_cfg
    return bool(distributed and distributed.enable and config.clock_probe_cfg.enabled)


class ClockProbeLifecycle:
    """Own clock-probe configuration, process commands, and failure policy."""

    def __init__(
        self,
        config: VAPConfig,
        master_runner: DockerRunner,
        log_path: str,
        date_str: str,
    ):
        self.config = config
        self.master_runner = master_runner
        self.log_path = log_path
        self.date_str = date_str
        self._active = False

    @property
    def enabled(self) -> bool:
        return clock_probe_enabled(self.config)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def required(self) -> bool:
        return self.config.clock_probe_cfg.required

    @property
    def local_session_path(self) -> Path:
        return Path(self.log_path) / "clock-probe" / "clock-session.json"

    def start(self) -> None:
        if not self.enabled:
            return

        self.master_runner.files.ensure_directory(CLOCK_PROBE_CONTAINER_DIR)
        self.master_runner.files.write_text(
            CLOCK_PROBE_CONFIG_PATH,
            self.config_json(),
        )
        try:
            result = self.master_runner.process.run(
                [
                    "python3",
                    "-m",
                    "vap.clock_probe",
                    "start",
                    "--config",
                    CLOCK_PROBE_CONFIG_PATH,
                ],
                environment=self.environment(),
                demux=True,
            )
        except Exception as exc:
            message = f"Clock probe failed to start: {exc}"
            if self.required:
                raise RuntimeError(message) from exc
            logger.warning(message)
            return
        if result.exit_code != 0:
            message = f"Clock probe failed to start: {result.combined_text.strip()}"
            if self.required:
                raise RuntimeError(message)
            logger.warning(message)
            return
        self._active = True
        logger.info(
            "Clock probe started; artifacts will be saved under %s",
            os.path.join(self.log_path, "clock-probe"),
        )

    def stop(self) -> None:
        if not self._active:
            return

        try:
            result = self.master_runner.process.run(
                [
                    "python3",
                    "-m",
                    "vap.clock_probe",
                    "stop",
                    "--config",
                    CLOCK_PROBE_CONFIG_PATH,
                    "--output",
                    CLOCK_PROBE_SESSION_PATH,
                ],
                environment=self.environment(),
                demux=True,
            )
        except Exception as exc:
            message = f"Clock probe failed to stop: {exc}"
            if self.required:
                raise RuntimeError(message) from exc
            logger.warning(message)
            return
        finally:
            self._active = False

        if result.exit_code != 0:
            message = f"Clock probe failed to stop: {result.combined_text.strip()}"
            if self.required:
                raise RuntimeError(message)
            logger.warning(message)
            return
        if not self.master_runner.files.is_file(CLOCK_PROBE_SESSION_PATH):
            message = f"Clock probe did not produce {CLOCK_PROBE_SESSION_PATH}"
            if self.required:
                raise RuntimeError(message)
            logger.warning(message)
            return
        try:
            logger.info(
                "Clock probe fitting summary:\n%s",
                self.summarize_session(),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            message = f"Cannot summarize clock probe session: {exc}"
            if self.required:
                raise RuntimeError(message) from exc
            logger.warning(message)
        logger.info(
            "Clock model session saved to %s",
            self.local_session_path,
        )

    def cleanup(self) -> None:
        if not self._active or not self.master_runner.is_started:
            return
        try:
            self.stop()
        except Exception as exc:
            logger.warning("Failed to stop clock probe during cleanup: %s", exc)

    def probe_config(self) -> ProbeConfig:
        config = self.config.clock_probe_cfg
        mode = config.mode
        if mode == "auto" and not (
            config.hardware_interface and config.hardware_phc_device
        ):
            logger.info(
                "Clock probe auto mode has no explicit PHC mapping; "
                "using software probing"
            )
            mode = "software"

        kwargs = {
            "ray_address": config.ray_address,
            "mode": mode,
            "hardware_ptp_logs": dict(config.hardware_ptp_logs),
            "hardware_interval_ms": config.hardware_interval_ms,
            "port": config.port,
            "interval_ms": config.interval_ms,
            "working_dir": CLOCK_PROBE_SOURCE_DIR,
            "raw_output_root": f"{CLOCK_PROBE_CONTAINER_DIR}/raw",
            "session_id": self.date_str,
            "output": CLOCK_PROBE_SESSION_PATH,
        }
        if config.hardware_interface:
            kwargs["hardware_interface"] = config.hardware_interface
        if config.hardware_phc_device:
            kwargs["hardware_phc_device"] = config.hardware_phc_device
        return ProbeConfig(**kwargs)

    def config_json(self) -> str:
        return (
            json.dumps(
                asdict(self.probe_config()),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def environment(self) -> dict[str, str]:
        configured_path = (self.config.container_cfg.env_vars or {}).get("PYTHONPATH")
        python_path = CLOCK_PROBE_SOURCE_DIR
        if configured_path:
            python_path = f"{python_path}:{configured_path}"
        return {"PYTHONPATH": python_path}

    def summarize_session(self) -> str:
        session = json.loads(self.local_session_path.read_text(encoding="utf-8"))
        return format_clock_summary(session)
