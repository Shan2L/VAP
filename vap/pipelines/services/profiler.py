from __future__ import annotations

import logging

from vap.config import VAPConfig
from vap.runners import DockerRunner

logger = logging.getLogger("VAP")

PROFILE_HTTP_TIMEOUT_SEC = 10


def profiler_enabled(config: VAPConfig) -> bool:
    return bool(config.profiler_cfg.enable)


class ProfilerLifecycle:
    """Own vLLM /start_profile and /stop_profile independently of benchmark."""

    def __init__(self, config: VAPConfig, master_runner: DockerRunner):
        self.config = config
        self.master_runner = master_runner
        self._active = False

    @property
    def enabled(self) -> bool:
        return profiler_enabled(self.config)

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        if not self.enabled:
            logger.info("Torch profiler is disabled")
            return
        if self._active:
            raise RuntimeError("vLLM profiler is already active")
        status = self.master_runner.network.http_status(
            self._profile_url("start_profile"),
            method="POST",
            timeout_sec=PROFILE_HTTP_TIMEOUT_SEC,
        )
        if not _http_ok(status):
            raise RuntimeError(f"Failed to start vLLM profiler: HTTP {status}")
        self._active = True
        logger.info("vLLM profiler started")

    def stop(self) -> None:
        if not self._active:
            return
        try:
            status = self.master_runner.network.http_status(
                self._profile_url("stop_profile"),
                method="POST",
                timeout_sec=PROFILE_HTTP_TIMEOUT_SEC,
            )
            if not _http_ok(status):
                raise RuntimeError(
                    f"Failed to stop vLLM profiler cleanly: HTTP {status}"
                )
            logger.info("vLLM profiler stopped")
        finally:
            self._active = False

    def cleanup(self) -> None:
        if not self._active or not self.master_runner.is_started:
            return
        try:
            self.stop()
        except Exception as exc:
            logger.warning("Failed to stop vLLM profiler during cleanup: %s", exc)

    def _profile_url(self, action: str) -> str:
        return f"http://127.0.0.1:{self.config.vllm_port}/{action}"


def _http_ok(status: int | None) -> bool:
    return status is not None and 200 <= status < 300
