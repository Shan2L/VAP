import argparse
import json
import logging
import os
import shutil
import signal
from datetime import datetime

from vap.pipelines.torch_profiling_pipeline import TorchProfilingPipeline

from .config import VAPConfig, expand_run_name
from .runtime_paths import ASSET_DIR, VAP_LOGS_DIR, ensure_vap_home
from .validation import build_legacy_config_warnings, validate_config_or_raise

logger = logging.getLogger("VAP")


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config_json = json.load(f)
    config = VAPConfig.model_validate(config_json)
    warnings = validate_config_or_raise(config)
    for notice in build_legacy_config_warnings(config_json):
        logger.warning("Config format [%s]: %s", notice["path"], notice["message"])
    logger.info(
        "Config loaded: model=%s docker_image=%s vllm=%s:%s",
        config.model_cfg.model_name,
        config.docker_image,
        config.vllm_host,
        config.vllm_port,
    )
    for warning in warnings:
        logger.warning(
            "Config security warning [%s]: %s", warning["path"], warning["message"]
        )
    return config


def setup_logging(log_path: str, debug: bool = False) -> logging.Logger:
    log_file = os.path.join(log_path, "vap_log.txt")
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("VAP")


def clean(log_dir: str):
    target = os.path.abspath(log_dir)
    logs_root = os.path.abspath(str(VAP_LOGS_DIR))
    if target != logs_root:
        raise ValueError(f"Refusing to clean non-VAP logs directory: {target}")
    if os.path.isdir(target):
        shutil.rmtree(target)
        print(f"Removed {target}")
    else:
        print(f"{target} does not exist; nothing to clean")


class _RunInterrupted(BaseException):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def run_dir_name(date_str: str, config_path: str) -> str:
    """The start time plus the config's expanded run_name; just the start time
    when the config cannot be read (load_config then reports why)."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return date_str
    suffix = expand_run_name(payload) if isinstance(payload, dict) else ""
    return f"{date_str}_{suffix}" if suffix else date_str


def run(args, log_dir: str):
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, run_dir_name(date_str, args.config))
    os.makedirs(log_path, mode=0o700, exist_ok=True)
    os.chmod(log_path, 0o700)
    run_config_copy = os.path.join(log_path, "config.json")
    shutil.copy2(args.config, run_config_copy)
    os.chmod(run_config_copy, 0o600)

    run_logger = setup_logging(
        log_path, debug=True if os.getenv("VAP_DEBUG") == "1" else False
    )
    config = load_config(args.config)
    run_logger.info("VAP started")

    pipeline = TorchProfilingPipeline(
        config,
        log_path,
        date_str,
        visualization_host=args.visualization_host,
    )

    def signal_handler(signum, _frame):
        raise _RunInterrupted(signum)

    previous_sigint = signal.signal(signal.SIGINT, signal_handler)
    previous_sigterm = signal.signal(signal.SIGTERM, signal_handler)
    try:
        pipeline.run_pipeline()
    except _RunInterrupted as interrupted:
        # Cleanup runs through the pipeline's normal finally path. Restore the
        # default handlers so a second stop request can terminate a stuck
        # Docker/SSH cleanup instead of being ignored forever.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        run_logger.info(
            "Signal %s received; normal pipeline cleanup requested",
            interrupted.signum,
        )
        try:
            pipeline.cleanup()
        except Exception:
            run_logger.exception(
                "Cleanup failed after signal %s",
                interrupted.signum,
            )
        raise SystemExit(128 + interrupted.signum) from None
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv: list[str] | None = None) -> None:
    ensure_vap_home()
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run VAP")
    subparsers.add_parser("clean", help="Clean VAP")
    run_parser.add_argument(
        "--config", type=str, default=str(ASSET_DIR / "example-config.json")
    )
    run_parser.add_argument("--visualization-host", default="127.0.0.1")
    args = argparser.parse_args(argv)

    log_dir = str(VAP_LOGS_DIR)

    if args.command == "run":
        run(args, log_dir)
    elif args.command == "clean":
        clean(log_dir)
    else:
        argparser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
