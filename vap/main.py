import argparse
import json
import logging
import os
import shutil
import signal
from datetime import datetime

from .config import VAPConfig
from .runtime_paths import VAP_LOGS_DIR, ensure_vap_home
from .validation import validate_config_or_raise
from vap.pipelines.torch_profiling_pipeline import TorchProfilingPipeline

logger = logging.getLogger("VAP")


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config_json = json.load(f)
    config = VAPConfig.model_validate(config_json)
    warnings = validate_config_or_raise(config)
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


def is_machine_connected(node: str) -> bool:
    logger.warning("Checking machine connection is not implemented yet.")
    return False


def check_remote_assets(node: str, asset_path: str) -> bool:
    logger.warning("Checking remote assets is not implemented yet.")
    return False


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


def run(args, log_dir: str):
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, date_str)
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

    if config.distributed_cfg is not None:
        run_logger.warning(
            "distributed_cfg is present but distributed execution is not "
            "supported yet; continuing with a local run"
        )

    pipeline = TorchProfilingPipeline(
        config,
        log_path,
        date_str,
        visualization_host=args.visualization_host,
    )

    def signal_handler(signum, frame):
        run_logger.info("Signal %s received. Cleaning up...", signum)
        pipeline.runner.remove_container()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    pipeline.run_pipeline()


def main(argv: list[str] | None = None) -> None:
    ensure_vap_home()
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run VAP")
    subparsers.add_parser("clean", help="Clean VAP")
    run_parser.add_argument("--config", type=str, default="example-config.json")
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
