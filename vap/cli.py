from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

from . import main as vap_workflow
from .runtime_paths import ASSET_DIR, VAP_CONFIG_PATH, VAP_LOGS_DIR, ensure_vap_home
from .server.app import main as start_server
from .server.settings import DEFAULT_SERVER_HOST


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["attribute"]:
        from .analysis import attribution

        raise SystemExit(attribution.main(argv[1:]))

    parser = argparse.ArgumentParser(prog="vap", description="VAP command line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start the VAP web UI server")
    start_parser.add_argument(
        "--host",
        default=DEFAULT_SERVER_HOST,
        help="Bind host (default: 0.0.0.0; use 127.0.0.1 for local-only access)",
    )
    start_parser.add_argument("--port", type=int, default=8899)

    run_parser = subparsers.add_parser("run", help="Run the VAP workflow")
    run_parser.add_argument("--config", default=str(VAP_CONFIG_PATH))
    run_parser.add_argument("--visualization-host", default="127.0.0.1")

    clean_parser = subparsers.add_parser("clean", help="Remove generated VAP logs")
    clean_parser.add_argument("--logs-dir", default=str(VAP_LOGS_DIR))

    attribute_parser = subparsers.add_parser(
        "attribute",
        help="Per-layer RCCL vs compute attribution of rank traces",
        add_help=False,
    )
    attribute_parser.add_argument("args", nargs=argparse.REMAINDER)

    uninstall_parser = subparsers.add_parser(
        "uninstall",
        help="Uninstall VAP while preserving config and logs by default",
    )
    uninstall_parser.add_argument(
        "--purge",
        action="store_true",
        help="Also remove config, logs, and all files under VAP_HOME",
    )
    uninstall_parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Also remove the managed bootstrap source checkout",
    )
    uninstall_parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not ask for interactive confirmation",
    )

    args = parser.parse_args(argv)
    if args.command == "uninstall":
        uninstall_script = ASSET_DIR / "uninstall.sh"
        if not uninstall_script.is_file():
            raise FileNotFoundError(f"Uninstall script not found: {uninstall_script}")

        command = ["bash", str(uninstall_script)]
        if args.purge:
            command.append("--purge")
        if args.remove_source:
            command.append("--remove-source")
        if args.yes:
            command.append("--yes")

        # Replace the current vap process so uninstall.sh can safely remove the
        # virtual environment containing this CLI executable.
        os.execvp(command[0], command)
        return

    ensure_vap_home()
    if not VAP_CONFIG_PATH.is_file():
        VAP_CONFIG_PATH.write_text(
            (ASSET_DIR / "example-config.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        VAP_CONFIG_PATH.chmod(0o600)
    if args.command == "start":
        start_server(["--host", args.host, "--port", str(args.port)])
    elif args.command == "run":
        vap_workflow.run(
            SimpleNamespace(
                config=args.config,
                visualization_host=args.visualization_host,
            ),
            str(VAP_LOGS_DIR),
        )
    elif args.command == "clean":
        vap_workflow.clean(args.logs_dir)


if __name__ == "__main__":
    main()
