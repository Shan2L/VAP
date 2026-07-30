from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import main as vap_workflow
import server as vap_server
from runtime_paths import APP_DIR, VAP_CONFIG_PATH, VAP_LOGS_DIR, ensure_vap_home


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vap", description="VAP command line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start the VAP web UI server")
    start_parser.add_argument(
        "--host",
        default=vap_server.DEFAULT_SERVER_HOST,
        help="Bind host (default: 0.0.0.0; use 127.0.0.1 for local-only access)",
    )
    start_parser.add_argument("--port", type=int, default=8899)

    run_parser = subparsers.add_parser("run", help="Run the VAP workflow")
    run_parser.add_argument("--config", default=str(VAP_CONFIG_PATH))
    run_parser.add_argument("--visualization-host", default="127.0.0.1")

    clean_parser = subparsers.add_parser("clean", help="Remove generated VAP logs")
    clean_parser.add_argument("--logs-dir", default=str(VAP_LOGS_DIR))

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
        uninstall_script = APP_DIR / "uninstall.sh"
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
            (APP_DIR / "example-config.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        VAP_CONFIG_PATH.chmod(0o600)
    if args.command == "start":
        vap_server.main(["--host", args.host, "--port", str(args.port)])
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
