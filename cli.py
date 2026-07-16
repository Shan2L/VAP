from __future__ import annotations

import argparse
from types import SimpleNamespace

import main as vap_workflow
import server as vap_server
from runtime_paths import APP_DIR, VAP_CONFIG_PATH, VAP_LOGS_DIR, ensure_vap_home


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vap", description="VAP command line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start the VAP web UI server")
    start_parser.add_argument("--host", default="127.0.0.1")
    start_parser.add_argument("--port", type=int, default=8899)

    run_parser = subparsers.add_parser("run", help="Run the VAP workflow")
    run_parser.add_argument("--config", default=str(VAP_CONFIG_PATH))

    clean_parser = subparsers.add_parser("clean", help="Remove generated VAP logs")
    clean_parser.add_argument("--logs-dir", default=str(VAP_LOGS_DIR))

    args = parser.parse_args(argv)
    ensure_vap_home()
    if not VAP_CONFIG_PATH.is_file():
        VAP_CONFIG_PATH.write_text(
            (APP_DIR / "example-config.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    if args.command == "start":
        vap_server.main(["--host", args.host, "--port", str(args.port)])
    elif args.command == "run":
        vap_workflow.run(SimpleNamespace(config=args.config), str(VAP_LOGS_DIR))
    elif args.command == "clean":
        vap_workflow.clean(os.path.abspath(args.logs_dir))


if __name__ == "__main__":
    main()
