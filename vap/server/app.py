from __future__ import annotations

import argparse
import atexit
import signal
from http.server import ThreadingHTTPServer

from vap.runtime_paths import VAP_HOME, ensure_vap_home, validate_runtime_assets
from vap.server import settings
from vap.server.artifacts import cleanup_old_profile_archives
from vap.server.auth import build_session_urls
from vap.server.handler import VAPConfigHandler
from vap.server.state import cleanup_active_run_on_server_exit, recover_orphaned_run


def main(argv: list[str] | None = None) -> None:
    validate_runtime_assets()
    ensure_vap_home()
    recover_orphaned_run()
    cleanup_old_profile_archives()
    parser = argparse.ArgumentParser(description="VAP config management UI")
    parser.add_argument(
        "--host",
        default=settings.DEFAULT_SERVER_HOST,
        help="Bind host (default: 0.0.0.0; use 127.0.0.1 for local-only access)",
    )
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args(argv)
    settings.SERVER_BIND_HOST = args.host

    atexit.register(cleanup_active_run_on_server_exit)

    def handle_shutdown_signal(signum: int, frame) -> None:
        print(f"Received signal {signum}; shutting down VAP config UI...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    server = ThreadingHTTPServer((args.host, args.port), VAPConfigHandler)
    print(f"VAP config UI started: http://{args.host}:{args.port}")
    print("VAP session URL candidates:")
    for label, url in build_session_urls(
        args.host, args.port, settings.SERVER_AUTH_TOKEN
    ):
        print(f"  {label}: {url}")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "WARNING: VAP is listening on all network interfaces without TLS. "
            "Use only on a trusted network and keep the session token private."
        )
        print(
            f"Remote reachability still requires DNS/routing and a firewall rule "
            f"allowing TCP port {args.port}."
        )
    print(f"VAP home: {VAP_HOME}")
    print(f"Temporary config files will be saved to: {settings.TEMP_CONFIG_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("VAP config UI is stopping...")
    finally:
        cleanup_active_run_on_server_exit()
        server.server_close()


if __name__ == "__main__":
    main()
