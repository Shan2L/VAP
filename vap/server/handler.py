from __future__ import annotations

import json
import secrets
import shutil
from http import HTTPStatus
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from vap.agent.analysis import resolve_latest_trace
from vap.agent.tools import get_agent_runtime, get_agent_status_payload
from vap.config import VAPConfig
from vap.server import settings
from vap.server.analysis import (
    ATTRIBUTION_FILE_TYPES,
    analyze_run,
    attribution_file,
    compare_two_runs,
    list_profile_runs,
    report_stream,
)
from vap.server.artifacts import (
    build_log_download,
    create_profile_archive,
    read_current_log_file,
    resolve_config_path,
    save_temp_config,
)
from vap.server.auth import parse_cookie_header, same_origin_allowed
from vap.server.checks import (
    check_config_clock_probe,
    check_config_container_resources,
    check_config_docker_resources,
    check_config_machines,
    check_config_model_resources,
    check_config_ports,
    check_config_resources,
    check_config_worker_docker_resources,
)
from vap.server.state import get_run_state_snapshot, start_vap_run, stop_vap_run
from vap.validation import validate_config_payload


class VAPConfigHandler(BaseHTTPRequestHandler):
    server_version = "VAPConfigServer/0.1"

    def token_from_request(self, parsed: Any) -> str | None:
        query_token = (
            parse_qs(parsed.query).get("token", [None])[0]
            if parsed.path == "/"
            else None
        )
        if query_token:
            return query_token
        header_token = self.headers.get("X-VAP-Token")
        if header_token:
            return header_token
        cookies = parse_cookie_header(self.headers.get("Cookie"))
        return cookies.get(settings.SERVER_COOKIE_NAME)

    def is_authenticated(self, parsed: Any) -> bool:
        token = self.token_from_request(parsed)
        if token and secrets.compare_digest(token, settings.SERVER_AUTH_TOKEN):
            if parse_qs(parsed.query).get("token", [None])[0] == token:
                self._issue_auth_cookie = True
            return True
        return False

    def require_authorized(
        self,
        parsed: Any,
        *,
        require_json: bool = False,
        require_same_origin: bool = False,
    ) -> bool:
        if not self.is_authenticated(parsed):
            self.send_json(
                {"message": "Unauthorized. Open the URL printed by vap start."},
                HTTPStatus.UNAUTHORIZED,
            )
            return False
        if require_same_origin and not same_origin_allowed(
            self.headers.get("Origin"), self.headers.get("Host")
        ):
            self.send_json({"message": "Origin is not allowed"}, HTTPStatus.FORBIDDEN)
            return False
        if require_json:
            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("application/json"):
                self.send_json(
                    {"message": "Content-Type must be application/json"},
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                )
                return False
        return True

    def send_auth_cookie_if_needed(self) -> None:
        if getattr(self, "_issue_auth_cookie", False):
            self.send_header(
                "Set-Cookie",
                f"{settings.SERVER_COOKIE_NAME}={settings.SERVER_AUTH_TOKEN}; Path=/; HttpOnly; SameSite=Strict",
            )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/favicon.svg":
            self.serve_static("favicon.svg")
            return
        if not self.require_authorized(parsed):
            return
        if parsed.path == "/":
            self.serve_static("index.html")
            return
        if parsed.path == "/tensorboard" or parsed.path.startswith("/tensorboard/"):
            self.handle_tensorboard_proxy(parsed)
            return
        if parsed.path == "/api/config":
            self.handle_get_config(parsed.query)
            return
        if parsed.path == "/api/log-file":
            self.handle_get_log_file(parsed.query)
            return
        if parsed.path == "/api/log-file/download":
            self.handle_log_download(parsed.query)
            return
        if parsed.path == "/api/run/status":
            self.handle_run_status()
            return
        if parsed.path == "/api/profile/archive":
            self.handle_profile_archive(parsed.query)
            return
        if parsed.path == "/api/profile/trace":
            self.handle_profile_trace(parsed.query)
            return
        if parsed.path == "/api/agent/status":
            self.handle_agent_status()
            return
        if parsed.path == "/api/attribution/file":
            self.handle_attribution_file(parsed.query)
            return
        if parsed.path == "/api/analysis/runs":
            self.send_json(list_profile_runs({"limit": 50}))
            return
        if parsed.path.startswith("/public/"):
            self.serve_static(parsed.path.removeprefix("/public/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        if not self.require_authorized(
            parsed, require_json=True, require_same_origin=True
        ):
            return
        if parsed.path != "/api/config":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            self.handle_put_config()
        except json.JSONDecodeError as exc:
            self.send_json(
                {"message": f"JSON parse failed: {exc}"}, HTTPStatus.BAD_REQUEST
            )
        except Exception as exc:
            self.send_json(
                {"message": f"Failed to save config: {exc}"}, HTTPStatus.BAD_REQUEST
            )

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        routes = {
            "/api/validate": self.handle_validate,
            "/api/temp-config": self.handle_save_temp_config,
            "/api/check-ports": self.handle_check_ports,
            "/api/check-machines": self.handle_check_machines,
            "/api/check-resources": self.handle_check_resources,
            "/api/check-model-resources": self.handle_check_model_resources,
            "/api/check-docker-resources": self.handle_check_docker_resources,
            "/api/check-worker-docker-resources": self.handle_check_worker_docker_resources,
            "/api/check-container-resources": self.handle_check_container_resources,
            "/api/check-clock-probe": self.handle_check_clock_probe,
            "/api/run": self.handle_run_start,
            "/api/run/stop": self.handle_run_stop,
            "/api/agent/unlock": self.handle_agent_unlock,
            "/api/agent/chat/stream": self.handle_agent_chat_stream,
            "/api/agent/approve": self.handle_agent_approve,
            "/api/agent/approve/stream": self.handle_agent_decision_stream,
            "/api/agent/cancel-action": self.handle_agent_cancel_action,
            "/api/analysis/layers": self.handle_analysis_layers,
            "/api/analysis/compare": self.handle_analysis_compare,
            "/api/analysis/report/stream": self.handle_analysis_report_stream,
        }
        handler = routes.get(parsed.path)
        if handler is None and (
            parsed.path == "/tensorboard" or parsed.path.startswith("/tensorboard/")
        ):
            if not self.require_authorized(parsed, require_same_origin=True):
                return
            self.handle_tensorboard_proxy(parsed)
            return
        if handler is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not self.require_authorized(
            parsed, require_json=True, require_same_origin=True
        ):
            return
        try:
            handler()
        except json.JSONDecodeError as exc:
            self.send_json(
                {"message": f"JSON parse failed: {exc}"}, HTTPStatus.BAD_REQUEST
            )
        except ValueError as exc:
            self.send_json({"message": str(exc)}, HTTPStatus.BAD_REQUEST)
        except RuntimeError as exc:
            self.send_json({"message": str(exc)}, HTTPStatus.CONFLICT)
        except TimeoutError as exc:
            self.send_json(
                {"message": f"Request timed out: {exc}"},
                HTTPStatus.GATEWAY_TIMEOUT,
            )
        except Exception as exc:
            if "timed out" in str(exc).lower():
                self.send_json(
                    {"message": f"Request timed out: {exc}"},
                    HTTPStatus.GATEWAY_TIMEOUT,
                )
                return
            self.send_json(
                {"message": f"Server handling failed: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def handle_get_config(self, query: str) -> None:
        try:
            params = parse_qs(query)
            source = params.get("source", ["current"])[0]
            if source == "example":
                config_path = settings.DEFAULT_CONFIG_PATH
            elif source == "current":
                raw_path = params.get("path", [None])[0]
                config_path = resolve_config_path(raw_path)
            else:
                raise ValueError("Config source must be 'current' or 'example'")

            with config_path.open("r", encoding="utf-8") as config_file:
                payload = json.load(config_file)
            self.send_json(
                {
                    "path": str(config_path),
                    "mtime_ns": str(config_path.stat().st_mtime_ns),
                    "config": payload,
                }
            )
        except Exception as exc:
            self.send_json(
                {"message": f"Failed to read config: {exc}"}, HTTPStatus.BAD_REQUEST
            )

    def handle_put_config(self) -> None:
        payload = self.read_json_body()
        validation = validate_config_payload(payload)
        if not validation["valid"]:
            self.send_json(
                {
                    "message": "Config validation failed before saving",
                    "errors": validation["errors"],
                },
                HTTPStatus.BAD_REQUEST,
            )
            return
        with settings.CONFIG_PATH.open("w", encoding="utf-8") as config_file:
            json.dump(payload, config_file, indent=4, ensure_ascii=False)
            config_file.write("\n")
        settings.CONFIG_PATH.chmod(0o600)
        self.send_json(
            {
                "message": "Config saved",
                "path": str(settings.CONFIG_PATH),
                "validation": validation,
            }
        )

    def handle_get_log_file(self, query: str) -> None:
        try:
            params = parse_qs(query)
            file_name = params.get("name", [""])[0]
            self.send_json(
                read_current_log_file(
                    file_name,
                    max_bytes=settings.MAX_STATUS_LOG_BYTES,
                )
            )
        except Exception as exc:
            self.send_json(
                {"message": f"Failed to read log: {exc}"}, HTTPStatus.BAD_REQUEST
            )

    def handle_log_download(self, query: str) -> None:
        try:
            params = parse_qs(query)
            file_name = params.get("name", [""])[0]
            download_name, content = build_log_download(file_name)
        except Exception as exc:
            self.send_json({"message": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self.send_binary(
            content,
            "text/plain; charset=utf-8",
            f'attachment; filename="{download_name}"',
        )

    def handle_run_status(self) -> None:
        self.send_json(
            {
                **get_run_state_snapshot(),
                "logs": {
                    name: read_current_log_file(
                        name,
                        max_bytes=settings.MAX_STATUS_LOG_BYTES,
                    )
                    for name in ("vap_log.txt", "vllm_deploy.log", "vllm_bench.log")
                },
            }
        )

    def handle_profile_archive(self, query: str) -> None:
        try:
            params = parse_qs(query)
            run_dir = params.get("run_dir", [None])[0]
            file_name, archive_path = create_profile_archive(run_dir)
        except ValueError as exc:
            self.send_json({"message": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        try:
            self.send_file(
                archive_path,
                "application/zip",
                f'attachment; filename="{file_name}"',
            )
        finally:
            archive_path.unlink(missing_ok=True)

    def handle_profile_trace(self, query: str) -> None:
        try:
            params = parse_qs(query)
            run_dir = params.get("run_dir", [None])[0]
            trace_info = resolve_latest_trace(
                {"preferred_name": "merged_trace", "run_dir": run_dir}
            )
            trace_path = Path(trace_info["trace_path"]).resolve()
            if not trace_path.is_relative_to(settings.LOGS_DIR.resolve()):
                raise ValueError("Trace path is outside VAP logs")
        except ValueError as exc:
            self.send_json({"message": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        content_type = (
            "application/gzip" if trace_path.suffix == ".gz" else "application/json"
        )
        self.send_file(
            trace_path,
            content_type,
            f'inline; filename="{trace_path.name}"',
        )

    def handle_tensorboard_proxy(self, parsed: Any) -> None:
        try:
            snapshot = get_run_state_snapshot()
            config_path = snapshot.get("config_path") or settings.CONFIG_PATH
            config_path = Path(config_path)
            config = VAPConfig.model_validate_json(
                config_path.read_text(encoding="utf-8")
            )
            port = config.profiler_cfg.tensorboard_port
            body = None
            if self.command in {"POST", "PUT", "PATCH"}:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else None
            path = parsed.path
            if parsed.query:
                path = f"{path}?{parsed.query}"
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower()
                not in {
                    "connection",
                    "host",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailers",
                    "transfer-encoding",
                    "upgrade",
                }
            }
            headers["Host"] = f"127.0.0.1:{port}"
            conn = HTTPConnection("127.0.0.1", port, timeout=30)
            conn.request(self.command, path, body=body, headers=headers)
            response = conn.getresponse()
            content = response.read()
        except ConnectionRefusedError:
            self.send_json(
                {"message": "TensorBoard is not running yet"},
                HTTPStatus.BAD_GATEWAY,
            )
            return
        except Exception as exc:
            self.send_json(
                {"message": f"TensorBoard proxy failed: {exc}"},
                HTTPStatus.BAD_GATEWAY,
            )
            return

        self.send_response(response.status)
        excluded_headers = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
        }
        for key, value in response.getheaders():
            if key.lower() in excluded_headers:
                continue
            if key.lower() == "content-length":
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(content)))
        self.send_auth_cookie_if_needed()
        self.end_headers()
        self.wfile.write(content)

    def handle_validate(self) -> None:
        payload = self.read_json_body()
        self.send_json(validate_config_payload(payload))

    def handle_agent_status(self) -> None:
        self.send_json(get_agent_status_payload())

    def handle_agent_unlock(self) -> None:
        payload = self.read_json_body()
        subscription_key = payload.get("subscription_key")
        if not isinstance(subscription_key, str):
            raise ValueError("subscription_key is required")
        self.send_json(
            {
                **get_agent_runtime().unlock(subscription_key),
                "server_session_id": settings.SERVER_SESSION_ID,
            }
        )

    def handle_agent_chat_stream(self) -> None:
        self.send_event_stream(get_agent_runtime().stream_chat(self.read_json_body()))

    def handle_agent_decision_stream(self) -> None:
        self.send_event_stream(
            get_agent_runtime().stream_decision(self.read_json_body())
        )

    def send_event_stream(self, events: Any) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def send(event: dict[str, Any]) -> None:
            self.wfile.write(
                f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
            )
            self.wfile.flush()

        try:
            for event in events:
                send(event)
        except (BrokenPipeError, ConnectionResetError):
            # The client went away; closing the generator stops the agent run.
            events.close()
        except Exception as exc:
            send({"type": "error", "message": str(exc)})

    def handle_analysis_layers(self) -> None:
        payload, report = analyze_run(self.read_json_body())
        self.send_json(
            {**payload, "report_markdown": report.read_text(encoding="utf-8")}
        )

    def handle_analysis_compare(self) -> None:
        payload, report = compare_two_runs(self.read_json_body())
        self.send_json(
            {**payload, "report_markdown": report.read_text(encoding="utf-8")}
        )

    def handle_analysis_report_stream(self) -> None:
        self.send_event_stream(report_stream(self.read_json_body()))

    def handle_attribution_file(self, query: str) -> None:
        try:
            params = parse_qs(query)
            path = attribution_file(
                params.get("run_dir", [None])[0], params.get("name", [None])[0]
            )
            content = path.read_bytes()
        except (OSError, ValueError) as exc:
            self.send_json({"message": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        disposition = "inline" if path.suffix == ".html" else "attachment"
        self.send_binary(
            content,
            ATTRIBUTION_FILE_TYPES[path.suffix],
            f'{disposition}; filename="{path.name}"',
            extra_headers={
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
            },
        )

    def handle_agent_approve(self) -> None:
        payload = self.read_json_body()
        approval_id = payload.get("approval_id")
        if not isinstance(approval_id, str):
            raise ValueError("approval_id is required")
        self.send_json(get_agent_runtime().approve(approval_id))

    def handle_agent_cancel_action(self) -> None:
        payload = self.read_json_body()
        approval_id = payload.get("approval_id")
        if not isinstance(approval_id, str):
            raise ValueError("approval_id is required")
        self.send_json(get_agent_runtime().cancel(approval_id))

    def handle_save_temp_config(self) -> None:
        payload = self.read_json_body()
        temp_path = save_temp_config(payload)
        validation = validate_config_payload(payload)
        self.send_json(
            {
                "path": str(temp_path),
                "file_name": temp_path.name,
                "validation": validation,
                "message": "Temporary config file generated.",
            }
        )

    def handle_check_ports(self) -> None:
        payload = self.read_json_body()
        self.send_json(check_config_ports(payload))

    def handle_check_machines(self) -> None:
        payload = self.read_json_body()
        self.send_json(check_config_machines(payload))

    def handle_check_resources(self) -> None:
        payload = self.read_json_body()
        self.send_json(check_config_resources(payload))

    def handle_check_model_resources(self) -> None:
        payload = self.read_json_body()
        self.send_json(check_config_model_resources(payload))

    def handle_check_docker_resources(self) -> None:
        payload = self.read_json_body()
        self.send_json(check_config_docker_resources(payload))

    def handle_check_worker_docker_resources(self) -> None:
        payload = self.read_json_body()
        self.send_json(check_config_worker_docker_resources(payload))

    def handle_check_container_resources(self) -> None:
        payload = self.read_json_body()
        self.send_json(check_config_container_resources(payload))

    def handle_check_clock_probe(self) -> None:
        payload = self.read_json_body()
        self.send_json(check_config_clock_probe(payload))

    def handle_run_start(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length > settings.MAX_JSON_BODY_BYTES:
            raise ValueError("Request body is too large")
        config_path = None
        if length > 0:
            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Run config must be a JSON object")
            validation = validate_config_payload(payload)
            if not validation["valid"]:
                self.send_json(
                    {
                        "message": "Run config validation failed",
                        "errors": validation["errors"],
                    },
                    HTTPStatus.BAD_REQUEST,
                )
                return
            config_path = save_temp_config(payload)
        self.send_json(start_vap_run(config_path))

    def handle_run_stop(self) -> None:
        self.send_json(stop_vap_run())

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > settings.MAX_JSON_BODY_BYTES:
            raise ValueError("Request body is too large")
        raw_body = self.rfile.read(length)
        payload = json.loads(raw_body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def serve_static(self, relative_path: str) -> None:
        static_path = (settings.STATIC_DIR / relative_path).resolve()
        if (
            not static_path.is_relative_to(settings.STATIC_DIR)
            or not static_path.is_file()
        ):
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content = static_path.read_bytes()
        content_type = "text/html; charset=utf-8"
        if static_path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif static_path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif static_path.suffix == ".svg":
            content_type = "image/svg+xml"

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        # The session token can appear in the page URL; keep it out of Referer headers.
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_auth_cookie_if_needed()
        self.end_headers()
        self.wfile.write(content)

    def send_json(
        self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_auth_cookie_if_needed()
        self.end_headers()
        self.wfile.write(content)

    def send_binary(
        self,
        content: bytes,
        content_type: str,
        content_disposition: str | None = None,
        status: HTTPStatus = HTTPStatus.OK,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        if content_disposition:
            self.send_header("Content-Disposition", content_disposition)
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.send_auth_cookie_if_needed()
        self.end_headers()
        self.wfile.write(content)

    def send_file(
        self,
        path: Path,
        content_type: str,
        content_disposition: str | None = None,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        if content_disposition:
            self.send_header("Content-Disposition", content_disposition)
        self.send_auth_cookie_if_needed()
        self.end_headers()
        with path.open("rb") as source:
            shutil.copyfileobj(source, self.wfile, length=1024 * 1024)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[VAP Config UI] {self.address_string()} - {fmt % args}")
