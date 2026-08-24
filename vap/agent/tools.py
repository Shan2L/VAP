from __future__ import annotations

import json
from typing import Any

from vap.agent.analysis import (
    inspect_latest_trace,
    load_skill_queries,
    object_schema,
    prepare_download_artifact,
    run_perfetto_sql,
    run_torchprofiler_skill,
    torchprofiler_skill_workflows,
)
from vap.agent.runtime import AgentTool, VAPAgentRuntime
from vap.server import settings
from vap.server.artifacts import (
    current_config_payload,
    read_current_log_file,
    save_temp_config,
)
from vap.server.checks import check_config_ports, check_config_resources
from vap.server.state import get_run_state_snapshot, start_vap_run, stop_vap_run
from vap.validation import validate_config_payload


def get_agent_runtime() -> VAPAgentRuntime:
    with settings.AGENT_TOOLS_LOCK:
        if not settings.AGENT_TOOLS_REGISTERED:
            register_vap_agent_tools(settings.AGENT_RUNTIME)
            settings.AGENT_TOOLS_REGISTERED = True
    return settings.AGENT_RUNTIME


def get_agent_status_payload() -> dict[str, Any]:
    return {
        **get_agent_runtime().status(),
        "server_session_id": settings.SERVER_SESSION_ID,
    }


def start_agent_run(args: dict[str, Any]) -> dict[str, Any]:
    payload = args.get("config")
    if not isinstance(payload, dict):
        return start_vap_run()

    validation = validate_config_payload(payload)
    if not validation["valid"]:
        raise ValueError(
            "Agent run config validation failed: "
            + json.dumps(validation["errors"], ensure_ascii=False)
        )
    return start_vap_run(save_temp_config(payload))


def register_vap_agent_tools(runtime: VAPAgentRuntime) -> None:
    runtime.register_tool(
        AgentTool(
            name="get_config",
            description="Read the saved VAP config payload.",
            safety="read_only",
            parameters=object_schema(),
            handler=lambda args: {"config": current_config_payload()},
        )
    )
    runtime.register_tool(
        AgentTool(
            name="get_run_status",
            description="Read current VAP run status without changing any process.",
            safety="read_only",
            parameters=object_schema(),
            handler=lambda args: get_run_state_snapshot(),
        )
    )
    runtime.register_tool(
        AgentTool(
            name="read_log_file",
            description="Read one current run log file.",
            safety="read_only",
            parameters=object_schema(
                {
                    "file_name": {
                        "type": "string",
                        "enum": ["vap_log.txt", "vllm_deploy.log", "vllm_bench.log"],
                    }
                },
                ["file_name"],
            ),
            handler=lambda args: read_current_log_file(str(args["file_name"])),
        )
    )
    runtime.register_tool(
        AgentTool(
            name="validate_config",
            description="Validate a VAP config payload. If config is omitted, validate the saved config.",
            safety="safe",
            parameters=object_schema({"config": {"type": "object"}}),
            handler=lambda args: validate_config_payload(
                args.get("config") or current_config_payload()
            ),
        )
    )
    runtime.register_tool(
        AgentTool(
            name="check_ports",
            description="Check local VAP service ports. If config is omitted, use the saved config.",
            safety="safe",
            parameters=object_schema({"config": {"type": "object"}}),
            handler=lambda args: check_config_ports(
                args.get("config") or current_config_payload()
            ),
        )
    )
    runtime.register_tool(
        AgentTool(
            name="check_resources",
            description="Check model paths, Docker image, devices, and mount sources.",
            safety="safe",
            parameters=object_schema({"config": {"type": "object"}}),
            handler=lambda args: check_config_resources(
                args.get("config") or current_config_payload()
            ),
        )
    )
    runtime.register_tool(
        AgentTool(
            name="inspect_latest_trace",
            description="Inspect the latest profiling trace metadata and a small preview. Defaults to merged_trace.",
            safety="read_only",
            parameters=object_schema(
                {
                    "preferred_name": {
                        "type": "string",
                        "description": "Preferred trace filename prefix. Defaults to merged_trace.",
                    },
                    "run_dir": {
                        "type": "string",
                        "description": "Optional run directory under logs.",
                    },
                }
            ),
            handler=inspect_latest_trace,
        )
    )
    runtime.register_tool(
        AgentTool(
            name="run_perfetto_sql",
            description="Run a whitelisted Perfetto SQL query on the latest trace. Use this for detailed trace analysis without loading raw JSON into the LLM.",
            safety="read_only",
            parameters=object_schema(
                {
                    "query_name": {
                        "type": "string",
                        "enum": sorted(load_skill_queries()),
                    },
                    "preferred_name": {
                        "type": "string",
                        "description": "Preferred trace filename prefix. Defaults to merged_trace.",
                    },
                    "run_dir": {
                        "type": "string",
                        "description": "Optional run directory under logs.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                ["query_name"],
            ),
            handler=run_perfetto_sql,
        )
    )
    runtime.register_tool(
        AgentTool(
            name="run_torchprofiler_skill",
            description="Run a TorchProfilerTraceSkill workflow made of VAP-owned Perfetto SQL presets.",
            safety="read_only",
            parameters=object_schema(
                {
                    "workflow": {
                        "type": "string",
                        "enum": sorted(torchprofiler_skill_workflows()),
                    },
                    "preferred_name": {
                        "type": "string",
                        "description": "Preferred trace filename prefix. Defaults to merged_trace.",
                    },
                    "run_dir": {
                        "type": "string",
                        "description": "Optional run directory under logs.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                ["workflow"],
            ),
            handler=run_torchprofiler_skill,
        )
    )
    runtime.register_tool(
        AgentTool(
            name="prepare_download_artifact",
            description="Prepare a safe download link for current run logs or the trace archive.",
            safety="safe",
            parameters=object_schema(
                {
                    "artifact": {
                        "type": "string",
                        "enum": [
                            "vap_log",
                            "vllm_deploy_log",
                            "vllm_bench_log",
                            "trace_archive",
                        ],
                    }
                },
                ["artifact"],
            ),
            handler=prepare_download_artifact,
        )
    )
    runtime.register_tool(
        AgentTool(
            name="start_run",
            description=(
                "Start a VAP run. Requires explicit user approval. "
                "profiler_cfg.torch_profiler_dir is immutable and must retain "
                "the value returned by get_config."
            ),
            safety="requires_approval",
            parameters=object_schema({"config": {"type": "object"}}),
            handler=start_agent_run,
        )
    )
    runtime.register_tool(
        AgentTool(
            name="stop_run",
            description="Stop the active VAP run. Requires explicit user approval.",
            safety="requires_approval",
            parameters=object_schema(),
            handler=lambda args: stop_vap_run(),
        )
    )
